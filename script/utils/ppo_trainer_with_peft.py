# PPO训练器模块 - 使用PEFT进行PPO强化学习训练的核心实现
#
# 本模块实现了PPO算法的完整训练流程，包括：
# 1. Actor-Critic模型的前向传播和反向传播
# 2. 经验数据的收集和处理
# 3. PPO损失函数的计算
# 4. 优势函数的计算和归一化
# 5. KL散度惩罚的实现
# 6. 多任务学习支持

import re
from typing import List, Tuple
import torch, os, sys, math, logging, random, time, warnings, shutil, copy
from tqdm import tqdm

# 添加上级目录到路径
sys.path.append("..")

# 导入Transformers相关模块
from transformers import Trainer,get_scheduler,PreTrainedModel,GenerationConfig
# 导入PyTorch相关模块
from torch.utils.data import DataLoader, RandomSampler
from accelerate import Accelerator
from torch.optim import AdamW,Adam
import torch.nn.functional as F
from pathlib import Path
# 导入PEFT相关模块
from peft import get_peft_model,get_peft_model_state_dict,PeftModel
import torch.nn as nn
from datasets import Dataset
from trl import AutoModelForCausalLMWithValueHead

# 设置日志记录器
logger = logging.getLogger(__name__)

# 常量定义
WEIGHTS_NAME = "adapter_model.bin"  # 模型权重文件名
TRAINING_ARGS_NAME = "training_args.bin"  # 训练参数文件名


class PPOModel(nn.Module):
    """
    PPO模型包装器 - 将actor模型和critic模型组合在一起

    这个类用于Accelerate分布式训练，它将actor和critic模型包装在一个模块中，
    以确保在分布式环境下的正确处理。

    参考: https://github.com/huggingface/accelerate/issues/668
    """
    def __init__(self, actor_model, critic_model):
        """
        初始化PPO模型

        Args:
            actor_model: Actor模型（策略网络）
            critic_model: Critic模型（价值网络）
        """
        super().__init__()
        self.actor_model = actor_model
        self.critic_model = critic_model

    def forward(self, sequences, extra_inputs=None):
        """
        前向传播

        Args:
            sequences: 包含input_ids和attention_mask的字典
            extra_inputs: 额外任务的输入（可选）

        Returns:
            actor_logits: Actor模型的输出logits
            critic_values: Critic模型的价值预测
            extra_loss: 额外任务的损失（如果有的话）
        """
        # sequences: dict containing input_ids and attention_mask
        # input_ids: [batch_size, seq_len] - token IDs
        # attention_mask: [batch_size, seq_len] - attention mask
        # Actor模型预测下一个token的概率分布
        # actor_logits: [batch_size, seq_len, vocab_size] - model output logits
        actor_logits = self.actor_model(**sequences, return_dict=True).logits
        # Critic模型预测每个位置的价值
        # critic_values: [batch_size, seq_len] - value predictions
        critic_values = self.critic_model(**sequences)[-1]

        # 处理额外任务的损失（用于多任务学习）
        if extra_inputs is not None:
            # extra_loss: [] - scalar extra task loss
            extra_loss = self.actor_model(**extra_inputs, return_dict=True).loss
        else:
            extra_loss = 0.0
        return actor_logits, critic_values, extra_loss
    
    

class PPOPeftTrainer(Trainer):
    def __init__(
        self, 
        args = None, 
        ppo_engine = None, 
        data_collator = None,
        train_dataset = None,
        tokenizer = None,
        extra_train_dataset = None,
        extra_data_collator = None, 
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        **kwargs
    ):
        self.args = args 
        if args.use_co_model:
            self.co_model = ppo_engine.model 
        else:
            self.actor_model = ppo_engine.actor_model 
            self.critic_model = ppo_engine.critic_model
            self.model = PPOModel(self.actor_model, self.critic_model)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision='fp16' if self.args.fp16 else None,
            log_with=self.args.report_to,
            
        )
        self.accelerator.init_trackers(
            project_name="ppo_train",
            config=self.args 
        )
        
        self.dataloader = DataLoader(
                                    train_dataset,
                                    batch_size=self.args.per_device_train_batch_size,
                                    collate_fn=data_collator,
                                    num_workers=self.args.dataloader_num_workers,
                                    shuffle=True,
                                    )
        self.dataloader = self.accelerator.prepare(self.dataloader)
        
        if extra_train_dataset is not None:
            self.extra_train_dataloader = DataLoader(
                extra_train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=extra_data_collator,
                num_workers=self.args.dataloader_num_workers,
                shuffle=True,
            )
            self.extra_train_dataloader = self.accelerator.prepare(self.extra_train_dataloader)
        else:
            self.extra_train_dataloader = None 
        
        self.tokenizer = tokenizer
        
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"
        self.device = self.accelerator.device

        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        if self.is_deepspeed_enabled:
            
            if not self.args.use_co_model:
                raise PermissionError(
                    "if you use deepspeed_plugin, you need to provide one model."
                )
                
            if getattr(self.args, "hf_deepspeed_config", None) is None:
                from transformers.deepspeed import HfTrainerDeepSpeedConfig
                ds_plugin = self.accelerator.state.deepspeed_plugin

                ds_plugin.hf_ds_config = HfTrainerDeepSpeedConfig(ds_plugin.hf_ds_config.config)
                ds_plugin.deepspeed_config = ds_plugin.hf_ds_config.config
                ds_plugin.hf_ds_config.trainer_config_process(self.args)


        ## get max_update_steps for lr_scheduler 
        if self.extra_train_dataloader is None:
            self.max_dataloader_iters = len(self.dataloader)
        else:
            self.max_dataloader_iters = min(len(self.dataloader), len(self.extra_train_dataloader))
        self.num_update_steps_per_epoch, self.max_update_steps = self.get_max_update_steps(args, self.max_dataloader_iters)
        
        
        ## create optimizer and scheduler 
        self.optimizer, self.lr_scheduler = optimizers
        if self.optimizer is None:
            self.optimizer = self.create_optimizer()
        
        if self.lr_scheduler is None:
            self.lr_scheduler = self.create_scheduler(self.optimizer, max_update_steps=self.max_update_steps)

        if self.args.use_co_model:
            self.co_model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.co_model, self.optimizer, self.lr_scheduler)
        else:
            self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)
            self.ator_model = self.accelerator.unwrap_model(self.model).actor_model 
            self.critic_model = self.accelerator.unwrap_model(self.model).critic_model
            

        
    def get_max_update_steps(self, args, dataloader_nums):
        num_update_steps_per_epoch = dataloader_nums * (args.per_device_train_batch_size / args.per_device_mini_train_batch_size) * args.ppo_epochs / args.gradient_accumulation_steps  
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        
        if args.max_steps > 0:
            max_update_steps = args.max_steps
        else:
            max_update_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        return num_update_steps_per_epoch, max_update_steps
        
        
    def get_parms(self, model, lr, weight_decay, eps=1e-8):
        params = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": weight_decay,
                "lr": lr,
                "eps": eps,
            }
        ]
        return params
    
    
    def create_optimizer(self):
        if self.args.use_co_model:
            params = self.get_parms(self.co_model, self.args.learning_rate, self.args.weight_decay)
        else:
            params = self.get_parms(self.actor_model, self.args.actor_lr, self.args.actor_weight_decay)
            params.extend(self.get_parms(self.critic_model, self.args.critic_lr, self.args.critic_weight_decay))

        optimizer = AdamW(params, betas=(0.9,0.95))
        
        return optimizer
    
    
    def create_scheduler(self, optimizer, max_update_steps):
        lr_scheduler = get_scheduler(self.args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=max_update_steps)
        return lr_scheduler


    def masked_mean(self, data, mask, dim=None, eps=1e-8):
        # data: tensor to compute mean on
        # mask: [batch_size, seq_len] - binary mask for valid elements
        data = data * mask
        if dim is not None:
            return data.sum(dim=dim) / (mask.sum(dim=dim) + eps)
        else:
            return data.sum() / (mask.sum() + eps) 
    
    def masked_var(self, data, mask, dim=None):
        # data: tensor to compute variance on
        # mask: [batch_size, seq_len] - binary mask for valid elements
        mean = self.masked_mean(data, mask, dim=dim)
        centered_values = data - mean
        var = self.masked_mean(centered_values**2, mask, dim=dim)
        return var


    def masked_whiten(self, data, mask, dim=None, shift_mean=True):
        # data: tensor to whiten
        # mask: [batch_size, seq_len] - binary mask for valid elements
        mean = data.sum() / mask.sum()
        var = torch.sum(((data - mean) ** 2).mul(mask)) / mask.sum()

        # whitened: tensor with same shape as data, but whitened values
        whitened = (data - mean) * torch.rsqrt(var + 1e-6)
        if not shift_mean:
            whitened += mean
        return whitened


    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        Recursively unwraps a model from potential containers (as used in distributed training).

        Args:
            model (`torch.nn.Module`): The model to unwrap.
        """
        # since there could be multiple levels of wrapping, unwrap recursively
        if hasattr(model, "module"):
            return self.unwrap_model(model.module)
        else:
            return model

    @torch.no_grad()
    def generate(
        self,
        prompts_ids,
        return_prompt: bool = True,
    ):
        # prompts_ids: [batch_size, prompt_len] - input prompt token IDs

        gen_kwargs = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "max_new_tokens": self.args.max_response_length,
            "min_new_tokens": self.args.min_response_length,
            "_from_model_config": False
        }

        if self.args.use_co_model:
            unwrapped_model = self.accelerator.unwrap_model(self.co_model)

            if self.unwrap_model(unwrapped_model) is not unwrapped_model:
                unwrapped_model = self.unwrap_model(unwrapped_model)

            if isinstance(unwrapped_model, AutoModelForCausalLMWithValueHead):
                unwrapped_model = unwrapped_model.pretrained_model

            if not hasattr(unwrapped_model, "generation_config"):
                raise AttributeError(
                    f" model object [{unwrapped_model.__class__.__name__}] has no attribute [generation_config] "
                )

            if unwrapped_model.generation_config._from_model_config:
                unwrapped_model.generation_config._from_model_config = False
            # sequences: [batch_size, prompt_len + response_len] - generated sequences
            sequences = unwrapped_model.generate(inputs=prompts_ids, **gen_kwargs)
        else:
            if self.actor_model.generation_config._from_model_config:
                self.actor_model.generation_config._from_model_config = False

            # sequences: [batch_size, prompt_len + response_len] - generated sequences
            sequences = self.actor_model.generate(inputs=prompts_ids, **gen_kwargs)

        if not return_prompt:
            # return: [batch_size, response_len] - response tokens only
            return sequences[:, prompts_ids.shape[1] :]

        # return: [batch_size, prompt_len + response_len] - full sequences
        return sequences


    def process_sequences(self, prompts_ids, responses_ids):
        # prompts_ids: [batch_size, max_len] - padded prompt sequences
        # responses_ids: [batch_size, max_len] - padded response sequences
        # seq: [0 0 0 0, prompt, response, 0 0 0 0] change to [prompt, response, 0 0 0 0]

        prompts_without_padding, responses_without_padding = [], []
        batch_size = prompts_ids.shape[0]
        for i in range(batch_size):
            # response: [max_len] - single response sequence
            response = responses_ids[i]
            # prompt: [max_len] - single prompt sequence
            prompt = prompts_ids[i]
            prompt_left_padding_length = (prompt == self.tokenizer.pad_token_id).sum().item()
            response_length = (response != self.tokenizer.pad_token_id).sum().item()
            # prompt_without_padding: [actual_prompt_len] - prompt without left padding
            prompt_without_padding = prompt[prompt_left_padding_length:]
            # response_without_padding: [actual_response_len] - response without right padding
            response_without_padding = response[:response_length]

            prompts_without_padding.append(prompt_without_padding.to(self.device))
            responses_without_padding.append(response_without_padding.to(self.device))

        # new_sequences: list of tensors - concatenated prompt+response sequences
        new_sequences = [torch.cat([q, r]) for q, r in zip(prompts_without_padding, responses_without_padding)]
        # sequences: [batch_size, max_seq_len] - padded concatenated sequences
        sequences = torch.nn.utils.rnn.pad_sequence(
            new_sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # sequences: dict containing processed sequences
        # input_ids: [batch_size, max_seq_len] - token IDs
        # attention_mask: [batch_size, max_seq_len] - attention mask
        sequences = dict(
            input_ids=sequences.to(self.device),
            attention_mask=sequences.ne(self.tokenizer.pad_token_id).long().to(self.device)
        )

        return prompts_without_padding, responses_without_padding, sequences
      
    
    def get_last_reward_score(self, values, responses_mask):
        # values: [batch_size, seq_len] - value predictions
        # responses_mask: [batch_size, seq_len] - mask for response tokens
        batch_size = values.shape[0]
        reward_score = []
        for i in range(batch_size):
            # value: [seq_len] - value predictions for a single sequence
            value = values[i]
            # end_index: int - index of the last response token
            end_index = responses_mask[i].nonzero()[-1].detach().item()
            reward_score.append(value[end_index])

        # rewards_score: [batch_size] - last token values for each sequence
        rewards_score = torch.stack(reward_score)

        return rewards_score
    
    
    def get_log_probs(self, logits, labels):
        # logits: [batch_size, seq_len, vocab_size] - model output logits
        # labels: [batch_size, seq_len] - target token IDs
        log_probs = F.log_softmax(logits, dim=-1)  # log_probs: [batch_size, seq_len, vocab_size]
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))  # [batch_size, seq_len, 1]
        return log_probs_labels.squeeze(-1)  # [batch_size, seq_len]


    def get_entropy(self, logits, mask):
        # logits: [batch_size, seq_len, vocab_size] - model output logits
        # mask: [batch_size, seq_len] - attention mask for valid tokens
        probs = torch.nn.functional.softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        # entropy: [] - scalar entropy value (masked mean)
        entropy = self.masked_mean(-torch.sum(probs * log_probs, dim=-1), mask)
        return entropy 
        
    
    def compute_rewards_with_kl_penalty(self, ref_values, actor_log_probs, ref_log_probs, responses_mask):
        # ref_values: [batch_size, seq_len] - reference model value predictions
        # actor_log_probs: [batch_size, seq_len] - current policy log probabilities
        # ref_log_probs: [batch_size, seq_len] - reference policy log probabilities
        # responses_mask: [batch_size, seq_len] - mask for response tokens
        masks = responses_mask[:, 1:]  # [batch_size, seq_len-1]
        # rewards_score: [batch_size] - reward scores for each sequence
        rewards_score = self.get_last_reward_score(ref_values, responses_mask)

        batch_size = rewards_score.shape[0]
        rewards_with_kl_penalty, kl_penalty_all = [], []
        for i in range(batch_size):
            mask = masks[i]

            # kl: [seq_len-1] - KL divergence between policies
            kl = actor_log_probs[i] - ref_log_probs[i]
            if self.args.kl_penalty_method == 'abs':
                kl = torch.abs(kl)
            elif self.args.kl_penalty_method == 'mse':
                kl = kl ** 2 * 0.5

            # kl_penalty: [seq_len-1] - KL penalty term
            kl_penalty = - self.args.kl_penalty_beta * kl
            kl_penalty_all.append(kl_penalty)

            if self.args.reward_score_clip is not None:
                rewards_score[i] = torch.clamp(rewards_score[i], -self.args.reward_score_clip, self.args.reward_score_clip)

            end_index = mask.nonzero()[-1].detach().item()
            kl_penalty[end_index] += rewards_score[i]

            rewards_with_kl_penalty.append(kl_penalty)
        # returns:
        # rewards_with_kl_penalty: [batch_size, seq_len-1] - rewards with KL penalty
        # kl_penalty_all: [batch_size, seq_len-1] - KL penalties only
        # rewards_score: [batch_size] - original reward scores
        return torch.stack(rewards_with_kl_penalty), torch.stack(kl_penalty_all), rewards_score 
    
    
    def get_advantages_and_returns(self, values, rewards, responses_mask):
        # Adopted from https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat
        # values: [batch_size, seq_len] - critic value predictions
        # rewards: [batch_size, seq_len] - reward signals
        # responses_mask: [batch_size, seq_len] - mask for response tokens
        masks = responses_mask[:, 1:]  # [batch_size, seq_len-1]

        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]

        for t in reversed(range(length)):
            # nextvalues: [batch_size] - next timestep values (or 0 for last timestep)
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            # delta: [batch_size] - TD error
            delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]
            # lastgaelam: [batch_size] - GAE advantage estimate
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        # advantages: [batch_size, seq_len] - GDE advantages
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        # returns: [batch_size, seq_len] - returns (advantages + values)
        returns = advantages + values

        if self.args.use_advantage_norm:
            advantages = self.masked_whiten(advantages, masks)

        return advantages.detach(), returns


    def get_responses_mask(self, sequences_mask, prompts_without_padding):
        # sequences_mask: [batch_size, seq_len] - mask for full sequences
        # prompts_without_padding: list of tensors - prompt tokens without padding
        batch_size = sequences_mask.shape[0]
        responses_mask = []
        for i in range(batch_size):
            # prompt: tensor - prompt tokens without padding
            prompt = prompts_without_padding[i]
            # response_mask: [seq_len] - mask for response tokens only
            response_mask = torch.zeros_like(sequences_mask[i])
            response_mask[len(prompt):] = sequences_mask[i][len(prompt):]
            responses_mask.append(response_mask)
        # return: [batch_size, seq_len] - masks for response tokens only
        return torch.stack(responses_mask)



    @torch.no_grad()
    def get_co_model_output(self, sequences):
        # sequences: dict containing input_ids and attention_mask
        # input_ids: [batch_size, seq_len] - token IDs
        # attention_mask: [batch_size, seq_len] - attention mask
        unwrap_model = self.accelerator.unwrap_model(self.co_model)
        # actor_logits: [batch_size, seq_len, vocab_size] - current policy logits
        # critic_values: [batch_size, seq_len] - critic value predictions
        actor_logits, _, critic_values = unwrap_model(**sequences, return_dict=True)

        if self.args.use_multi_adapters:
            unwrap_model.pretrained_model.set_adapter("critic")
            # critic_values: [batch_size, seq_len] - updated critic values with critic adapter
            critic_values = unwrap_model(**sequences, return_dict=True)[-1]
            unwrap_model.pretrained_model.set_adapter("default")

        with unwrap_model.pretrained_model.disable_adapter():
            ## the same as sft model
            # ref_logits: [batch_size, seq_len, vocab_size] - reference policy logits
            ref_logits, _, _ = unwrap_model(**sequences, return_dict=True)

            ## the same as reward model
            ## save current critic model v_head
            v_head_stat_dict = unwrap_model.v_head.state_dict()
            setattr(unwrap_model, "critic_head_weight", v_head_stat_dict["summary.weight"])
            setattr(unwrap_model, "critic_head_bias", v_head_stat_dict["summary.bias"])
            ## change to reward model v_head
            unwrap_model.v_head.load_state_dict({"summary.weight": getattr(unwrap_model, "reward_head_weight"), "summary.bias": getattr(unwrap_model, "reward_head_bias")})

            # ref_values: [batch_size, seq_len] - reference model value predictions
            ref_values = unwrap_model(**sequences)[-1]
            ## back to critic model v_head
            unwrap_model.v_head.load_state_dict({"summary.weight": getattr(unwrap_model, "critic_head_weight"), "summary.bias": getattr(unwrap_model, "critic_head_bias")})

        return actor_logits, critic_values, ref_logits, ref_values
        
        
        
    @torch.no_grad()
    def get_model_output(self, sequences):
        # sequences: dict containing input_ids and attention_mask
        # input_ids: [batch_size, seq_len] - token IDs
        # attention_mask: [batch_size, seq_len] - attention mask
        # actor_logits: [batch_size, seq_len, vocab_size] - current policy logits
        # critic_values: [batch_size, seq_len] - critic value predictions
        actor_logits, critic_values, _ = self.model(sequences)
        with self.actor_model.disable_adapter(): # 临时屏蔽掉 LoRA 层的权重, Actor 模型瞬间变回了训练前的基座模型（也就是 SFT 模型）
            ## the same as sft model
            # ref_logits: [batch_size, seq_len, vocab_size] - reference policy logits
            ref_logits = self.actor_model(**sequences, return_dict=True).logits

        with self.critic_model.pretrained_model.disable_adapter():
            ### Critic 模型和 Reward Model 共享同一个基座（Backbone），只是最后一层线性层（Head）不同
            ## the same as reward model
            ## save current critic model v_head
            v_head_stat_dict = self.critic_model.v_head.state_dict()
            ## 把当前 Critic 模型正在训练的那个“头”（v_head）的权重保存下来，暂存在内存里
            setattr(self.critic_model, "critic_head_weight", v_head_stat_dict["summary.weight"])
            setattr(self.critic_model, "critic_head_bias", v_head_stat_dict["summary.bias"])
            ## change to reward model v_head
            ## 把预先加载好的 Reward Model 的头（权重和偏置）强行加载到当前模型的 v_head
            self.critic_model.v_head.load_state_dict({"summary.weight": getattr(self.critic_model, "reward_head_weight"), "summary.bias": getattr(self.critic_model, "reward_head_bias")})

            ## ref_values: [batch_size, seq_len] - reference model value predictions
            ## 用这个临时拼凑出来的 Reward Model 进行一次前向传播，得到了该序列的真实奖励分数
            ref_values = self.critic_model(**sequences)[-1]
            ## back to critic model v_head
            ## 用完 Reward Head 后，必须立刻把之前保存的 Critic Head 装回去，并在退出 with 语句块后自动恢复 LoRA adapter
            self.critic_model.v_head.load_state_dict({"summary.weight": getattr(self.critic_model, "critic_head_weight"), "summary.bias": getattr(self.critic_model, "critic_head_bias")})

        return actor_logits, critic_values, ref_logits, ref_values
        
        
    def get_experience_data(self, prompts_ids):
        # prompts_ids: [batch_size, prompt_len] - input prompt token IDs

        # responses_ids: [batch_size, response_len] - generated response tokens
        responses_ids = self.generate(prompts_ids, return_prompt=False)
        prompts_without_padding, responses_without_padding, sequences = self.process_sequences(prompts_ids, responses_ids)

        ### 不同进程填充
        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            # input_ids: [batch_size, max_seq_len] - padded across processes
            sequences["input_ids"] = self.accelerator.pad_across_processes(
                sequences["input_ids"], dim=1, pad_index=self.tokenizer.pad_token_id, pad_first=pad_first
            )
            # attention_mask: [batch_size, max_seq_len] - padded across processes
            sequences["attention_mask"] = self.accelerator.pad_across_processes(
                sequences["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )

        if self.args.use_co_model:
            # actor_logits: [batch_size, seq_len, vocab_size] - current policy logits
            # critic_values: [batch_size, seq_len] - critic value predictions
            # ref_logits: [batch_size, seq_len, vocab_size] - reference policy logits
            # ref_values: [batch_size, seq_len] - reference value predictions
            actor_logits, critic_values, ref_logits, ref_values = self.get_co_model_output(sequences)
        else:
            actor_logits, critic_values, ref_logits, ref_values = self.get_model_output(sequences)

        # \pi_{\theta} (a|s)
        # actor_log_probs: [batch_size, seq_len-1] - current policy log probabilities
        actor_log_probs = self.get_log_probs(actor_logits[:, :-1, :], sequences["input_ids"][:, 1:])
        # actor_ce_loss: [batch_size] - actor cross-entropy loss
        actor_ce_loss = -self.masked_mean(actor_log_probs, sequences["attention_mask"][:, 1:], dim=-1)

        # \pi_{ref} (a|s)
        # ref_log_probs: [batch_size, seq_len-1] - reference policy log probabilities
        ref_log_probs = self.get_log_probs(ref_logits[:, :-1, :], sequences["input_ids"][:, 1:])
        # ref_ce_loss: [batch_size] - reference cross-entropy loss
        ref_ce_loss = -self.masked_mean(ref_log_probs, sequences["attention_mask"][:, 1:], dim=-1)

        # responses_mask: [batch_size, seq_len] - mask for response tokens only
        responses_mask = self.get_responses_mask(sequences["attention_mask"], prompts_without_padding).to(self.device)

        # R(s,a)
        # rewards_with_kl_penalty: [batch_size, seq_len-1] - rewards with KL penalty
        # kl_penalty: [batch_size, seq_len-1] - KL penalties only
        # rewards_score: [batch_size] - original reward scores
        rewards_with_kl_penalty, kl_penalty, rewards_score = self.compute_rewards_with_kl_penalty(ref_values, actor_log_probs, ref_log_probs, responses_mask)

        # V_{\phi}(s)
        # critic_values: [batch_size, seq_len-1] - masked critic values
        critic_values = critic_values[:, :-1] * responses_mask[:, 1:]
        # rewards_with_kl_penalty: [batch_size, seq_len-1] - masked rewards
        rewards_with_kl_penalty = rewards_with_kl_penalty * responses_mask[:, 1:]

        # GAE (Generalized Advantage Estimation)
        # advantages: [batch_size, seq_len-1] - GAE advantages
        # returns: [batch_size, seq_len-1] - returns
        advantages, returns = self.get_advantages_and_returns(critic_values, rewards_with_kl_penalty, responses_mask)

        return dict(
            prompts_ids=prompts_without_padding,           # list of tensors - prompts without padding
            responses_ids=responses_without_padding,         # list of tensors - responses without padding
            responses_mask=responses_mask,                   # [batch_size, seq_len] - response masks
            sequences_ids=sequences["input_ids"],           # [batch_size, seq_len] - sequence token IDs
            sequences_mask=sequences["attention_mask"],     # [batch_size, seq_len] - sequence attention masks
            actor_log_probs=actor_log_probs,                 # [batch_size, seq_len-1] - current policy log probs
            ref_log_probs=ref_log_probs,                     # [batch_size, seq_len-1] - reference policy log probs
            rewards_with_kl_penalty=rewards_with_kl_penalty, # [batch_size, seq_len-1] - rewards with KL penalty
            rewards_score=rewards_score,                     # [batch_size] - reward scores
            kl_penalty=kl_penalty,                           # [batch_size, seq_len-1] - KL penalties
            critic_values=critic_values,                     # [batch_size, seq_len-1] - critic values
            advantages=advantages,                           # [batch_size, seq_len-1] - advantages
            returns=returns,                                 # [batch_size, seq_len-1] - returns
            actor_ce_loss=actor_ce_loss,                     # [batch_size] - actor cross-entropy loss
            ref_ce_loss=ref_ce_loss,                         # [batch_size] - reference cross-entropy loss
        )


    def get_mini_dataset(self, data_buffer):
        # data_buffer: list of dicts containing experience data
        # Each dict contains 'exp' (experience_data) and 'extra' (batch_extra_data)

        mini_dataset = []
        # batch_size: int - size of the experience batches
        batch_size = data_buffer[0]["exp"]["sequences_ids"].shape[0]
        for item in data_buffer:
            # experience_data: dict - contains all PPO experience tensors
            # batch_extra_data: dict or None - contains extra task data
            experience_data, batch_extra_data = item['exp'], item['extra']
            index = 0
            while index < batch_size:
                dic = {}
                for k, v in experience_data.items():
                    if k in ["prompts_ids", "responses_ids"]:
                        # dic[k]: list of tensors - sliced prompts/responses
                        dic[k] = v[index : index + self.args.per_device_mini_train_batch_size]
                    else:
                        # dic[k]: tensor - sliced experience data moved to device
                        # Shape depends on key: e.g., [mini_batch_size, seq_len, ...]
                        dic[k] = v[index : index + self.args.per_device_mini_train_batch_size].to(self.device)

                if batch_extra_data is not None:
                    for k, v in batch_extra_data.items():
                        # dic[k]: tensor - sliced extra task data moved to device
                        # Shape depends on key: e.g., [mini_batch_size, ...]
                        dic[k] = v[index : index + self.args.per_device_mini_train_batch_size].to(self.device)

                mini_dataset.append(dic)
                index += self.args.per_device_mini_train_batch_size

        return mini_dataset 
        
        
    def actor_loss(self, actor_log_probs, mini_batch_actor_log_probs, advantages, mask):
        # actor_log_probs: [batch_size, seq_len] - log probabilities from old policy
        # mini_batch_actor_log_probs: [batch_size, seq_len] - log probabilities from current policy
        # advantages: [batch_size, seq_len] - advantage estimates
        # mask: [batch_size, seq_len] - attention mask
        ratio = torch.exp((mini_batch_actor_log_probs - actor_log_probs) * mask)  # [batch_size, seq_len]
        loss1 = -advantages * ratio  # [batch_size, seq_len]
        loss2 = -advantages * torch.clamp(ratio, 1.0 - self.args.ratio_clip,
                                             1.0 + self.args.ratio_clip)  # [batch_size, seq_len]

        # loss: [] - scalar actor loss (masked mean)
        loss = self.masked_mean(torch.max(loss1, loss2), mask)
        return loss, ratio 


    def critic_loss(self, critic_values, mini_batch_critic_values, returns, mask):
        # critic_values: [batch_size, seq_len] - value predictions from old critic
        # mini_batch_critic_values: [batch_size, seq_len] - value predictions from current critic
        # returns: [batch_size, seq_len] - target returns
        # mask: [batch_size, seq_len] - attention mask
        critic_values_clip = torch.clamp(
            mini_batch_critic_values,
            critic_values - self.args.value_clip,
            critic_values + self.args.value_clip,
        )  # [batch_size, seq_len]
        values_error = (mini_batch_critic_values - returns)**2  # [batch_size, seq_len]
        values_clip_error = (critic_values_clip - returns)**2  # [batch_size, seq_len]
        # loss: [] - scalar critic loss (masked mean)
        loss = 0.5 * self.masked_mean(torch.max(values_error, values_clip_error), mask)

        return loss, values_error 
    

    def get_state_dict(self, model):
        pretrained_model_state_dict = model.pretrained_model.state_dict()
        v_head_state_dict = model.v_head.state_dict()
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict 


    def save_checkpoint(self, model, output_dir, step, adapter_name="default", state_dict=None):

        if self.unwrap_model(model) is not model:
            model = self.unwrap_model(model)
            
        output_dir = os.path.join(output_dir, f"checkpoint-{step}")
        logger.info(f"Saving model checkpoint to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            if hasattr(model, "v_head"):
                state_dict = self.get_state_dict(model)
            else:
                state_dict = model.state_dict()

        if isinstance(model, PreTrainedModel):  
            model.save_pretrained(output_dir, state_dict=state_dict)
        else:
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            if hasattr(model, "peft_config"):
                adapter_state_dict = get_peft_model_state_dict(model, state_dict, adapter_name=adapter_name)
            elif isinstance(model, AutoModelForCausalLMWithValueHead):
                adapter_state_dict = get_peft_model_state_dict(model.pretrained_model, state_dict, adapter_name=adapter_name)

            if hasattr(model, "v_head"):
                ### add v_head (v_head not in modules_to_save)
                v_head_state_dict = model.v_head.state_dict()
                for k, v in v_head_state_dict.items():
                    adapter_state_dict[f"v_head.{k}"] = v 
            torch.save(adapter_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                
        try:
            if hasattr(model, "peft_config"):
                model.peft_config.save_pretrained(output_dir)
            elif isinstance(model, AutoModelForCausalLMWithValueHead):
                model.pretrained_model.peft_config.save_pretrained(output_dir)

        except AttributeError:
            if hasattr(model, "peft_config"):
                model.peft_config[adapter_name].save_pretrained(output_dir)
            else:
                model.pretrained_model.peft_config[adapter_name].save_pretrained(output_dir)


        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    
    def record_logs(self, batch):
        # batch: dict containing all PPO training data and computed losses

        # mask: [batch_size, seq_len-1] - response mask (excluding first token)
        mask = batch["responses_mask"][:, 1:]
        # prompt_lens: [batch_size] - lengths of prompts
        prompt_lens = torch.tensor([len(prompt) for prompt in batch["prompts_ids"]], dtype=torch.float)
        # response_lens: [batch_size] - lengths of responses
        response_lens = torch.tensor([len(response) for response in batch["responses_ids"]], dtype=torch.float)

        logs = dict()
        ## params
        logs["lr"] = self.optimizer.param_groups[0]['lr']

        ## loss
        # batch["actor_loss"]: [] - scalar actor loss
        logs["loss/actor"] = batch["actor_loss"]
        # batch["entropy"]: [] - scalar entropy
        logs["loss/entropy"] = batch["entropy"]
        # batch["critic_loss"]: [] - scalar critic loss
        logs["loss/critic"] = batch["critic_loss"]
        # batch["extra_loss"]: [] - scalar extra loss
        logs["loss/extra"] = batch["extra_loss"]

        # batch["rewards_score"]: [batch_size] - reward scores
        logs["exp_data/reward_score_mean"] = torch.mean(batch["rewards_score"])
        logs["exp_data/reward_score_var"] = torch.var(batch["rewards_score"])

        # batch["kl_penalty"]: [batch_size, seq_len-1] - KL penalties
        logs["exp_data/kl_penalty_mean"] = self.masked_mean(batch["kl_penalty"], mask)
        logs["exp_data/kl_penalty_var"] = self.masked_var(batch["kl_penalty"], mask)

        # batch["rewards_with_kl_penalty"]: [batch_size, seq_len-1] - rewards with KL penalty
        logs["exp_data/rewards_with_kl_penalty_mean"] = self.masked_mean(batch["rewards_with_kl_penalty"], mask)
        logs["exp_data/rewards_with_kl_penalty_var"] = self.masked_var(batch["rewards_with_kl_penalty"], mask)

        # batch["actor_ce_loss"]: [batch_size] - actor cross-entropy loss
        logs["exp_data/actor_perplexity"] = math.exp(torch.mean(batch["actor_ce_loss"]))
        # batch["ref_ce_loss"]: [batch_size] - reference cross-entropy loss
        logs["exp_data/ref_perplexity"] = math.exp(torch.mean(batch["ref_ce_loss"]))

        ## actor
        # batch["advantages"]: [batch_size, seq_len-1] - advantages
        logs["actor/advantages_mean"] = self.masked_mean(batch["advantages"], mask)
        logs["actor/advantages_var"] = self.masked_var(batch["advantages"], mask)

        # batch["ratio"]: [batch_size, seq_len-1] - probability ratios
        logs["actor/ratio_mean"] = self.masked_mean(batch["ratio"], mask)
        logs["actor/ratio_var"] = self.masked_var(batch["ratio"], mask)

        ## critic
        # batch["returns"]: [batch_size, seq_len-1] - returns
        logs["critic/returns_mean"] = self.masked_mean(batch["returns"], mask)
        logs["critic/returns_var"] = self.masked_var(batch["returns"], mask)

        # batch["values_error"]: [batch_size, seq_len-1] - value prediction errors
        logs["critic/values_error_mean"] = self.masked_mean(batch["values_error"], mask)
        logs["critic/values_error_var"] = self.masked_var(batch["values_error"], mask)

        ## length
        # prompt_lens: [batch_size] - prompt lengths
        logs["length/prompts_length_mean"] = torch.mean(prompt_lens)
        logs["length/prompts_length_var"] = torch.var(prompt_lens)

        # response_lens: [batch_size] - response lengths
        logs["length/responses_length_mean"] = torch.mean(response_lens)
        logs["length/responses_length_var"] = torch.var(response_lens)

        return logs


    def print_logs(self, all_logs, update_steps):
        # all_logs: list of dicts - logs from multiple training steps
        # update_steps: int - current update step count

        # all_logs_merged: dict - merged logs with tensor values
        all_logs_merged = {}
        for key in all_logs[0]:
            # Create tensor from log values and move to device
            # tensor shape: [num_logs] -> scalar after mean
            all_logs_merged[key] = torch.mean(torch.tensor([log[key] for log in all_logs])).to(self.device)

        if self.is_distributed:
            logs = {}
            torch.distributed.barrier()
            for k, v in all_logs_merged.items():
                if not isinstance(v, torch.Tensor):
                    warnings.warn(f"the log of {k} need to be tensors")
                    continue
                # v: tensor - log value to be reduced across processes
                torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
                v /= self.accelerator.num_processes
                logs[k] = v
            all_logs_merged = copy.deepcopy(logs)

        if self.accelerator.is_main_process:
            logs = {}
            for k, v in all_logs_merged.items():
                # Convert tensor to numpy scalar for logging
                # v: tensor -> logs[k]: float
                logs[k] = v.cpu().numpy().item()
            self.accelerator.log(logs, step=int(update_steps))

            if update_steps > 0 and update_steps % self.args.logging_steps == 0:
                # Extract scalar values for printing
                actor_loss = logs["loss/actor"]
                critic_loss = logs["loss/critic"]
                extra_loss = logs["loss/extra"]
                rewards_with_kl_penalty_mean = logs["exp_data/rewards_with_kl_penalty_mean"]
                lr = logs["lr"]
                print(f'update_steps:{update_steps}|lr:{lr}|actor_loss:{actor_loss}, critic_loss:{critic_loss}, extra_loss:{extra_loss}, rewards_with_kl_penalty_mean:{rewards_with_kl_penalty_mean}')
    
    
    def train_step(self, batch_mini_data, extra_inputs, step):

        extra_loss_weight_warmup = self.args.extra_loss_weight
        if self.args.extra_warmup_steps_ratio is not None:
            extra_warmup_steps = int(self.args.extra_warmup_steps_ratio * self.max_steps)
        ## get extra_loss_weight
        if self.args.extra_warmup_steps_ratio is not None:
            if step < extra_warmup_steps:
                extra_loss_weight_warmup = step / extra_warmup_steps * self.args.extra_loss_weight
            else:
                extra_loss_weight_warmup = extra_loss_weight_warmup ** 1.001

        # responses_mask: [batch_size, seq_len] - mask for response tokens only
        responses_mask = batch_mini_data["responses_mask"]
        # sequences: dict containing input_ids and attention_mask
        # input_ids: [batch_size, seq_len] - token IDs for the full sequences
        # attention_mask: [batch_size, seq_len] - attention mask for the full sequences
        sequences = {"input_ids": batch_mini_data["sequences_ids"], "attention_mask": batch_mini_data["sequences_mask"]}

        if self.args.use_co_model:
            with self.accelerator.accumulate(self.co_model):
                unwrap_model = self.accelerator.unwrap_model(self.co_model)
                # mini_batch_actor_logits: [batch_size, seq_len, vocab_size] - model output logits
                # mini_batch_critic_values: [batch_size, seq_len] - value predictions
                mini_batch_actor_logits, _, mini_batch_critic_values = unwrap_model(**sequences, return_dict=True)
                # extra_loss: [] - scalar extra task loss
                _, extra_loss, _ = unwrap_model(**extra_inputs, return_dict=True)

                if self.args.use_multi_adapters:
                    unwrap_model.pretrained_model.set_adapter("critic")
                    # mini_batch_critic_values: [batch_size, seq_len] - updated value predictions with critic adapter
                    mini_batch_critic_values = unwrap_model(**sequences, return_dict=True)[-1]
                    unwrap_model.pretrained_model.set_adapter("default")
        else:
            with self.accelerator.accumulate(self.model):
                # mini_batch_actor_logits: [batch_size, seq_len, vocab_size] - model output logits
                # mini_batch_critic_values: [batch_size, seq_len] - value predictions
                # extra_loss: [] - scalar extra task loss
                mini_batch_actor_logits, mini_batch_critic_values, extra_loss = self.model(sequences, extra_inputs)

        # mini_batch_actor_log_probs: [batch_size, seq_len-1] - log probabilities of actions
        mini_batch_actor_log_probs = self.get_log_probs(mini_batch_actor_logits[:, :-1, :], batch_mini_data["sequences_ids"][:, 1:])
        # entropy: [] - scalar entropy value
        entropy = self.get_entropy(mini_batch_actor_logits[:, :-1, :], responses_mask[:, 1:])

        # actor_loss: [] - scalar actor loss
        # ratio: [batch_size, seq_len-1] - probability ratio between current and old policy
        actor_loss, ratio = self.actor_loss(batch_mini_data["actor_log_probs"], mini_batch_actor_log_probs, batch_mini_data["advantages"], responses_mask[:, 1:])

        # critic_loss: [] - scalar critic loss
        # values_error: [batch_size, seq_len-1] - squared error between predicted and target values
        critic_loss, values_error = self.critic_loss(batch_mini_data["critic_values"], mini_batch_critic_values[:, :-1], batch_mini_data["returns"], responses_mask[:, 1:])
  
        if extra_inputs is not None:
            # loss: [] - scalar total loss combining all components
            loss = self.args.actor_loss_weight * actor_loss + self.args.entropy_beta * entropy + self.args.critic_loss_weight * critic_loss + extra_loss_weight_warmup * extra_loss
        else:
            # loss: [] - scalar total loss combining main components
            loss = self.args.actor_loss_weight * actor_loss + self.args.entropy_beta * entropy + self.args.critic_loss_weight * critic_loss

        self.accelerator.backward(loss)

        if self.args.max_grad_norm is not None:
            if self.args.use_co_model:
                # params: list of trainable parameters from the co-model
                params = [p for n, p in self.co_model.named_parameters() if p.requires_grad]
            else:
                # params: concatenated list of trainable parameters from actor and critic models
                params = [p for n, p in self.actor_model.named_parameters() if p.requires_grad] + [p for n, p in self.critic_model.named_parameters() if p.requires_grad]

            torch.nn.utils.clip_grad_norm_(
                parameters=params,
                max_norm=self.args.max_grad_norm
            )

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return dict(
            # all_loss: [] - scalar detached total loss
            all_loss=loss.detach(),
            # actor_loss: [] - scalar detached actor loss
            actor_loss=actor_loss.detach(),
            # critic_loss: [] - scalar detached critic loss
            critic_loss=critic_loss.detach(),
            # extra_loss: [] - scalar detached extra loss (or 0.0 if no extra inputs)
            extra_loss=extra_loss.detach() if extra_inputs is not None else 0.0,
            # entropy: [] - scalar detached entropy
            entropy=entropy.detach(),
            # ratio: [batch_size, seq_len-1] - detached probability ratios
            ratio=ratio.detach(),
            # values_error: [batch_size, seq_len-1] - detached value prediction errors
            values_error=values_error.detach(),

        )
        
        
    def train(self):

        total_train_batch_size = (
            self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        )
        num_examples = self.num_examples(self.dataloader)
        if self.extra_train_dataloader is not None:
            extra_data_num_examples = self.num_examples(self.extra_train_dataloader)
        else:
            extra_data_num_examples = 0

        if self.args.max_steps > 0:
            self.num_train_epochs = self.args.max_steps // self.num_update_steps_per_epoch + int(
                self.args.max_steps % self.num_update_steps_per_epoch > 0
            )
            self.max_steps = self.max_update_steps * self.args.gradient_accumulation_steps
        else:
            self.num_train_epochs = math.ceil(self.args.num_train_epochs)
            self.max_steps = self.max_update_steps * self.args.gradient_accumulation_steps

        if self.is_world_process_zero():
            # Train!
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}, Extra task examples = {extra_data_num_examples}")
            logger.info(f"  Num Epochs = {self.num_train_epochs:,}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total steps = {self.max_steps}, Total optimization steps = {self.max_update_steps}")


        progress_bar = tqdm(total=self.max_steps, disable=not self.is_world_process_zero())
        step = 0
        # data_buffer: list - buffer for experience data
        data_buffer = list()
        # all_logs: list - buffer for logging data
        all_logs = list()
        for epoch in range(int(self.num_train_epochs)):
            if self.extra_train_dataloader is None:
                self.extra_train_dataloader = [None] * len(self.dataloader)

            for i, (batch_data, batch_extra_data) in enumerate(zip(self.dataloader, self.extra_train_dataloader)):
                if i >= self.max_dataloader_iters:
                    break

                # prompts_ids: [batch_size, prompt_len] - input prompt token IDs
                prompts_ids = batch_data["input_ids"]
                # experience_data: dict - contains all PPO experience tensors
                experience_data = self.get_experience_data(prompts_ids)

                self.accelerator.wait_for_everyone()
                data_buffer.append({'exp': experience_data, 'extra': batch_extra_data})
                if len(data_buffer) == self.args.mini_data_buffer_nums:
                    # mini_dataset: list - list of mini-batch dictionaries
                    mini_dataset = self.get_mini_dataset(data_buffer)
                    random.shuffle(mini_dataset)
                    data_buffer.clear()

                    for ppo_epoch in range(self.args.ppo_epochs):

                        for j, batch_mini_data in enumerate(mini_dataset):
                            step += 1

                            if batch_extra_data is not None:
                                # extra_inputs: dict - extra task inputs for multi-task learning
                                extra_inputs = {"input_ids": batch_mini_data["input_ids"], "labels": batch_mini_data["labels"]}
                            else:
                                extra_inputs = None


                            # result: dict - training results (losses, metrics)
                            result = self.train_step(batch_mini_data, extra_inputs, step)
                            batch_mini_data.update(result)

                            progress_bar.update(1)

                            # logs: dict - training metrics for this step
                            logs = self.record_logs(batch_mini_data)
                            all_logs.append(logs)

                            update_steps = step / self.args.gradient_accumulation_steps

                            if step > 0 and step % self.args.gradient_accumulation_steps == 0:
                                self.print_logs(all_logs, update_steps)
                                all_logs.clear()

                            if update_steps > 0 and (update_steps % self.args.save_steps) == 0:

                                if self.is_world_process_zero():
                                    if self.args.use_co_model:
                                        # unwrapped_model: model - unwrapped model for saving
                                        unwrapped_model = self.accelerator.unwrap_model(self.co_model)
                                        self.save_checkpoint(unwrapped_model, self.args.output_dir, int(update_steps))
                                        if self.args.use_multi_adapters:
                                            self.save_checkpoint(unwrapped_model, self.args.critic_output_dir, int(update_steps), adapter_name="critic")
                                    else:
                                        self.save_checkpoint(self.actor_model, self.args.output_dir, int(update_steps))
                                        self.save_checkpoint(self.critic_model, self.args.critic_output_dir, int(update_steps))

                        random.shuffle(mini_dataset)
                        torch.cuda.empty_cache()

        progress_bar.close()
        self.accelerator.end_training()

