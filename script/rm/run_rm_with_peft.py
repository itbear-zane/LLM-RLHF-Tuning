# 奖励模型训练脚本 - 使用PEFT(参数高效微调)技术
# 该脚本用于训练一个奖励模型，用于RLHF(人类反馈强化学习)中的奖励函数

import os,sys,torch,logging,math
import numpy as np
from typing import Dict
import transformers
from transformers import AutoConfig,AutoTokenizer,LlamaForCausalLM,LlamaTokenizer,Trainer,DataCollatorWithPadding,AutoModelForCausalLM,BitsAndBytesConfig

# 添加父目录到系统路径，以便导入自定义模块
sys.path.append("..")
from peft import LoraConfig,PeftModel,TaskType,get_peft_model
from pathlib import Path
from datasets import load_dataset,concatenate_datasets
from itertools import chain
from utils.parser_args import parser_arguments
from utils.metrics import compute_metrics_for_pair
from utils.trainer import PeftTrainer,RMPeftTrainer
from trl import AutoModelForCausalLMWithValueHead
from utils.data_collator import PairDataCollatorWithPadding
from utils.utils import PROMPT_TEMPLATE


logger = logging.getLogger(__name__)
IGNORE_INDEX = -100  # 用于在标签中忽略不需要计算损失的token

# 支持的模型类型配置
# key: 模型类型名称, value: (配置类, 分词器类, 模型类)
MODEL_CLASSES = {
    "llama": (AutoConfig, LlamaTokenizer, LlamaForCausalLM),
    "auto": (AutoConfig, AutoTokenizer, AutoModelForCausalLM),
}



def print_trainable_params(model: torch.nn.Module) -> None:
    """
    打印模型的可训练参数信息

    参数:
        model: 要分析的模型

    功能:
        - 计算总参数数量和可训练参数数量
        - 打印可训练参数占比
        - 支持DeepSpeed Zero 3的分布式训练场景
    """
    # Adopted from https://github.com/LLaMA-Efficient-Tuning-main/src/utils/other.py
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # 如果使用DS Zero 3且权重初始化为空，需要特殊处理
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
                trainable_params, all_param, 100 * trainable_params / all_param))



def create_model(model_args, data_args, training_args):
    """
    创建和配置奖励模型

    参数:
        model_args: 模型相关参数
        data_args: 数据相关参数
        training_args: 训练相关参数

    返回:
        model: 配置好的奖励模型
        tokenizer: 分词器

    功能:
        - 加载预训练模型和分词器
        - 配置PEFT(LoRA)参数高效微调
        - 添加价值头(value head)用于奖励预测
        - 支持4bit量化和梯度检查点优化
    """

    ## 加载模型和分词器
    config_class, tokenizer_class, model_class = MODEL_CLASSES[model_args.model_type]

    # 加载分词器，支持自定义分词器路径
    if model_args.tokenizer_name_or_path is None:
        tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_args.tokenizer_name_or_path, use_fast=model_args.use_fast_tokenizer)

    # 设置pad token，如果没有则使用unk token (ID为0)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    # 模型配置参数
    config_kwargs = {
        "trust_remote_code": True,  # 信任远程代码
        "torch_dtype": model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype),  # 数据类型
        "low_cpu_mem_usage": True  # 低内存使用模式
    }

    # 配置4bit量化（用于减少内存占用）
    if model_args.load_in_4bit:
        config_kwargs["load_in_4bit"] = True
        config_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用bfloat16
            bnb_4bit_use_double_quant=True,  # 使用双重量化
            bnb_4bit_quant_type="nf4"  # 使用NF4量化类型
        )

    # 加载预训练模型
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),  # 检查是否为TensorFlow检查点
        **config_kwargs
    )

    # 配置PEFT(参数高效微调)
    if model_args.peft_path is not None:
        # 如果提供了预训练的PEFT模型路径，则加载它
        logger.info(f"Load pre-trained model: {model_args.peft_path}" )
        model = PeftModel.from_pretrained(model, model_args.peft_path, is_trainable=True)

    else:
        # 初始化新的LoRA配置
        logger.info("Init new peft model")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
            inference_mode=False,  # 非推理模式（即为训练模式）
            target_modules=training_args.lora_target.split(','),  # 目标模块列表
            r=training_args.lora_rank,  # LoRA秩
            lora_alpha=training_args.lora_alpha,  # LoRA缩放因子
            lora_dropout=training_args.lora_dropout,  # LoRA dropout率
        )
        model = get_peft_model(model, peft_config=lora_config)

    # 添加价值头(value head) - 这是奖励模型的核心组件
    # 用于将语言模型的输出转换为奖励分数
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    # 如果加载了预训练的PEFT模型，也需要加载对应的价值头权重
    if model_args.peft_path is not None:
        lora_state_dict = torch.load(os.path.join(model_args.peft_path, 'adapter_model.bin'))
        model.v_head.load_state_dict({
                    "summary.weight": lora_state_dict["v_head.summary.weight"],
                    "summary.bias": lora_state_dict["v_head.summary.bias"]
                })

    # 打印模型参数信息
    print('*********************model*******************')
    print_trainable_params(model)

    # 启用梯度检查点以节省内存
    model.gradient_checkpointing_enable()
    # 禁用缓存以节省内存
    model.config.use_cache = False

    return model, tokenizer
    


def process_data(model_args, data_args, training_args, tokenizer):
    """
    数据预处理函数 - 将原始数据转换为模型训练所需的格式

    参数:
        model_args: 模型相关参数
        data_args: 数据相关参数
        training_args: 训练相关参数
        tokenizer: 分词器

    返回:
        all_datasets: 处理好的数据集，包含训练集和测试集

    功能:
        - 支持两种数据格式：成对比较格式和指令-输出格式
        - 将文本转换为token ID序列
        - 处理长度限制和填充
        - 构建成对比较的训练样本
    """

    def process_tokenize(examples):
        """
        将单个批次的数据进行token化处理

        参数:
            examples: 原始数据批次

        返回:
            model_inputs: 包含input_ids和label_ids的字典
        """
        model_inputs = {"input_ids": [], "label_ids": []}
        columns = list(examples.keys())
        template = PROMPT_TEMPLATE[data_args.template]  # 获取提示模板

        for index in range(len(examples[columns[0]])):
            # 根据数据格式解析数据
            if 'chosen' not in columns or 'rejected' not in columns:
                # 格式1: instruction-input-output格式
                assert 'instruction' in columns and 'input' in columns and 'output' in columns

                instruction, input, output = examples['instruction'][index], examples['input'][index], examples['output'][index]
                if input is not None and input != "":
                    instruction = instruction + '\n' + input
                assert len(output) > 1
                prompt, chosen, rejected = instruction, output[0], output[1]  # 第一个输出是好的，第二个是差的
            else:
                # 格式2: prompt-chosen-rejected格式（标准成对比较格式）
                assert 'prompt' in columns and 'rejected' in columns and 'chosen' in columns
                prompt, chosen, rejected = examples['prompt'][index], examples['chosen'][index], examples['rejected'][index]

            # 使用模板格式化提示
            source = template.format_map({'instruction':prompt})
            source_ids = tokenizer.encode(text=source, add_special_tokens=False)
            accepts_ids = tokenizer.encode(text=chosen, add_special_tokens=False)
            rejects_ids = tokenizer.encode(text=rejected, add_special_tokens=False)

            # 长度截断 - 确保不超过最大长度限制
            if len(source_ids) > training_args.max_prompt_length - 1:
                source_ids = source_ids[:training_args.max_prompt_length - 1]
            if len(accepts_ids) > training_args.max_response_length - 1:
                accepts_ids = accepts_ids[:training_args.max_response_length - 1]
            if len(rejects_ids) > training_args.max_response_length - 1:
                rejects_ids = rejects_ids[:training_args.max_response_length - 1]


            # 构建完整的输入序列：source + bos + response + eos
            source_accepts_ids = source_ids + [tokenizer.bos_token_id] + accepts_ids + [tokenizer.eos_token_id]
            source_accepts_labels = [IGNORE_INDEX] * len(source_ids) + [tokenizer.bos_token_id] + accepts_ids + [tokenizer.eos_token_id]
            source_rejects_ids = source_ids + [tokenizer.bos_token_id] + rejects_ids + [tokenizer.eos_token_id]
            source_rejects_labels = [IGNORE_INDEX] * len(source_ids) + [tokenizer.bos_token_id] + rejects_ids + [tokenizer.eos_token_id]

            # 对齐长度 - 使用padding确保两个序列长度相同
            source_accepts_length, source_rejects_length = len(source_accepts_ids), len(source_rejects_ids)
            max_length = max(source_accepts_length, source_rejects_length)

            source_accepts_ids = source_accepts_ids + [tokenizer.pad_token_id] * (max_length - source_accepts_length)
            source_accepts_labels = source_accepts_labels + [IGNORE_INDEX] * (max_length - source_accepts_length)
            source_rejects_ids = source_rejects_ids + [tokenizer.pad_token_id] * (max_length - source_rejects_length)
            source_rejects_labels = source_rejects_labels + [IGNORE_INDEX] * (max_length - source_rejects_length)

            # 合并成对样本：先接受样本，后拒绝样本
            inputs_ids = source_accepts_ids + source_rejects_ids
            labels = source_accepts_labels + source_rejects_labels

            model_inputs["input_ids"].append(inputs_ids)
            model_inputs["label_ids"].append(labels)

        return model_inputs

    ### 处理数据集
    logger.info("process datasets")
    with training_args.main_process_first(desc="process datasets"):
        if data_args.dataset_dir is not None:
            # 从目录加载多个JSON文件
            all_datasets = []
            path = Path(data_args.dataset_dir)
            files = [file.name for file in path.glob("*.json")]
            for file in files:
                data_path = os.path.join(path, file)
                raw_dataset = load_dataset(
                    "json",
                    data_files=data_path,
                )
                columns = list(raw_dataset.column_names.values())[0]
                # 对数据进行token化处理
                tokenized_data = raw_dataset.map(
                    process_tokenize,
                    batched=True,
                    num_proc=training_args.dataloader_num_workers,
                    remove_columns=columns,
                    load_from_cache_file=True
                )
                all_datasets.append(tokenized_data['train'])

            # 合并多个数据集
            if len(all_datasets) == 1:
                all_datasets = all_datasets[0]
            else:
                all_datasets = concatenate_datasets(all_datasets)

            # 划分训练集和测试集
            all_datasets = all_datasets.train_test_split(test_size=data_args.split_ratio)
        elif data_args.train_file is not None and data_args.validation_file is not None:
            # 从单独的训练和验证文件加载
            all_datasets = {}
            raw_train_datasets = load_dataset(
                "json",
                data_files=data_args.train_file,
                cache_dir=data_args.data_cache_dir
            )
            columns = list(raw_train_datasets.column_names.values())[0]
            # 处理训练数据
            all_datasets['train'] = raw_train_datasets.map(
                process_tokenize,
                batched=True,
                num_proc=training_args.dataloader_num_workers,
                remove_columns=columns,
                load_from_cache_file=True
            )['train']

            raw_valid_datasets = load_dataset(
                "json",
                data_files=data_args.validation_file,
                cache_dir=data_args.data_cache_dir
            )
            # 处理验证数据
            all_datasets['test'] = raw_valid_datasets.map(
                process_tokenize,
                batched=True,
                num_proc=training_args.dataloader_num_workers,
                remove_columns=columns,
                load_from_cache_file=True
            )['train']
        else:
            # 数据加载参数错误
            raise ValueError(
                "数据集文件路径不正确。"
                "你可以提供 --dataset_dir 目录，或者提供 --train_file 和 --validation_file 两个文件。"
            )

    return all_datasets


def main():
    """
    主函数 - 训练奖励模型的完整流程

    功能:
        1. 解析命令行参数
        2. 设置随机种子
        3. 创建和配置模型
        4. 处理训练数据
        5. 创建训练器并开始训练
        6. 保存训练结果和模型
    """

    # 解析命令行参数
    model_args, data_args, training_args = parser_arguments(logger)
    # 设置随机种子以确保结果可重现
    transformers.set_seed(training_args.seed)

    # 步骤1: 创建模型和分词器
    model, tokenizer = create_model(model_args, data_args, training_args)

    # 步骤2: 处理训练数据
    all_datasets = process_data(model_args, data_args, training_args, tokenizer)

    # 步骤3: 创建奖励模型训练器
    trainer = RMPeftTrainer(
        model=model,
        args=training_args,
        train_dataset=all_datasets['train'] if training_args.do_train else None,
        eval_dataset=all_datasets['test'] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=PairDataCollatorWithPadding(tokenizer=tokenizer),  # 成对数据整理器
        compute_metrics=compute_metrics_for_pair,  # 成对比较的评价指标
    )

    # 步骤4: 开始训练（如果设置了训练标志）
    if training_args.do_train:
        output = trainer.train()
        # 记录训练指标
        trainer.log_metrics("train", output.metrics)
        trainer.save_metrics("train", output.metrics)
        # 保存训练状态和模型
        trainer.save_state()
        trainer.save_model()


if __name__ == "__main__":
    # 运行主函数
    main()




