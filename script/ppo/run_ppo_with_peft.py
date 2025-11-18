# PPO训练主脚本 - 使用PEFT(参数高效微调)进行PPO强化学习训练
#
# 本脚本实现了基于PPO (Proximal Policy Optimization) 算法的LLM强化学习训练流程，
# 支持使用LoRA等PEFT技术来高效微调大型语言模型。

import datetime
import os,sys,torch,logging
import numpy as np
from typing import Dict
import transformers

# 添加上级目录到路径，以便导入自定义模块
sys.path.append('..')

# 导入自定义工具模块
from utils.parser_args import parser_arguments  # 参数解析
from utils.data_collator import PPODataCollatorWithPadding,DataCollatorForSupervisedDataset  # 数据整理器
from utils.ppo_models import PPOEngine_CO, PPOEngine  # PPO模型引擎
from utils.ppo_trainer_with_peft import PPOPeftTrainer  # PPO训练器
from utils.utils import PROMPT_TEMPLATE  # 提示词模板

# 导入HuggingFace Transformers相关模块
from transformers import (
    AutoConfig,AutoTokenizer,LlamaForCausalLM,LlamaTokenizer,
    Trainer,AutoModelForCausalLM,get_scheduler,default_data_collator
)

# 导入PEFT相关模块
from peft import LoraConfig,PeftModel,TaskType,get_peft_model

# 导入数据处理相关模块
from pathlib import Path
from datasets import load_dataset,concatenate_datasets
from itertools import chain

# 设置日志记录器
logger = logging.getLogger(__name__)
# 设置忽略标签索引（用于损失计算中的掩码）
IGNORE_INDEX = -100
# 模型类型映射表
MODEL_CLASSES = {
    "llama": (AutoConfig, LlamaTokenizer, LlamaForCausalLM),
    "auto": (AutoConfig, AutoTokenizer, AutoModelForCausalLM),
}



def process_data(model_args, data_args, training_args, tokenizer):
    """
    处理训练数据，将原始文本数据转换为PPO训练所需的格式

    Args:
        model_args: 模型相关参数
        data_args: 数据相关参数
        training_args: 训练相关参数
        tokenizer: 分词器

    Returns:
        all_datasets: 处理后的PPO训练数据集
        extra_datasets: 额外的训练数据集（用于多任务学习）
    """

    def process_tokenize(examples):
        """
        处理PPO数据集的tokenization

        Args:
            examples: 原始数据样本

        Returns:
            model_inputs: 包含input_ids和label_ids的字典
        """
        model_inputs = {"input_ids": [], "label_ids": []}
        columns = list(examples.keys())
        template = PROMPT_TEMPLATE[data_args.template]

        for index in range(len(examples[columns[0]])):
            # 处理两种数据格式：prompt格式 或 instruction/input/output格式
            if 'prompt' not in columns:
                assert 'instruction' in columns and 'input' in columns and 'output' in columns

                instruction, input, output = examples['instruction'][index], examples['input'][index], examples['output'][index]
                # 如果有输入内容，将其拼接到指令后面
                if input is not None and input != "":
                    instruction = instruction + '\n' + input
                prompt = instruction
                # 处理输出格式
                if len(output) > 1:
                    response = output[0]
                else:
                    response = output
            else:
                assert 'prompt' in columns
                prompt, response = examples['prompt'][index], examples['chosen'][index]

            # 使用模板格式化提示词
            source = template.format_map({'instruction':prompt})
            # 对提示词和响应分别进行编码
            source_ids = tokenizer.encode(text=source, add_special_tokens=False)
            target_ids = tokenizer.encode(text=response, add_special_tokens=False)

            # 截断过长的序列
            if len(source_ids) > training_args.max_prompt_length - 1:
                source_ids = source_ids[:training_args.max_prompt_length - 1]
            if len(target_ids) > training_args.max_response_length - 1:
                target_ids = target_ids[:training_args.max_response_length - 1]

            # 构建输入序列和标签序列（PPO格式：分离prompt和response）
            input_ids = source_ids + [tokenizer.bos_token_id]
            labels = target_ids + [tokenizer.bos_token_id]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["label_ids"].append(labels)
        return model_inputs

    # 处理主要的PPO训练数据集
    logger.info("process prompt datasets")
    with training_args.main_process_first(desc="process prompt datasets"):
        if data_args.dataset_dir is not None:
            all_datasets = []
            path = Path(data_args.dataset_dir)
            # 获取目录下所有JSON文件
            files = [file.name for file in path.glob("*.json")]
            for file in files:
                data_path = os.path.join(path, file)
                # 加载JSON格式的数据集
                raw_dataset = load_dataset(
                    "json",
                    data_files=data_path,
                    cache_dir=data_args.data_cache_dir
                )
                columns = list(raw_dataset.column_names.values())[0]
                # 对数据进行tokenization处理
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

            # 可选：划分训练集和验证集
            # all_datasets = all_datasets.train_test_split(test_size=data_args.split_ratio)

    
    def process_tokenize_for_pt(examples):
        """
        处理预训练数据的tokenization（用于额外的预训练任务）

        Args:
            examples: 包含text字段的原始数据

        Returns:
            result: 固定长度的输入序列和标签序列
        """
        text_input_ids = tokenizer(examples["text"])["input_ids"]
        # 将所有文本连接成一个长序列
        concatenated_ids = list(chain(*text_input_ids))
        total_length = len(concatenated_ids)
        # 按照block_size进行分割，丢弃不足一个block的部分
        if total_length >= data_args.block_size:
            total_length = (total_length // data_args.block_size) * data_args.block_size
        result = [concatenated_ids[i : i + data_args.block_size] for i in range(0, total_length, data_args.block_size)]
        return {"input_ids": result, "label_ids": result.copy()}


    def process_tokenize_for_sft(examples):
        """
        处理SFT（监督微调）数据的tokenization（用于额外的SFT任务）

        Args:
            examples: SFT格式的数据样本

        Returns:
            model_inputs: 包含完整序列和标签的字典
        """
        template = PROMPT_TEMPLATE[data_args.template]
        model_inputs = {"input_ids": [], "label_ids": []}
        columns = list(examples.keys())

        for index in range(len(examples[columns[0]])):
            # 处理两种数据格式
            if 'prompt' not in columns:
                assert 'instruction' in columns and 'input' in columns and 'output' in columns

                instruction, input, output = examples['instruction'][index], examples['input'][index], examples['output'][index]
                if input is not None and input != "":
                    instruction = instruction + '\n' + input
                prompt = instruction
                if len(output) > 1:
                    response = output[0]
                else:
                    response = output
            else:
                assert 'prompt' in columns
                prompt, response = examples['prompt'][index], examples['chosen'][index]

            # 使用模板格式化
            source = template.format_map({'instruction':prompt})
            source_ids = tokenizer.encode(text=source, add_special_tokens=False)
            target_ids = tokenizer.encode(text=response, add_special_tokens=False)

            # 截断过长的序列
            if len(source_ids) > training_args.max_prompt_length - 1:
                source_ids = source_ids[:training_args.max_prompt_length - 1]
            if len(target_ids) > training_args.max_response_length - 1:
                target_ids = target_ids[:training_args.max_response_length - 1]

            # 构建完整的SFT序列（prompt + response）和相应的标签
            input_ids = source_ids + [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]
            # 标签中prompt部分用IGNORE_INDEX掩码，只对response部分计算损失
            labels = [IGNORE_INDEX] * len(source_ids) + [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["label_ids"].append(labels)

        return model_inputs


    # 处理额外的数据集（用于多任务学习）
    extra_datasets = []
    if data_args.extra_dataset_dir is not None:
        logger.info("process extra data")
        with training_args.main_process_first(desc="process extra data"):
            path = Path(data_args.extra_dataset_dir)

            # 根据数据集类型选择处理方式
            if training_args.extra_dataset_type == 'sft':
                # 处理SFT格式的JSON数据
                files = [file.name for file in path.glob("*.json")]
                for file in files:
                    data_path = os.path.join(path, file)
                    raw_dataset = load_dataset(
                        "json",
                        data_files=data_path,
                    )
                    columns = list(raw_dataset.column_names.values())[0]
                    tokenized_data = raw_dataset.map(
                        process_tokenize_for_sft,
                        batched=True,
                        num_proc=training_args.dataloader_num_workers,
                        remove_columns=columns,
                    )
                    extra_datasets.append(tokenized_data['train'])
            else:
                # 处理预训练格式的TXT数据
                files = [file.name for file in path.glob("*.txt")]
                for file in files:
                    data_path = os.path.join(path, file)
                    raw_dataset = load_dataset(
                        "text",
                        data_files=data_path
                    )
                    tokenized_data = raw_dataset.map(
                        process_tokenize_for_pt,
                        batched=True,
                        num_proc=training_args.dataloader_num_workers,
                        remove_columns="text"
                    )
                    extra_datasets.append(tokenized_data['train'])

            # 合并多个额外数据集
            if len(extra_datasets) == 1:
                extra_datasets = extra_datasets[0]
            else:
                extra_datasets = concatenate_datasets(extra_datasets)

    return all_datasets, extra_datasets


def main():
    """
    主函数：执行PPO训练流程
    """
    # 解析命令行参数
    model_args, data_args, training_args = parser_arguments(logger)
    # 设置随机种子确保可复现性
    transformers.set_seed(training_args.seed)

    # 获取模型配置类、tokenizer类和模型类
    config_class, tokenizer_class, model_class = MODEL_CLASSES[model_args.model_type]
    # 加载tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_args.sft_model_path, use_fast=model_args.use_fast_tokenizer)
    # 设置pad_token_id（如果没有设置，使用0作为默认值，通常对应<unk> token）
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    # 处理训练数据
    all_datasets, extra_datasets = process_data(model_args, data_args, training_args, tokenizer)

    logger.info("training")

    # 创建数据整理器
    data_collator = PPODataCollatorWithPadding(tokenizer)
    if data_args.extra_dataset_dir is not None:
        # 为额外数据集选择相应的整理器
        if training_args.extra_dataset_type == 'sft':
            extra_data_collator = DataCollatorForSupervisedDataset(tokenizer)
        else:
            extra_data_collator = default_data_collator

    ## 加载PPO模型引擎
    logger.info("load model")
    if training_args.use_co_model:
        # 使用CO（Composition）模型架构（单一模型同时处理actor和critic）
        ppo_engine = PPOEngine_CO(model_args, training_args)
    else:
        # 使用标准的分离模型架构（独立的actor和critic模型）
        ppo_engine = PPOEngine(model_args, training_args)

    # 创建PPO训练器
    trainer = PPOPeftTrainer(
        args = training_args,
        ppo_engine = ppo_engine,
        train_dataset = all_datasets,
        data_collator = data_collator,
        tokenizer = tokenizer,
        extra_train_dataset = extra_datasets if data_args.extra_dataset_dir is not None else None,
        extra_data_collator = extra_data_collator if data_args.extra_dataset_dir is not None else None,
    )

    # 开始训练
    if training_args.do_train:
        trainer.train() 


# 程序入口点
if __name__ == "__main__":
    main()

