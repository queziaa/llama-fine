# https://zhuanlan.zhihu.com/p/8362625598
# https://github.com/andersonbcdefg/dpo-lora
# https://zhuanlan.zhihu.com/p/26128254223
# https://github.com/gulabpatel/LLMs/blob/main/04_trl_SFTTrainer_RewardTrainer_PPO.ipynb
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# python train_Datawhale-R1_unsloth.py --config Datawhale-R1_unsloth.yaml
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
from unsloth import FastLanguageModel, PatchFastRL
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List
from datasets import load_dataset
# from swanlab.integration.transformers import SwanLabCallback
# os.environ["SWANLAB_MODE"] = "disabled"
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
PatchFastRL("GRPO", FastLanguageModel)  # 对 TRL 进行补丁处理
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

# 配置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)  # 设置日志格式

logger.addHandler(handler)

def format_reward_func(completions, **kwargs):
    return [0,0,0,0]
    """
    格式奖励函数，检查模型输出格式是否匹配: <think>...</think><answer>...</answer>

    参数:
        completions (list[str]): 生成的输出
    返回:
        list[float]: 奖励分数
    """
    # 初始化奖励列表
    rewards = []
    # 遍历生成的输出
    for completion in completions:
        try:
            # 在生成的输出前添加<think>标签，便于后续正则表达式匹配
            completion = "<think>" + completion
            if random.random() < 0.1:  # 1% 的概率将生成输出写入文件
                # 创建生成输出目录（如果不存在）
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)  # 写入生成的输出
            # 定义正则表达式模式，用于匹配 <think> 和 <answer> 标签
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)  # 使用正则表达式进行匹配
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)  # 如果格式不正确，奖励为 0
            else:
                rewards.append(1.0)  # 如果格式正确，奖励为 1
        except Exception:
            rewards.append(0.0)  # 如果发生异常，奖励为 0
    return rewards

# def equation_reward_func(prompts, completions, target, nums, **kwargs):
def equation_reward_func(prompts, **kwargs):
    return [0,0,0,0]
    """
    方程奖励函数，检查计算结果是否正确，数字是否符合使用要求（每个数字只用一次，只使用所提供的数字）

    参数:
        completions (list[str]): 生成的输出
        target (list[str]): 预期的答案
        nums (list[str]): 可用的数字

    返回:
        list[float]: 奖励分数
    """
    # 初始化奖励列表
    rewards = []
    # 遍历生成的输出、预期的答案和可用的数字
    for prompt, completion, gt, numbers in zip(prompts, completions, target, nums):
        try:
            # 在生成的输出前添加 <think> 标签，便于后续正则表达式匹配
            completion = "<think>" + completion
            # 定义正则表达式模式，用于匹配 <answer> 标签
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)  # 如果没有匹配到 <answer> 标签，奖励为 0
                continue
            equation = match.group(1).strip()  # 提取 <answer> 标签中的内容
            # 提取方程中的所有数字
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            # 检查所有数字是否被使用且只使用一次
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue
            # 定义允许的字符模式，只允许数字、运算符、括号和空白字符
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)  # 如果方程包含不允许的字符，奖励为 0
                continue
            # 计算方程的结果
            result = eval(equation, {"__builtins__": None}, {})

            if random.random() < 0.3:  # 30% 的概率打印出来
                print('-'*20, f"\nQuestion:\n{prompt}",
                    f"\nCompletion:\n{completion}", f"\nResult:\n{result}", f"\nTarget:\n{gt}", f"\nNumbers:\n{numbers}")
            # 检查方程是否正确且与预期答案匹配（误差小于 1e-5）
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)  # 如果正确，奖励为 1                
                # 10% 的概率将成功的样本写入文件
                if random.random() < 0.10:
                    # 创建生成输出目录（如果不存在）
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join(
                        "completion_samples", "success_completion_samples.txt"
                    )
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(f"\nQuestion:\n{prompt}\nCompletion:\n{completion}\nResult:\n{result}\nTarget:\n{gt}\nNumbers:\n{numbers}")  # 写入生成的输出
                    # print('-'*20, f"\nQuestion:\n{prompt}", f"\nCompletion:\n{completion}", f"\nResult:\n{result}", f"\nTarget:\n{gt}", f"\nNumbers:\n{numbers}")
            else:
                rewards.append(0.0)  # 如果不正确，奖励为 0
        except Exception:
            rewards.append(0.0)  # 如果评估失败，奖励为 0
    return rewards





parser = TrlParser((ModelConfig, GRPOConfig))
model_args, training_args = (parser.parse_args_and_config())

# 记录模型参数
logger.info(f"Model parameters {model_args}")
# 记录训练/评估参数
logger.info(f"Training/evaluation parameters {training_args}")

# 从预训练模型加载模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_args.model_name_or_path,  # 模型名称或路径
    # fast_inference=True,  # 启用 vLLM 快速推理
    load_in_4bit=True,  # 是否以 4 位加载模型，False 表示使用 LoRA 16 位
    max_lora_rank=model_args.lora_r,  # 设置 LoRA 的最大秩
    max_seq_length=training_args.max_completion_length,  # 设置最大序列长度
    gpu_memory_utilization=training_args.vllm_gpu_memory_utilization,  # GPU 内存利用率，若内存不足可减少
    attn_implementation=model_args.attn_implementation, # 设置注意力实现方式 flash attention
) 

# PEFT 模型
model = FastLanguageModel.get_peft_model(
    model,
    r = model_args.lora_r,  # 选择任意大于 0 的数字！建议使用 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],  # 如果内存不足，可以移除 QKVO
    lora_alpha = model_args.lora_alpha,  # 设置 LoRA 的 alpha 值
    use_gradient_checkpointing = "unsloth",  # 启用长上下文微调
    random_state = training_args.seed,  # 设置随机种子
)

def generate_token(x):
    r1_prefix = [
        {
            "role": "user",
            "content": f"使用给定的数字 1，创建一个等于 1 的方程。你可以使用基本算术运算（+、-、*、/）一次或多次，但每个数字只能使用一次。在 <think> </think> 标签中展示你的思考过程，并在 <answer> </answer> 标签中返回最终方程，例如 <answer> (1 + 2) / 3 </answer>。在 <think> 标签中逐步思考。",
        },
        {
            "role": "assistant",
            "content": "让我们逐步解决这个问题。\n<think>",  # 结尾使用 `<think>` 促使模型开始思考
        },
    ]
    # 提示词，continue_final_message=True 表示将提示词中的最后一个消息继续到最终的输出中
    return {
        "prompt": tokenizer.apply_chat_template(
            r1_prefix, tokenize=False, continue_final_message=True
        ),  
    }


from tool import dataset_DEAL
seed = 3407
WORK = 3      #  3:仇恨目标搜索微调  31:仇恨目标奖励微调  
WORKFILE = 'outputnew_3_CC.json'
train_dataset = dataset_DEAL(WORKFILE,WORK,seed,test_size=-1)
# train_dataset = train_dataset.map(lambda x: generate_token(x))
# 设置 GRPOTrainer
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    # 奖励函数列表，用于计算奖励分数
    reward_funcs=[
        format_reward_func,  # 格式奖励函数
        equation_reward_func,  # 方程奖励函数
    ],
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=test_dataset,
)
# last_checkpoint = get_checkpoint(training_args)  # 检查最后一个检查点
# print("Last Checkpoint",last_checkpoint)
# # 如果检测到检查点且指定从检查点恢复训练，则记录信息
# if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
#     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
logger.info(
    f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
)
# 训练模型
train_result = trainer.train()




metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
logger.info("*** Training complete ***")
logger.info("*** Save model ***")
trainer.model.config.use_cache = True
model.save_lora(training_args.output_dir)
logger.info(f"Model saved to {training_args.output_dir}")
tokenizer.save_pretrained(training_args.output_dir)
logger.info(f"Tokenizer saved to {training_args.output_dir}")
logger.info("*** Training complete! ***")