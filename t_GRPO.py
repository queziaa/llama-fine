from unsloth import FastLanguageModel, PatchFastRL
import os
os.environ["VLLM_USE_V1"] = '0'
import logging
from datetime import datetime
from tool import soft_match_score,REWARD
# from swanlab.integration.transformers import SwanLabCallback
# os.environ["SWANLAB_MODE"] = "disabled"
PatchFastRL("GRPO", FastLanguageModel)  # 对 TRL 进行补丁处理
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
) 
logger.addHandler(handler)
LOGLOGfilf = open('log.txt', 'a')

Reward = REWARD()

def format_reward_func(completions, **kwargs):
    reward_list = []
    for completion in completions:
        reward_list.append(Reward.conut_format_reward(completion))
    print('format_reward_func', reward_list)
    LOGLOGfilf.write('format_reward_func' + str(reward_list) + '\n')
    return reward_list

def format_reward_func_2(completions, **kwargs):
    reward_list = []
    LOGLOGfilf.write(str(completions))
    for completion in completions:
        reward_list.append(Reward.conut_format_reward_2(completion))
    print('format_reward_func_2', reward_list)
    LOGLOGfilf.write('format_reward_func_2' + str(reward_list) + '\n')
    return reward_list

def conut_format_reward_1(completions, target, **kwargs):
    reward_list = []
    for completion, target in zip(completions, target):
        reward_list.append(Reward.conut_format_reward_1(completion, target))
    print('conut_format_reward_1', reward_list)
    LOGLOGfilf.write('conut_format_reward_1' + str(reward_list) + '\n')
    return reward_list


def equation_reward_func(completions, **kwargs):
    target_list = kwargs["target"]
    reward_list = []
    pr_target_list = []
    for completion, target in zip(completions, target_list):
        rw,pr_target = Reward.conut_reward(completion, target)
        pr_target_list.append(pr_target)
        reward_list.append(rw)
    print('targets', target_list)
    LOGLOGfilf.write('targets' + str(target_list) + '\n')
    print('pr_target_list', pr_target_list)
    LOGLOGfilf.write('pr_target_list' + str(pr_target_list) + '\n')
    print('equation_reward_func', reward_list)
    LOGLOGfilf.write('equation_reward_func' + str(reward_list) + '\n')
    print('identical', kwargs['id'])
    LOGLOGfilf.write('identical' + str(kwargs['id']) + '\n')
    return reward_list



parser = TrlParser((ModelConfig, GRPOConfig))
model_args, training_args = (parser.parse_args_and_config())
logger.info(f"Model parameters {model_args}")
logger.info(f"Training/evaluation parameters {training_args}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_args.model_name_or_path,  # 模型名称或路径
    load_in_4bit=True,  # 是否以 4 位加载模型，False 表示使用 LoRA 16 位
    max_lora_rank=model_args.lora_r,  # 设置 LoRA 的最大秩
    max_seq_length=training_args.max_completion_length,  # 设置最大序列长度
    attn_implementation=model_args.attn_implementation, # 设置注意力实现方式 flash attention

    fast_inference=training_args.use_vllm,  # 是否使用 VLLM 进行快速推理
    gpu_memory_utilization=training_args.vllm_gpu_memory_utilization,  # GPU 内存利用率，若内存不足可减少
) 
# PEFT 模型
model = FastLanguageModel.get_peft_model(
    model,
    r = model_args.lora_r,  # 选择任意大于 0 的数字！建议使用 8, 16, 32, 64, 128
    lora_alpha = model_args.lora_alpha,  # 设置 LoRA 的 alpha 值
    lora_dropout = model_args.lora_dropout,  # 设置 LoRA 的 dropout 值
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], 
    use_gradient_checkpointing = "unsloth",  # 启用长上下文微调
    random_state = training_args.seed,  # 设置随机种子
)

from tool import dataset_DEAL
seed = 3407
WORK = 31      #  3:仇恨目标搜索微调  31:仇恨目标奖励微调  
WORKFILE = 'outputnew_3_CC.json'
train_dataset,test_dataset = dataset_DEAL(WORKFILE,WORK,seed)
# train_dataset 删除前100条数据  
train_dataset = train_dataset.select(range(100, len(train_dataset)))
def generate_r1_prompt(prompt, target):
    return {"prompt": tokenizer.apply_chat_template(prompt, tokenize=False, continue_final_message=True), "target": target}
train_dataset = train_dataset.map(lambda x: generate_r1_prompt(x["prompt"], x["target"]))
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs=[
        # format_reward_func,  # 格式奖励函数
        format_reward_func_2,  # 格式奖励函数2
        conut_format_reward_1,  # 格式奖励函数1
        equation_reward_func,  # 方程奖励函数
    ],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
logger.info(
    f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
)
train_result = trainer.train()

metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
logger.info("*** Training complete ***")
logger.info("*** Save model ***")
trainer.model.config.use_cache = True
# 将模型lora层合并到主模型中
model.merge_and_unload()
model.save_lora(training_args.output_dir)
logger.info(f"Model saved to {training_args.output_dir}")
tokenizer.save_pretrained(training_args.output_dir)
logger.info(f"Tokenizer saved to {training_args.output_dir}")
logger.info("*** Training complete! ***")