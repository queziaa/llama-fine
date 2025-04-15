from unsloth import FastLanguageModel, PatchFastRL
import logging
from datetime import datetime
from tool import soft_match_score,conut_reward,conut_format_reward
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
def format_reward_func(completions, **kwargs):
    LOGLOGfilf.write(str(completions))
    reward_list = []
    for completion in completions:
        reward_list.append(conut_format_reward(completion))
    print('format_reward_func', reward_list)
    return reward_list

def equation_reward_func(completions, **kwargs):
    target_list = kwargs["target"]
    reward_list = []
    pr_target_list = []
    for completion, target in zip(completions, target_list):
        rw,pr_target = conut_reward(completion, target)
        pr_target_list.append(pr_target)
        reward_list.append(rw / 10)
    print('targets', target_list)
    print('pr_target_list', pr_target_list)
    print('equation_reward_func', reward_list)
    return reward_list



parser = TrlParser((ModelConfig, GRPOConfig))
model_args, training_args = (parser.parse_args_and_config())

logger.info(f"Model parameters {model_args}")
logger.info(f"Training/evaluation parameters {training_args}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_args.model_name_or_path,  # 模型名称或路径
    # fast_inference=True,  # 启用 vLLM 快速推理
    load_in_4bit=True,  # 是否以 4 位加载模型，False 表示使用 LoRA 16 位
    max_lora_rank=model_args.lora_r,  # 设置 LoRA 的最大秩
    max_seq_length=training_args.max_completion_length,  # 设置最大序列长度
    # gpu_memory_utilization=training_args.vllm_gpu_memory_utilization,  # GPU 内存利用率，若内存不足可减少
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

from tool import dataset_DEAL
seed = 3407
WORK = 31      #  3:仇恨目标搜索微调  31:仇恨目标奖励微调  
WORKFILE = 'outputnew_3_CC.json'
train_dataset = dataset_DEAL(WORKFILE,WORK,seed,test_size=-1)
def generate_r1_prompt(prompt, target):
    return {"prompt": tokenizer.apply_chat_template(prompt, tokenize=False, continue_final_message=True), "target": target}
train_dataset = train_dataset.map(lambda x: generate_r1_prompt(x["prompt"], x["target"]))

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs=[
        format_reward_func,  # 格式奖励函数
        equation_reward_func,  # 方程奖励函数
    ],
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=test_dataset,
)
# last_checkpoint = get_checkpoint(training_args)  # 检查最后一个检查点
# print("Last Checkpoint",last_checkpoint) # 如果检测到检查点且指定从检查点恢复训练，则记录信息
# if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
#     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
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
model.save_lora(training_args.output_dir)
logger.info(f"Model saved to {training_args.output_dir}")
tokenizer.save_pretrained(training_args.output_dir)
logger.info(f"Tokenizer saved to {training_args.output_dir}")
logger.info("*** Training complete! ***")