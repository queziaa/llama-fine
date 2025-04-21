from unsloth import FastLanguageModel, PatchFastRL
import os
# os.environ["VLLM_USE_V1"] = '0'
# os.environ["TORCHDYNAMO_DISABLE"] = "1"  # 完全禁用dynamo
import logging
from datetime import datetime
from tool import REWARD
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
from tool import dataset_DEAL
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
seed = 3407


parser = TrlParser((ModelConfig, GRPOConfig))
ALL = {
    'MODEL': './4outputnew4CC',
    'WORK': '41',
    'WORKFILE': './DATA/outputnew4CC.json',
}

model_args, training_args = (parser.parse_args_and_config())
_, training_args = (parser.parse_args_and_config())


MODEL_name_or_path = ALL['MODEL']
WORK = ALL['WORK']
WORKFILE = ALL['WORKFILE']
train_dataset = dataset_DEAL(WORKFILE,WORK)
training_args.output_dir = 'GRPO_check_' + str(WORK)



Reward = REWARD(WORK)
# def len_HateTargetJudgment(completions, **kwargs):
#     reward_list = []
#     for completion_i in completions:
#         reward_list.append(Reward.len_HateTargetJudgment(completion_i))
#     print('仇恨判断输出长度奖励', reward_list)
#     LOGLOGfilf.write('仇恨判断输出长度奖励' + str(reward_list) + '\n')
#     return reward_list

# def three_stage(completions, **kwargs):
#     reward_list = []
#     for completion_i in completions:
#         reward_list.append(Reward.three_stage(completion_i))
#     print('三段式思考输出奖励', reward_list)
#     LOGLOGfilf.write('三段式思考输出奖励' + str(reward_list) + '\n')
#     return reward_list

# def out_number_matching(completions, target,**kwargs):
#     reward_list = []
#     for completion_i, target_i in zip(completions, target):
#         reward_list.append(Reward.out_number_matching(completion_i, target_i))
#     print('输出和预测数量奖励', reward_list)
#     LOGLOGfilf.write('输出和预测数量奖励' + str(reward_list) + '\n')
    # return reward_list

# def intercepted_in_text(completions, **kwargs):
#     reward_list = []
#     target_list = kwargs["target"]
#     for completion_i, target in zip(completions, target_list):
#         reward_list.append(Reward.intercepted_in_text(completion_i,target))
#     print('输出是否为文中截取奖励', reward_list)
#     LOGLOGfilf.write('输出是否为文中截取奖励' + str(reward_list) + '\n')
#     return reward_list

def Final_matching(completions, **kwargs):
    if WORK == '3':
        target_list = kwargs["target"]
    else:
        target_list = kwargs["Argument"]
    reward_list = []
    pr_target_list = []
    for completion_i, target in zip(completions, target_list):
        rw,pr_target = Reward.Final_matching(completion_i, target)
        pr_target_list.append(pr_target)
        reward_list.append(rw)
    print('黄金', target_list)
    LOGLOGfilf.write('黄金' + str(target_list) + '\n')
    print('预测', pr_target_list)
    LOGLOGfilf.write('预测' + str(pr_target_list) + '\n')
    print('**预测奖励**', reward_list)
    LOGLOGfilf.write('**预测奖励**' + str(reward_list) + '\n')
    print('编号', kwargs['id'])
    LOGLOGfilf.write('编号' + str(kwargs['id']) + '\n')
    LOGLOGfilf.write(str(completions))
    LOGLOGfilf.write('\n---------------------------------------------------------------------\n')
    return reward_list



# logger.info(f"Model parameters {model_args}")
logger.info(f"Training/evaluation parameters {training_args}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_name_or_path,
    load_in_4bit=True,  # 是否以 4 位加载模型，False 表示使用 LoRA 16 位
    # max_lora_rank=model_args.lora_r,  # 设置 LoRA 的最大秩
    max_seq_length=2048,  # 设置最大序列长度
    repetition_penalty=1.2,  # 设置重复惩罚
    attn_implementation='flash_attention_2', # 设置注意力实现方式 flash attention
) 
# # PEFT 模型
# model = FastLanguageModel.get_peft_model(
#     model,
#     r = model_args.lora_r,  # 选择任意大于 0 的数字！建议使用 8, 16, 32, 64, 128
#     lora_alpha = model_args.lora_alpha,  # 设置 LoRA 的 alpha 值
#     lora_dropout = model_args.lora_dropout,  # 设置 LoRA 的 dropout 值
#     target_modules = [
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ], 
#     use_gradient_checkpointing = "unsloth",  # 启用长上下文微调
#     random_state = training_args.seed,  # 设置随机种子
# )

# def generate_r1_prompt(prompt, target):
    # return {
        # "prompt": 
            # tokenizer.apply_chat_template(prompt, tokenize=False, continue_final_message=True).replace("assistant\n<|im_end|>\n", "assistant\n"), 
        # "target": 
            # target
    # }


# train_dataset = train_dataset.map(lambda x: generate_r1_prompt(x["prompt"], x["target"]))
# test_dataset = test_dataset.map(lambda x: generate_r1_prompt(x["prompt"], x["target"]))
training_args.repetition_penalty = 1.2
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs=[
        # len_HateTargetJudgment,
        # three_stage,
        # out_number_matching,
        # intercepted_in_text,
        Final_matching
    ],
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=test_dataset,
)
logger.info(
    f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
)


# train_result = trainer.train(resume_from_checkpoint=True)  # 自动查找最新的checkpoint
train_result = trainer.train()

# metrics = train_result.metrics
# metrics["train_samples"] = len(train_dataset)
# trainer.log_metrics("train", metrics)
# trainer.save_metrics("train", metrics)
# trainer.save_state()
# logger.info("*** Training complete ***")
# logger.info("*** Save model ***")
# trainer.model.config.use_cache = True
# # 将模型lora层合并到主模型中
# model.merge_and_unload()
# model.save_lora(training_args.output_dir)
# logger.info(f"Model saved to {training_args.output_dir}")
# tokenizer.save_pretrained(training_args.output_dir)
# logger.info(f"Tokenizer saved to {training_args.output_dir}")
# logger.info("*** Training complete! ***")