import swanlab
swanlab.login(api_key="f7jcdTmoq9NOkGigx50Wl", save=True)
Lora_dir = './lora'
Lora_MODEL = None
from unsloth import FastLanguageModel
import numpy as np
import random
from peft import LoftQConfig
import torch
from tool import dataset_DEAL

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--work', type=int)
parser.add_argument('--workfile', type=str)
parser.add_argument('--Lora_MODEL', type=str)
args = parser.parse_args()
WORK = args.work
WORKFILE = args.workfile
Lora_MODEL = args.Lora_MODEL
if Lora_MODEL == "None":
    Lora_MODEL = None

# Lora_MODEL =  './12outputnew4CC'
# WORK = 3       #  3:仇恨目标搜索微调  31:仇恨目标奖励微调   12 俚语分析微调
# WORKFILE = './DATA/outputnew3CC.json'  

max_seq_length = 2048
seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if Lora_MODEL != None:
    lora_model,tokenizer = FastLanguageModel.from_pretrained(
        model_name = Lora_MODEL,
        max_seq_length = max_seq_length,
        max_lora_rank = 256
    )
else:
    base_model,tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
        max_seq_length = max_seq_length,
        max_lora_rank = 256
    )
    lora_model = FastLanguageModel.get_peft_model(
        base_model,
        r = 128, 
        lora_alpha = 256, 
        lora_dropout = 0.2, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",],
        bias = "lora_only",   #["none", "all", "lora_only"]
        use_gradient_checkpointing = "unsloth",
        random_state = seed,
        use_rslora = True,
        loftq_config = LoftQConfig(loftq_bits = 8, loftq_iter = 2)
    )
dataset_train = dataset_DEAL(WORKFILE,WORK)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
# 清除梯度
lora_model.gradient_checkpointing_enable()
# 清除缓存
torch.cuda.empty_cache()
trainer = SFTTrainer(
    model = lora_model,
    tokenizer = tokenizer,
    train_dataset = dataset_train,
    dataset_text_field = "prompt",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, 
        max_steps = -1, # 移除或设置为-1，让模型训练完整一个epoch
        learning_rate = 2e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        # optim = "adamw_torch_fused",
        optim = "paged_adamw_32bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = seed,
        output_dir = Lora_dir,
    ),
)
trainer_stats = trainer.train()

import os
checkpoint_dirs = [d for d in os.listdir(Lora_dir) if d.startswith('checkpoint-')]
if not checkpoint_dirs:
    print("No checkpoint directories found.")
latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
checkpoint_dir = os.path.join(Lora_dir, latest_checkpoint)
if checkpoint_dir:
    new_checkpoint_path = os.path.join('./', str(WORK) + WORKFILE.split('/')[-1].split('.')[0])
    os.rename(checkpoint_dir, new_checkpoint_path)
    print(f"Renamed {checkpoint_dir} to {new_checkpoint_path}")
import shutil
shutil.rmtree(Lora_dir)