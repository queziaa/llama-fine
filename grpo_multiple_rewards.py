from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 创建配置时可以指定多个奖励模型
config = GRPOConfig(
    model_id="gpt2",  # 指定基础模型
    reward_model_ids=["reward-model-1", "reward-model-2"],  # 指定多个奖励模型
    reward_weights=[0.5, 0.5],  # 可选：指定每个模型的权重
    learning_rate=5e-5,  # 设置学习率
    batch_size=16,  # 设置批量大小
    max_length=128,  # 设置最大序列长度
)

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(config.model_id)
tokenizer = AutoTokenizer.from_pretrained(config.model_id)

# 创建GRPOTrainer
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
)

# 开始训练
trainer.train()