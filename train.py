import torch
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from config import Config
from data_processor import load_and_process_data


def main() -> None:
    cfg = Config()

    print("加载并处理数据...")
    train_dataset = load_and_process_data(cfg.train_file, max_samples=500)
    print(f"实际用于训练的数据条数：{len(train_dataset)}")
    assert len(train_dataset) <= 500, f"数据截断未生效，当前样本数={len(train_dataset)}。"

    print("加载模型与分词器...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",  # Windows 上禁用 flash_attn，使用 sdpa
    )
    model.config.pretraining_tp = 1

    print("配置 LoRA...")
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("配置训练参数...")
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=10,                 # 每 10 步保存一次
        save_total_limit=3,            # 最多保留 3 个 checkpoint
        save_strategy="steps",         # 按步保存
        save_safetensors=True,         # 使用 safetensors
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to="none",
    )

    print("初始化 SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field=cfg.dataset_text_field,
    )

    print("🚀 开始训练...")
    trainer.train()

    print("✅ 训练完成！")


if __name__ == "__main__":
    main()