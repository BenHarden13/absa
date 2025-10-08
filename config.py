import os

class Config:
    # 数据路径
    data_dir = "/content/drive/MyDrive/absa-c-cos/data"
    train_file = os.path.join(data_dir, "dailydialog_train.json")
    valid_file = os.path.join(data_dir, "dailydialog_valid.json")  # 新增验证集
    test_file = os.path.join(data_dir, "dailydialog_test.json")
    
    # 模型配置
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir = "/content/drive/MyDrive/absa-c-cos/models/qwen_absa_v2"  # v2版本
    
    # LoRA配置
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # 训练超参数 (v2增强)
    num_train_epochs = 3  # 从1轮增加到3轮
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 2  # 新增评估批次大小
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    max_seq_length = 1024
    warmup_steps = 100
    logging_steps = 10
    save_steps = 100
    eval_steps = 50  # 新增：每50步评估一次
    evaluation_strategy = "steps"  # 新增：评估策略
    load_best_model_at_end = True  # 新增：加载最佳模型
    
    # 推理配置 (新增)
    inference_max_retries = 3  # 推理时最多重试3次
    inference_max_new_tokens = 256
    inference_temperature = 0.7
    inference_do_sample = True