import argparse
import json
import os
from typing import Dict

import torch
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint

from config import Config
from data_processor import create_bulletproof_prompt_and_answer


def _prepare_quant_config(cfg: Config) -> BitsAndBytesConfig:
    compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _generate(model, tokenizer, prompt: str, cfg: Config) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_seq_length,
    ).to(model.device)
    with torch.inference_mode():
        do_sample = cfg.inference_do_sample

        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.inference_max_new_tokens,
            temperature=cfg.inference_temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_model_output(output: str) -> Dict:
    """解析模型输出为JSON格式"""
    candidate = output.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and start < end:
            snippet = candidate[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
    return {"raw_output": candidate}


def _load_model_and_tokenizer(cfg: Config):
    quant_config = _prepare_quant_config(cfg)

    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    adapter_path = None
    if os.path.isdir(cfg.output_dir):
        adapter_path = get_last_checkpoint(cfg.output_dir) or cfg.output_dir

    if adapter_path is None or not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"在 {cfg.output_dir} 未找到微调后的 LoRA 权重，请先运行训练脚本（确保保存 checkpoint）。"
        )

    print(f"加载 LoRA 适配器：{adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def run_inference_v2(dialogue_text: str, target_utterance: str, cfg: Config = None) -> Dict:
    """
    v2版本的推理函数
    使用防弹指令和重试机制
    """
    if cfg is None:
        cfg = Config()
    
    model, tokenizer = _load_model_and_tokenizer(cfg)
    
    # 创建统一的数据点结构（用于生成prompt）
    unified_datapoint = {
        "dialogue_text": dialogue_text,
        "target_utterance": target_utterance,
        "elements": {}  # 推理时不需要预填充
    }
    
    # 创建防弹指令（只使用prompt部分）
    prompt_data = create_bulletproof_prompt_and_answer(unified_datapoint)
    prompt = prompt_data["prompt"]
    
    # 重试机制
    for attempt in range(cfg.inference_max_retries):
        try:
            print(f"\n--- 推理尝试 {attempt + 1}/{cfg.inference_max_retries} ---")
            output_text = _generate(model, tokenizer, prompt, cfg)
            print(f"模型输出: {output_text[:200]}...")  # 显示前200字符
            
            result = parse_model_output(output_text)
            
            # 验证结果是否包含所有必需字段
            required_fields = ["holder", "target", "opinion", "sentiment", "cause", "flipping", "trigger"]
            if all(field in result for field in required_fields):
                print("✓ 成功解析完整的JSON结果")
                return result
            else:
                missing = [f for f in required_fields if f not in result]
                print(f"✗ 缺少字段: {missing}")
                if attempt == cfg.inference_max_retries - 1:
                    return result  # 最后一次尝试，返回不完整的结果
        except Exception as e:
            print(f"✗ 推理失败: {e}")
            if attempt == cfg.inference_max_retries - 1:
                return {"error": str(e)}
    
    return {"error": "达到最大重试次数"}


def run_inference(prompt: str, target: str = None) -> None:
    """
    简单的推理接口（向后兼容）
    """
    cfg = Config()
    
    # 将简单文本转换为对话格式
    dialogue_text = f"A: {prompt}"
    target_utterance = target if target else prompt
    
    print("运行 ABSA v2.0 推理...")
    result = run_inference_v2(dialogue_text, target_utterance, cfg)
    
    print("\n--- 推理结果 ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABSA v2.0 inference")
    parser.add_argument("--dialogue", type=str, help="对话文本", default=None)
    parser.add_argument("--target", type=str, help="目标语句", default=None)
    parser.add_argument("--merge", action="store_true", help="将 LoRA 权重合并并导出成独立模型")
    parser.add_argument("--export_dir", type=str, default="./output_model/merged", help="合并模型的输出目录")
    args = parser.parse_args()

    cfg = Config()

    if args.merge:
        model, tokenizer = _load_model_and_tokenizer(cfg)
        print("合并 LoRA 到基础模型...")
        merged_model = model.merge_and_unload()
        os.makedirs(args.export_dir, exist_ok=True)
        merged_model.save_pretrained(args.export_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.export_dir)
        print(f"✅ 合并完成，模型已保存到 {args.export_dir}")
        return

    if args.dialogue:
        # 使用提供的对话
        dialogue_text = args.dialogue
        target_utterance = args.target if args.target else dialogue_text.split('\n')[-1]
        result = run_inference_v2(dialogue_text, target_utterance, cfg)
        print("\n--- 推理结果 ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # 使用默认测试示例
        print("使用默认测试示例...")
        test_dialogue = "A: I really enjoyed the movie last night.\nB: Oh really? I thought it was quite boring."
        test_target = "I thought it was quite boring."
        result = run_inference_v2(test_dialogue, test_target, cfg)
        print("\n--- 推理结果 ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
