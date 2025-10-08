import argparse
import json
import os
from typing import Dict, List

import torch
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint

from config import Config
from data_processor import (
    build_dialogue_context,
    create_prompt_for_step1,
    create_prompt_for_step2,
    create_prompt_for_step3,
    create_prompt_for_step4,
    create_prompt_for_step5,
)


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
        do_sample = cfg.generation_temperature > 0.0

        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.generation_max_new_tokens,
            temperature=cfg.generation_temperature,
            top_p=cfg.generation_top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _parse_model_output(output: str) -> Dict:
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


def run_c_cos_chain(model, tokenizer, dialogue: List[Dict[str, str]], target: str, cfg: Config) -> Dict:
    context = build_dialogue_context(dialogue)
    meta = {
        "conversation_id": "inference",
        "turn_index": len(dialogue) - 1,
        "speaker": dialogue[-1].get("speaker", "User"),
        "utterance": dialogue[-1].get("text", ""),
    }

    chain_outputs: Dict[str, Dict] = {}
    annotation_stub: Dict = {}

    step_sequence = [
        ("step1", create_prompt_for_step1),
        ("step2", create_prompt_for_step2),
        ("step3", create_prompt_for_step3),
        ("step4", create_prompt_for_step4),
        ("step5", create_prompt_for_step5),
    ]

    for step_name, step_fn in step_sequence:
        instruction, _ = step_fn(context, target, annotation_stub, meta)
        if chain_outputs:
            instruction = (
                f"{instruction}\n\n前序步骤摘要："
                f"\n{json.dumps(chain_outputs, ensure_ascii=False)}"
            )
        output_text = _generate(model, tokenizer, instruction, cfg)
        parsed_output = _parse_model_output(output_text)
        chain_outputs[step_name] = parsed_output
        if isinstance(parsed_output, dict):
            annotation_stub.update(parsed_output)

    return chain_outputs


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


def run_inference(prompt: str, target: str) -> None:
    cfg = Config()
    model, tokenizer = _load_model_and_tokenizer(cfg)

    sample_dialogue = [
        {"speaker": "User", "text": prompt},
    ]

    print("运行 C-CoS 五步推理链...")
    final_result = run_c_cos_chain(model, tokenizer, sample_dialogue, target, cfg)

    print("\n--- 最终推理结果 ---")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABSA C-CoS inference")
    parser.add_argument("--dialogue", type=str, help="以换行分隔的对话，格式：Speaker: Utterance", default=None)
    parser.add_argument("--target", type=str, help="关注的目标或对象", default="overall experience")
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
        dialogue_lines = [line.strip() for line in args.dialogue.split("\n") if line.strip()]
        dialogue_list = []
        for line in dialogue_lines:
            if ":" in line:
                speaker, text = line.split(":", 1)
                dialogue_list.append({"speaker": speaker.strip(), "text": text.strip()})
            else:
                dialogue_list.append({"speaker": "User", "text": line})
        model, tokenizer = _load_model_and_tokenizer(cfg)
        print("运行 C-CoS 五步推理链...")
        final_result = run_c_cos_chain(model, tokenizer, dialogue_list, args.target, cfg)
        print("\n--- 最终推理结果 ---")
        print(json.dumps(final_result, indent=2, ensure_ascii=False))
    else:
        default_prompt = "今天试运行 MiniCPM 的适配效果，感觉对话小助手很聪明。"
        run_inference(default_prompt, args.target)


if __name__ == "__main__":
    main()
