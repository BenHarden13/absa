import json
import re
from datasets import Dataset

# 情感映射保持不变
EMOTION_TO_POLARITY = {
    'happiness': 'positive', 'joy': 'positive', 'love': 'positive', 
    'excitement': 'positive', 'pride': 'positive', 'admiration': 'positive',
    'sadness': 'negative', 'anger': 'negative', 'fear': 'negative', 
    'disgust': 'negative', 'disappointment': 'negative', 'embarrassment': 'negative',
    'surprise': 'neutral', 'neutral': 'neutral', 'anticipation': 'neutral'
}

def extract_dialogue_text(conversation_turns):
    """提取对话文本"""
    dialogue_parts = []
    for i, turn in enumerate(conversation_turns):
        speaker = 'A' if i % 2 == 0 else 'B'
        dialogue_parts.append(f"{speaker}: {turn['text']}")
    return "\n".join(dialogue_parts)

def create_unified_datapoint_v2(full_dialogue_turns, target_idx):
    """
    v2版本的数据处理函数，创建统一的数据点
    针对防弹指令和结构化输出进行优化
    """
    dialogue_text = extract_dialogue_text(full_dialogue_turns)
    target_turn = full_dialogue_turns[target_idx]
    target_utterance = target_turn['text']
    
    # 确定情感持有者
    holder = 'A' if target_idx % 2 == 0 else 'B'
    
    # 提取基本情感信息
    emotion = target_turn.get('emotion', 'neutral')
    sentiment = EMOTION_TO_POLARITY.get(emotion.lower(), 'neutral')
    
    # 提取或推断其他要素
    target_entity = target_turn.get('target', 'conversation topic')
    aspect = target_turn.get('aspect', 'general')
    opinion = target_utterance  # 使用整个目标语句作为观点表达
    
    # 分析情感原因
    cause = "previous dialogue context" if target_idx > 0 else "current situation"
    
    # 检测情感翻转
    is_flipping = False
    trigger = "N/A"
    if target_idx > 0:
        prev_emotion = full_dialogue_turns[target_idx-1].get('emotion', 'neutral')
        prev_sentiment = EMOTION_TO_POLARITY.get(prev_emotion.lower(), 'neutral')
        if prev_sentiment != sentiment and sentiment != 'neutral':
            is_flipping = True
            trigger = "response to previous statement"
    
    return {
        "dialogue_text": dialogue_text,
        "target_utterance": target_utterance,
        "elements": {
            "holder": holder,
            "target": target_entity,
            "opinion": opinion,
            "sentiment": sentiment,
            "cause": cause,
            "flipping": str(is_flipping),
            "trigger": trigger
        }
    }

def create_bulletproof_prompt_and_answer(unified_datapoint):
    """
    创建防弹指令和标准答案
    这是v2.0版本的核心改进功能
    """
    dialogue = unified_datapoint['dialogue_text']
    target = unified_datapoint['target_utterance']
    elements = unified_datapoint['elements']
    
    # 防弹指令模板 (Bulletproof Prompt)
    prompt = f"""### 任务说明
你是一个专业的对话情感分析专家。你的唯一任务是从给定的对话中提取情感分析要素，并严格按照指定的JSON格式输出结果。

### 输出格式要求
你必须输出一个完整的JSON对象，包含以下字段：
- "holder": 情感持有者，必须是"A"或"B"
- "target": 情感指向的目标对象
- "opinion": 表达情感的具体句子
- "sentiment": 情感极性，必须是"positive"、"negative"或"neutral"之一
- "cause": 导致该情感的原因
- "flipping": 情感是否发生翻转，必须是"True"或"False"
- "trigger": 情感翻转的触发因素，如果未翻转则为"N/A"

### JSON格式示例
{{
  "holder": "A",
  "target": "movie",
  "opinion": "I really enjoyed that film",
  "sentiment": "positive",
  "cause": "personal experience",
  "flipping": "False",
  "trigger": "N/A"
}}

### 对话内容
{dialogue}

### 需要分析的目标语句
"{target}"

### 要求
1. 仅输出JSON格式的结果
2. 不要添加任何解释或额外文字
3. 确保JSON格式完全正确
4. 所有字段都必须包含

### JSON输出:
"""

    # 标准答案 - 严格的JSON格式
    answer = json.dumps(elements, ensure_ascii=False, indent=None)
    
    return {"prompt": prompt, "answer": answer}

def load_and_process_data_v2(json_path):
    """
    v2版本的主数据处理函数
    增强了数据处理的鲁棒性和训练效果
    """
    print(f"Loading data from: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_path} not found!")
        return Dataset.from_list([])
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_path}: {e}")
        return Dataset.from_list([])
    
    multitask_data = []
    processed_count = 0
    
    for conversation_id, conversation_list in raw_data.items():
        if not isinstance(conversation_list, list) or len(conversation_list) == 0:
            continue
            
        conversation_turns = conversation_list[0] if isinstance(conversation_list[0], list) else conversation_list
        
        for i, turn_data in enumerate(conversation_turns):
            if not isinstance(turn_data, dict):
                continue
                
            # 只处理包含非中性情感的语句
            if 'emotion' in turn_data and turn_data['emotion'] != 'neutral':
                try:
                    unified_dp = create_unified_datapoint_v2(conversation_turns, i)
                    processed_sample = create_bulletproof_prompt_and_answer(unified_dp)
                    
                    # 为SFTTrainer创建text字段
                    training_text = processed_sample['prompt'] + processed_sample['answer']
                    multitask_data.append({
                        "text": training_text,
                        "prompt": processed_sample['prompt'],
                        "answer": processed_sample['answer']
                    })
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing turn {i} in conversation {conversation_id}: {e}")
                    continue
    
    print(f"Successfully processed {processed_count} samples")
    return Dataset.from_list(multitask_data)

def validate_dataset(dataset):
    """验证数据集的质量"""
    if len(dataset) == 0:
        print("Warning: Dataset is empty!")
        return False
    
    print(f"Dataset size: {len(dataset)}")
    
    # 检查前几个样本
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n--- Sample {i+1} ---")
        print(f"Text length: {len(sample['text'])}")
        if 'answer' in sample:
            try:
                # 验证答案是否为有效JSON
                json.loads(sample['answer'])
                print("✓ Valid JSON answer")
            except json.JSONDecodeError:
                print("✗ Invalid JSON answer")
    
    return True
        response.raise_for_status()
        return response.json()

    path = Path(json_source)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{json_source}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_and_process_data(
    json_path: str,
    max_samples: int = None,
    shuffle: bool = False,
    seed: int = 42,
) -> Dataset:
    raw_data = _load_raw_data(json_path)

    multitask_data: List[Dict[str, str]] = []

    for conversation_id, conversation_list in raw_data.items():
        if not isinstance(conversation_list, list) or not conversation_list:
            continue

        turns = conversation_list[0] if isinstance(conversation_list[0], list) else conversation_list
        for turn_index, turn in enumerate(turns):
            context_turns = turns[: turn_index + 1]
            dialogue_context = build_dialogue_context(context_turns)
            meta = {
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "speaker": turn.get("speaker") or f"Speaker {turn_index % 2 + 1}",
                "utterance": turn.get("utterance") or "",
            }

            emotion = turn.get("emotion")
            if emotion:
                annotation = {
                    "emotion": emotion,
                    "evidence": meta["utterance"],
                }
                target = "整体体验"
                annotation.setdefault("polarity", ensure_polarity(annotation))

                for step_fn in (
                    create_prompt_for_step1,
                    create_prompt_for_step2,
                    create_prompt_for_step3,
                    create_prompt_for_step4,
                    create_prompt_for_step5,
                ):
                    instruction, answer = step_fn(dialogue_context, target, annotation, meta)
                    multitask_data.append({"text": format_instruction_response(instruction, answer)})

    if not multitask_data:
        raise ValueError("未从提供的数据中提取到任何带标注的样本，请检查数据格式。")

    dataset = Dataset.from_list(multitask_data)
    total = len(dataset)

    if max_samples is not None and total > max_samples:
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(max_samples))
        print(f"✅ 已根据配置采样 {len(dataset)} / {total} 条数据用于训练。")
    elif max_samples is not None:
        print(f"✅ 数据量 {total} 条，未达到采样上限 {max_samples}。")

    return dataset