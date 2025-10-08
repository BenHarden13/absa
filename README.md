# ABSA-C-CoS

面向小型语言模型的因果链式情感分析（Aspect-Based Sentiment Analysis with Causal Chain-of-Sentiment, 简称 C-CoS）项目脚手架。

## 项目简介

该项目提供了一个完整的多任务指令微调范例，目标是在 MiniCPM 等小型语言模型上，通过五步“因果情感链”推理过程，实现结构化的对话情感分析。训练部分使用 LoRA 参数高效微调与 4-bit 量化加载，推理脚本展示了如何串联五步推理链获取最终结果。

## 目录结构

```text
absa-c-cos/
├── config.py              # 全局配置
├── data_processor.py      # 数据读取与五步链条样本构建
├── inference.py           # 推理演示脚本
├── train.py               # 主训练入口
├── requirements.txt       # 依赖列表
├── README.md
├── data/
│   ├── dailydialog_train.json
│   └── dailydialog_valid.json
└── output_model/          # 训练输出目录
```

## 使用 Anaconda 创建环境

```powershell
conda create -n absa-c-cos python=3.10 -y
conda activate absa-c-cos
pip install -r requirements.txt
```

> 💡 **提示**：若使用 GPU，请确保安装了兼容的 CUDA 驱动，并根据硬件情况调整 `torch` 安装方式。

## 准备数据

- 默认训练数据路径：`data/dailydialog_train.json`
- 可以将真实的 ABSA 数据转换为相同的字典结构：`{conversation_id: [turn, ...]}`，其中带情感标注的轮次需包含 `annotations` 列表。

样本结构示例：

```json
{
  "speaker": "B",
  "text": "It was great, I finally got promoted!",
  "annotations": [
    {
      "target": "career progress",
      "emotion": "joy",
      "cause": "Received a promotion at work",
      "evidence": "It was great, I finally got promoted!",
      "confidence": "high"
    }
  ]
}
```

## 启动训练

```powershell
python train.py
```

训练完成后，会在 `output_model/final_checkpoint/` 下保存 LoRA 适配器与分词器信息。

## 运行推理链条

```powershell
python inference.py
```

推理脚本会加载基础模型与 LoRA 适配器，执行 C-CoS 五步推理，并输出 JSON 化的链式结果。

## 下一步建议

- 根据实际业务数据扩展 `EMOTION_TO_POLARITY` 字典。
- 在真实语料上丰富五步标签，以提升模型的泛化表现。
- 将评估脚本整合到 `train.py` 中，结合验证集监控训练过程。
