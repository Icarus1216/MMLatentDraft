# NLD: Native Latent Draft

**基于 VLM 原生隐空间的自适应多步推理解码**

> Native Latent Draft — 在视觉语言模型的隐空间中进行多步推理演化，无需外挂模块，等价于 COCONUT 的连续思考过程。

## 核心思想

NLD 在 VLM 的 step boundary 处插入隐空间思考步骤，将 Transformer 当作 RNN cell 使用：

```
Input → [Segment 1] → <|latent|> → [Hidden Thinking ×N steps] → <|/latent|> → [Segment 2] → Answer
                                         ↑                ↓
                                    step_embedding    thought_output
                                         ↑                ↓
                                    KV cache ←── text_model() (全部 36 层)
```

**与 COCONUT 的等价性**：
- COCONUT: last hidden → 过 VLM 全部层 → 新 hidden (不解码) → 循环
- NLD: last hidden → text_model() (全部 36 层 + norm, with KV cache) → 新 hidden → 循环
- 两者都把 Transformer 当 RNN cell 用，hidden state 循环反馈

**关键设计**：
- **单 token recurrence**：每步只用最后一个 token 的 hidden state 作为"查询探针"，历史信息全在 KV cache 中
- **双流推理**：隐式推理（hidden space）+ 显式推理（自然语言 CoT）构成统一序列
- **自适应退出**：退出 token 预测 OR 饱和信号（双保险）
- **训推一致**：训练和推理使用完全相同的 forward 路径

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    NLDModelForVL (model_v2.py)              │
│                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────┐   │
│  │  Qwen3-VL    │    │   NativeLatentThinker             │   │
│  │  (8.3B)      │    │   (latent_thinker.py)             │   │
│  │              │    │                                    │   │
│  │  - Vision    │    │   - step_embedding (标识演化步数)  │   │
│  │  - Text      │    │   - output_norm (prefix 归一化)    │   │
│  │  - lm_head   │    │   - visual_probe (视觉信息探针)    │   │
│  │              │    │                                    │   │
│  └──────┬───────┘    └──────────────┬───────────────────┘   │
│         │                           │                       │
│         │    Segment-wise Forward   │                       │
│         │    + Latent Thinking      │                       │
│         └───────────┬───────────────┘                       │
│                     ▼                                       │
│              Unified Sequence                               │
│    [visual] [text] [thought×N] [text] ... [answer]          │
└─────────────────────────────────────────────────────────────┘
```

## 监督信号

| Loss | 作用 | 描述 |
|:---|:---|:---|
| **CE Loss** | 主损失 | 标准 next-token prediction，thought 位置 labels=-100 不参与 |
| **Exit Token Loss** | 退出信号 | 最后一步 hidden → lm_head (frozen) → 应预测 `<|/latent|>` |
| **SCA Loss** | 语义锚定 | Soft Concept Anchoring — hidden → lm_head (frozen) → KL 散度对齐 key tokens 的语义邻域 |
| **Complementarity Loss** | 互补性 | 鼓励 thought 与 CoT 互补而非冗余 |
| **Visual Probe Loss** | 视觉保持 | 确保 thought hidden 编码视觉信息 |

**lm_head 冻结策略**：
- CE loss 路径：lm_head 正常训练（需要学习新 token 的输出权重）
- SCA / Exit Token Loss 路径：lm_head 冻结（detach），梯度只回传到 hidden states

## 项目结构

```
LatentDraft/
├── rld/                          # 核心模块
│   ├── model_v2.py               # NLDModelForVL — 主模型，segment-wise forward
│   ├── latent_thinker.py         # NativeLatentThinker — 隐空间多步推理模块
│   ├── data.py                   # NLDDataset + NLDCollator — 数据加载与预处理
│   ├── trainer_nld.py            # NLDTrainer — 自定义 Trainer（双学习率、监控）
│   └── __init__.py
├── configs/
│   ├── nld_train_phase1.yaml     # Phase 1 训练配置
│   └── fsdp_config.json          # FSDP 分布式配置
├── scripts/
│   ├── train_nld.py              # 训练入口
│   ├── inference.py              # 推理脚本
│   ├── generate_vcr_latent_cot.py  # VCR Latent CoT 数据生成
│   ├── run_train_nld.sh          # 训练启动脚本
│   └── run_generate_vcr*.sh      # 数据生成脚本
├── start_training.sh             # 一键启动训练（含环境激活）
├── requirements.txt
└── .gitignore
```

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.1+
- 8× GPU（FSDP 分布式训练）
- Flash Attention 2

### 安装

```bash
pip install -r requirements.txt
```

### 训练

```bash
# 默认启动 Phase 1 训练（8卡 FSDP）
bash start_training.sh

# 恢复训练
bash start_training.sh train --resume /path/to/checkpoint

# 指定配置文件
bash start_training.sh train --config configs/nld_train_phase1.yaml
```

### 推理

```bash
bash start_training.sh inference --checkpoint /path/to/checkpoint
```

## 训练配置

| 参数 | 值 | 说明 |
|:---|:---|:---|
| 基座模型 | Qwen3-VL-8B-Instruct | 8.3B 参数 VLM |
| 训练数据 | VCR Latent CoT 20K | 视觉常识推理 |
| 有效 batch size | 32 | 1 × 4 (grad_accum) × 8 (GPUs) |
| 学习率 (VLM) | 2e-5 | cosine scheduler |
| 学习率 (Thinker) | 1e-4 | 从头训练，较大 lr |
| 训练轮数 | 2 epochs | ~1250 steps |
| 最大隐空间步数 | 5 | 覆盖 99.98% 的 num_stages 分布 |
| 分布式策略 | FSDP full_shard | 替代 DeepSpeed，避免 KV cache 冲突 |

## 特殊 Token

| Token | 作用 |
|:---|:---|
| `<\|latent\|>` | 进入隐空间推理模式 |
| `<\|/latent\|>` | 退出隐空间推理模式 |
| `<\|pause\|>` | 原始数据中的 step boundary 标记（预处理时替换） |

## 数据格式

训练数据为 JSON 格式，每条样本包含：

```json
{
  "image": "vcr1images/xxx.jpg",
  "question": "Why is person1 looking at person2?",
  "answer": "Because ...",
  "reasoning_for_training": "Stage 1: ... <|pause|> Stage 2: ... <|pause|> ...",
  "num_stages": 3,
  "latent_key_tokens": [
    {"stage": "Stage 1", "tokens": ["keyword1", "keyword2", "keyword3"]},
    {"stage": "Stage 2", "tokens": ["keyword4", "keyword5", "keyword6"]}
  ]
}
```

预处理时 `<|pause|>` 被替换为 `<|latent|> <|/latent|>`，模型学习在正确位置输出这些 token，中间的隐空间推理在 hidden space 中完成。

## License

MIT
