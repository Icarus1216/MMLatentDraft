"""
RLD (Reflective Latent Draft) —— 面向 Qwen3-VL 的步级"回看-验证"隐空间草稿

核心模块:
- modules: 基础组件 (Resampler, CrossAttention, Embedding Projector 等)
- controller: RLD 控制器，编排 draft 生命周期
- model: 包装 Qwen3-VL 的 RLD 模型
- data: 数据集与 Collator
- trainer: 自定义 Trainer (支持步段展开训练)
"""

from .modules import (
    CrossAttentionBlock,
    EvidenceResampler,
    StepResampler,
    TraceUpdater,
    ReflectionModule,
    DraftUpdater,
    ResidualFlowDraftUpdater,
    DraftReadoutAdapter,
    MultiLayerDraftReadout,
    InSituDraftInjector,
)
from .controller import RLDController
from .model import RLDModel
from .data import RLDDataset, RLDCollator
from .trainer import RLDTrainer

__version__ = "0.1.0"
