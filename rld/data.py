"""
RLD 数据处理 (适配 Qwen3-VL-8B-Instruct)

Dataset: 加载图像+问答数据，使用 Qwen3-VL-Instruct 原生格式:
         "Step 1: ...\nStep 2: ...\nFinal Answer: ..."
         在 "Step N:" 边界触发 RLD Controller 更新 Z_d
         支持三种推理链来源:
           1. free_reasoning: 自由推理正确 (最高质量, 分布最匹配)
           2. corrected_free_reasoning: 纠正式推理 (含"犯错→纠正"模式)
           3. fabricated (fallback): 自动用 answer 碎片构造占位推理链

         格式设计动机:
           Qwen3-VL-8B-Instruct 不支持 <think></think> 格式 (OOD token),
           但能稳定生成 "Step N:" + "Final Answer:" 格式。
           使用模型原生格式确保:
           - 基座 hidden states 在分布内, Controller 提取的 step summary 质量更高
           - 推理时模型能自主生成标准格式, 无需硬编码注入
           - 训推格式完全一致, 消除 exposure bias

         Labels 策略 (分段监督):
           - 高质量推理链 (free_reasoning/corrected_free_reasoning/correct_cot):
             推理步骤参与 loss → 为 Controller 提供密集梯度信号
           - 低质量/伪造推理链 (wrong_cot/fabricated 等):
             推理步骤 labels = -100 → 只监督 final answer
           - prompt 部分: 始终 -100
           - final answer: 始终参与 loss

         数据混合 (correct_cot 混合训练):
           已移至预处理阶段 (scripts/prepare_training_data.py) 静态完成。
           从原始数据集中选择拒绝采样未覆盖的样本，转换格式后与 wrong_cot 混合。
           训练集 JSON 中已包含混合好的 wrong_cot + correct_cot 样本，
           data.py 不再做运行时混合。

         Answer 标准化 (方案 A):
           对 GT answer 做预处理标准化，消除格式差异:
           - LaTeX 包裹去除: \(24 + 8π\) → 24 + 8π
           - 选择题前缀清理: "D: 193.3" → "D"
           - 数值格式统一: 去除尾部零
Collator: 批处理，检测 step 边界位置
"""

import json
import os
import re
import random
import warnings
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor


class _SampleTooLongError(Exception):
    """样本序列超长且无法截断时抛出，触发 _get_item_safe 重试其他样本"""
    pass

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("[RLD Data] ⚠️ qwen_vl_utils 未安装，请运行: pip install qwen-vl-utils")
    process_vision_info = None


# ====== RLD System Prompt (Qwen3-VL-Instruct 原生格式) ======
# Qwen3-VL-8B-Instruct 不支持 <think></think> 格式 (OOD token),
# 但能稳定生成 "Step N:" + "Final Answer:" 格式。
# 使用与 rejection sampling 完全一致的 prompt, 确保训推格式一致。
# Step boundary 检测: 遇到 "Step N:" (N≥2) 或 "Final Answer:" 时触发 Z_d 更新。
# Step 1 使用初始 Z_d_0 (来自 prefill), 不触发更新。
#
# 设计理念: 保持推理简洁但不施加硬性长度限制, 不人为要求 verify/backtrack。
# 验证和纠错由 RLD 的 Controller 在隐空间完成, 不需要模型在显式推理链中做。
# 更多 step boundary = Controller 有更多修正机会。
RLD_SYSTEM_PROMPT = """You are a visual reasoning assistant. Think step by step. Be concise.

Rules:
1. Use numbered steps: "Step 1:", "Step 2:", etc.
2. Each step should focus on one reasoning action (observe, calculate, deduce, etc.).
3. Do NOT add introductions, greetings, summaries, or filler words.
4. End with "Final Answer:" on its own line. This is MANDATORY.

Example:
Step 1: The triangle has base 6 and height 8.
Step 2: Area = 0.5 × 6 × 8 = 24.
Final Answer: 24 square units."""


class RLDDataset(Dataset):
    """
    RLD 数据集 (Qwen3-VL 适配)
    
    训练数据格式 (基本): 图像 + query + answer
    ```json
    {
        "image": "/path/to/image.jpg",
        "question": "What is the area of the triangle?",
        "answer": "The area is 24 square units."
    }
    ```
    
    训练数据格式 (预生成, 推荐): 额外携带 think_chain 字段
    ```json
    {
        "image": "/path/to/image.jpg",
        "question": "What is the area of the triangle?",
        "answer": "The area is 24 square units.",
        "think_chain": "<think>\\nI see a triangle...\\n</step>\\nArea = ...\\n</step>\\n</think>\\n",
        "think_chain_status": "valid"
    }
    ```
    
    think 块来源优先级:
      1. 数据中的 "think_chain" 字段 (由 pregenerate_think.py 用 vLLM 预生成)
      2. 伪造: 用 answer 碎片自动构造占位 think 块 (fallback)
    
    训练时的序列构造 (符合 Qwen3-VL 官方 chat_template):
      [prompt]<|im_start|>assistant\n<think>\n{steps}\n</step>\n</think>\n\n{answer}<|im_end|>
      其中 `<|im_start|>assistant\n` 由 apply_chat_template(add_generation_prompt=True) 自动生成,
      我们手动拼接 `<think>\n{think_inner}</think>\n\n` 包裹推理过程,
      `</think>\n\n` 作为过渡衔接 think 块和 final answer。
      
    labels 策略 (分段监督, 根据 think_chain_source 决定):
      - prompt 部分: -100 (不参与 loss)
      - <think>...</think> 部分:
          * free_reasoning / corrected_free_reasoning: 真实 token id (参与 loss)
          * fabricated / 其他: -100 (不参与 loss)
      - </think> 后的 final answer + <|im_end|>: 真实 token id (始终参与 loss)
    
    推理时:
      模型根据 system prompt 的显式指令，输出 <think>...</think> 包裹的推理过程，
      遇到 </step> 触发 RLD 更新，</think>\n\n 后输出最终答案。
      推理时在 generation prompt 后追加 `<think>\n` 确保训推一致。
    
    Args:
        json_path: 数据 JSON 文件路径
        processor: AutoProcessor
        auto_split_steps: 是否自动生成 <think> 块中的伪推理步骤 (仅 fallback 时使用)
        max_tokens_per_step: 自动分段时每段最大 token 数 (fallback 按此长度自然切分 answer)
        use_system_prompt: 是否在对话中注入 RLD system prompt (推荐 True)
        system_prompt: 自定义 system prompt (None 则使用默认 RLD_SYSTEM_PROMPT)
    """

    # Step boundary 检测标记
    # 新格式: "Step N:" + "Final Answer:" (Qwen3-VL-Instruct 原生格式)
    # Step boundary 触发点: "Step 2:", "Step 3:", ..., "Final Answer:"
    # Step 1 不触发更新 (使用初始 Z_d_0)
    STEP_BOUNDARY_PATTERN = re.compile(
        r'(?:^|\n)\s*(?:Step\s+(\d+)\s*[:.]|Final\s+Answer\s*:)',
        re.IGNORECASE
    )
    FINAL_ANSWER_PATTERN = re.compile(
        r'(?:^|\n)\s*Final\s+Answer\s*:\s*',
        re.IGNORECASE
    )
    # 保留旧常量用于兼容数据解析 (旧格式数据中仍有 <think>...</think>)
    STEP_DELIMITER = "</step>"
    THINK_START = "<think>"
    THINK_END = "</think>"

    def __init__(
        self,
        json_path: str,
        processor,
        auto_split_steps: bool = True,
        max_tokens_per_step: int = 64,
        use_system_prompt: bool = True,
        system_prompt: str = None,
        max_seq_len: int = 4096,
        skip_image_check: bool = False,
        hard_sample_max_ratio: float = 0.22,
    ):
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        if skip_image_check:
            # 跳过图片存在性检查 (数据已通过 scripts/prefilter_data.py 预过滤)
            # 仅做最轻量的字段检查: 移除无 image 字段的样本
            valid_data = [item for item in raw_data if item.get('image', '')]
            no_image_count = len(raw_data) - len(valid_data)
            if no_image_count > 0:
                print(f"[RLDDataset] ⚠️ 已过滤 {no_image_count} 个无图像字段的样本")
                print(f"[RLDDataset]   保留 {len(valid_data)}/{len(raw_data)}")
            print(f"[RLDDataset] ✅ 已跳过图片存在性检查 (skip_image_check=True)")
        else:
            # 动态过滤: 逐个检查图片路径是否存在 (较慢，约 4 分钟/297K 样本)
            valid_data = []
            invalid_count = 0
            no_image_count = 0
            for item in raw_data:
                img = item.get('image', '')
                if not img:
                    no_image_count += 1
                    continue
                if not os.path.exists(img):
                    invalid_count += 1
                    continue
                valid_data.append(item)
            if invalid_count > 0:
                print(f"[RLDDataset] ⚠️ 已过滤 {invalid_count} 个图片缺失的样本")
            if no_image_count > 0:
                print(f"[RLDDataset] ⚠️ 已过滤 {no_image_count} 个无图像的样本")
            if invalid_count > 0 or no_image_count > 0:
                print(f"[RLDDataset]   保留 {len(valid_data)}/{len(raw_data)}")
        self.data = valid_data
        
        # 过滤超高段数样本 (>14 段的样本会导致显存峰值过高, 触发 CUDA 错误)
        MAX_STEPS_FILTER = 14
        before_filter = len(self.data)
        self.data = [
            item for item in self.data
            if self._count_steps_in_chain(item.get('think_chain', '')) <= MAX_STEPS_FILTER
        ]
        filtered_by_steps = before_filter - len(self.data)
        if filtered_by_steps > 0:
            print(f"[RLDDataset] ⚠️ 已过滤 {filtered_by_steps} 个超高段数(>{MAX_STEPS_FILTER})的样本")
            print(f"[RLDDataset]   保留 {len(self.data)}/{before_filter}")

        # ====== P0: 下采样 pass_rate=0 的困难样本, 防止 draft 学坏 ======
        # pass_rate=0 表示基座 4 次采样全错, hidden states 偏差过大,
        # draft controller 难以从中学到有效修正, 过多会导致学到激进的修正策略。
        # 将 pr=0 样本下采样到不超过总数据的 hard_sample_max_ratio 比例。
        self.hard_sample_max_ratio = hard_sample_max_ratio
        if hard_sample_max_ratio < 1.0:
            hard_samples = [item for item in self.data
                            if item.get('rejection_sampling_meta', {}).get('pass_rate', -1) == 0]
            non_hard_samples = [item for item in self.data
                                if item.get('rejection_sampling_meta', {}).get('pass_rate', -1) != 0]
            if len(hard_samples) > 0:
                # 目标: hard 样本数 ≤ non_hard * ratio / (1 - ratio)
                max_hard = int(len(non_hard_samples) * hard_sample_max_ratio / max(1.0 - hard_sample_max_ratio, 0.01))
                if len(hard_samples) > max_hard:
                    random.seed(42)  # 确保可复现
                    hard_samples_downsampled = random.sample(hard_samples, max_hard)
                    print(f"[RLDDataset] 🎯 P0 困难样本下采样: pass_rate=0 从 {len(hard_samples)} → {max_hard} "
                          f"(目标比例 ≤{hard_sample_max_ratio:.0%})")
                    self.data = non_hard_samples + hard_samples_downsampled
                    random.shuffle(self.data)  # 打乱顺序
                    print(f"[RLDDataset]   下采样后总样本数: {len(self.data)}")
                else:
                    print(f"[RLDDataset] ✅ P0: pass_rate=0 样本 ({len(hard_samples)}) "
                          f"已在目标比例内 (≤{hard_sample_max_ratio:.0%}), 无需下采样")

        # 注意: correct_cot 混合已移至预处理阶段 (scripts/prepare_training_data.py)
        # 训练集 JSON 中已包含混合好的 wrong_cot + correct_cot (dataset_converted) 样本
        # data.py 不再做运行时混合, 直接加载即可

        self.processor = processor
        self.auto_split_steps = auto_split_steps
        self.max_tokens_per_step = max_tokens_per_step
        self.use_system_prompt = use_system_prompt
        self.system_prompt = system_prompt or RLD_SYSTEM_PROMPT
        self.max_seq_len = max_seq_len

        # 确认图片分辨率限制已生效 (由 train.py 在 processor 上设置)
        if hasattr(self.processor, 'image_processor'):
            _ip = self.processor.image_processor
            _min_px = getattr(_ip, 'min_pixels', None)
            _max_px = getattr(_ip, 'max_pixels', None)
            print(f"[RLDDataset] 图片分辨率: min_pixels={_min_px}, max_pixels={_max_px}")
            if _max_px and _max_px > 2_000_000:
                print(f"[RLDDataset] ⚠️ max_pixels={_max_px} 过大! 建议设为 1003520 以避免 ViT 编码缓慢")

        # 确保 tokenizer 配置
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

        # Qwen3-VL 的 image placeholder token id (从 processor 配置获取)
        # Qwen3-VL: 151655
        try:
            self.IMAGE_TOKEN_ID = self.processor.image_token_id
        except AttributeError:
            self.IMAGE_TOKEN_ID = 151655  # Qwen3-VL 默认值

        # 高质量来源: think 块全部参与 loss
        # correct_cot: 从拒绝采样数据中提取的正确推理链 (混合训练)
        self.SUPERVISED_THINK_SOURCES = {"free_reasoning", "corrected_free_reasoning", "dataset_converted", "correct_cot"}

        # 统计预生成覆盖率 (按来源分类)
        source_counts = {}
        for item in self.data:
            src = item.get('think_chain_source', 'fabricated')
            if item.get('think_chain') and item.get('think_chain_status') == 'valid':
                source_counts[src] = source_counts.get(src, 0) + 1
        
        num_pregenerated = sum(source_counts.values())
        num_supervised_think = sum(
            v for k, v in source_counts.items() if k in self.SUPERVISED_THINK_SOURCES
        )
        self.pregenerated_ratio = num_pregenerated / max(len(self.data), 1)

        print(f"[RLDDataset] 加载 {len(self.data)} 个样本 from {json_path}")
        print(f"[RLDDataset] Qwen3-VL 格式, 分段监督 labels")
        print(f"[RLDDataset] 预生成推理链覆盖率: {num_pregenerated}/{len(self.data)} "
              f"({self.pregenerated_ratio:.1%})")
        if source_counts:
            print(f"[RLDDataset] 按来源统计:")
            for src, cnt in sorted(source_counts.items()):
                supervised = "✅ think+answer loss" if src in self.SUPERVISED_THINK_SOURCES else "⚠️ answer-only loss"
                print(f"  {src}: {cnt} → {supervised}")
        print(f"[RLDDataset] think 块参与 loss 的样本: {num_supervised_think} "
              f"({num_supervised_think/max(len(self.data),1):.1%})")
        if num_pregenerated == 0:
            print(f"[RLDDataset] ⚠️ 无预生成推理链, 全部使用伪造 think 块 "
                  f"(建议运行 scripts/pregenerate_think.py 预生成)")

        # 预检查
        self._preflight_check(min(5, len(self.data)))

    @staticmethod
    def _count_steps_in_chain(think_chain: str) -> int:
        """统计推理链中的步骤数 (兼容新旧格式)"""
        if not think_chain:
            return 0
        # 新格式: 统计 "Step N:" 的数量
        step_n_count = len(re.findall(r'(?:^|\n)\s*Step\s+\d+\s*[:.\s]', think_chain, re.IGNORECASE))
        if step_n_count > 0:
            return step_n_count
        # 旧格式: 统计 </step> 的数量
        return think_chain.count('</step>')

    def _preflight_check(self, n: int):
        """预检查前 n 个样本"""
        for i in range(n):
            item = self.data[i]
            answer = item.get('answer', '')
            if not answer.strip():
                print(f"[RLDDataset] ⚠️ 样本 {i} 的 answer 为空")
            # 检查预生成 think_chain 格式
            think_chain = item.get('think_chain')
            if think_chain:
                num_steps = self._count_steps_in_chain(think_chain)
                if num_steps == 0:
                    print(f"[RLDDataset] ⚠️ 样本 {i} 的 think_chain 无法检测到步骤边界")
        print(f"[RLDDataset] 预检查完成: {n} 个样本")

    def _build_cot_block(self, answer: str) -> str:
        """
        构造推理链 (fallback: 训练数据中无预生成 think_chain 时使用)
        
        使用 Qwen3-VL-Instruct 原生格式: "Step 1: ...\nStep 2: ...\n"
        按 max_tokens_per_step 自然切分 answer 文本为多步。
        
        注意: 返回值不包含 "Final Answer:" 部分!
        外层 _get_item_inner 会拼接 "Final Answer: {answer}"。
        
        这些 token 不参与 loss (fabricated 来源)，但为 RLD Controller
        提供 step 边界触发更新的机会。
        
        Args:
            answer: 原始的最终答案文本
            
        Returns:
            cot_text: 推理链文本, 格式: "Step 1: {step1}\nStep 2: {step2}\n"
        """
        steps = []
        
        if self.auto_split_steps and answer.strip():
            # 按句子拆分，然后按 max_tokens_per_step 贪心合并
            sentences = self._split_sentences(answer)
            current_step_tokens = 0
            current_step_parts = []
            
            for sentence in sentences:
                # 估算 token 数 (粗略: 按字符数 / 4 近似)
                est_tokens = max(len(sentence) // 4, 1)
                
                if current_step_parts and current_step_tokens + est_tokens > self.max_tokens_per_step:
                    # 当前步已满，收割为一步
                    steps.append(' '.join(current_step_parts))
                    current_step_parts = [sentence.strip()]
                    current_step_tokens = est_tokens
                else:
                    current_step_parts.append(sentence.strip())
                    current_step_tokens += est_tokens
            
            # 收割剩余
            if current_step_parts:
                steps.append(' '.join(current_step_parts))
            
            # 至少保证 1 步
            if not steps:
                steps = [answer.strip()]
        else:
            # 不自动分步，整个 answer 作为单步占位
            steps = [answer.strip() if answer.strip() else "Analyzing the image."]

        # 组装推理链: "Step 1: {step1}\nStep 2: {step2}\n"
        cot_parts = []
        for i, step_text in enumerate(steps, 1):
            cot_parts.append(f"Step {i}: {step_text}")
        cot_text = "\n".join(cot_parts) + "\n"
        
        return cot_text

    def _split_sentences(self, text: str) -> list:
        """按句子边界拆分文本"""
        sentences = []
        current = []
        for char in text:
            current.append(char)
            if char in '.。!！?？':
                sentence = ''.join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []
        if current:
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)
        return sentences if sentences else [text]

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """
        方案 A: Answer 预处理标准化
        
        将 GT answer 标准化为模型最可能生成的格式，消除格式差异导致的
        token 不匹配问题。标准化规则 (按优先级):
        
        1. LaTeX 包裹去除: \\(24 + 8\\pi\\) → 24 + 8π
        2. 选择题前缀清理: "D: 193.3" → "D" (只保留字母)
        3. 数值尾部零清理: "193.30" → "193.3"
        4. Markdown 格式清理: **answer** → answer
        5. 多余空白清理
        
        注意: 此方法是幂等的，多次调用结果不变。
        """
        if not answer:
            return answer
        
        text = answer.strip()
        
        # ---- 1. LaTeX 包裹去除 ----
        # \(...\) 格式 (JSON 加载后 \\( 变成 \()
        text = re.sub(r'\\\((.+?)\\\)', r'\1', text)
        # $...$ 格式 (单 $ 行内公式)
        if text.startswith('$') and text.endswith('$') and len(text) > 2:
            text = text[1:-1].strip()
        # \[...\] 格式 (display math)
        text = re.sub(r'\\\[(.+?)\\\]', r'\1', text)
        # 常见 LaTeX 命令简化
        text = text.replace('\\pi', 'π')
        text = text.replace('\\sqrt', '√')
        text = text.replace('\\times', '×')
        text = text.replace('\\div', '÷')
        text = text.replace('\\pm', '±')
        text = text.replace('\\leq', '≤')
        text = text.replace('\\geq', '≥')
        text = text.replace('\\neq', '≠')
        text = text.replace('\\approx', '≈')
        text = text.replace('\\infty', '∞')
        # \frac{a}{b} → a/b
        text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)
        # \text{...} → ... (去掉 \text 包裹)
        text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
        # \, (LaTeX 细空格) → 空格
        text = text.replace('\\,', ' ')
        
        # ---- 2. Markdown 格式清理 ----
        # **bold** → bold
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        # *italic* → italic (但不影响乘号 *)
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
        
        # ---- 3. 选择题格式标准化 ----
        # 如果 GT 是单字母选择题 (A/B/C/D/E)，直接保留字母
        stripped = text.strip()
        if len(stripped) == 1 and stripped.upper() in 'ABCDE':
            return stripped.upper()
        
        # 如果 GT 是 "D: 193.3" 或 "A. xxx" 格式，只保留字母
        # 但仅当第一个字符是选择题字母时
        m = re.match(r'^([A-Ea-e])\s*[:.]\s*.+', stripped)
        if m:
            # 检查是否确实是选择题格式 (字母后跟冒号/句号+内容)
            return m.group(1).upper()
        
        # ---- 4. 数值格式标准化 ----
        # 去除尾部零: "193.30" → "193.3", "24.00" → "24"
        # 仅对纯数值或数值+单位的情况处理
        m_num = re.match(r'^(-?[\d]+\.[\d]+)(.*)', text)
        if m_num:
            num_str = m_num.group(1)
            suffix = m_num.group(2)
            # 去除尾部零
            cleaned = num_str.rstrip('0').rstrip('.')
            text = cleaned + suffix
        
        # ---- 5. 多余空白清理 ----
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._get_item_safe(idx)

    def _get_item_safe(self, idx, _retry=0):
        """带容错的样本获取，图片加载失败或序列超长时随机选另一个样本"""
        import random
        if _retry > 10:
            raise RuntimeError(f"[RLDDataset] 连续 {_retry} 次采样失败，请检查数据集图片路径或 max_seq_len 设置")
        try:
            return self._get_item_inner(idx)
        except _SampleTooLongError as e:
            # 超长样本: 静默跳过, 随机替换 (不打印警告, 避免日志刷屏)
            if _retry == 0:
                warnings.warn(f"[RLDDataset] 样本 {idx} 超长被跳过: {e}")
            new_idx = random.randint(0, len(self.data) - 1)
            return self._get_item_safe(new_idx, _retry + 1)
        except (FileNotFoundError, OSError) as e:
            warnings.warn(f"[RLDDataset] 样本 {idx} 加载失败: {e}, 随机替换")
            new_idx = random.randint(0, len(self.data) - 1)
            return self._get_item_safe(new_idx, _retry + 1)

    def _convert_old_format_to_new(self, think_chain: str) -> str:
        """
        将旧格式 (<think>...</step>...</think>) 转换为新格式 (Step N: ... + Final Answer: ...)
        
        旧格式: "<think>\nstep1_content\n</step>\nstep2_content\n</step>\n</think>\nAnswer: xxx"
        新格式: "Step 1: step1_content\nStep 2: step2_content\n"
        
        注意: 返回值只包含推理步骤部分, 不包含 "Final Answer:" 行。
        "Final Answer:" 由外层 _get_item_inner 拼接。
        """
        text = think_chain
        
        # 1) 剥离 <think>...</think> 包裹及其后面的内容 (如 "\nAnswer: 12")
        think_end_pos = text.rfind(self.THINK_END)
        if think_end_pos != -1:
            text = text[:think_end_pos]
        if text.startswith(self.THINK_START):
            text = text[len(self.THINK_START):]
        text = text.strip('\n')
        
        # 2) 按 </step> 分割为步骤
        parts = text.split(self.STEP_DELIMITER)
        steps = [p.strip() for p in parts if p.strip()]
        
        if not steps:
            return "Step 1: Analyzing the image.\n"
        
        # 3) 重新编号为 "Step N: ..." 格式
        cot_parts = []
        for i, step_text in enumerate(steps, 1):
            cot_parts.append(f"Step {i}: {step_text}")
        return "\n".join(cot_parts) + "\n"

    def _extract_cot_from_chain(self, think_chain: str) -> str:
        """
        从 think_chain 中提取推理步骤部分 (兼容新旧格式)
        
        新格式 (Step N: + Final Answer:): 直接提取 Final Answer: 之前的部分
        旧格式 (<think>...</step>...</think>): 转换为新格式
        
        返回值: 推理步骤文本 (不含 Final Answer: 行), 以 "\n" 结尾
        """
        if not think_chain:
            return "Step 1: Analyzing the image.\n"
        
        # 检测是否为旧格式 (包含 <think> 标签)
        if self.THINK_START in think_chain:
            return self._convert_old_format_to_new(think_chain)
        
        # 新格式: 提取 Final Answer: 之前的部分
        fa_match = self.FINAL_ANSWER_PATTERN.search(think_chain)
        if fa_match:
            cot_text = think_chain[:fa_match.start()].strip()
        else:
            cot_text = think_chain.strip()
        
        # 确保有 Step N: 格式
        if not re.search(r'Step\s+\d+\s*[:.\s]', cot_text, re.IGNORECASE):
            # 没有 Step N: 格式, 尝试按行分割并添加编号
            lines = [l.strip() for l in cot_text.split('\n') if l.strip()]
            if lines:
                cot_parts = [f"Step {i}: {line}" for i, line in enumerate(lines, 1)]
                cot_text = "\n".join(cot_parts)
            else:
                cot_text = "Step 1: Analyzing the image."
        
        if not cot_text.endswith('\n'):
            cot_text += '\n'
        return cot_text

    def _get_item_inner(self, idx):
        item = self.data[idx]

        image_path = item.get('image', '')
        question = item.get('question', '')
        answer = item.get('answer', '')

        # ---- 1. 构造消息 (含 system prompt) ----
        messages = []

        # 注入 system prompt: 引导模型使用 "Step N:" + "Final Answer:" 格式
        if self.use_system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            })

        # 用户消息: 图像 + 问题
        user_content = []
        if image_path:
            user_content.append({"type": "image", "image": image_path})
        user_content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": user_content})

        # 处理视觉信息
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs, video_inputs = None, None

        # Tokenize prompt (包含到 generation prompt 为止)
        prompt_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        # ---- 2. 构造回答: "Step 1: ...\nStep 2: ...\nFinal Answer: {answer}" ----
        # 新格式: 使用 Qwen3-VL-Instruct 原生格式, 不再使用 <think>...</think>
        # generation prompt 输出: `<|im_start|>assistant\n`
        # 我们拼接: "{cot_steps}Final Answer: {final_answer}<|im_end|>"
        
        think_chain = item.get('think_chain')
        think_chain_status = item.get('think_chain_status', '')
        
        if think_chain and think_chain_status == 'valid':
            # 使用预生成的推理链 (兼容新旧格式)
            cot_text = self._extract_cot_from_chain(think_chain)
        else:
            # Fallback: 用 answer 碎片构造占位推理链
            cot_text = self._build_cot_block(answer)
        
        # 最终答案 (参与 loss) — 方案 A: 标准化 GT answer 格式
        final_answer = self._normalize_answer(answer)
        
        im_end_token = "<|im_end|>"
        
        # 分别 tokenize 推理步骤和 final answer，以便精确设置 labels
        # cot_text = "Step 1: xxx\nStep 2: yyy\n" (推理步骤部分)
        # answer_text = "Final Answer: {final_answer}<|im_end|>" (答案部分)
        cot_tokens = self.processor.tokenizer(
            cot_text,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # ---- 精确计算 step boundary 在 cot_ids 中的位置 ----
        # Step boundary 触发点: "Step 2:", "Step 3:", ..., "Final Answer:"
        # 即: 遇到 Step N (N>=2) 的起始位置时, 说明前一个 step 结束, 触发 Z_d 更新
        # Step 1 不触发更新 (使用初始 Z_d_0)
        # "Final Answer:" 也是一个 boundary (最后一个 step 结束)
        #
        # 使用 offset_mapping 在字符级精确定位每个 boundary 的起始 token 位置
        cot_step_positions = []  # 相对于 cot_ids 起始位置的偏移
        try:
            cot_result = self.processor.tokenizer(
                cot_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = cot_result['offset_mapping']  # [(char_start, char_end), ...]
            
            # 查找所有 "Step N:" (N>=2) 的起始字符位置
            for m in re.finditer(r'(?:^|\n)\s*(Step\s+(\d+)\s*[:.\s])', cot_text, re.IGNORECASE):
                step_num = int(m.group(2))
                if step_num >= 2:
                    # boundary 位置 = "Step N:" 的起始字符位置
                    start_char = m.start(1)  # group(1) 是 "Step N:" 部分
                    # 找到覆盖 start_char 的 token
                    for tok_idx, (cs, ce) in enumerate(offsets):
                        if cs <= start_char < ce:
                            cot_step_positions.append(tok_idx)
                            break
        except Exception:
            # Fallback: 累积 decode 匹配
            cot_ids_list = cot_tokens['input_ids'][0].tolist()
            cumulative_text = ""
            for tok_idx, tid in enumerate(cot_ids_list):
                cumulative_text += self.processor.tokenizer.decode([tid])
                # 检测 "Step N:" (N>=2)
                for m in re.finditer(r'Step\s+(\d+)\s*[:.\s]', cumulative_text, re.IGNORECASE):
                    step_num = int(m.group(1))
                    if step_num >= 2 and tok_idx not in cot_step_positions:
                        cot_step_positions.append(tok_idx)

        # "Final Answer:" 的 boundary 位置将在 answer_tokens 中处理
        # (它是 answer 部分的第一个 token, 即 cot_ids 的长度位置)
        # 在后面拼接时, 我们把 "Final Answer:" 的起始位置也加入 step_positions

        answer_text = "Final Answer: " + final_answer + im_end_token
        answer_tokens = self.processor.tokenizer(
            answer_text,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # ---- 3. 拼接 prompt + cot_steps + final_answer ----
        # processor 返回带 batch 维的 tensor，需要 squeeze
        prompt_ids = prompt_inputs['input_ids'].squeeze(0)       # [prompt_len]
        prompt_mask = prompt_inputs['attention_mask'].squeeze(0)  # [prompt_len]
        think_ids = cot_tokens['input_ids'][0]            # [cot_len] (变量名保持 think_ids 兼容下游)
        think_mask = cot_tokens['attention_mask'][0]      # [cot_len]
        answer_ids = answer_tokens['input_ids'][0]        # [answer_len]
        answer_mask = answer_tokens['attention_mask'][0]  # [answer_len]

        # ---- 序列长度硬截断: 确保总长度 ≤ max_seq_len ----
        # 策略: 绝不触碰 prompt (含图像 token) 和 answer, 只截断 cot 步骤部分。
        # 如果 prompt + answer 已经超限, 直接跳过该样本。
        if self.max_seq_len > 0:
            prompt_len_val = len(prompt_ids)
            answer_len_val = len(answer_ids)
            max_think_len = self.max_seq_len - prompt_len_val - answer_len_val

            if max_think_len <= 0:
                raise _SampleTooLongError(
                    f"prompt({prompt_len_val}) + answer({answer_len_val}) = "
                    f"{prompt_len_val + answer_len_val} > max_seq_len {self.max_seq_len}, "
                    f"无法容纳 cot 步骤, 跳过样本"
                )

            if len(think_ids) > max_think_len:
                # 截断 cot 步骤: 保留前半段 + 后半段
                orig_think_len = len(think_ids)
                half = max_think_len // 2
                think_ids = torch.cat([think_ids[:half], think_ids[-half:]], dim=0)
                think_mask = torch.cat([think_mask[:half], think_mask[-half:]], dim=0)
                # 重映射 cot_step_positions
                new_positions = []
                for pos in cot_step_positions:
                    if pos < half:
                        new_positions.append(pos)
                    elif pos >= orig_think_len - half:
                        new_pos = half + (pos - (orig_think_len - half))
                        if new_pos < max_think_len:
                            new_positions.append(new_pos)
                cot_step_positions = new_positions

        # 拼接完整序列 (此时保证 ≤ max_seq_len)
        input_ids = torch.cat([prompt_ids, think_ids, answer_ids], dim=0)
        attention_mask = torch.cat([prompt_mask, think_mask, answer_mask], dim=0)

        # 断言: 硬性保证不超限
        assert self.max_seq_len <= 0 or len(input_ids) <= self.max_seq_len, \
            f"BUG: 截断后仍超限 {len(input_ids)} > {self.max_seq_len}"

        # Labels 策略 (分段监督):
        #   - prompt 部分: -100 (不参与 loss)
        #   - cot 步骤部分 ("Step 1: ...\nStep 2: ...\n"): 根据 think_chain_source 决定
        #       * free_reasoning / corrected_free_reasoning / dataset_converted → 真实 token id (密集梯度信号)
        #       * wrong_cot / fabricated / 其他 → -100 (错误推理链, 不监督)
        #   - "Final Answer: {answer}<|im_end|>": 真实 token id (始终监督)
        think_chain_source = item.get('think_chain_source', 'fabricated')
        if think_chain_source in self.SUPERVISED_THINK_SOURCES:
            # 高质量推理链: cot 步骤也参与 loss
            think_labels = think_ids.clone()
        else:
            # wrong_cot / 低质量 / 伪造: cot 步骤不参与 loss
            think_labels = torch.full_like(think_ids, -100)

        # ====== P2: 构建 answer_weight_mask (answer token 加权) ======
        # 当 answer 很短 (如单字符 "B") 时, 有效监督 token 极少,
        # 通过给 answer token 更高权重来补偿梯度信号稀疏问题。
        # weight_mask: prompt=-100标记的不参与, think部分=1.0, answer部分=加权
        #
        # ★ 方案 A 改进: 使用 think 总长度 (而非有效 label 数) 计算 answer_weight
        # 原逻辑: think_valid_count = (think_labels != -100).sum() → 错误 CoT 为 0 → answer_weight=1.0
        # 新逻辑: think_total_len = len(think_ids) → 错误 CoT 也能获得加权
        # 这样错误 CoT 的 answer token 也能获得与 think 块等量的梯度贡献,
        # 让 Controller 从错误 CoT 中获得更强的"纠错信号"。
        think_total_len = len(think_ids)
        answer_len_val = len(answer_ids)
        if answer_len_val > 0 and think_total_len > 0:
            answer_weight = min(max(think_total_len / answer_len_val, 1.0), 10.0)
        else:
            answer_weight = 1.0
        
        # 构建 per-token weight mask (与 labels 对齐)
        weight_mask = torch.cat([
            torch.zeros(len(prompt_ids)),          # prompt: 不参与
            torch.ones(len(think_ids)),             # think 块: 权重 1.0
            torch.full((len(answer_ids),), answer_weight),  # answer: 加权
        ], dim=0)

        labels = torch.cat([
            torch.full_like(prompt_ids, -100),  # prompt: 不监督
            think_labels,                        # cot 步骤: 按来源决定
            answer_ids.clone(),                  # "Final Answer: {answer}<|im_end|>": 监督 ✅
        ], dim=0)

        # pixel_values 和 image_grid_thw 不需要 squeeze batch 维
        pixel_values = prompt_inputs.get('pixel_values')
        image_grid_thw = prompt_inputs.get('image_grid_thw')

        # 记录 prompt 长度
        prompt_len = len(prompt_ids)

        # 将 cot_step_positions 转换为完整序列中的位置 (加上 prompt_len 偏移)
        step_positions = [pos + prompt_len for pos in cot_step_positions]
        # 添加 "Final Answer:" 的 boundary 位置
        # "Final Answer:" 是 answer_ids 的第一个 token, 即完整序列中的 prompt_len + len(think_ids)
        final_answer_boundary = prompt_len + len(think_ids)
        step_positions.append(final_answer_boundary)

        # ====== Contrastive Draft Learning: 已弃用 (当前数据无 wrong_cot + correct_cot 配对) ======

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "has_image": pixel_values is not None,
            "prompt_len": prompt_len,
            "step_positions": step_positions,
            "loss_weight_mask": weight_mask,  # P2: per-token loss 权重
            "think_chain_source": think_chain_source,  # 样本类型: dataset_converted / wrong_cot 等
            "num_think_tokens": int(len(think_ids)),
            "num_answer_tokens": int(len(answer_ids)),
            "num_supervised_think_tokens": int((think_labels != -100).sum().item()),  # think 中参与 loss 的 token 数
            "answer_weight": float(answer_weight),
        }


class RLDCollator:
    """
    RLD Data Collator
    
    职责:
    1. Left padding 对齐批次
    2. 拼接 pixel_values 和 image_grid_thw
    3. 检测并返回每个样本的 step boundary 位置
    
    Args:
        processor: AutoProcessor
        step_delimiter_ids: </step> 的 token id 序列 (可能是多个 token)
    """

    def __init__(self, processor, step_delimiter_ids: list):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        self.step_delimiter_ids = step_delimiter_ids

    def _left_pad(self, sequences, pad_value):
        """Left padding"""
        batch_size = len(sequences)
        max_len = max(seq.size(0) for seq in sequences)
        padded = torch.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            padded[i, max_len - seq.size(0):] = seq
        return padded

    def _left_pad_float(self, sequences, pad_value):
        """Left padding for float tensors (P2: loss weight mask)"""
        batch_size = len(sequences)
        max_len = max(seq.size(0) for seq in sequences)
        padded = torch.full((batch_size, max_len), pad_value, dtype=torch.float32)
        for i, seq in enumerate(sequences):
            padded[i, max_len - seq.size(0):] = seq
        return padded

    def __call__(self, batch):
        # Left pad
        input_ids = self._left_pad([item['input_ids'] for item in batch], self.pad_token_id)
        attention_mask = self._left_pad([item['attention_mask'] for item in batch], 0)
        labels = self._left_pad([item['labels'] for item in batch], -100)
        # P2: loss weight mask (left pad with 0.0)
        loss_weight_mask = self._left_pad_float([item['loss_weight_mask'] for item in batch], 0.0)

        # 图像处理 (所有样本都有图像，已在 Dataset 中过滤)
        pixel_values_list = [item['pixel_values'] for item in batch if item['has_image']]
        image_grid_thw_list = [item['image_grid_thw'] for item in batch if item['has_image']]
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0)
            image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
        else:
            pixel_values = None
            image_grid_thw = None

        # 计算 left padding 偏移后的 prompt_len (per-sample)
        # left pad 后，原始序列右对齐，padding 偏移 = max_len - orig_len
        max_len = input_ids.shape[1]
        prompt_lens = []
        for item in batch:
            orig_len = len(item['input_ids'])
            pad_offset = max_len - orig_len
            prompt_lens.append(pad_offset + item['prompt_len'])

        # 使用 __getitem__ 中预计算的 step_positions (精确, 不受 BPE 合并影响)
        # 只需加上 left-padding 偏移即可
        step_boundaries = []
        for b_idx, item in enumerate(batch):
            orig_len = len(item['input_ids'])
            pad_offset = max_len - orig_len
            # item['step_positions'] 是相对于原始序列的位置，加上 pad_offset 得到 padded 序列位置
            positions = [pos + pad_offset for pos in item.get('step_positions', [])]
            step_boundaries.append(positions)

        # 收集样本级元信息 (用于 trainer 日志监控)
        sample_meta = {
            "think_chain_sources": [item.get('think_chain_source', 'unknown') for item in batch],
            "num_think_tokens": [item.get('num_think_tokens', 0) for item in batch],
            "num_answer_tokens": [item.get('num_answer_tokens', 0) for item in batch],
            "num_supervised_think_tokens": [item.get('num_supervised_think_tokens', 0) for item in batch],
            "answer_weights": [item.get('answer_weight', 1.0) for item in batch],
        }

        # ====== Contrastive Draft Learning: 已弃用 ======

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "step_boundaries": step_boundaries,
            "prompt_lens": prompt_lens,
            "loss_weight_mask": loss_weight_mask,  # P2: per-token loss 权重
            "sample_meta": sample_meta,  # 样本级元信息 (不参与模型计算)
        }
