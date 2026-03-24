"""
RLD 数据处理 (适配 Qwen3-VL-8B-Instruct)

Dataset: 加载图像+问答数据，利用 Qwen3-VL 原生 <think></think> 格式
         在 <think> 块内部使用 </step> 做细粒度分步控制 RLD 更新
         支持三种 think 块来源 (由 pregenerate_think.py 三阶段流水线生成):
           1. free_reasoning: 自由推理正确 (最高质量, 分布最匹配)
           2. corrected_free_reasoning: 纠正式推理 (含"犯错→纠正"模式)
           3. fabricated (fallback): 自动用 answer 碎片构造占位 think 块

         Labels 策略 (分段监督):
           - 高质量推理链 (free_reasoning/corrected_free_reasoning):
             think 块参与 loss → 为 Controller 提供密集梯度信号
           - 低质量/伪造推理链 (fabricated 等):
             think 块 labels = -100 → 只监督 final answer
           - prompt 部分: 始终 -100
           - final answer: 始终参与 loss
Collator: 批处理，检测 step 边界位置
"""

import json
import os
import re
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


# ====== RLD System Prompt (Qwen3-VL 格式) ======
# Qwen3-VL-8B-Instruct 是 Instruct 模型，chat_template 的 add_generation_prompt
# 只输出 `<|im_start|>assistant\n`，**不会**自动添加 `<think>\n`。
# （与 Qwen3 文本模型不同，Qwen3-VL 不支持 enable_thinking 参数。）
# 因此我们在训练时手动拼接 `<think>\n...\n</think>\n\n` 包裹推理过程，
# system prompt 显式要求模型: 1) 用 <think></think> 包裹思维链  2) 用 </step> 分步  3) </think> 后输出最终答案
# 这保证了训推一致性: 训练数据中硬编码了 <think>...</think>\n\n{answer} 格式,
# 推理时模型也会被 prompt 引导自主输出相同格式。
RLD_SYSTEM_PROMPT = """You are a visual reasoning assistant. You must structure your response as follows:

1. Wrap your reasoning process inside <think> and </think> tags.
2. Inside the thinking block, break your reasoning into clear steps, ending each step with </step>.
3. After </think>, output your final answer directly.

Example format:
<think>
I observe that the triangle has a base of 6 and height of 8.
</step>
Using the formula: Area = 0.5 × base × height = 0.5 × 6 × 8 = 24.
</step>
</think>

The area of the triangle is 24 square units."""


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
            if item.get('think_chain', '').count('</step>') <= MAX_STEPS_FILTER
        ]
        filtered_by_steps = before_filter - len(self.data)
        if filtered_by_steps > 0:
            print(f"[RLDDataset] ⚠️ 已过滤 {filtered_by_steps} 个超高段数(>{MAX_STEPS_FILTER})的样本")
            print(f"[RLDDataset]   保留 {len(self.data)}/{before_filter}")
        
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

        # 高质量来源: think 块参与 loss
        self.SUPERVISED_THINK_SOURCES = {"free_reasoning", "corrected_free_reasoning", "dataset_converted"}

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
                if self.THINK_START not in think_chain or self.THINK_END not in think_chain:
                    print(f"[RLDDataset] ⚠️ 样本 {i} 的 think_chain 缺少 <think>/<\\think> 标签")
                if self.STEP_DELIMITER not in think_chain:
                    print(f"[RLDDataset] ⚠️ 样本 {i} 的 think_chain 缺少 </step> 边界")
        print(f"[RLDDataset] 预检查完成: {n} 个样本")

    def _build_think_block(self, answer: str) -> str:
        """
        构造 think 块内部内容 (fallback: 训练数据中无预生成 think_chain 时使用)
        
        按 max_tokens_per_step 自然切分 answer 文本为多步，
        不再硬性指定固定段数。切分逻辑:
        1. 将 answer 按句子拆分
        2. 按 max_tokens_per_step 贪心合并句子为步骤
        3. 每个步骤后跟 </step>
        
        注意: 返回值不包含 <think>/</think> 标签!
        外层 _get_item_inner 会在拼接时手动添加 <think>\n 前缀和 </think>\n\n 后缀，
        因为 Qwen3-VL-8B-Instruct 的 chat_template 不会自动输出 <think>。
        
        这些 token 不参与 loss，但为 RLD Controller
        提供 step 边界触发更新的机会。
        
        Args:
            answer: 原始的最终答案文本
            
        Returns:
            think_inner: think 块内部内容 (不含 <think>/</think> 标签)，
                         格式: "{step1}\n</step>\n{step2}\n</step>\n"
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
            steps = [answer.strip() if answer.strip() else "Analyzing the image..."]

        # 组装 think 内部内容: 每个步骤后跟 </step>
        # 注意: 不包裹 <think>/</think> 标签!
        # 外层 _get_item_inner 会手动添加 `<think>\n` 前缀和 `</think>\n\n` 后缀。
        think_content = f"\n{self.STEP_DELIMITER}\n".join(steps)
        think_inner = f"{think_content}\n{self.STEP_DELIMITER}\n"
        
        return think_inner

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

    def _get_item_inner(self, idx):
        item = self.data[idx]

        image_path = item.get('image', '')
        question = item.get('question', '')
        answer = item.get('answer', '')

        # ---- 1. 构造消息 (含 system prompt) ----
        messages = []

        # 注入 system prompt: 引导模型使用 </step> 分步推理格式
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

        # ---- 2. 构造回答: <think>...</think> + final answer ----
        # 优先使用预生成的推理链，fallback 到伪造 think 块
        
        # 获取 think 内部内容: 预生成 > 伪造
        # 注意: Qwen3-VL-8B-Instruct 的 chat_template 只输出 `<|im_start|>assistant\n`，
        # 不会自动添加 `<think>\n`。我们在下方拼接 think_with_close 时手动添加。
        # 预生成的 think_chain 格式为 `<think>\n...\n</step>\n</think>\n`，
        # 需要剥离外层 <think>/</think> 标签，只保留内部内容。
        think_chain = item.get('think_chain')
        think_chain_status = item.get('think_chain_status', '')
        
        if think_chain and think_chain_status == 'valid':
            # 使用预生成的推理链 (由 pregenerate_think.py 生成)
            # 剥离 <think>...</think> 包裹及其后面的一切内容 (如 "Answer: ...")，
            # 只保留 <think> 与 </think> 之间的内部内容。
            think_inner = think_chain

            # 1) 先定位 </think>，截掉它及其后面所有内容 (如 "\nAnswer: 12")
            think_end_pos = think_inner.rfind(self.THINK_END)
            if think_end_pos != -1:
                think_inner = think_inner[:think_end_pos]

            # 2) 再剥离开头的 <think> 标签
            if think_inner.startswith(self.THINK_START):
                think_inner = think_inner[len(self.THINK_START):]

            # 3) 去除首尾多余换行
            think_inner = think_inner.strip('\n')

            # 确保末尾有换行 (后面还要拼 </think>\n\n)
            if not think_inner.endswith('\n'):
                think_inner = think_inner + '\n'
        else:
            # Fallback: 用 answer 碎片构造占位 think 内部内容 (不含 <think>/</think>)
            think_inner = self._build_think_block(answer)
        
        # 最终答案 (参与 loss)
        final_answer = answer.strip()
        
        # 完整回答部分 (拼接在 generation prompt 之后):
        # generation prompt 只输出: `<|im_start|>assistant\n`
        # 我们手动拼接: <think>\n{think_inner}</think>\n\n{final_answer}<|im_end|>
        im_end_token = "<|im_end|>"
        
        # 分别 tokenize think 块 (含 <think>/</think> 标签) 和 final answer，以便精确设置 labels
        # think_with_close = "<think>\n{think_inner}</think>\n\n" — 包含完整的 think 包裹
        think_with_close = self.THINK_START + "\n" + think_inner + self.THINK_END + "\n\n"
        think_tokens = self.processor.tokenizer(
            think_with_close,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # ---- 精确计算 </step> 在 think_ids 中的边界位置 ----
        # BPE tokenizer 会根据上下文合并 token (如 ">" + "\n" → ">\n"),
        # 导致基于 token 子序列匹配的方式无法找到 delimiter。
        # 使用 offset_mapping 在字符级精确定位每个 </step> 的最后一个 token 位置。
        think_step_positions = []  # 相对于 think_ids 起始位置的偏移
        if self.STEP_DELIMITER in think_with_close:
            try:
                # 方法1 (首选): 使用 tokenizer 的 offset_mapping 精确映射字符→token 位置
                think_result = self.processor.tokenizer(
                    think_with_close,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offsets = think_result['offset_mapping']  # [(char_start, char_end), ...]
                
                for m in re.finditer(re.escape(self.STEP_DELIMITER), think_with_close):
                    end_char = m.end()  # </step> 在文本中的字符级结束位置
                    # 找到覆盖 end_char 的 token (即 </step> 的最后一个 token)
                    for tok_idx, (cs, ce) in enumerate(offsets):
                        if cs < end_char and ce >= end_char:
                            think_step_positions.append(tok_idx)
                            break
            except Exception:
                # 方法2 (fallback): 累积 decode 匹配
                think_ids_list = think_tokens['input_ids'][0].tolist()
                cumulative_text = ""
                delim_text = self.STEP_DELIMITER
                for tok_idx, tid in enumerate(think_ids_list):
                    cumulative_text += self.processor.tokenizer.decode([tid])
                    while delim_text in cumulative_text:
                        think_step_positions.append(tok_idx)
                        pos = cumulative_text.find(delim_text)
                        cumulative_text = cumulative_text[pos + len(delim_text):]
                        break
        answer_tokens = self.processor.tokenizer(
            final_answer + im_end_token,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # ---- 3. 拼接 prompt + think_block + final_answer ----
        # processor 返回带 batch 维的 tensor，需要 squeeze
        prompt_ids = prompt_inputs['input_ids'].squeeze(0)       # [prompt_len]
        prompt_mask = prompt_inputs['attention_mask'].squeeze(0)  # [prompt_len]
        think_ids = think_tokens['input_ids'][0]          # [think_len]
        think_mask = think_tokens['attention_mask'][0]    # [think_len]
        answer_ids = answer_tokens['input_ids'][0]        # [answer_len]
        answer_mask = answer_tokens['attention_mask'][0]  # [answer_len]

        # ---- 序列长度硬截断: 确保总长度 ≤ max_seq_len ----
        # 策略: 绝不触碰 prompt (含图像 token) 和 answer, 只截断 think 块。
        # 如果 prompt + answer 已经超限, 直接跳过该样本 (不截断 prompt, 避免 image token 不匹配)。
        if self.max_seq_len > 0:
            prompt_len_val = len(prompt_ids)
            answer_len_val = len(answer_ids)
            max_think_len = self.max_seq_len - prompt_len_val - answer_len_val

            if max_think_len <= 0:
                # prompt + answer 已超 max_seq_len, 无法容纳任何 think 块
                # 直接跳过, 不尝试截断 prompt (会破坏 image token 匹配)
                raise _SampleTooLongError(
                    f"prompt({prompt_len_val}) + answer({answer_len_val}) = "
                    f"{prompt_len_val + answer_len_val} > max_seq_len {self.max_seq_len}, "
                    f"无法容纳 think 块, 跳过样本"
                )

            if len(think_ids) > max_think_len:
                # 截断 think 块: 保留前半段 + 后半段, 使 </step> 边界仍有一些
                orig_think_len = len(think_ids)
                half = max_think_len // 2
                think_ids = torch.cat([think_ids[:half], think_ids[-half:]], dim=0)
                think_mask = torch.cat([think_mask[:half], think_mask[-half:]], dim=0)
                # 重映射 think_step_positions
                new_positions = []
                for pos in think_step_positions:
                    if pos < half:
                        new_positions.append(pos)
                    elif pos >= orig_think_len - half:
                        new_pos = half + (pos - (orig_think_len - half))
                        if new_pos < max_think_len:
                            new_positions.append(new_pos)
                think_step_positions = new_positions

        # 拼接完整序列 (此时保证 ≤ max_seq_len)
        input_ids = torch.cat([prompt_ids, think_ids, answer_ids], dim=0)
        attention_mask = torch.cat([prompt_mask, think_mask, answer_mask], dim=0)

        # 断言: 硬性保证不超限
        assert self.max_seq_len <= 0 or len(input_ids) <= self.max_seq_len, \
            f"BUG: 截断后仍超限 {len(input_ids)} > {self.max_seq_len}"

        # Labels 策略 (分段监督):
        #   - prompt 部分: -100 (不参与 loss)
        #   - <think>...</think> 部分: 根据 think_chain_source 决定
        #       * free_reasoning / corrected_free_reasoning / dataset_converted → 真实 token id (密集梯度信号)
        #       * fabricated / 其他 → -100 (低质量, 不监督)
        #   - final answer + <|im_end|>: 真实 token id (始终监督)
        think_chain_source = item.get('think_chain_source', 'fabricated')
        if think_chain_source in self.SUPERVISED_THINK_SOURCES:
            # 高质量推理链: think 块也参与 loss
            think_labels = think_ids.clone()
        else:
            # 低质量/伪造: think 块不参与 loss
            think_labels = torch.full_like(think_ids, -100)

        labels = torch.cat([
            torch.full_like(prompt_ids, -100),  # prompt: 不监督
            think_labels,                        # think 块: 按来源决定
            answer_ids.clone(),                  # final answer: 监督 ✅
        ], dim=0)

        # pixel_values 和 image_grid_thw 不需要 squeeze batch 维
        pixel_values = prompt_inputs.get('pixel_values')
        image_grid_thw = prompt_inputs.get('image_grid_thw')

        # 记录 prompt 长度
        prompt_len = len(prompt_ids)

        # 将 think_step_positions 转换为完整序列中的位置 (加上 prompt_len 偏移)
        step_positions = [pos + prompt_len for pos in think_step_positions]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "has_image": pixel_values is not None,
            "prompt_len": prompt_len,
            "step_positions": step_positions,
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

    def __call__(self, batch):
        # Left pad
        input_ids = self._left_pad([item['input_ids'] for item in batch], self.pad_token_id)
        attention_mask = self._left_pad([item['attention_mask'] for item in batch], 0)
        labels = self._left_pad([item['labels'] for item in batch], -100)

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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "step_boundaries": step_boundaries,
            "prompt_lens": prompt_lens,
        }
