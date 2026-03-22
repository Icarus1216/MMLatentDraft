"""
RLD 数据处理 (方案 B: Qwen3.5 适配)

Dataset: 加载图像+问答数据，利用 Qwen3.5 原生 <think></think> 格式
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
import warnings
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("[RLD Data] ⚠️ qwen_vl_utils 未安装，请运行: pip install qwen-vl-utils")
    process_vision_info = None


# ====== RLD System Prompt (方案 B: Qwen3.5 格式) ======
# Qwen3.5 的 thinking 模式是内建的 — chat_template 的 add_generation_prompt
# 会自动输出 `<|im_start|>assistant\n<think>\n`，模型已被训练为在 <think> 内思考。
# 因此 system prompt 不需要教模型 <think>/<\think> 格式（冗余且可能冲突），
# 只需要教 RLD 自定义的 </step> 分步规范。
RLD_SYSTEM_PROMPT = """You are a visual reasoning assistant. When thinking through problems, break your reasoning into clear steps separated by </step>.

For each distinct observation, calculation, or deduction, end that step with </step> before moving on."""


class RLDDataset(Dataset):
    """
    RLD 数据集 (方案 B: Qwen3.5 适配)
    
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
    
    训练时的序列构造 (符合 Qwen3.5 官方 chat_template):
      [prompt]<|im_start|>assistant\n<think>\n{steps}\n</step>\n</think>\n\n{answer}<|im_end|>
      其中 `<|im_start|>assistant\n<think>\n` 由 apply_chat_template(add_generation_prompt=True) 自动生成,
      think_block 只包含 think 内部内容 (不含 <think>/</think> 标签),
      `</think>\n\n` 作为过渡衔接 think 块和 final answer。
      
    labels 策略 (分段监督, 根据 think_chain_source 决定):
      - prompt 部分: -100 (不参与 loss)
      - <think>...</think> 部分:
          * free_reasoning / corrected_free_reasoning: 真实 token id (参与 loss)
          * fabricated / 其他: -100 (不参与 loss)
      - </think> 后的 final answer + <|im_end|>: 真实 token id (始终参与 loss)
    
    推理时:
      模型根据 system prompt 自主在 <think> 内生成推理过程，
      遇到 </step> 触发 RLD 更新，</think> 后输出最终答案。
    
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
        
        self.processor = processor
        self.auto_split_steps = auto_split_steps
        self.max_tokens_per_step = max_tokens_per_step
        self.use_system_prompt = use_system_prompt
        self.system_prompt = system_prompt or RLD_SYSTEM_PROMPT
        self.max_seq_len = max_seq_len


        # 确保 tokenizer 配置
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

        # Qwen3.5 的 image placeholder token id (从 processor 配置获取)
        # Qwen3.5: 248056, Qwen3-VL: 151655
        try:
            self.IMAGE_TOKEN_ID = self.processor.image_token_id
        except AttributeError:
            self.IMAGE_TOKEN_ID = 248056  # Qwen3.5 默认值

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
        print(f"[RLDDataset] 方案 B: Qwen3.5 格式, 分段监督 labels")
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
        apply_chat_template 的 generation prompt 已输出 <think>\n，
        外层 _get_item_inner 会在末尾追加 </think>\n\n 衔接 final answer。
        
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
        # apply_chat_template(add_generation_prompt=True) 已输出 `<think>\n`,
        # 外层 _get_item_inner 会在末尾追加 `</think>\n\n` 衔接 final answer。
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
        """带容错的样本获取，图片加载失败时随机选另一个样本"""
        import random
        if _retry > 5:
            raise RuntimeError(f"[RLDDataset] 连续 {_retry} 次采样失败，请检查数据集图片路径")
        try:
            return self._get_item_inner(idx)
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
        # 注意: apply_chat_template(add_generation_prompt=True) 已在 prompt 末尾
        # 输出 `<|im_start|>assistant\n<think>\n`，因此 think_inner 不含 <think> 标签。
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
        # generation prompt 已输出: `<|im_start|>assistant\n<think>\n`
        # 我们拼接: {think_inner}</think>\n\n{final_answer}<|im_end|>
        # 其中 </think>\n\n 符合 Qwen3.5 官方模板格式
        im_end_token = "<|im_end|>"
        
        # 分别 tokenize think 内部内容 + </think>\n\n 过渡 和 final answer，以便精确设置 labels
        # think_with_close = "{think_inner}</think>\n\n" — 包含 think 内容和关闭标签
        think_with_close = think_inner + self.THINK_END + "\n\n"
        think_tokens = self.processor.tokenizer(
            think_with_close,
            add_special_tokens=False,
            return_tensors="pt",
        )
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

        # 完整序列
        input_ids = torch.cat([prompt_ids, think_ids, answer_ids], dim=0)
        attention_mask = torch.cat([prompt_mask, think_mask, answer_mask], dim=0)

        # ---- 序列长度截断: 防止极长样本导致 NCCL 超时 ----
        if self.max_seq_len > 0 and len(input_ids) > self.max_seq_len:
            # 截断策略: 优先保留 prompt 和 answer，截断 think 块中间部分
            prompt_len_val = len(prompt_ids)
            answer_len_val = len(answer_ids)
            max_think_len = self.max_seq_len - prompt_len_val - answer_len_val
            if max_think_len > 0 and len(think_ids) > max_think_len:
                # 截断 think 块 (保留前半段+后半段，使得 </step> 边界仍有一些)
                half = max_think_len // 2
                think_ids = torch.cat([think_ids[:half], think_ids[-half:]], dim=0)
                think_mask = torch.cat([think_mask[:half], think_mask[-half:]], dim=0)
                # 重新拼接
                input_ids = torch.cat([prompt_ids, think_ids, answer_ids], dim=0)
                attention_mask = torch.cat([prompt_mask, think_mask, answer_mask], dim=0)
            elif max_think_len <= 0:
                # 极端: prompt + answer 已超限，直接截断总序列
                input_ids = input_ids[:self.max_seq_len]
                attention_mask = attention_mask[:self.max_seq_len]



        # Labels 策略 (分段监督):
        #   - prompt 部分: -100 (不参与 loss)
        #   - <think>...</think> 部分: 根据 think_chain_source 决定
        #       * free_reasoning / corrected_free_reasoning / dataset_converted → 真实 token id (密集梯度信号)
        #       * fabricated / 其他 → -100 (低质量, 不监督)
        #   - final answer + <|im_end|>: 真实 token id (始终监督)
        think_chain_source = item.get('think_chain_source', 'fabricated')
        if think_chain_source in self.SUPERVISED_THINK_SOURCES:
            # 高质量推理链: think 块也参与 loss
            # 为 Controller 提供密集梯度信号 (每个 token 位置都有梯度)
            think_labels = think_ids.clone()
        else:
            # 低质量/伪造: think 块不参与 loss
            think_labels = torch.full_like(think_ids, -100)

        labels = torch.cat([
            torch.full_like(prompt_ids, -100),  # prompt: 不监督
            think_labels,                        # think 块: 按来源决定
            answer_ids.clone(),                  # final answer: 监督 ✅
        ], dim=0)

        # labels 也需要同步截断 (使用实际截断后的长度)
        actual_length = len(input_ids)
        if len(labels) > actual_length:
            labels = labels[:actual_length]

        # pixel_values 和 image_grid_thw 不需要 squeeze batch 维
        # 因为 collator 中会用 torch.cat 沿 dim=0 拼接
        pixel_values = prompt_inputs.get('pixel_values')
        image_grid_thw = prompt_inputs.get('image_grid_thw')
        
        # pixel_values: [num_patches, C] (已经没有 batch 维)
        # image_grid_thw: [num_images, 3]

        # 记录 prompt 长度 (用于 model.forward 中将 prompt 独立 prefill)
        prompt_len = len(prompt_ids)

        # ---- 截断安全检查: image token 与 visual features 数量必须匹配 ----
        # 截断后如果 image token 数量与 visual features 不一致，说明截断保护失效。
        # 此时不能清除 pixel_values (会导致 ZeRO-3 下 ViT forward 路径不一致 → NCCL 死锁)。
        # 安全做法: 完全不截断该样本 (恢复原始序列)，容忍可能的 OOM。
        SPATIAL_MERGE_SIZE = 2   # Qwen3.5 默认 spatial_merge_size
        if pixel_values is not None and image_grid_thw is not None:
            num_image_tokens = (input_ids == self.IMAGE_TOKEN_ID).sum().item()
            num_image_features = (
                image_grid_thw.prod(-1) // (SPATIAL_MERGE_SIZE ** 2)
            ).sum().item()
            if num_image_tokens != num_image_features:
                # 截断保护失效，回退到不截断
                input_ids = torch.cat([prompt_ids, think_ids, answer_ids], dim=0)
                attention_mask = torch.cat([prompt_mask, think_mask, answer_mask], dim=0)
                labels_prompt = torch.full_like(prompt_ids, -100)
                labels_think = think_labels
                labels_answer = answer_ids.clone()
                labels = torch.cat([labels_prompt, labels_think, labels_answer], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "has_image": pixel_values is not None,
            "prompt_len": prompt_len,
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

        # 检测 step boundaries (padding 后的位置，支持多 token delimiter)
        step_boundaries = []
        delim = self.step_delimiter_ids
        delim_len = len(delim)
        for b in range(input_ids.shape[0]):
            positions = []
            seq = input_ids[b].tolist()
            # 只在 prompt 之后搜索 delimiter (prompt 中不可能有 </step>)
            search_start = max(prompt_lens[b], delim_len - 1)
            for i in range(search_start, len(seq)):
                if seq[i - delim_len + 1 : i + 1] == delim:
                    positions.append(i)
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
