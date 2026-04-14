"""
NLD 数据处理 (适配 Qwen3-VL-8B-Instruct + VCR Latent Reasoning)

Dataset: 加载图像+问答数据，使用 VCR Latent Reasoning 格式:
         reasoning_for_training 中的 <|latent|> 标记隐空间思考触发点
         仅在 <|latent|> 位置触发 NativeLatentThinker (Final Answer 后不触发)

         训练序列格式:
           [prompt]<|im_start|>assistant\n
           {reasoning_for_training}\nFinal Answer:\n{answer}<|im_end|>
         
         其中 reasoning_for_training 包含 <|latent|> token，
         模型在该位置触发隐空间思考。
         注意: 只有 <|latent|> 位置触发隐空间推理，Final Answer 后不触发。
         每个 <|latent|> 的隐空间迭代步数由数据中 latent_key_tokens 的 stage 数决定。

         Labels 策略:
           - prompt 部分: -100 (不参与 loss)
           - reasoning 部分: 参与 loss (高质量 GPT 生成推理链)
           - <|latent|> token: 参与 loss (模型需学会在正确位置输出)
           - "\\nFinal Answer:\\n" 部分: 参与 loss
           - answer + <|im_end|>: 参与 loss

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


# <|latent|> 特殊 token (需要在 tokenizer 中注册)
LATENT_TOKEN = "<|latent|>"
# <|/latent|> 退出 token: 模型在 latent 推理完成后输出此 token 退出隐空间
LATENT_END_TOKEN = "<|/latent|>"

# Final Answer 分隔符
FINAL_ANSWER_SEP = "\nFinal Answer:\n"


class _SampleTooLongError(Exception):
    """样本序列超长且无法截断时抛出，触发 _get_item_safe 重试其他样本"""
    pass

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("[NLD Data] ⚠️ qwen_vl_utils 未安装，请运行: pip install qwen-vl-utils")
    process_vision_info = None


# ====== NLD System Prompt (VCR Latent Reasoning) ======
# 引导模型进行视觉推理，在需要深度思考时输出 <|latent|> token
# <|latent|> 触发隐空间思考，模型在隐空间中完成复杂推理后继续生成
NLD_SYSTEM_PROMPT = """You are a visual reasoning assistant. Analyze images carefully and think deeply.

Rules:
1. Describe your reasoning process clearly and concisely.
2. When you need to perform high-level visual thinking — such as mentally reconstructing 3D layouts from 2D views, reasoning about spatial relationships, simulating physical dynamics, resolving occlusions, or imagining counterfactual scenes — output <|latent|> to enter deep visual reasoning in latent space.
3. When your latent reasoning is complete, output <|/latent|> to exit latent space and continue generating.
4. After latent reasoning, continue with your conclusion grounded in the visual evidence.
5. End with "Final Answer:" on its own line, followed by your final answer.

Example:
The scene shows a ball mid-flight toward a glass window. To predict the outcome, I need to mentally simulate the collision dynamics and trace the impact geometry. <|latent|> <|/latent|> Based on the trajectory angle and estimated velocity, the ball will shatter the lower-left pane on impact, sending glass fragments inward in a cone pattern.
Final Answer:
The ball will break through the lower-left window pane, scattering glass fragments inward."""


class NLDDataset(Dataset):
    """
    NLD 数据集 (VCR Latent Reasoning 格式)
    
    训练数据格式:
    ```json
    {
        "image": "vcr1images/xxx.jpg",
        "image_path": "data/nld_phase1/raw/vcr/vcr1images/xxx.jpg",
        "question": "...",
        "answer": "...",
        "reasoning_for_training": "... <|pause|> ...",
        "task_type": "dynamic_simulation"
    }
    ```
    
    训练时的序列构造:
      [prompt]<|im_start|>assistant\n
      {reasoning}\nFinal Answer:\n{answer}<|im_end|>
      
      其中 reasoning 中的 <|pause|> 被替换为 <|latent|> (已注册的特殊 token)
    
    Labels 策略:
      - prompt 部分: -100 (不参与 loss)
      - reasoning + Final Answer + answer + <|im_end|>: 参与 loss
    
    Step boundary:
      - <|latent|> token 位置: 触发隐空间思考
      - "Final Answer:" 起始位置: 最后一个 boundary
    
    Args:
        json_path: 数据 JSON 文件路径
        processor: AutoProcessor (tokenizer 中需已注册 <|latent|> 特殊 token)
        image_base_dir: 图片基础路径 (用于拼接相对路径)
        use_system_prompt: 是否在对话中注入 NLD system prompt
        system_prompt: 自定义 system prompt
        max_seq_len: 最大序列长度
        skip_image_check: 是否跳过图片存在性检查
    """

    def __init__(
        self,
        json_path: str,
        processor,
        image_base_dir: str = None,
        use_system_prompt: bool = True,
        system_prompt: str = None,
        max_seq_len: int = 4096,
        skip_image_check: bool = False,
        **kwargs,  # 兼容旧参数，忽略不用的
    ):
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        self.image_base_dir = image_base_dir
        
        # 过滤无效样本
        valid_data = []
        invalid_count = 0
        no_image_count = 0
        no_reasoning_count = 0
        
        for item in raw_data:
            # 检查必要字段
            # 优先使用 'image' 字段 (短相对路径如 vcr1images/xxx.jpg)
            # 'image_path' 字段已含 data/nld_phase1/raw/vcr/ 前缀，与 image_base_dir 拼接会导致双重路径
            img = item.get('image') or item.get('image_path', '')
            if not img:
                no_image_count += 1
                continue
            
            # 拼接图片路径
            if self.image_base_dir and not os.path.isabs(img):
                full_path = os.path.join(self.image_base_dir, img)
            else:
                full_path = img
            item['_resolved_image_path'] = full_path
            
            # 检查 reasoning_for_training 字段
            if not item.get('reasoning_for_training', '').strip():
                no_reasoning_count += 1
                continue
            
            if not skip_image_check and not os.path.exists(full_path):
                invalid_count += 1
                continue
            
            valid_data.append(item)
        
        if invalid_count > 0:
            print(f"[NLDDataset] ⚠️ 已过滤 {invalid_count} 个图片缺失的样本")
        if no_image_count > 0:
            print(f"[NLDDataset] ⚠️ 已过滤 {no_image_count} 个无图像的样本")
        if no_reasoning_count > 0:
            print(f"[NLDDataset] ⚠️ 已过滤 {no_reasoning_count} 个无 reasoning 的样本")
        if invalid_count > 0 or no_image_count > 0 or no_reasoning_count > 0:
            print(f"[NLDDataset]   保留 {len(valid_data)}/{len(raw_data)}")
        
        self.data = valid_data
        self.processor = processor
        self.use_system_prompt = use_system_prompt
        self.system_prompt = system_prompt or NLD_SYSTEM_PROMPT
        self.max_seq_len = max_seq_len

        # 确保 tokenizer 配置
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

        # 获取 <|latent|> token id (必须已注册)
        self.latent_token_id = self.processor.tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
        if self.latent_token_id == self.processor.tokenizer.unk_token_id:
            raise ValueError(
                f"[NLDDataset] ❌ {LATENT_TOKEN} 未在 tokenizer 中注册! "
                f"请在创建 Dataset 之前调用 tokenizer.add_special_tokens()"
            )

        # 获取 <|/latent|> token id (退出 token, 必须已注册)
        self.latent_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(LATENT_END_TOKEN)
        if self.latent_end_token_id == self.processor.tokenizer.unk_token_id:
            raise ValueError(
                f"[NLDDataset] ❌ {LATENT_END_TOKEN} 未在 tokenizer 中注册! "
                f"请在创建 Dataset 之前调用 tokenizer.add_special_tokens()"
            )

        # Qwen3-VL 的 image placeholder token id
        try:
            self.IMAGE_TOKEN_ID = self.processor.image_token_id
        except AttributeError:
            self.IMAGE_TOKEN_ID = 151655  # Qwen3-VL 默认值

        # 确认图片分辨率限制已生效
        if hasattr(self.processor, 'image_processor'):
            _ip = self.processor.image_processor
            _min_px = getattr(_ip, 'min_pixels', None)
            _max_px = getattr(_ip, 'max_pixels', None)
            print(f"[NLDDataset] 图片分辨率: min_pixels={_min_px}, max_pixels={_max_px}")

        # 统计 task_type 分布
        task_counts = {}
        for item in self.data:
            tt = item.get('task_type', 'unknown')
            task_counts[tt] = task_counts.get(tt, 0) + 1

        print(f"[NLDDataset] 加载 {len(self.data)} 个样本 from {json_path}")
        print(f"[NLDDataset] VCR Latent Reasoning 格式, <|latent|> token id = {self.latent_token_id}, <|/latent|> token id = {self.latent_end_token_id}")
        if task_counts:
            print(f"[NLDDataset] 任务类型分布:")
            for tt, cnt in sorted(task_counts.items(), key=lambda x: -x[1]):
                print(f"  {tt}: {cnt} ({cnt/len(self.data):.1%})")

        # 预检查
        self._preflight_check(min(3, len(self.data)))

    def _preflight_check(self, n: int):
        """预检查前 n 个样本"""
        for i in range(n):
            item = self.data[i]
            answer = item.get('answer', '')
            if not answer.strip():
                print(f"[NLDDataset] ⚠️ 样本 {i} 的 answer 为空")
            reasoning = item.get('reasoning_for_training', '')
            if '<|pause|>' not in reasoning and LATENT_TOKEN not in reasoning:
                print(f"[NLDDataset] ⚠️ 样本 {i} 的 reasoning 中无 <|pause|>/<|latent|> 标记")
        print(f"[NLDDataset] 预检查完成: {n} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._get_item_safe(idx)

    def _get_item_safe(self, idx, _retry=0):
        """带容错的样本获取"""
        if _retry > 10:
            raise RuntimeError(f"[NLDDataset] 连续 {_retry} 次采样失败")
        try:
            return self._get_item_inner(idx)
        except _SampleTooLongError as e:
            if _retry == 0:
                warnings.warn(f"[NLDDataset] 样本 {idx} 超长被跳过: {e}")
            new_idx = random.randint(0, len(self.data) - 1)
            return self._get_item_safe(new_idx, _retry + 1)
        except (FileNotFoundError, OSError) as e:
            warnings.warn(f"[NLDDataset] 样本 {idx} 加载失败: {e}, 随机替换")
            new_idx = random.randint(0, len(self.data) - 1)
            return self._get_item_safe(new_idx, _retry + 1)

    def _get_item_inner(self, idx):
        item = self.data[idx]

        image_path = item.get('_resolved_image_path', item.get('image', item.get('image_path', '')))
        question = item.get('question', '')
        answer = item.get('answer', '')
        reasoning = item.get('reasoning_for_training', '')

        # 将 <|pause|> 替换为 <|latent|> (数据中使用 <|pause|>，训练使用 <|latent|>)
        reasoning = reasoning.replace('<|pause|>', LATENT_TOKEN)
        
        # 在每个 <|latent|> 后面插入 <|/latent|>
        # 训练序列: ... <|latent|> <|/latent|> ...
        # 模型学习: 输出 <|latent|> 进入隐空间 → latent 推理 → 输出 <|/latent|> 退出
        # <|/latent|> 紧跟在 <|latent|> 后面，中间的 latent 推理步骤在 hidden space 中完成
        reasoning = reasoning.replace(LATENT_TOKEN, LATENT_TOKEN + ' ' + LATENT_END_TOKEN)

        # ---- 1. 构造消息 (含 system prompt) ----
        messages = []
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

        # Tokenize prompt
        prompt_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        # ---- 2. 构造回答: "{reasoning}\nFinal Answer:\n{answer}<|im_end|>" ----
        im_end_token = "<|im_end|>"
        
        # 分别 tokenize reasoning 和 answer 部分
        # reasoning 部分包含 <|latent|> token
        reasoning_tokens = self.processor.tokenizer(
            reasoning,
            add_special_tokens=False,
            return_tensors="pt",
        )
        
        # Final Answer 分隔 + answer + im_end
        answer_text = FINAL_ANSWER_SEP + answer + im_end_token
        answer_tokens = self.processor.tokenizer(
            answer_text,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # ---- 3. 定位 <|latent|> 在 reasoning_ids 中的位置 ----
        reasoning_ids = reasoning_tokens['input_ids'][0]  # [reasoning_len]
        reasoning_mask = reasoning_tokens['attention_mask'][0]
        answer_ids = answer_tokens['input_ids'][0]  # [answer_len]
        answer_mask = answer_tokens['attention_mask'][0]
        
        # 查找 <|latent|> token 的位置 (在 reasoning_ids 中)
        latent_positions_in_reasoning = (reasoning_ids == self.latent_token_id).nonzero(as_tuple=True)[0].tolist()

        # ---- 4. 拼接 prompt + reasoning + answer ----
        prompt_ids = prompt_inputs['input_ids'].squeeze(0)
        prompt_mask = prompt_inputs['attention_mask'].squeeze(0)

        # 序列长度硬截断: 只截断 reasoning 部分
        if self.max_seq_len > 0:
            prompt_len_val = len(prompt_ids)
            answer_len_val = len(answer_ids)
            max_reasoning_len = self.max_seq_len - prompt_len_val - answer_len_val

            if max_reasoning_len <= 0:
                raise _SampleTooLongError(
                    f"prompt({prompt_len_val}) + answer({answer_len_val}) = "
                    f"{prompt_len_val + answer_len_val} > max_seq_len {self.max_seq_len}"
                )

            if len(reasoning_ids) > max_reasoning_len:
                # 截断 reasoning: 保留前半段 + 后半段
                orig_len = len(reasoning_ids)
                half = max_reasoning_len // 2
                reasoning_ids = torch.cat([reasoning_ids[:half], reasoning_ids[-half:]], dim=0)
                reasoning_mask = torch.cat([reasoning_mask[:half], reasoning_mask[-half:]], dim=0)
                # 重映射 latent_positions
                new_positions = []
                for pos in latent_positions_in_reasoning:
                    if pos < half:
                        new_positions.append(pos)
                    elif pos >= orig_len - half:
                        new_pos = half + (pos - (orig_len - half))
                        if new_pos < max_reasoning_len:
                            new_positions.append(new_pos)
                latent_positions_in_reasoning = new_positions

        # 拼接完整序列
        input_ids = torch.cat([prompt_ids, reasoning_ids, answer_ids], dim=0)
        attention_mask = torch.cat([prompt_mask, reasoning_mask, answer_mask], dim=0)

        assert self.max_seq_len <= 0 or len(input_ids) <= self.max_seq_len, \
            f"BUG: 截断后仍超限 {len(input_ids)} > {self.max_seq_len}"

        # ---- 5. Labels: reasoning + answer 全部参与 loss ----
        prompt_len = len(prompt_ids)
        
        labels = torch.cat([
            torch.full_like(prompt_ids, -100),   # prompt: 不监督
            reasoning_ids.clone(),                # reasoning (含 <|latent|>): 监督 ✅
            answer_ids.clone(),                   # "\nFinal Answer:\n{answer}<|im_end|>": 监督 ✅
        ], dim=0)

        # ---- 6. Loss weight mask (answer 加权) ----
        reasoning_len = len(reasoning_ids)
        answer_len_val = len(answer_ids)
        if answer_len_val > 0 and reasoning_len > 0:
            answer_weight = min(max(reasoning_len / answer_len_val, 1.0), 10.0)
        else:
            answer_weight = 1.0
        
        weight_mask = torch.cat([
            torch.zeros(prompt_len),                              # prompt: 不参与
            torch.ones(reasoning_len),                            # reasoning: 权重 1.0
            torch.full((answer_len_val,), answer_weight),         # answer: 加权
        ], dim=0)

        # ---- 7. Step positions (boundary 位置) ----
        # 只有 <|latent|> 位置触发隐空间推理，Final Answer 后不触发
        step_positions = [pos + prompt_len for pos in latent_positions_in_reasoning]

        # ---- 8. 从 latent_key_tokens 获取每个 <|latent|> 的隐空间迭代步数 ----
        # latent_key_tokens 是一个 list of stages，每个 stage 对应一个认知阶段
        # 隐空间迭代步数 = stage 数量 (不同数据的 stage 数不同)
        latent_key_tokens = item.get('latent_key_tokens', [])
        num_stages = item.get('num_stages', len(latent_key_tokens) if latent_key_tokens else 0)
        # 为每个 <|latent|> 触发点分配迭代步数
        # 当前数据中每条只有 1 个 <|latent|>，所以 latent_think_steps 是一个列表
        # 如果没有 stage 信息，默认使用 0 (由模型的 max_think_steps 兜底)
        latent_think_steps = [num_stages] * len(latent_positions_in_reasoning) if num_stages > 0 else []

        # ---- 9. Stage Concept Tokens: 将 latent_key_tokens 的关键词 tokenize ----
        # 每个 stage 的关键词 tokens 会被 tokenize，用于 Stage-Aligned Concept Supervision
        # 格式: List[List[int]] — 每个 stage 对应一个 token id 列表
        stage_concept_token_ids = []
        if latent_key_tokens:
            for stage_info in latent_key_tokens:
                # 将该 stage 的所有关键词拼接为一个字符串，然后 tokenize
                keywords = stage_info.get('tokens', [])
                if keywords:
                    keyword_text = ', '.join(keywords)
                    token_ids = self.processor.tokenizer(
                        keyword_text,
                        add_special_tokens=False,
                        return_tensors=None,
                    )['input_ids']
                    stage_concept_token_ids.append(token_ids)
                else:
                    stage_concept_token_ids.append([])

        # pixel_values 和 image_grid_thw
        pixel_values = prompt_inputs.get('pixel_values')
        image_grid_thw = prompt_inputs.get('image_grid_thw')

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "has_image": pixel_values is not None,
            "prompt_len": prompt_len,
            "step_positions": step_positions,
            "latent_think_steps": latent_think_steps,
            "loss_weight_mask": weight_mask,
            "task_type": item.get('task_type', 'unknown'),
            "num_reasoning_tokens": int(reasoning_len),
            "num_answer_tokens": int(answer_len_val),
            "num_latent_triggers": len(latent_positions_in_reasoning),
            "answer_weight": float(answer_weight),
            "num_stages": num_stages,
            "stage_concept_token_ids": stage_concept_token_ids,
        }


class NLDCollator:
    """
    NLD Data Collator
    
    职责:
    1. Left padding 对齐批次
    2. 拼接 pixel_values 和 image_grid_thw
    3. 返回每个样本的 step boundary 位置
    
    Args:
        processor: AutoProcessor
        latent_token_id: <|latent|> 的 token id
    """

    def __init__(self, processor, latent_token_id: int = None, **kwargs):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        self.latent_token_id = latent_token_id

    def _left_pad(self, sequences, pad_value):
        """Left padding"""
        batch_size = len(sequences)
        max_len = max(seq.size(0) for seq in sequences)
        padded = torch.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            padded[i, max_len - seq.size(0):] = seq
        return padded

    def _left_pad_float(self, sequences, pad_value):
        """Left padding for float tensors"""
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
        loss_weight_mask = self._left_pad_float([item['loss_weight_mask'] for item in batch], 0.0)

        # 图像处理
        pixel_values_list = [item['pixel_values'] for item in batch if item['has_image']]
        image_grid_thw_list = [item['image_grid_thw'] for item in batch if item['has_image']]
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0)
            image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
        else:
            pixel_values = None
            image_grid_thw = None

        # 计算 left padding 偏移后的 prompt_len
        max_len = input_ids.shape[1]
        prompt_lens = []
        for item in batch:
            orig_len = len(item['input_ids'])
            pad_offset = max_len - orig_len
            prompt_lens.append(pad_offset + item['prompt_len'])

        # step_positions 加上 left-padding 偏移
        step_boundaries = []
        for b_idx, item in enumerate(batch):
            orig_len = len(item['input_ids'])
            pad_offset = max_len - orig_len
            positions = [pos + pad_offset for pos in item.get('step_positions', [])]
            step_boundaries.append(positions)

        # 每个样本的 latent_think_steps (per-boundary 的隐空间迭代步数)
        latent_think_steps = [item.get('latent_think_steps', []) for item in batch]

        # Stage Concept Token IDs: 每个样本的每个 stage 的关键词 token ids
        # 格式: List[List[List[int]]] — [batch][stage][token_ids]
        stage_concept_token_ids = [item.get('stage_concept_token_ids', []) for item in batch]

        # 样本级元信息
        sample_meta = {
            "task_types": [item.get('task_type', 'unknown') for item in batch],
            "num_reasoning_tokens": [item.get('num_reasoning_tokens', 0) for item in batch],
            "num_answer_tokens": [item.get('num_answer_tokens', 0) for item in batch],
            "num_latent_triggers": [item.get('num_latent_triggers', 0) for item in batch],
            "answer_weights": [item.get('answer_weight', 1.0) for item in batch],
            "num_stages": [item.get('num_stages', 0) for item in batch],
        }

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "step_boundaries": step_boundaries,
            "latent_think_steps": latent_think_steps,
            "prompt_lens": prompt_lens,
            "loss_weight_mask": loss_weight_mask,
            "stage_concept_token_ids": stage_concept_token_ids,
            "sample_meta": sample_meta,
        }
