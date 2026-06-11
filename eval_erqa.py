#!/usr/bin/env python3
"""
ERQA 全数据集纯推理测试脚本（无可视化）

功能:
  1. 加载 NLD 模型 + checkpoint
  2. 对 ERQA 全部 400 条数据进行推理（支持多图输入）
  3. 使用与训练一致的 system prompt，让模型自适应触发 latent 推理
  4. 从 "Final Answer:" 后面提取最终选项字母，与 ground truth 对比
  5. 输出总体准确率和按 question_type 分类的准确率
"""
import os, sys, json, argparse, re, time
from typing import Optional, List, Dict, Tuple
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("[Eval] ⚠️ qwen_vl_utils 未安装，多图支持可能受限")
    process_vision_info = None


# ====== 分布式辅助 ======
def _ddp_env():
    """读取 RANK / WORLD_SIZE / LOCAL_RANK"""
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    return rank, world_size, local_rank


def _is_main():
    return int(os.environ.get('RANK', '0')) == 0


def _maybe_init_dist():
    rank, world_size, local_rank = _ddp_env()
    if world_size > 1 and not dist.is_initialized():
        # 使用 NCCL 后端进行小量元数据同步
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def load_nld_model(model_path, checkpoint=None, device="cuda", dtype=torch.bfloat16):
    """加载 NLD 模型"""
    from transformers import AutoProcessor
    from rld.model_v2 import NLDModel
    from rld.data import LATENT_TOKEN, LATENT_END_TOKEN, NLD_SYSTEM_PROMPT

    if _is_main():
        print(f"[Eval] 加载基座模型: {model_path}")
    model = NLDModel(model_path=model_path, torch_dtype=dtype, attn_implementation="flash_attention_2")
    processor = AutoProcessor.from_pretrained(model_path)

    # 注册特殊 token
    num_added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [LATENT_TOKEN, LATENT_END_TOKEN]}
    )
    # ---- 关键: embedding 大小适配, 与 diagnose_latent_ab.py / train_nld.py 一致 ----
    # Qwen3-VL 原生 embed_tokens.shape[0] = 151936 (含 ~265 个预留 special token 槽位),
    # 而 len(tokenizer) 通常只有 ~151671. 若直接 resize_token_embeddings(len(tokenizer))
    # 会把 embedding 从 151936 **缩小**到 151671, 截断 Qwen 预留权重, 同时
    # 与训练 ckpt 的 [151936, H] 形状不匹配, 加载时报 size mismatch.
    # 正确做法是 max(len(tokenizer), V_old): 训练 ckpt 与推理模型形状始终对齐.
    _V_old = model.base_model.get_input_embeddings().weight.shape[0]
    _target_vocab = max(len(processor.tokenizer), _V_old)
    if _target_vocab != _V_old:
        model.base_model.resize_token_embeddings(_target_vocab)
    if _is_main():
        _V_new = model.base_model.get_input_embeddings().weight.shape[0]
        print(f"[Eval] 注册特殊 token: {LATENT_TOKEN}, {LATENT_END_TOKEN} (新增 {num_added} 个, 词表大小: {len(processor.tokenizer)})")
        print(f"[Eval] embedding: {_V_old} -> {_V_new} (target=max(tokenizer={len(processor.tokenizer)}, old={_V_old}))")
    model.set_processor(processor)

    if checkpoint:
        if _is_main():
            print(f"[Eval] 加载 checkpoint: {checkpoint}")
        model.load_pretrained(checkpoint)

    model = model.to(device)
    model.eval()
    return model, processor


def extract_final_answer(text: str) -> str:
    """
    从模型完整输出中提取 "Final Answer:" 后面的内容

    训练格式: 模型输出 reasoning + "\\nFinal Answer:\\n" + 最终答案
    """
    # 查找 "Final Answer:" (不区分大小写，允许前后空白)
    patterns = [
        r'Final\s*Answer\s*:\s*\n?(.*)',
        r'final\s*answer\s*:\s*\n?(.*)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    # 如果没有 Final Answer 标记，返回整个文本（fallback）
    return text.strip()


def extract_answer_letter(text: str) -> str:
    """
    从 Final Answer 后的文本中提取选项字母 (A/B/C/D)

    策略（按优先级）:
      1. 文本非常短（去除非字母后只有一个A-D），直接返回
      2. 开头匹配 "A." 或 "(A)" 或 "A " 等
      3. 匹配 "answer is X" 模式
      4. 第一个独立出现的 A-D 大写字母
      5. 任何位置的 A-D
    """
    text = text.strip()
    if not text:
        return ""

    # 策略1: 文本非常短，直接提取
    clean = re.sub(r'[^A-Da-d]', '', text)
    if len(clean) == 1:
        return clean.upper()

    # 策略2: 开头匹配 "A." 或 "A" 或 "(A)" 或 "[A]"
    m = re.match(r'^[\s\(\[]*([A-Da-d])[\s\.\)\]\,:]', text)
    if m:
        return m.group(1).upper()
    # 单独一个字母开头
    m = re.match(r'^([A-Da-d])$', text.split('\n')[0].strip())
    if m:
        return m.group(1).upper()

    # 策略3: "answer is X" 或 "option X" 或 "choice X"
    m = re.search(r'(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?([A-Da-d])[\)\]\.]?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 策略4: 第一个独立出现的 A-D 大写字母（word boundary）
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)

    # 策略5: 任何位置的 A-D（最后兜底）
    m = re.search(r'([A-Da-d])', text)
    if m:
        return m.group(1).upper()

    return ""


def run_eval(args):
    """运行 ERQA 全数据集评估 (支持多卡数据并行)"""
    from PIL import Image
    from rld.data import NLD_SYSTEM_PROMPT, LATENT_TOKEN, LATENT_END_TOKEN

    # ===== 分布式初始化 =====
    rank, world_size, local_rank = _maybe_init_dist()
    is_main = (rank == 0)
    args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # 加载模型
    dm = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    model, processor = load_nld_model(args.model_path, args.checkpoint, args.device, dm[args.dtype])

    # 加载数据
    data_file = args.data_file
    data_root = args.data_root
    if is_main:
        print(f"[Eval] 加载数据: {data_file}")
        print(f"[Eval] 数据根目录: {data_root}")
        print(f"[Eval] world_size={world_size}")

    samples_all = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples_all.append(json.loads(line))

    if args.max_samples > 0:
        samples_all = samples_all[:args.max_samples]

    # 多卡分片: rank i 处理 i, i+world_size, i+2*world_size...
    samples = samples_all[rank::world_size] if world_size > 1 else samples_all
    n_global = len(samples_all)
    n_local = len(samples)

    # 统计多图样本
    if is_main:
        multi_img_count = sum(1 for s in samples_all if s.get("num_images", 1) > 1)
        print(f"[Eval] 全局样本: {n_global} 条 (单图: {n_global-multi_img_count}, 多图: {multi_img_count})")
        print(f"[Eval] 本 rank 负责: {n_local} 条")
        print(f"[Eval] skip_latent={args.skip_latent}")
        print(f"[Eval] max_new_tokens={args.max_new_tokens}")
        print(f"[Eval] 使用训练 system prompt: {'是' if not args.no_system_prompt else '否'}")
        print()
    # 注意: 方案 B (DDP-free) 下不再 dist.barrier() —— 各 rank 已独立完成数据分片,
    # 也不需要全员同步起跑. 任何 collective 都会在异步推理结束时撞超时.

    # 评估
    results = []
    correct = 0
    total = 0
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    latent_stats = {"triggered": 0, "not_triggered": 0, "triggered_correct": 0, "not_triggered_correct": 0}
    errors = []
    # ===== latent 健康度诊断聚合 =====
    diag_agg = {
        'num_thought_steps': [],          # 每次 latent 触发实际推理步数
        'h_first_last_cos': [],            # 跨步演化幅度 (越低说明 hidden 在动)
        'h_norm_mean': [],                 # hidden L2 范数
        'h_adj_min': [],                   # 相邻步 cos 最小 (>0.99 即原地踏步)
        'h_sat_step1': [],                 # step0->step1 饱和度 (>0.99 即固定第二步退出)
        'h_sat_early_exit_ratio': [],      # 早退比例
        'topk_hit_ratio': [],              # SW-SRS topk 命中率
        'h_stage_diag_score': [],          # stage 对应步对齐分
        'h_stage_monotonic': [],           # stage 单调性
        'h_stage_shift_kl': [],            # stage 间分布偏移
    }
    exit_reason_counter = Counter()

    start_time = time.time()

    for idx, sample in enumerate(samples):
        qid = sample.get("question_id", f"Q_{idx}")
        question = sample["question"]
        answer_gt = sample["answer"].strip().upper()
        q_type = sample.get("question_type", "Unknown")
        num_images = sample.get("num_images", 1)

        # 获取所有图像路径
        image_paths = []
        if "image_paths" in sample:
            paths_list = sample["image_paths"] if isinstance(sample["image_paths"], list) else [sample["image_paths"]]
            for img_rel in paths_list:
                if data_root and not os.path.isabs(img_rel):
                    img_path = os.path.join(data_root, img_rel)
                else:
                    img_path = img_rel
                image_paths.append(img_path)
        elif "image" in sample:
            img_rel = sample["image"]
            if data_root and not os.path.isabs(img_rel):
                img_path = os.path.join(data_root, img_rel)
            else:
                img_path = img_rel
            image_paths.append(img_path)
        else:
            print(f"  [r{rank} {idx+1}/{n_local}] {qid}: ⚠️ 缺少图像字段，跳过", flush=True)
            continue

        # 检查图像是否存在
        missing = [p for p in image_paths if not os.path.exists(p)]
        if missing:
            print(f"  [r{rank} {idx+1}/{n_local}] {qid}: ⚠️ 图像不存在: {missing[0]}，跳过", flush=True)
            continue

        try:
            # 构建消息（与训练时格式一致）
            # 1. System prompt（与训练时一致，引导模型自适应触发 latent）
            messages = []
            if not args.no_system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": NLD_SYSTEM_PROMPT}],
                })

            # 2. User message（多图 + 问题）
            user_content = []
            for img_path in image_paths:
                user_content.append({"type": "image", "image": img_path})
            user_content.append({"type": "text", "text": question})
            messages.append({"role": "user", "content": user_content})

            # 3. 使用 processor 处理输入
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 4. 处理图像（使用 qwen_vl_utils 支持多图）
            if process_vision_info:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                # fallback: 手动加载图像
                image_inputs = [Image.open(p).convert('RGB') for p in image_paths]
                video_inputs = None

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            ).to(args.device)

            # 5. 使用 generate_with_fallback() 进行推理 (LatentDraft 默认行为)
            # ---- LatentDraft 默认推理时行为: 一旦命中 hit_max / no Final Answer
            # / ngram 重复, 自动用 directly-answer prompt + 强制 'Final Answer: '
            # 前缀重跑一次, 防止 latent 病态思考耗尽 max_new_tokens. ----
            from rld.inference_utils import generate_with_fallback
            generated_ids, prompt_len, latent_diags, _fb_meta = generate_with_fallback(
                model, processor, messages, inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                do_sample=False,
                return_diagnostics=True,
                enable_fallback=True,
            )

            # 6. 解码生成结果
            generated_tokens = generated_ids[0, prompt_len:]

            # 保留特殊 token 的完整输出（用于分析 latent 触发）
            full_output_raw = processor.tokenizer.decode(generated_tokens, skip_special_tokens=False)
            # 干净输出（去除特殊 token）
            full_output_clean = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # 7. 统计 latent 触发
            num_latent_triggers = full_output_raw.count(LATENT_TOKEN)

            # 7.5 聚合 latent 健康度诊断
            sample_diag_summary = []
            for diag in latent_diags:
                hs = diag.get('hidden_stats', {}) or {}
                sw = diag.get('sw_srs_stats', {}) or {}
                exit_reason_counter[diag.get('exit_reason', 'unknown')] += 1
                # 累积全局指标
                if 'num_thought_steps' in diag:
                    diag_agg['num_thought_steps'].append(diag['num_thought_steps'])
                for key in ['h_first_last_cos', 'h_norm_mean', 'h_adj_min',
                            'h_sat_step1', 'h_sat_early_exit_ratio',
                            'h_stage_diag_score', 'h_stage_monotonic', 'h_stage_shift_kl']:
                    if key in hs:
                        diag_agg[key].append(hs[key])
                if 'topk_hit_ratio' in sw:
                    diag_agg['topk_hit_ratio'].append(sw['topk_hit_ratio'])
                # 仅保留每次触发的简短信息
                sample_diag_summary.append({
                    'steps': diag.get('num_thought_steps'),
                    'exit': diag.get('exit_reason'),
                    'first_last_cos': hs.get('h_first_last_cos'),
                    'sat_step1': hs.get('h_sat_step1'),
                    'topk_hit': sw.get('topk_hit_ratio'),
                })

            # 8. 从 "Final Answer:" 后提取最终答案
            final_answer_text = extract_final_answer(full_output_clean)
            pred = extract_answer_letter(final_answer_text)
            is_correct = (pred == answer_gt)

            total += 1
            if is_correct:
                correct += 1
            type_stats[q_type]["total"] += 1
            if is_correct:
                type_stats[q_type]["correct"] += 1

            # Latent 触发统计
            if num_latent_triggers > 0:
                latent_stats["triggered"] += 1
                if is_correct:
                    latent_stats["triggered_correct"] += 1
            else:
                latent_stats["not_triggered"] += 1
                if is_correct:
                    latent_stats["not_triggered_correct"] += 1

            results.append({
                "question_id": qid,
                "question_type": q_type,
                "question": question,
                "num_images": num_images,
                "answer_gt": answer_gt,
                "answer_pred": pred,
                "final_answer_text": final_answer_text,
                "full_output": full_output_clean,
                "full_output_raw": full_output_raw,
                "num_latent_triggers": num_latent_triggers,
                "num_generated_tokens": len(generated_tokens),
                "correct": is_correct,
                "latent_diags": sample_diag_summary,
            })

            # 打印进度 (多卡时打印本 rank 进度 + 全局估计)
            acc_so_far = correct / total * 100
            status = "✅" if is_correct else "❌"
            latent_info = f"🧠×{num_latent_triggers}" if num_latent_triggers > 0 else "  "
            if (idx + 1) % 10 == 0 or not is_correct:
                tag = f"r{rank} " if world_size > 1 else ""
                print(f"  [{tag}{idx+1}/{n_local}] {qid} {status} {latent_info} pred={pred} gt={answer_gt} "
                      f"acc={acc_so_far:.1f}% | FA: \"{final_answer_text[:60]}\"", flush=True)

        except Exception as e:
            print(f"  [r{rank} {idx+1}/{n_local}] {qid}: ❌ 错误: {e}", flush=True)
            errors.append({"question_id": qid, "error": str(e)})
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time

    # ===== 多卡聚合 (方案 B: 每 rank 落盘 + 主 rank 轮询合并, 无任何 NCCL collective) =====
    # 设计动机:
    #   原方案用 dist.barrier + dist.gather_object, 但 LatentDraft 推理耗时方差极大
    #   (latent 触发次数因样本而异, fallback 也会重 prefill), 快 rank 先到 barrier
    #   会启动 ALLREDUCE 等慢 rank, 600s 后 NCCL watchdog 触发 SIGABRT 全员崩溃.
    #
    # 方案 B: 完全摆脱 NCCL collective —— 每 rank 把 local_payload 写到独立的
    # shard json, 非主 rank 写完即退; 主 rank 写完后轮询等待所有分片落盘, 收齐
    # 后读盘合并. 慢 rank 不阻塞快 rank, 快 rank 提前退出释放显存, 撞超时无任何风险.
    if world_size > 1:
        # 打包本 rank 产出 (与原 gather_object 路径完全一致的字段)
        local_payload = {
            'results': results,
            'errors': errors,
            'correct': correct,
            'total': total,
            'type_stats': {k: dict(v) for k, v in type_stats.items()},
            'latent_stats': dict(latent_stats),
            'diag_agg': {k: list(v) for k, v in diag_agg.items()},
            'exit_reason_counter': dict(exit_reason_counter),
            'elapsed': elapsed,
            'rank': rank,
        }

        # ---- 每 rank 各自把 shard 写到 output_dir ----
        # 用 .tmp -> rename 实现 atomic write, 避免主 rank 读到半截文件.
        os.makedirs(args.output_dir, exist_ok=True)
        shard_path = os.path.join(
            args.output_dir, f".shard_rank{rank}_of{world_size}.json"
        )
        shard_tmp = shard_path + ".tmp"
        with open(shard_tmp, "w", encoding="utf-8") as f:
            json.dump(local_payload, f, ensure_ascii=False)
        os.replace(shard_tmp, shard_path)
        print(f"[Eval] [r{rank}] shard 已落盘: {shard_path} "
              f"(correct={correct}/{total}, elapsed={elapsed:.1f}s)", flush=True)

        if not is_main:
            # 非主 rank 写完分片就退出, 不参与合并, 不发任何 collective
            # 注意: 不主动 destroy_process_group, 让 main() 的 finally 兜底
            return

        # ---- 主 rank: 轮询等待所有分片落盘, 然后合并 ----
        expected = [
            os.path.join(args.output_dir, f".shard_rank{r}_of{world_size}.json")
            for r in range(world_size)
        ]
        # 超时设置: 单 rank 最长跑完时间 + 安全余量. ERQA 400 条/8 卡 ≈ 50 题/rank,
        # 单题最坏 ~30s (1024 token + fallback) → 25min 一 rank, 这里给 4 小时.
        WAIT_TIMEOUT_S = float(os.environ.get('SHARD_WAIT_TIMEOUT_S', '14400'))
        WAIT_INTERVAL_S = 5.0
        REPORT_INTERVAL_S = 30.0
        wait_start = time.time()
        last_report = wait_start
        while True:
            missing = [p for p in expected if not os.path.exists(p)]
            if not missing:
                print(f"[Eval] [r0] 所有 {world_size} 个 shard 已就绪, 开始合并",
                      flush=True)
                break
            now = time.time()
            if now - wait_start > WAIT_TIMEOUT_S:
                print(f"[Eval] [r0] ⚠️ 等待 shard 超时 ({WAIT_TIMEOUT_S}s), "
                      f"仍缺 {len(missing)} 个分片, 用现有分片合并",
                      flush=True)
                break
            if now - last_report > REPORT_INTERVAL_S:
                ready = world_size - len(missing)
                print(f"[Eval] [r0] 等待 shard 中... {ready}/{world_size} 就绪 "
                      f"(已等 {now-wait_start:.0f}s, 缺: "
                      f"{[os.path.basename(p) for p in missing]})",
                      flush=True)
                last_report = now
            time.sleep(WAIT_INTERVAL_S)

        # ---- 读盘 + 合并 (与原 for payload in gathered 段语义一致) ----
        gathered = []
        for p in expected:
            if not os.path.exists(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    gathered.append(json.load(f))
            except Exception as _err:
                print(f"[Eval] [r0] ⚠️ 读取 shard 失败 {p}: {_err}", flush=True)

        # rank 0 合并所有 shard
        results = []
        errors = []
        correct = 0
        total = 0
        type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        latent_stats = {"triggered": 0, "not_triggered": 0, "triggered_correct": 0, "not_triggered_correct": 0}
        diag_agg = {k: [] for k in diag_agg.keys()}
        exit_reason_counter = Counter()
        max_elapsed = 0.0
        for payload in gathered:
            if payload is None:
                continue
            results.extend(payload['results'])
            errors.extend(payload['errors'])
            correct += payload['correct']
            total += payload['total']
            for q_type, s in payload['type_stats'].items():
                type_stats[q_type]['correct'] += s['correct']
                type_stats[q_type]['total'] += s['total']
            for k, v in payload['latent_stats'].items():
                latent_stats[k] += v
            for k, vlist in payload['diag_agg'].items():
                if k in diag_agg:
                    diag_agg[k].extend(vlist)
            for k, v in payload['exit_reason_counter'].items():
                exit_reason_counter[k] += v
            max_elapsed = max(max_elapsed, payload['elapsed'])
        elapsed = max_elapsed

        # ---- 合并完成: 清理临时 shard 文件 ----
        for p in expected:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        samples_total_for_print = n_global
    else:
        samples_total_for_print = n_global

    # 汇总结果
    print()
    print("=" * 70)
    print("  ERQA 评估结果")
    print("=" * 70)
    print(f"  总样本数: {samples_total_for_print}")
    print(f"  有效推理: {total}")
    print(f"  错误跳过: {len(errors)}")
    print(f"  总耗时:   {elapsed:.1f}s ({elapsed/max(total,1):.2f}s/样本, world_size={world_size})")
    print()

    if total > 0:
        overall_acc = correct / total * 100
        print(f"  📊 总体准确率: {correct}/{total} = {overall_acc:.2f}%")
        print()

        # Latent 触发分析
        print(f"  🧠 Latent 触发分析:")
        lt = latent_stats["triggered"]
        lnt = latent_stats["not_triggered"]
        print(f"    触发 latent:   {lt}/{total} ({lt/total*100:.1f}%)")
        print(f"    未触发 latent: {lnt}/{total} ({lnt/total*100:.1f}%)")
        if lt > 0:
            lt_acc = latent_stats["triggered_correct"] / lt * 100
            print(f"    触发时准确率:  {latent_stats['triggered_correct']}/{lt} = {lt_acc:.2f}%")
        if lnt > 0:
            lnt_acc = latent_stats["not_triggered_correct"] / lnt * 100
            print(f"    未触发准确率:  {latent_stats['not_triggered_correct']}/{lnt} = {lnt_acc:.2f}%")
        print()

        # 按题型分类
        print(f"  📋 按题型分类:")
        print(f"  {'题型':<25s} {'正确':>5s} {'总数':>5s} {'准确率':>8s}")
        print(f"  {'-'*50}")
        for q_type in sorted(type_stats.keys()):
            s = type_stats[q_type]
            acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
            print(f"  {q_type:<25s} {s['correct']:>5d} {s['total']:>5d} {acc:>7.2f}%")

        # 生成长度统计
        gen_lens = [r["num_generated_tokens"] for r in results]
        if gen_lens:
            print()
            print(f"  📝 生成统计:")
            print(f"    平均生成 tokens: {sum(gen_lens)/len(gen_lens):.0f}")
            print(f"    最短: {min(gen_lens)}, 最长: {max(gen_lens)}")
            latent_counts = [r["num_latent_triggers"] for r in results]
            print(f"    平均 latent 触发次数: {sum(latent_counts)/len(latent_counts):.2f}")

        # ===== Latent 健康度诊断 =====
        print()
        print("  " + "=" * 66)
        print("  🧠 Latent 健康度诊断 (推理期触发次数: {})".format(sum(len(diag_agg[k]) for k in ['num_thought_steps']) // 1 if diag_agg['num_thought_steps'] else 0))
        print("  " + "=" * 66)

        def _stat(name, vals, fmt="{:.4f}", warn_lo=None, warn_hi=None, hint=None):
            if not vals:
                print(f"  {name:<30s}: 无数据")
                return
            mn = min(vals); mx = max(vals); mean = sum(vals)/len(vals)
            warn = ""
            if warn_lo is not None and mean < warn_lo:
                warn = f"  ⚠️ 偏低 ({hint})"
            if warn_hi is not None and mean > warn_hi:
                warn = f"  ⚠️ 偏高 ({hint})"
            print(f"  {name:<30s}: mean={fmt.format(mean)}  min={fmt.format(mn)}  max={fmt.format(mx)}  n={len(vals)}{warn}")

        # 1. 推理步数分布 (诊断 early-exit 坍塌)
        if diag_agg['num_thought_steps']:
            steps_cnt = Counter(diag_agg['num_thought_steps'])
            print("  推理步数分布 (num_thought_steps):")
            total_t = sum(steps_cnt.values())
            for s in sorted(steps_cnt.keys()):
                pct = steps_cnt[s] / total_t * 100
                bar = '█' * int(pct / 2)
                print(f"    steps={s:>2d}: {steps_cnt[s]:>4d} ({pct:>5.1f}%) {bar}")
            mean_steps = sum(diag_agg['num_thought_steps'])/len(diag_agg['num_thought_steps'])
            print(f"    平均推理步数: {mean_steps:.2f}")
            if mean_steps < 2.5:
                print("    ⚠️ early-exit 坍塌警报: 平均推理步数过低 (期望 3-9)")

        # 2. 退出原因分布
        if exit_reason_counter:
            print("  退出原因分布 (exit_reason):")
            total_t = sum(exit_reason_counter.values())
            for reason, cnt in exit_reason_counter.most_common():
                pct = cnt / total_t * 100
                bar = '█' * int(pct / 2)
                marker = ''
                if reason == 'exit_token' and pct > 50:
                    marker = '  ⚠️ exit_token 主导'
                if reason == 'saturation' and pct > 50:
                    marker = '  ⚠️ saturation 主导'
                print(f"    {reason:<14s}: {cnt:>4d} ({pct:>5.1f}%) {bar}{marker}")

        # 3. Hidden 几何 (坍塌检测)
        print("  Hidden 几何:")
        _stat("h_first_last_cos", diag_agg['h_first_last_cos'], warn_hi=0.95,
              hint="跨步演化幅度不足, hidden 几乎不动")
        _stat("h_adj_min", diag_agg['h_adj_min'], warn_hi=0.99,
              hint="某些相邻步原地踏步")
        _stat("h_norm_mean", diag_agg['h_norm_mean'], warn_lo=10.0, warn_hi=300.0,
              hint="hidden 范数异常")

        # 4. Saturation (推理期 early-exit 预警)
        print("  Saturation 指标:")
        _stat("h_sat_step1", diag_agg['h_sat_step1'], warn_hi=0.95,
              hint="第一步就饱和，固定二步退出坍塌")
        _stat("h_sat_early_exit_ratio", diag_agg['h_sat_early_exit_ratio'], warn_hi=0.5,
              hint="过多样本推理期早退")

        # 5. SW-SRS topk 命中率 (hidden 语义推理质量)
        print("  Hidden 推理语义:")
        _stat("topk_hit_ratio", diag_agg['topk_hit_ratio'], warn_lo=0.3,
              hint="hidden argmax 压不中 stage key tokens")

        # 6. Stage 对齐
        print("  Stage 对齐:")
        _stat("h_stage_diag_score", diag_agg['h_stage_diag_score'], warn_lo=0.4,
              hint="step↔stage 对齐偏弱")
        _stat("h_stage_monotonic", diag_agg['h_stage_monotonic'], warn_lo=0.6,
              hint="stage 路径单调性偏弱")
        _stat("h_stage_shift_kl", diag_agg['h_stage_shift_kl'], warn_lo=0.05,
              hint="stage 间分布变化过小")

        # 7. 总体判决
        print()
        verdicts = []
        if diag_agg['num_thought_steps']:
            ms = sum(diag_agg['num_thought_steps'])/len(diag_agg['num_thought_steps'])
            if ms < 2.3:
                verdicts.append(f"❌ 推理步数坍塌 (avg={ms:.2f})")
            elif ms < 3.0:
                verdicts.append(f"⚠️ 推理步数偏低 (avg={ms:.2f})")
            else:
                verdicts.append(f"✅ 推理步数正常 (avg={ms:.2f})")
        if diag_agg['h_first_last_cos']:
            fc = sum(diag_agg['h_first_last_cos'])/len(diag_agg['h_first_last_cos'])
            if fc > 0.97:
                verdicts.append(f"❌ hidden 几乎不演化 (first_last_cos={fc:.3f})")
            elif fc > 0.93:
                verdicts.append(f"⚠️ hidden 演化偏弱 (first_last_cos={fc:.3f})")
            else:
                verdicts.append(f"✅ hidden 跨步有变化 (first_last_cos={fc:.3f})")
        if 'exit_token' in exit_reason_counter and exit_reason_counter:
            t = sum(exit_reason_counter.values())
            r = exit_reason_counter['exit_token'] / t
            if r > 0.7:
                verdicts.append(f"❌ exit_token 退出主导 ({r*100:.0f}%)")
            elif r > 0.4:
                verdicts.append(f"⚠️ exit_token 退出偏多 ({r*100:.0f}%)")
        if diag_agg['topk_hit_ratio']:
            th = sum(diag_agg['topk_hit_ratio'])/len(diag_agg['topk_hit_ratio'])
            if th < 0.2:
                verdicts.append(f"❌ hidden 不走 stage keys (topk_hit={th:.2f})")
            elif th < 0.4:
                verdicts.append(f"⚠️ hidden topk 命中偏低 ({th:.2f})")
            else:
                verdicts.append(f"✅ hidden 命中 stage keys ({th:.2f})")
        print("  🔍 诊断结论:")
        for v in verdicts:
            print(f"    {v}")

    print("=" * 70)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, "erqa_eval_results.json")
    summary = {
        "config": {
            "model_path": args.model_path,
            "checkpoint": args.checkpoint,
            "skip_latent": args.skip_latent,
            "no_system_prompt": args.no_system_prompt,
            "max_new_tokens": args.max_new_tokens,
            "data_file": args.data_file,
        },
        "summary": {
            "total_samples": samples_total_for_print,
            "valid_inferences": total,
            "correct": correct,
            "accuracy": correct / total * 100 if total > 0 else 0,
            "errors": len(errors),
            "elapsed_seconds": elapsed,
            "world_size": world_size,
            "latent_triggered": latent_stats["triggered"],
            "latent_not_triggered": latent_stats["not_triggered"],
        },
        "type_accuracy": {
            q_type: {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
            }
            for q_type, s in sorted(type_stats.items())
        },
        "latent_health": {
            "exit_reason_distribution": dict(exit_reason_counter),
            "step_distribution": dict(Counter(diag_agg['num_thought_steps'])),
            "aggregates": {
                k: ({
                    "mean": sum(v)/len(v),
                    "min": min(v),
                    "max": max(v),
                    "n": len(v),
                } if v else None)
                for k, v in diag_agg.items()
            },
        },
        "results": results,
        "errors": errors,
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 结果已保存: {result_file}")


def main():
    pa = argparse.ArgumentParser(description="ERQA 全数据集纯推理测试")
    pa.add_argument('--model_path', type=str, required=True, help='基座模型路径')
    pa.add_argument('--checkpoint', type=str, default=None, help='NLD checkpoint 路径')
    pa.add_argument('--data_file', type=str, default='./data/erqa/erqa_test.jsonl', help='ERQA 测试数据文件')
    pa.add_argument('--data_root', type=str, default='./data/erqa', help='数据根目录（用于拼接相对图像路径）')
    pa.add_argument('--output_dir', type=str, default='./outputs/erqa_eval', help='结果输出目录')
    pa.add_argument('--max_new_tokens', type=int, default=2048, help='最大生成 token 数（支持长链推理输出）')
    pa.add_argument('--max_samples', type=int, default=0, help='最大测试样本数（0=全部）')
    pa.add_argument('--skip_latent', action='store_true', help='跳过 latent thinking（用于对比基线）')
    pa.add_argument('--no_system_prompt', action='store_true', help='不使用 system prompt（用于对比基线）')
    pa.add_argument('--device', type=str, default='cuda')
    pa.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    args = pa.parse_args()

    try:
        run_eval(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
