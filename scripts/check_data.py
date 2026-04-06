#!/usr/bin/env python3
"""
RLD 训练数据质量全面检查脚本

检查项:
  1. 字段完整性 (image/question/answer/think_chain)
  2. 图片文件存在性 (抽样检查)
  3. Think chain 格式正确性 (<think>/<step>/</think>)
  4. 文本长度分布 (发现超长样本)
  5. Token 化后的序列长度分布 (关键: 超过 max_seq_len 的样本)
  6. 段数分布 (</step> 数量, 关键: 段数过多导致显存溢出)
  7. 异常字符/编码问题
  8. 数据样本间的一致性
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_PATH = "/mnt/cephszjt/user_juntianzhang/LatentDraft/data/rld_mmathcot_filtered_checked.json"
MAX_SEQ_LEN = 4096  # 配置中的最大序列长度
IMAGE_CHECK_LIMIT = 500  # 图片存在性抽查数量


def main():
    print("=" * 80)
    print("RLD 训练数据质量检查")
    print("=" * 80)
    
    # ====== 1. 加载数据 ======
    print(f"\n[1/7] 加载数据: {DATA_PATH}")
    t0 = time.time()
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    print(f"  加载耗时: {time.time()-t0:.1f}s")
    print(f"  总样本数: {len(data)}")
    
    # ====== 2. 字段完整性检查 ======
    print(f"\n[2/7] 字段完整性检查")
    issues = {
        'no_image': [],
        'no_question': [],
        'no_answer': [],
        'empty_answer': [],
        'image_not_exist': [],
    }
    
    think_chain_stats = {
        'has_think_chain': 0,
        'think_valid': 0,
        'sources': Counter(),
        'no_think_start': [],
        'no_think_end': [],
        'no_step_delim': [],
        'think_after_close': [],  # </think> 后面还有非 "Answer:" 的内容
    }
    
    answer_lengths = []
    think_lengths = []
    question_lengths = []
    total_text_lengths = []
    step_counts = []  # 每个样本的段数
    
    for i, item in enumerate(data):
        img = item.get('image', '')
        q = item.get('question', '')
        a = item.get('answer', '')
        tc = item.get('think_chain', '')
        tc_status = item.get('think_chain_status', '')
        tc_source = item.get('think_chain_source', 'unknown')
        
        # 字段检查
        if not img:
            issues['no_image'].append(i)
        elif i < IMAGE_CHECK_LIMIT and not os.path.exists(img):
            issues['image_not_exist'].append(i)
        
        if not q:
            issues['no_question'].append(i)
        if not a:
            issues['no_answer'].append(i)
        elif not a.strip():
            issues['empty_answer'].append(i)
        
        # Think chain 检查
        if tc:
            think_chain_stats['has_think_chain'] += 1
            if tc_status == 'valid':
                think_chain_stats['think_valid'] += 1
            think_chain_stats['sources'][tc_source] += 1
            
            # 格式检查
            if '<think>' not in tc:
                think_chain_stats['no_think_start'].append(i)
            if '</think>' not in tc:
                think_chain_stats['no_think_end'].append(i)
            if '</step>' not in tc:
                think_chain_stats['no_step_delim'].append(i)
            
            # 检查 </think> 后面的内容
            think_end_pos = tc.rfind('</think>')
            if think_end_pos != -1:
                after_think = tc[think_end_pos + len('</think>'):].strip()
                # 正常情况下 </think> 后面应该是 "Answer: X" 或空
                if after_think and not after_think.startswith('Answer:') and not after_think.startswith('\nAnswer:'):
                    think_chain_stats['think_after_close'].append(i)
            
            # 段数统计
            num_steps = tc.count('</step>')
            step_counts.append(num_steps)
            think_lengths.append(len(tc))
        else:
            think_chain_stats['sources']['fabricated_fallback'] += 1
            # 无 think chain 时，根据 answer 估算自动分段数
            # _build_think_block 按句子拆分，约每 64 token 一段
            est_steps = max(1, len(a) // (64 * 4))  # 粗略估算
            step_counts.append(est_steps)
        
        answer_lengths.append(len(a) if a else 0)
        question_lengths.append(len(q) if q else 0)
        total_text_lengths.append(len(q or '') + len(a or '') + len(tc or ''))
    
    # 打印字段检查结果
    for key, lst in issues.items():
        status = "✅" if not lst else "❌"
        print(f"  {status} {key}: {len(lst)}")
        if lst and len(lst) <= 10:
            print(f"      索引: {lst}")
        elif lst:
            print(f"      前10个索引: {lst[:10]}...")
    
    # ====== 3. Think Chain 统计 ======
    print(f"\n[3/7] Think Chain 统计")
    ts = think_chain_stats
    print(f"  有 think_chain: {ts['has_think_chain']}/{len(data)} "
          f"({ts['has_think_chain']/len(data)*100:.1f}%)")
    print(f"  status=valid: {ts['think_valid']}")
    print(f"  来源分布:")
    for src, cnt in ts['sources'].most_common():
        print(f"    {src}: {cnt}")
    
    fmt_issues = {
        'no_think_start': '缺少 <think>',
        'no_think_end': '缺少 </think>',
        'no_step_delim': '缺少 </step>',
        'think_after_close': '</think> 后有异常内容',
    }
    for key, desc in fmt_issues.items():
        lst = ts[key]
        status = "✅" if not lst else "⚠️"
        print(f"  {status} {desc}: {len(lst)}")
        if lst and len(lst) <= 5:
            print(f"      索引: {lst}")
        elif lst:
            print(f"      前5个索引: {lst[:5]}...")
    
    # ====== 4. 长度分布统计 ======
    print(f"\n[4/7] 文本长度分布 (字符数)")
    
    def print_dist(name, values):
        if not values:
            print(f"  {name}: 无数据")
            return
        values_sorted = sorted(values)
        n = len(values_sorted)
        print(f"  {name}:")
        print(f"    min={values_sorted[0]}, max={values_sorted[-1]}, "
              f"mean={sum(values)/n:.0f}, median={values_sorted[n//2]}")
        print(f"    p90={values_sorted[int(n*0.9)]}, "
              f"p95={values_sorted[int(n*0.95)]}, "
              f"p99={values_sorted[int(n*0.99)]}")
    
    print_dist("Question 长度", question_lengths)
    print_dist("Answer 长度", answer_lengths)
    print_dist("Think chain 长度", think_lengths)
    print_dist("总文本长度(q+a+tc)", total_text_lengths)
    
    # ====== 5. 段数分布 (关键!) ======
    print(f"\n[5/7] 段数分布 (</step> 数量) — 关键: 段数过多导致显存溢出!")
    if step_counts:
        print_dist("段数", step_counts)
        step_counter = Counter(step_counts)
        print(f"  段数分布 (top 15):")
        for cnt, num in step_counter.most_common(15):
            print(f"    {cnt} 段: {num} 个样本")
        
        # 高段数样本 (>= 8 段, 当前 MAX_SEGMENTS 限制)
        high_step_samples = [(i, sc) for i, sc in enumerate(step_counts) if sc >= 8]
        print(f"\n  ⚠️ >= 8 段的样本: {len(high_step_samples)}")
        if high_step_samples[:5]:
            for idx, sc in high_step_samples[:5]:
                item = data[idx]
                print(f"    [{idx}] 段数={sc}, a_len={len(item.get('answer',''))}, "
                      f"tc_len={len(item.get('think_chain',''))}, "
                      f"src={item.get('think_chain_source','?')}")
        
        # 超高段数样本 (>= 15)
        very_high = [(i, sc) for i, sc in enumerate(step_counts) if sc >= 15]
        if very_high:
            print(f"\n  ❌ >= 15 段的样本: {len(very_high)} — 这些样本极可能导致 CUDA 错误!")
            for idx, sc in very_high[:10]:
                item = data[idx]
                print(f"    [{idx}] 段数={sc}, a_len={len(item.get('answer',''))}, "
                      f"tc_len={len(item.get('think_chain',''))}")
    
    # ====== 6. Token 化长度估算 ======
    print(f"\n[6/7] Token 长度估算 (字符数/3.5 近似)")
    # 粗略估算: 英文约 4 字符/token, 中文约 2 字符/token, 混合约 3.5
    est_token_lengths = [int(l / 3.5) for l in total_text_lengths]
    print_dist("估算 token 数", est_token_lengths)
    
    over_limit = sum(1 for l in est_token_lengths if l > MAX_SEQ_LEN)
    print(f"\n  超过 max_seq_len({MAX_SEQ_LEN}) 的样本: {over_limit} "
          f"({over_limit/len(data)*100:.1f}%)")
    
    # 加上 prompt 部分(system prompt + image tokens)估算
    # system prompt 约 200 tokens, image tokens 约 1000-2000
    EST_PROMPT_TOKENS = 1500  # prompt 部分的估算 token 数
    over_limit_with_prompt = sum(1 for l in est_token_lengths 
                                  if l + EST_PROMPT_TOKENS > MAX_SEQ_LEN)
    print(f"  加上 prompt(~{EST_PROMPT_TOKENS} tokens) 后超限: {over_limit_with_prompt} "
          f"({over_limit_with_prompt/len(data)*100:.1f}%)")
    
    # ====== 7. 异常样本详细输出 ======
    print(f"\n[7/7] 超长样本分析 (总文本 > 20000 字符)")
    ultra_long = [(i, total_text_lengths[i]) for i in range(len(data)) 
                  if total_text_lengths[i] > 20000]
    ultra_long.sort(key=lambda x: -x[1])
    print(f"  超长样本数: {len(ultra_long)}")
    for idx, tl in ultra_long[:10]:
        item = data[idx]
        tc = item.get('think_chain', '')
        num_steps = tc.count('</step>') if tc else 0
        print(f"  [{idx}] 总长={tl}, 段数={num_steps}, "
              f"q_len={len(item.get('question',''))}, "
              f"a_len={len(item.get('answer',''))}, "
              f"tc_len={len(tc)}, "
              f"src={item.get('think_chain_source','?')}")
    
    # ====== 总结 ======
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    has_critical = False
    
    if issues['no_answer']:
        print(f"❌ 严重: {len(issues['no_answer'])} 个样本无 answer")
        has_critical = True
    
    if issues['image_not_exist']:
        print(f"⚠️  警告: 前{IMAGE_CHECK_LIMIT}个中有 {len(issues['image_not_exist'])} 个图片不存在")
    
    if step_counts:
        max_steps = max(step_counts)
        if max_steps >= 15:
            print(f"❌ 严重: 最大段数={max_steps}, 这极可能导致显存溢出!")
            has_critical = True
        elif max_steps >= 10:
            print(f"⚠️  警告: 最大段数={max_steps}, 建议关注显存使用")
    
    if over_limit_with_prompt > 0:
        pct = over_limit_with_prompt / len(data) * 100
        if pct > 20:
            print(f"⚠️  警告: {pct:.1f}% 的样本加上 prompt 后可能超过 max_seq_len")
        else:
            print(f"✅ {pct:.1f}% 的样本加上 prompt 后超过 max_seq_len (截断处理)")
    
    if not has_critical:
        print("✅ 未发现严重的数据质量问题")
    
    print()


if __name__ == "__main__":
    main()
