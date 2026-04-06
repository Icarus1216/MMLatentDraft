"""
RLD 拒绝采样数据质量验证脚本

验证维度:
1. 基本字段完整性 (image, question, answer, think_chain, think_chain_source, think_chain_status)
2. think_chain 格式合规性 (<think>...</think>, </step> 边界)
3. answer 质量 (非空, 合理长度)
4. rejection_sampling_meta 统计一致性
5. correct_cot 覆盖率 (用于后续混合训练)
6. 图片路径存在性抽检
7. step 数分布
8. pass_rate 分布
9. 异常样本检测 (answer 含噪声, think_chain 过短/过长)
10. 与 data.py 训练加载兼容性检查

用法:
    python scripts/validate_rejection_data.py [json_path]
"""

import json
import os
import sys
import re
import collections
import random

# ============================================================
# 配置
# ============================================================
DEFAULT_PATH = "data/rejection_sampling/rld_50k_rejection_sampled.json"
IMAGE_SPOT_CHECK_N = 200  # 图片存在性抽检数量
MAX_STEPS_ALLOWED = 14    # data.py 中的硬限制
SUPERVISED_THINK_SOURCES = {"free_reasoning", "corrected_free_reasoning", "dataset_converted"}

# ============================================================
# 工具函数
# ============================================================
def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_bar(label, count, total, width=40):
    pct = count / max(total, 1) * 100
    bar_len = int(pct / 100 * width)
    bar = "█" * bar_len + "░" * (width - bar_len)
    print(f"  {label:>20}: {count:>6} ({pct:5.1f}%) |{bar}|")

def print_histogram(values, bins, labels, title=""):
    if title:
        print(f"\n  {title}")
    total = len(values)
    for i in range(len(labels)):
        lo, hi = bins[i], bins[i+1]
        count = sum(1 for v in values if lo <= v < hi)
        pct = count / max(total, 1) * 100
        bar = "█" * max(1, int(pct / 2)) if count > 0 else ""
        print(f"    {labels[i]:>12}: {count:>6} ({pct:5.1f}%) {bar}")

# ============================================================
# 主验证逻辑
# ============================================================
def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    
    print(f"📂 加载数据: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"✅ 成功加载 {len(data)} 条样本")

    # ============================================================
    # 1. 基本字段完整性检查
    # ============================================================
    print_section("1. 基本字段完整性检查")
    
    required_fields = ["image", "question", "answer", "think_chain", "think_chain_status", "think_chain_source"]
    optional_fields = ["rejection_sampling_meta", "correct_cot"]
    
    field_missing = {f: 0 for f in required_fields + optional_fields}
    field_empty = {f: 0 for f in required_fields}
    
    for item in data:
        for f in required_fields + optional_fields:
            if f not in item:
                field_missing[f] += 1
        for f in required_fields:
            if f in item and not item[f]:
                field_empty[f] += 1
    
    all_ok = True
    for f in required_fields:
        missing = field_missing[f]
        empty = field_empty[f]
        status = "✅" if missing == 0 and empty == 0 else "❌"
        if missing > 0 or empty > 0:
            all_ok = False
        print(f"  {status} {f:>25}: 缺失={missing}, 空值={empty}")
    
    for f in optional_fields:
        missing = field_missing[f]
        status = "✅" if missing == 0 else "⚠️"
        print(f"  {status} {f:>25}: 缺失={missing} (可选)")
    
    if all_ok:
        print(f"\n  ✅ 所有必需字段完整!")
    else:
        print(f"\n  ❌ 存在字段缺失或空值问题!")

    # ============================================================
    # 2. think_chain 格式合规性
    # ============================================================
    print_section("2. think_chain 格式合规性")
    
    fmt_issues = {
        "missing_think_start": 0,
        "missing_think_end": 0,
        "missing_step_delim": 0,
        "status_not_valid": 0,
        "source_not_wrong_cot": 0,
        "nested_think_tags": 0,
    }
    
    source_counts = collections.Counter()
    status_counts = collections.Counter()
    
    for item in data:
        tc = item.get("think_chain", "")
        status = item.get("think_chain_status", "")
        source = item.get("think_chain_source", "")
        
        source_counts[source] += 1
        status_counts[status] += 1
        
        if not tc.strip().startswith("<think>"):
            fmt_issues["missing_think_start"] += 1
        if "</think>" not in tc:
            fmt_issues["missing_think_end"] += 1
        if "</step>" not in tc:
            fmt_issues["missing_step_delim"] += 1
        if status != "valid":
            fmt_issues["status_not_valid"] += 1
        if source != "wrong_cot":
            fmt_issues["source_not_wrong_cot"] += 1
        if tc.count("<think>") > 1:
            fmt_issues["nested_think_tags"] += 1
    
    for issue, count in fmt_issues.items():
        status = "✅" if count == 0 else "❌"
        print(f"  {status} {issue:>30}: {count}")
    
    print(f"\n  think_chain_source 分布:")
    for src, cnt in source_counts.most_common():
        supervised = "think+answer loss" if src in SUPERVISED_THINK_SOURCES else "answer-only loss"
        print(f"    {src:>30}: {cnt:>6} → {supervised}")
    
    print(f"\n  think_chain_status 分布:")
    for st, cnt in status_counts.most_common():
        print(f"    {st:>30}: {cnt:>6}")

    # ============================================================
    # 3. Answer 质量检查
    # ============================================================
    print_section("3. Answer 质量检查")
    
    answer_lens = []
    answer_issues = {
        "empty": 0,
        "too_short_lt3": 0,
        "too_long_gt500": 0,
        "contains_note": 0,       # 含 "Note:" 等噪声
        "contains_incorrect": 0,  # 含 "incorrect" 等自我否定
        "contains_none": 0,       # 答案是 "None" 类
    }
    noisy_answer_examples = []
    
    for i, item in enumerate(data):
        ans = item.get("answer", "")
        answer_lens.append(len(ans))
        
        if not ans.strip():
            answer_issues["empty"] += 1
        elif len(ans.strip()) < 3:
            answer_issues["too_short_lt3"] += 1
        
        if len(ans) > 500:
            answer_issues["too_long_gt500"] += 1
        
        ans_lower = ans.lower()
        if "note:" in ans_lower or "note :" in ans_lower:
            answer_issues["contains_note"] += 1
            if len(noisy_answer_examples) < 5:
                noisy_answer_examples.append((i, ans[:200]))
        if "incorrect" in ans_lower:
            answer_issues["contains_incorrect"] += 1
            if len(noisy_answer_examples) < 5:
                noisy_answer_examples.append((i, ans[:200]))
        if ans.strip().lower() in ("none", "n/a", "null", ""):
            answer_issues["contains_none"] += 1
    
    for issue, count in answer_issues.items():
        status = "✅" if count == 0 else ("⚠️" if count < len(data) * 0.05 else "❌")
        print(f"  {status} {issue:>25}: {count}")
    
    if answer_lens:
        s = sorted(answer_lens)
        print(f"\n  Answer 长度统计 (chars):")
        print(f"    N={len(s)}, Avg={sum(s)//len(s)}, Min={min(s)}, Max={max(s)}")
        print(f"    P50={s[len(s)//2]}, P90={s[int(len(s)*0.9)]}, P95={s[int(len(s)*0.95)]}")
    
    if noisy_answer_examples:
        print(f"\n  ⚠️ 噪声 Answer 示例 (前5个):")
        for idx, ans in noisy_answer_examples:
            print(f"    [{idx}] {ans}")

    # ============================================================
    # 4. rejection_sampling_meta 统计
    # ============================================================
    print_section("4. rejection_sampling_meta 统计")
    
    pass_rates = []
    num_correct_list = []
    num_wrong_list = []
    num_invalid_list = []
    num_samples_list = []
    meta_issues = {
        "missing_meta": 0,
        "inconsistent_counts": 0,
        "zero_samples": 0,
    }
    
    for item in data:
        meta = item.get("rejection_sampling_meta")
        if not meta:
            meta_issues["missing_meta"] += 1
            continue
        
        ns = meta.get("num_samples", 0)
        nv = meta.get("num_valid", 0)
        nc = meta.get("num_correct", 0)
        nw = meta.get("num_wrong", 0)
        ni = meta.get("num_invalid", 0)
        pr = meta.get("pass_rate", 0)
        
        num_samples_list.append(ns)
        num_correct_list.append(nc)
        num_wrong_list.append(nw)
        num_invalid_list.append(ni)
        pass_rates.append(pr)
        
        if ns == 0:
            meta_issues["zero_samples"] += 1
        
        # 一致性检查: num_valid + num_invalid == num_samples
        if nv + ni != ns:
            meta_issues["inconsistent_counts"] += 1
        # 一致性检查: num_correct + num_wrong == num_valid
        if nc + nw != nv:
            meta_issues["inconsistent_counts"] += 1
    
    for issue, count in meta_issues.items():
        status = "✅" if count == 0 else "❌"
        print(f"  {status} {issue:>25}: {count}")
    
    if pass_rates:
        print(f"\n  Pass Rate 统计:")
        print(f"    N={len(pass_rates)}, Avg={sum(pass_rates)/len(pass_rates):.3f}")
        print(f"    Min={min(pass_rates):.3f}, Max={max(pass_rates):.3f}")
        
        # Pass Rate 分布
        pr_bins = [0, 0.001, 0.25, 0.5, 0.75, 1.0, 1.001]
        pr_labels = ["=0 (全错)", "0<pr<0.25", "0.25≤pr<0.5", "0.5≤pr<0.75", "0.75≤pr<1.0", "=1.0 (全对)"]
        print(f"\n  Pass Rate 分布:")
        for i in range(len(pr_labels)):
            lo, hi = pr_bins[i], pr_bins[i+1]
            if i == 0:
                count = sum(1 for p in pass_rates if p == 0)
            elif i == len(pr_labels) - 1:
                count = sum(1 for p in pass_rates if p >= 1.0)
            else:
                count = sum(1 for p in pass_rates if lo < p < hi or (lo <= p < hi and p > 0))
                # 更精确的区间
                count = sum(1 for p in pass_rates if lo <= p < hi and not (i == 0 and p == 0))
            pct = count / len(pass_rates) * 100
            bar = "█" * max(1, int(pct / 2)) if count > 0 else ""
            print(f"    {pr_labels[i]:>16}: {count:>6} ({pct:5.1f}%) {bar}")
        
        # 简化的 pass_rate 分桶
        pr_0 = sum(1 for p in pass_rates if p == 0)
        pr_low = sum(1 for p in pass_rates if 0 < p <= 0.25)
        pr_mid = sum(1 for p in pass_rates if 0.25 < p <= 0.5)
        pr_high = sum(1 for p in pass_rates if 0.5 < p < 1.0)
        pr_1 = sum(1 for p in pass_rates if p >= 1.0)
        
        print(f"\n  简化分桶:")
        print_bar("全错 (pr=0)", pr_0, len(pass_rates))
        print_bar("低 (0<pr≤0.25)", pr_low, len(pass_rates))
        print_bar("中 (0.25<pr≤0.5)", pr_mid, len(pass_rates))
        print_bar("高 (0.5<pr<1.0)", pr_high, len(pass_rates))
        print_bar("全对 (pr=1.0)", pr_1, len(pass_rates))
    
    if num_samples_list:
        print(f"\n  采样次数分布:")
        for ns_val, cnt in collections.Counter(num_samples_list).most_common():
            print(f"    num_samples={ns_val}: {cnt}")

    # ============================================================
    # 5. correct_cot 覆盖率 (用于混合训练)
    # ============================================================
    print_section("5. correct_cot 覆盖率 (用于混合训练)")
    
    has_correct_cot = sum(1 for item in data if item.get("correct_cot"))
    no_correct_cot = len(data) - has_correct_cot
    
    print_bar("有 correct_cot", has_correct_cot, len(data))
    print_bar("无 correct_cot", no_correct_cot, len(data))
    
    # 按 pass_rate 分析 correct_cot 覆盖
    has_cc_by_pr = collections.Counter()
    no_cc_by_pr = collections.Counter()
    for item in data:
        meta = item.get("rejection_sampling_meta", {})
        pr = meta.get("pass_rate", 0)
        bucket = "pr=0" if pr == 0 else ("0<pr≤0.5" if pr <= 0.5 else "pr>0.5")
        if item.get("correct_cot"):
            has_cc_by_pr[bucket] += 1
        else:
            no_cc_by_pr[bucket] += 1
    
    print(f"\n  按 pass_rate 分析 correct_cot 覆盖:")
    for bucket in ["pr=0", "0<pr≤0.5", "pr>0.5"]:
        has = has_cc_by_pr.get(bucket, 0)
        no = no_cc_by_pr.get(bucket, 0)
        total = has + no
        pct = has / max(total, 1) * 100
        print(f"    {bucket:>12}: {has}/{total} ({pct:.1f}%) 有 correct_cot")
    
    # correct_cot 格式检查
    cc_fmt_issues = 0
    for item in data:
        cc = item.get("correct_cot", "")
        if cc and ("<think>" not in cc or "</think>" not in cc or "</step>" not in cc):
            cc_fmt_issues += 1
    if cc_fmt_issues > 0:
        print(f"\n  ⚠️ {cc_fmt_issues} 条 correct_cot 格式不合规 (缺少 <think>/<\\think>/</step>)")
    else:
        print(f"\n  ✅ 所有 correct_cot 格式合规")

    # ============================================================
    # 6. 图片路径存在性抽检
    # ============================================================
    print_section("6. 图片路径存在性抽检")
    
    all_images = [item.get("image", "") for item in data if item.get("image")]
    unique_images = set(all_images)
    print(f"  总图片引用: {len(all_images)}")
    print(f"  唯一图片数: {len(unique_images)}")
    
    # 抽检
    check_n = min(IMAGE_SPOT_CHECK_N, len(all_images))
    sample_images = random.sample(all_images, check_n)
    exists_count = sum(1 for img in sample_images if os.path.exists(img))
    missing_count = check_n - exists_count
    
    status = "✅" if missing_count == 0 else "❌"
    print(f"  {status} 抽检 {check_n} 张图片: {exists_count} 存在, {missing_count} 缺失")
    
    if missing_count > 0:
        missing_examples = [img for img in sample_images if not os.path.exists(img)][:5]
        print(f"  缺失示例:")
        for img in missing_examples:
            print(f"    {img}")
    
    # 图片路径前缀分布
    prefix_counts = collections.Counter()
    for img in all_images:
        parts = img.split("/")
        if len(parts) >= 6:
            prefix = "/".join(parts[:6])
        else:
            prefix = img
        prefix_counts[prefix] += 1
    
    print(f"\n  图片路径前缀分布 (top 10):")
    for prefix, cnt in prefix_counts.most_common(10):
        print(f"    {prefix}: {cnt}")

    # ============================================================
    # 7. Step 数分布
    # ============================================================
    print_section("7. Step 数分布")
    
    step_counts = []
    over_limit = 0
    for item in data:
        tc = item.get("think_chain", "")
        n_steps = tc.count("</step>")
        step_counts.append(n_steps)
        if n_steps > MAX_STEPS_ALLOWED:
            over_limit += 1
    
    if step_counts:
        s = sorted(step_counts)
        print(f"  N={len(s)}, Avg={sum(s)/len(s):.1f}, Min={min(s)}, Max={max(s)}")
        print(f"  P50={s[len(s)//2]}, P90={s[int(len(s)*0.9)]}, P95={s[int(len(s)*0.95)]}")
        
        status = "✅" if over_limit == 0 else "⚠️"
        print(f"  {status} 超过 {MAX_STEPS_ALLOWED} 步的样本: {over_limit} (会被 data.py 过滤)")
        
        # 分桶
        step_bins = [0, 1, 3, 5, 7, 10, 14, 100]
        step_labels = ["0步", "1-2步", "3-4步", "5-6步", "7-9步", "10-14步", ">14步"]
        print_histogram(step_counts, step_bins, step_labels, "Step 数分布:")
    
    # 对比 meta 中的 wrong_num_steps
    meta_step_match = 0
    meta_step_mismatch = 0
    for item in data:
        meta = item.get("rejection_sampling_meta", {})
        meta_steps = meta.get("wrong_num_steps", None)
        if meta_steps is not None:
            tc = item.get("think_chain", "")
            actual_steps = tc.count("</step>")
            if actual_steps == meta_steps:
                meta_step_match += 1
            else:
                meta_step_mismatch += 1
    
    if meta_step_match + meta_step_mismatch > 0:
        status = "✅" if meta_step_mismatch == 0 else "⚠️"
        print(f"\n  {status} meta.wrong_num_steps 与实际 step 数一致性: "
              f"匹配={meta_step_match}, 不匹配={meta_step_mismatch}")

    # ============================================================
    # 8. think_chain 长度分布 (chars)
    # ============================================================
    print_section("8. think_chain 长度分布")
    
    cot_lens = [len(item.get("think_chain", "")) for item in data if item.get("think_chain")]
    
    if cot_lens:
        s = sorted(cot_lens)
        print(f"  N={len(s)}, Avg={sum(s)//len(s)}, Min={min(s)}, Max={max(s)}")
        print(f"  P50={s[len(s)//2]}, P90={s[int(len(s)*0.9)]}, P95={s[int(len(s)*0.95)]}")
        
        len_bins = [0, 100, 200, 500, 1000, 2000, 3000, 5000, 100000]
        len_labels = ["<100", "100-200", "200-500", "500-1k", "1k-2k", "2k-3k", "3k-5k", ">=5k"]
        print_histogram(cot_lens, len_bins, len_labels, "长度分布:")

    # ============================================================
    # 9. 异常样本检测
    # ============================================================
    print_section("9. 异常样本检测")
    
    anomalies = {
        "think_chain_too_short_lt50": [],
        "think_chain_too_long_gt5000": [],
        "answer_contains_explanation": [],
        "wrong_gen_answer_matches_gt": [],
        "zero_steps": [],
    }
    
    for i, item in enumerate(data):
        tc = item.get("think_chain", "")
        ans = item.get("answer", "")
        meta = item.get("rejection_sampling_meta", {})
        
        # think_chain 过短
        if len(tc) < 50:
            anomalies["think_chain_too_short_lt50"].append(i)
        
        # think_chain 过长
        if len(tc) > 5000:
            anomalies["think_chain_too_long_gt5000"].append(i)
        
        # answer 含解释性文本 (应该只是简短答案)
        if len(ans) > 200:
            anomalies["answer_contains_explanation"].append(i)
        
        # wrong_gen_answer 与 GT answer 相同 (数据标注可能有误)
        wrong_ans = meta.get("wrong_gen_answer", "")
        if wrong_ans and ans.strip() and wrong_ans.strip().lower() == ans.strip().lower():
            anomalies["wrong_gen_answer_matches_gt"].append(i)
        
        # 0 步
        if tc.count("</step>") == 0:
            anomalies["zero_steps"].append(i)
    
    for anomaly, indices in anomalies.items():
        count = len(indices)
        status = "✅" if count == 0 else ("⚠️" if count < len(data) * 0.05 else "❌")
        print(f"  {status} {anomaly:>40}: {count}")
        if count > 0 and count <= 5:
            for idx in indices[:3]:
                item = data[idx]
                print(f"      [{idx}] answer={item.get('answer','')[:80]}...")

    # ============================================================
    # 10. 与 data.py 训练加载兼容性检查
    # ============================================================
    print_section("10. 与 data.py 训练加载兼容性检查")
    
    compat_issues = []
    
    # 检查1: think_chain_source 是否被 data.py 正确识别
    for src in source_counts:
        if src in SUPERVISED_THINK_SOURCES:
            compat_issues.append(f"⚠️ source '{src}' 会让 think 块参与 loss (是否预期?)")
        elif src == "wrong_cot":
            pass  # 预期: answer-only loss
        else:
            compat_issues.append(f"⚠️ 未知 source '{src}', 将被视为 answer-only loss")
    
    # 检查2: think_chain 格式是否能被 data.py 正确解析
    parse_ok = 0
    parse_fail = 0
    for item in data[:1000]:  # 抽检前1000条
        tc = item.get("think_chain", "")
        # 模拟 data.py 的解析逻辑
        think_inner = tc
        think_end_pos = think_inner.rfind("</think>")
        if think_end_pos != -1:
            think_inner = think_inner[:think_end_pos]
        if think_inner.startswith("<think>"):
            think_inner = think_inner[len("<think>"):]
        think_inner = think_inner.strip('\n')
        
        if think_inner and "</step>" in think_inner:
            parse_ok += 1
        else:
            parse_fail += 1
    
    status = "✅" if parse_fail == 0 else "❌"
    print(f"  {status} data.py 解析兼容性 (抽检1000条): 成功={parse_ok}, 失败={parse_fail}")
    
    # 检查3: 是否所有样本都有 image (data.py 会过滤无图样本)
    no_image = sum(1 for item in data if not item.get("image"))
    status = "✅" if no_image == 0 else "⚠️"
    print(f"  {status} 无图片样本: {no_image} (会被 data.py 过滤)")
    
    # 检查4: 预估训练后的 loss 来源分布
    answer_only_loss = sum(1 for item in data if item.get("think_chain_source", "fabricated") not in SUPERVISED_THINK_SOURCES)
    think_answer_loss = len(data) - answer_only_loss
    print(f"\n  预估 loss 来源分布:")
    print_bar("answer-only loss", answer_only_loss, len(data))
    print_bar("think+answer loss", think_answer_loss, len(data))
    
    if compat_issues:
        print(f"\n  兼容性提示:")
        for issue in compat_issues:
            print(f"    {issue}")

    # ============================================================
    # 总结
    # ============================================================
    print_section("📊 总结")
    
    total_issues = 0
    
    # 关键指标
    checks = [
        ("字段完整性", all_ok),
        ("think_chain 格式", fmt_issues["missing_think_start"] == 0 and fmt_issues["missing_think_end"] == 0),
        ("所有样本为 wrong_cot", fmt_issues["source_not_wrong_cot"] == 0),
        ("所有样本 status=valid", fmt_issues["status_not_valid"] == 0),
        ("meta 一致性", meta_issues["inconsistent_counts"] == 0),
        ("图片存在性", missing_count == 0),
        ("step 数 ≤ 14", over_limit == 0),
        ("data.py 解析兼容", parse_fail == 0),
    ]
    
    for name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            total_issues += 1
        print(f"  {status:>10}  {name}")
    
    print(f"\n  数据规模: {len(data)} 条")
    print(f"  correct_cot 覆盖率: {has_correct_cot}/{len(data)} ({has_correct_cot/max(len(data),1)*100:.1f}%)")
    print(f"  平均 pass_rate: {sum(pass_rates)/max(len(pass_rates),1):.3f}")
    print(f"  平均 step 数: {sum(step_counts)/max(len(step_counts),1):.1f}")
    
    if total_issues == 0:
        print(f"\n  🎉 数据质量验证通过! 可以用于 Phase 1 SFT 训练。")
    else:
        print(f"\n  ⚠️ 发现 {total_issues} 个问题，请检查后再用于训练。")
    
    # 训练建议
    print_section("💡 Phase 1 SFT 训练建议")
    
    print(f"  1. 当前数据全部为 wrong_cot ({len(data)} 条)")
    print(f"     → think 块 labels=-100, 只监督 final answer")
    print(f"  2. correct_cot 覆盖率: {has_correct_cot/max(len(data),1)*100:.1f}%")
    if has_correct_cot > 0:
        print(f"     → 建议展开 correct_cot 样本，按 7:3 混合训练")
        print(f"     → 可额外获得 ~{has_correct_cot} 条正确 CoT 样本")
    else:
        print(f"     → ⚠️ 无 correct_cot，建议先只用 wrong_cot 训练")
    
    pr_0_pct = pr_0 / max(len(pass_rates), 1) * 100
    if pr_0_pct > 50:
        print(f"  3. ⚠️ {pr_0_pct:.1f}% 样本 pass_rate=0 (基座完全做不对)")
        print(f"     → 这些样本的 hidden states 偏差可能过大，Draft 难以修正")
        print(f"     → 建议考虑过滤部分 pass_rate=0 的样本，或降低其采样权重")
    else:
        print(f"  3. pass_rate=0 占比 {pr_0_pct:.1f}%，分布合理")


if __name__ == "__main__":
    main()
