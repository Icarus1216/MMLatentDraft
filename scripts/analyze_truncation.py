import json
import collections
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "data/rejection_sampling/rld_debug_rejection_raw_outputs.json"

with open(filepath) as f:
    data = json.load(f)

total = 0
truncated = 0
text_lens = []
has_fa_count = 0

for entry in data:
    for out in entry["outputs"]:
        total += 1
        raw = out["raw_text"]
        text_lens.append(len(raw))
        if len(raw) >= 2990:
            truncated += 1
        if out.get("has_final_answer"):
            has_fa_count += 1

print("=== Raw Output 统计 ===")
print(f"总输出数: {total}")
print(f"平均长度 (chars): {sum(text_lens)/len(text_lens):.0f}")
print(f"最大长度 (chars): {max(text_lens)}")
print(f"最小长度 (chars): {min(text_lens)}")
print(f"中位数长度 (chars): {sorted(text_lens)[len(text_lens)//2]}")
print(f"截断数 (>=2990 chars): {truncated}/{total} ({truncated/total*100:.1f}%)")
print(f"有 Final Answer: {has_fa_count}/{total} ({has_fa_count/total*100:.1f}%)")
print()

labels = ["<500", "500-1000", "1000-1500", "1500-2000", "2000-2500", "2500-3000", ">=3000"]
buckets = collections.OrderedDict((l, 0) for l in labels)
for length in text_lens:
    if length < 500: buckets["<500"] += 1
    elif length < 1000: buckets["500-1000"] += 1
    elif length < 1500: buckets["1000-1500"] += 1
    elif length < 2000: buckets["1500-2000"] += 1
    elif length < 2500: buckets["2000-2500"] += 1
    elif length < 3000: buckets["2500-3000"] += 1
    else: buckets[">=3000"] += 1

print("长度分布:")
for k, v in buckets.items():
    bar = "#" * (v // 2)
    print(f"  {k:>12}: {v:>4} ({v/total*100:5.1f}%) {bar}")
