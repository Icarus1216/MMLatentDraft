import json
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "data/rejection_sampling/rld_debug_rejection.json"

with open(filepath) as f:
    data = json.load(f)

cot_lens = []
for item in data:
    tc = item.get("think_chain", "")
    if tc:
        cot_lens.append(len(tc))

if not cot_lens:
    print("No think_chain found!")
    sys.exit(0)

s = sorted(cot_lens)
print(f"=== 训练数据 think_chain 长度 (chars) ===")
print(f"N={len(cot_lens)}")
print(f"Avg={sum(cot_lens)//len(cot_lens)}")
print(f"Min={min(cot_lens)} Max={max(cot_lens)}")
print(f"P50={s[len(s)//2]} P90={s[int(len(s)*0.9)]} P95={s[int(len(s)*0.95)]}")
print(f">=500: {sum(1 for l in cot_lens if l>=500)}/{len(cot_lens)}")
print(f">=1k: {sum(1 for l in cot_lens if l>=1000)}/{len(cot_lens)}")
print(f">=2k: {sum(1 for l in cot_lens if l>=2000)}/{len(cot_lens)}")
print(f">=3k: {sum(1 for l in cot_lens if l>=3000)}/{len(cot_lens)}")
print(f">=5k: {sum(1 for l in cot_lens if l>=5000)}/{len(cot_lens)}")

# 分桶
import collections
labels = ["<200","200-500","500-1k","1k-2k","2k-3k","3k-5k",">=5k"]
bounds = [0,200,500,1000,2000,3000,5000,999999]
buckets = collections.OrderedDict((l,0) for l in labels)
for length in cot_lens:
    for i in range(len(bounds)-1):
        if bounds[i] <= length < bounds[i+1]:
            buckets[labels[i]] += 1
            break
print("\n长度分布:")
for k,v in buckets.items():
    bar = "#" * max(1, v//1) if v > 0 else ""
    print(f"  {k:>8}: {v:>4} ({v/len(cot_lens)*100:5.1f}%) {bar}")