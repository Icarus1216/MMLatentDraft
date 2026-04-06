#!/usr/bin/env python3
"""测试 </step> delimiter 在 tokenizer 中的编码和匹配"""
import sys
sys.path.insert(0, '.')

from transformers import AutoTokenizer

MODEL_PATH = "/mnt/wfs/mmchongqingssdwfssz/project_luban_infra/luban_infra/model_factory/for_lucas_Qwen3-VL-8B-Instruct_export"

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 1. 单独编码 </step>
delim = '</step>'
ids = tokenizer.encode(delim, add_special_tokens=False)
print(f'\n=== 1. 单独编码 "{delim}" ===')
print(f'  token ids: {ids}')
print(f'  len: {len(ids)}')
for tid in ids:
    print(f'  {tid} -> repr={repr(tokenizer.decode([tid]))}')

# 2. 上下文中编码（模拟真实数据）
think_text = '<think>\nStep 1 reasoning.\n</step>\nStep 2 reasoning.\n</step>\n</think>\n\n'
think_ids = tokenizer.encode(think_text, add_special_tokens=False)
print(f'\n=== 2. 上下文中编码 ===')
print(f'  text: {repr(think_text)}')
print(f'  ids ({len(think_ids)} tokens): {think_ids}')
print(f'  decoded: {repr(tokenizer.decode(think_ids))}')

# 3. 在上下文中匹配 delimiter
delim_len = len(ids)
print(f'\n=== 3. 匹配 delimiter (len={delim_len}) ===')
found = []
for i in range(delim_len - 1, len(think_ids)):
    if think_ids[i - delim_len + 1 : i + 1] == ids:
        found.append(i)
        print(f'  Found at pos {i} (tokens[{i-delim_len+1}:{i+1}]={think_ids[i-delim_len+1:i+1]})')
print(f'  Total found: {len(found)} (expected: 2)')

# 4. 如果匹配失败，分析原因
if len(found) < 2:
    print(f'\n=== 4. 编码差异分析 ===')
    # 在上下文中逐 token 解码
    for i, tid in enumerate(think_ids):
        decoded = repr(tokenizer.decode([tid]))
        print(f'  [{i:3d}] {tid:6d} -> {decoded}')
    
    # 尝试在上下文中匹配
    print(f'\n  --- 尝试匹配上下文中的 </step> ---')
    # 手动查找 "</step>" 对应的子序列
    target_text = '\n</step>\n'
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    print(f'  "\\n</step>\\n" 单独编码: {target_ids}')
    for tid in target_ids:
        print(f'    {tid} -> {repr(tokenizer.decode([tid]))}')

print('\n完成!')
