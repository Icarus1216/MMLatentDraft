"""prepare_benchmark.py - 7 个 VLM benchmark 的 parquet → ERQA 格式预处理工具

已支持:
  - MMStar         (1500, choice_abcd, 单图)
  - RealWorldQA    (~765, open, 单图)
  - BLINK          (val ~1.9k, choice_abcde, 多图, 14 subtasks)
  - MUIRBench      (~2k, choice_abcde, 多图)
  - MMBench-EN     (dev ~4k, choice_abcd 含循环重排, 单图)
  - HallusionBench (~1.1k, open yes/no, 单图)
  - SimpleVQA      (open, 单图 单词/短语回答)
快速用法:
  python3 scripts/prepare_benchmark.py --dataset MMBench
  python3 scripts/prepare_benchmark.py --dataset HallusionBench
  python3 scripts/prepare_benchmark.py --dataset SimpleVQA
  python3 scripts/prepare_benchmark.py --dataset ALL
"""
import os, sys, json, argparse
from pathlib import Path
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


def _write_image(img_bytes, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        with open(out_path, 'wb') as f:
            f.write(img_bytes)


def _resolve_image_ext(img_bytes, hint_path=""):
    if img_bytes[:8].startswith(b'\x89PNG'):
        return '.png'
    if img_bytes[:3] == b'\xff\xd8\xff':
        return '.jpg'
    if img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
        return '.webp'
    if hint_path and '.' in hint_path:
        return '.' + hint_path.rsplit('.', 1)[-1].lower()
    return '.jpg'


def _iter_parquet(paths):
    for p in paths:
        pf = pq.ParquetFile(str(p))
        for batch in pf.iter_batches(batch_size=32):
            cols = batch.schema.names
            for i in range(batch.num_rows):
                yield {c: batch.column(c)[i].as_py() for c in cols}


# ============================================================
# MMStar: 1500 条, 单图, question 已含 A-D 选项, answer 为 A/B/C/D
# ============================================================
def prepare_mmstar(out_dir):
    print("\n[MMStar] 输出 →", out_dir, flush=True)
    src = DATA_ROOT / "MMStar" / "mmstar.parquet"
    assert src.exists(), f"{src} 不存在"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "mmstar_test.jsonl"

    n = 0
    with open(jp, 'w', encoding='utf-8') as fout:
        for row in _iter_parquet([src]):
            idx = row['index']
            question = (row.get('question') or "").strip()
            answer = (row.get('answer') or "").strip().upper()
            img_bytes = row['image']
            if not img_bytes or not answer:
                continue
            ext = _resolve_image_ext(img_bytes)
            img_rel = f"images/{idx:05d}{ext}"
            _write_image(img_bytes, img_dir / f"{idx:05d}{ext}")
            rec = {
                "question_id": f"mmstar_{idx:05d}",
                "question": question + "\nAnswer with just the option letter (A/B/C/D).",
                "answer": answer,
                "image_paths": [img_rel],
                "question_type": f"{row.get('category','')}/{row.get('l2_category','')}",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"  已处理 {n} 条...", flush=True)
    print(f"[MMStar] ✅ {n} 条 → {jp}", flush=True)
    return jp, n


# ============================================================
# RealWorldQA: 单图, answer 多为单词/数字, 部分为 A-D
# ============================================================
def prepare_realworldqa(out_dir):
    print("\n[RealWorldQA] 输出 →", out_dir, flush=True)
    src_dir = DATA_ROOT / "RealWorldQA" / "data"
    srcs = sorted(src_dir.glob("test-*.parquet"))
    assert srcs, f"{src_dir} 下没有 test-*.parquet"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "realworldqa_test.jsonl"

    n = 0
    with open(jp, 'w', encoding='utf-8') as fout:
        for row in _iter_parquet(srcs):
            question = (row.get('question') or "").strip()
            ans_raw = (row.get('answer') or "").strip()
            imf = row.get('image') or {}
            imb = imf.get('bytes') if isinstance(imf, dict) else None
            if not imb or not ans_raw:
                continue
            ext = _resolve_image_ext(imb, imf.get('path') or "")
            img_rel = f"images/{n:05d}{ext}"
            _write_image(imb, img_dir / f"{n:05d}{ext}")

            if len(ans_raw) == 1 and ans_raw.upper() in "ABCDE":
                answer = ans_raw.upper()
            else:
                answer = ans_raw

            rec = {
                "question_id": f"rwqa_{n:05d}",
                "question": question,
                "answer": answer,
                "image_paths": [img_rel],
                "question_type": "RealWorldQA",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[RealWorldQA] ✅ {n} 条 → {jp}", flush=True)
    return jp, n


# ============================================================
# BLINK: 14 子任务, 多图 (image_1..4), answer 为 "(A)"..."(E)"
# ============================================================
BLINK_SUBTASKS = [
    "Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence",
    "IQ_Test", "Jigsaw", "Multi-view_Reasoning", "Object_Localization",
    "Relative_Depth", "Relative_Reflectance", "Semantic_Correspondence",
    "Spatial_Relation", "Visual_Correspondence", "Visual_Similarity",
]


def prepare_blink(out_dir, subtasks=None, split="val"):
    subtasks = subtasks or BLINK_SUBTASKS
    print(f"\n[BLINK] 输出 → {out_dir}  subtasks={subtasks}  split={split}", flush=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"blink_{split}.jsonl"

    total, per_sub = 0, {}
    with open(jp, 'w', encoding='utf-8') as fout:
        for sub in subtasks:
            src = DATA_ROOT / "BLINK" / sub / f"{split}-00000-of-00001.parquet"
            if not src.exists():
                print(f"  [skip] {src} 不存在", flush=True)
                continue
            sub_n = 0
            for row in _iter_parquet([src]):
                idx = row.get('idx') or f"{sub}_{sub_n}"
                prompt = (row.get('prompt') or row.get('question') or "").strip()
                ans_raw = (row.get('answer') or "").strip()
                # answer 形如 "(A)" → 取首个 A-E 字母
                letter = ""
                for c in ans_raw:
                    if c.upper() in "ABCDE":
                        letter = c.upper()
                        break
                if not letter:
                    continue

                img_rels = []
                for k in range(1, 5):
                    imf = row.get(f'image_{k}')
                    if not imf:
                        continue
                    imb = imf.get('bytes') if isinstance(imf, dict) else None
                    if not imb:
                        continue
                    ext = _resolve_image_ext(imb, imf.get('path') or "")
                    rel = f"images/{sub}/{idx}_img{k}{ext}"
                    _write_image(imb, img_dir / f"{sub}/{idx}_img{k}{ext}")
                    img_rels.append(rel)
                if not img_rels:
                    continue

                rec = {
                    "question_id": f"blink_{sub}_{idx}",
                    "question": prompt + "\nAnswer with just the option letter.",
                    "answer": letter,
                    "image_paths": img_rels,
                    "question_type": f"BLINK/{sub}",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sub_n += 1
                total += 1
            per_sub[sub] = sub_n
            print(f"  [{sub}] +{sub_n}", flush=True)
    print(f"[BLINK] ✅ {total} 条 → {jp}", flush=True)
    return jp, total


# ============================================================
# MUIRBench: 多图 image_list, question 含 <image> 占位符, answer A-E
# ============================================================
def _replace_image_placeholder(s):
    out, k, i = [], 1, 0
    while i < len(s):
        if s[i:i + 7] == '<image>':
            out.append(f'[IMG {k}]')
            k += 1
            i += 7
        else:
            out.append(s[i])
            i += 1
    return ''.join(out)


def prepare_muirbench(out_dir):
    print("\n[MUIRBench] 输出 →", out_dir, flush=True)
    src_dir = DATA_ROOT / "MUIRBENCH" / "data"
    srcs = sorted(src_dir.glob("test-*.parquet"))
    assert srcs, f"{src_dir} 下没有 test-*.parquet"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "muirbench_test.jsonl"

    n = 0
    with open(jp, 'w', encoding='utf-8') as fout:
        for row in _iter_parquet(srcs):
            idx = row.get('idx') or str(n)
            task = (row.get('task') or "").strip()
            image_type = (row.get('image_type') or "").strip()
            q_raw = (row.get('question') or "").strip()
            options = list(row.get('options') or [])
            ans_raw = (row.get('answer') or "").strip()
            image_list = list(row.get('image_list') or [])
            if not options or not image_list:
                continue

            letter = ""
            for c in ans_raw:
                if c.upper() in "ABCDE":
                    letter = c.upper()
                    break
            if not letter:
                continue

            img_rels = []
            for k, imf in enumerate(image_list, 1):
                if not isinstance(imf, dict):
                    continue
                imb = imf.get('bytes')
                if not imb:
                    continue
                ext = _resolve_image_ext(imb, imf.get('path') or "")
                rel = f"images/{idx}/img{k}{ext}"
                _write_image(imb, img_dir / f"{idx}/img{k}{ext}")
                img_rels.append(rel)
            if not img_rels:
                continue

            q_rep = _replace_image_placeholder(q_raw)
            opts_rep = [_replace_image_placeholder(o) for o in options]
            letters = "ABCDE"
            opts_str = "\n".join(
                f"{letters[i]}: {opts_rep[i]}"
                for i in range(min(len(opts_rep), 5))
            )
            full_q = (
                f"{q_rep}\nOptions:\n{opts_str}\n"
                f"Answer with just the option letter (A/B/C/D/E)."
            )

            rec = {
                "question_id": f"muir_{idx}",
                "question": full_q,
                "answer": letter,
                "image_paths": img_rels,
                "question_type": f"MUIRBench/{task}/{image_type}",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 100 == 0:
                print(f"  已处理 {n} 条...", flush=True)
    print(f"[MUIRBench] ✅ {n} 条 → {jp}", flush=True)
    return jp, n


# ============================================================
# 通用 schema 自适应小工具
# 不同上游版本的 MMBench/HallusionBench/SimpleVQA 字段命名不一致,
# 用 dict get-by-fallback 屏蔽差异
# ============================================================
def _g(row, *names, default=None):
    """从 row 中按候选字段名取第一个非空值"""
    for n in names:
        if n in row and row[n] not in (None, "", b""):
            return row[n]
    return default


def _img_bytes_from_field(v):
    """图像字段可能是: bytes / dict{bytes,path} / list[dict]"""
    if v is None:
        return None, ""
    if isinstance(v, dict):
        return v.get("bytes"), (v.get("path") or "")
    if isinstance(v, (bytes, bytearray)):
        return v, ""
    if isinstance(v, list) and v:
        return _img_bytes_from_field(v[0])
    return None, ""


def _list_parquets(d, splits):
    """在目录 d 下找 split 的 parquet (支持 'split-XXXXX-of-YYYYY.parquet' / 'data/split.parquet' 等多种命名)

    去重用 (parent_resolve, name) 组合作唯一 key, 同时对每个 pattern 的结果先入 set 再合并.
    避免同一文件被 'dev-*.parquet' 和 '**/dev-*.parquet' 两个 pattern 重复匹配后出现两次.
    """
    seen_keys = set()
    uniq = []
    for split in splits:
        for pat in (f"{split}-*.parquet",
                    f"{split}.parquet",
                    f"data/{split}-*.parquet",
                    f"data/{split}.parquet",
                    f"**/{split}-*.parquet"):
            try:
                matches = sorted(d.glob(pat))
            except Exception:
                matches = []
            for p in matches:
                # 用 (绝对路径父目录, 文件名) 作 key, 兼容 mount 下 resolve 不稳定的情况
                try:
                    key = (str(p.parent.resolve()), p.name)
                except Exception:
                    key = (str(p.parent), p.name)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                uniq.append(p)
    return uniq


# ============================================================
# MMBench-EN: ~4k dev 条目
#   字段(不同版本可能有差异): index / question / A / B / C / D / answer /
#                            category / l2-category / hint / image
#   官方原版评测: circular evaluation (4 次重排选项, 全对才算对)
#   这里默认走 single-pass 评测 (与同项目 MMStar/BLINK 风格一致); 如需 circular,
#   后续可用 --mmbench_circular 参数扩展.
# ============================================================
def prepare_mmbench(out_dir, split="dev"):
    print(f"\n[MMBench-EN] 输出 → {out_dir}  split={split}", flush=True)
    src_dir = DATA_ROOT / "MMBench"
    # 上游版本可能把 EN 子集放在 en/ 或 直接放 split.parquet
    # 这里显式枚举可能的精确路径, 避免 glob 在多 pattern / 父子目录下重复匹配同一文件.
    cand_files = [
        src_dir / "en" / f"{split}-00000-of-00001.parquet",
        src_dir / "EN" / f"{split}-00000-of-00001.parquet",
        src_dir / f"en_{split}.parquet",
        src_dir / f"{split}.parquet",
        src_dir / "MMBench_DEV_EN" / f"{split}.parquet",
    ]
    srcs = []
    for f in cand_files:
        if f.exists() and f not in srcs:
            srcs.append(f)
    # 若上面都没找到, 再 fallback 到通用 _list_parquets (但只允许根 dir 一级)
    if not srcs:
        for d in [src_dir / "en", src_dir / "EN", src_dir]:
            if d.exists():
                cands = sorted(d.glob(f"{split}-*.parquet"))
                if cands:
                    srcs = cands
                    break
    assert srcs, f"未在 {src_dir} 下找到 MMBench-EN {split} 的 parquet"
    print(f"  使用 {len(srcs)} 个 parquet 文件: {[p.name for p in srcs]}", flush=True)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"mmbench_en_{split}.jsonl"

    n = 0
    with open(jp, 'w', encoding='utf-8') as fout:
        for row in _iter_parquet(srcs):
            idx = _g(row, 'index', 'idx', 'id', default=n)
            question = (_g(row, 'question', 'question_text', default="") or "").strip()
            hint = (_g(row, 'hint', default="") or "").strip()
            answer = (_g(row, 'answer', 'gt_answer', default="") or "").strip().upper()
            category = (_g(row, 'category', default="") or "").strip()
            l2 = (_g(row, 'L2-category', 'l2-category', 'l2_category', default="") or "").strip()
            # 选项: A/B/C/D 列, 或 options=list
            options = _g(row, 'options', default=None)
            if options is None:
                options = []
                for letter in 'ABCD':
                    v = _g(row, letter, default=None)
                    if v is None:
                        break
                    options.append(str(v))
            else:
                options = [str(o) for o in list(options)]
            if len(options) < 2 or not answer or answer not in 'ABCDE':
                continue

            imb, hint_path = _img_bytes_from_field(_g(row, 'image', 'image_1', default=None))
            if not imb:
                continue
            ext = _resolve_image_ext(imb, hint_path)
            img_rel = f"images/{idx}{ext}"
            _write_image(imb, img_dir / f"{idx}{ext}")

            opts_str = "\n".join(
                f"{'ABCDE'[i]}: {options[i]}"
                for i in range(min(len(options), 5))
            )
            full_q_parts = []
            if hint:
                full_q_parts.append(f"Hint: {hint}")
            full_q_parts.append(question)
            full_q_parts.append(f"Options:\n{opts_str}")
            full_q_parts.append("Answer with just the option letter (A/B/C/D).")
            full_q = "\n".join(full_q_parts)

            rec = {
                "question_id": f"mmbench_en_{split}_{idx}",
                "question": full_q,
                "answer": answer,
                "image_paths": [img_rel],
                "question_type": f"MMBench-EN/{category}/{l2}",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 500 == 0:
                print(f"  已处理 {n} 条...", flush=True)
    print(f"[MMBench-EN] ✅ {n} 条 → {jp}", flush=True)
    return jp, n


# ============================================================
# HallusionBench: ~1.1k 条
#   字段(常见): set_id / figure_id / question / gt_answer (yes/no/0/1) /
#              category / subcategory / visual_input / filename / image
#   评测方式: open yes/no (官方还有 figure-level / question-pair 聚合,
#            这里输出 single-pass yes/no, 后续如需 figure 级聚合可二次脚本处理)
# ============================================================
def prepare_hallusionbench(out_dir):
    print(f"\n[HallusionBench] 输出 → {out_dir}", flush=True)
    src_dir = DATA_ROOT / "HallusionBench"
    srcs = _list_parquets(src_dir, ["test", "val", "train", "image"])
    assert srcs, f"未在 {src_dir} 下找到 HallusionBench 的 parquet"
    print(f"  使用 {len(srcs)} 个 parquet 文件: {[p.name for p in srcs]}", flush=True)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "hallusionbench_test.jsonl"

    def _norm_yn(v):
        s = str(v).strip().lower()
        if s in ('1', 'yes', 'y', 'true', 't'):
            return 'yes'
        if s in ('0', 'no', 'n', 'false', 'f'):
            return 'no'
        return s  # 可能本身就是 'yes'/'no' 或其它

    n = 0
    with open(jp, 'w', encoding='utf-8') as fout:
        for i, row in enumerate(_iter_parquet(srcs)):
            qid_raw = _g(row, 'question_id', 'qid',
                         default=f"hb_{_g(row, 'category', default='')}_"
                                 f"{_g(row, 'subcategory', default='')}_"
                                 f"{_g(row, 'set_id', default='')}_"
                                 f"{_g(row, 'figure_id', default='')}_"
                                 f"{_g(row, 'question_id', default=i)}")
            question = (_g(row, 'question', 'question_text', default="") or "").strip()
            ans_raw = _g(row, 'gt_answer', 'answer', 'gt_answer_details', default="")
            answer = _norm_yn(ans_raw)
            if not question or answer not in ('yes', 'no'):
                # 部分版本将无图样本 (visual_input=0) 也保留为纯文本; 这里只保留 yes/no
                continue
            visual_input = str(_g(row, 'visual_input', default='1')).strip()
            imb, hint_path = _img_bytes_from_field(_g(row, 'image', default=None))
            if not imb:
                # 无图样本跳过 (我们做 VLM 评测)
                continue
            ext = _resolve_image_ext(imb, hint_path)
            img_rel = f"images/{n:05d}{ext}"
            _write_image(imb, img_dir / f"{n:05d}{ext}")

            full_q = (
                f"{question}\n"
                f"Please answer with a single word: yes or no."
            )
            category = (_g(row, 'category', default='') or '').strip()
            subcat = (_g(row, 'subcategory', default='') or '').strip()
            rec = {
                "question_id": f"hb_{n:05d}",
                "question": full_q,
                "answer": answer,
                "image_paths": [img_rel],
                "question_type": f"HallusionBench/{category}/{subcat}",
                # 保留聚合所需的元信息
                "set_id": str(_g(row, 'set_id', default='')),
                "figure_id": str(_g(row, 'figure_id', default='')),
                "visual_input": visual_input,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"  已处理 {n} 条...", flush=True)
    print(f"[HallusionBench] ✅ {n} 条 → {jp}", flush=True)
    return jp, n


# ============================================================
# SimpleVQA: 简短开放回答 benchmark (单图)
#   字段(常见): question_id / question / answer / category / image
#   评测方式: open (substring / 单词 / 数字 匹配; 不再做 letter 选择)
# ============================================================
def prepare_simplevqa(out_dir):
    print(f"\n[SimpleVQA] 输出 → {out_dir}", flush=True)
    src_dir = DATA_ROOT / "SimpleVQA"
    srcs = _list_parquets(src_dir, ["test", "val", "train"])
    assert srcs, f"未在 {src_dir} 下找到 SimpleVQA 的 parquet"
    print(f"  使用 {len(srcs)} 个 parquet 文件: {[p.name for p in srcs]}", flush=True)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "simplevqa_test.jsonl"

    n = 0
    with open(jp, 'w', encoding='utf-8') as fout:
        for i, row in enumerate(_iter_parquet(srcs)):
            qid = _g(row, 'question_id', 'qid', 'id', default=f"svqa_{i:05d}")
            question = (_g(row, 'question', 'question_text', default="") or "").strip()
            answer = _g(row, 'answer', 'gt_answer', 'answers', default="")
            # answer 可能是 list (多个 alternative); 取第一个为主
            if isinstance(answer, list):
                if not answer:
                    continue
                answer_str = str(answer[0]).strip()
                alt_answers = [str(a).strip() for a in answer if str(a).strip()]
            else:
                answer_str = str(answer).strip()
                alt_answers = [answer_str] if answer_str else []
            if not question or not answer_str:
                continue

            imb, hint_path = _img_bytes_from_field(_g(row, 'image', 'images', default=None))
            if not imb:
                continue
            ext = _resolve_image_ext(imb, hint_path)
            img_rel = f"images/{n:05d}{ext}"
            _write_image(imb, img_dir / f"{n:05d}{ext}")

            category = (_g(row, 'category', default='') or '').strip()
            full_q = (
                f"{question}\n"
                f"Please answer briefly with a single word or short phrase."
            )
            rec = {
                "question_id": str(qid) if not str(qid).startswith("svqa_") else str(qid),
                "question": full_q,
                "answer": answer_str,
                "alt_answers": alt_answers,  # 供 open 匹配时多选项命中
                "image_paths": [img_rel],
                "question_type": f"SimpleVQA/{category}",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 500 == 0:
                print(f"  已处理 {n} 条...", flush=True)
    print(f"[SimpleVQA] ✅ {n} 条 → {jp}", flush=True)
    return jp, n


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["MMStar", "RealWorldQA", "BLINK", "MUIRBench",
                                 "MMBench", "HallusionBench", "SimpleVQA", "ALL"])
    parser.add_argument("--out_root", default=str(DATA_ROOT))
    parser.add_argument("--subtasks", nargs="+", default=None)
    parser.add_argument("--blink_split", default="val", choices=["val", "test"])
    parser.add_argument("--mmbench_split", default="dev",
                        help="MMBench split: dev (有 GT, 默认) / test (需上传服务器)")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    todo = ([args.dataset] if args.dataset != "ALL"
            else ["MMStar", "RealWorldQA", "BLINK", "MUIRBench",
                  "MMBench", "HallusionBench", "SimpleVQA"])

    summary = []
    for ds in todo:
        out_dir = out_root / f"{ds}_eval"
        if ds == "MMStar":
            jp, n = prepare_mmstar(out_dir)
        elif ds == "RealWorldQA":
            jp, n = prepare_realworldqa(out_dir)
        elif ds == "BLINK":
            jp, n = prepare_blink(out_dir, subtasks=args.subtasks, split=args.blink_split)
        elif ds == "MUIRBench":
            jp, n = prepare_muirbench(out_dir)
        elif ds == "MMBench":
            jp, n = prepare_mmbench(out_dir, split=args.mmbench_split)
        elif ds == "HallusionBench":
            jp, n = prepare_hallusionbench(out_dir)
        elif ds == "SimpleVQA":
            jp, n = prepare_simplevqa(out_dir)
        else:
            raise ValueError(ds)
        summary.append((ds, str(jp), n))

    print("\n" + "=" * 70)
    print("  📋 预处理汇总")
    print("=" * 70)
    for ds, jp, n in summary:
        print(f"  {ds:<14s}  {n:>5d} 条  →  {jp}")


if __name__ == "__main__":
    main()
