#!/usr/bin/env python3
"""
无第三方依赖的 TFEvents 解析器（仅支持 simple_value scalar）。
events 文件格式（TFRecord）：
  [uint64 length-le][uint32 masked_crc][payload bytes][uint32 masked_crc]
payload = Event protobuf.
我们只需要：
  Event.step (int64, field 2, wiretype VARINT)
  Event.summary.value[*].tag (string)
  Event.summary.value[*].simple_value (float)
其他字段全部跳过。
"""
import argparse
import os
import struct
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------- minimal protobuf reader ---------------------
def _read_varint(buf, i):
    shift = 0
    result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, i


def _skip(buf, i, wire):
    if wire == 0:
        _, i = _read_varint(buf, i)
    elif wire == 1:
        i += 8
    elif wire == 2:
        ln, i = _read_varint(buf, i)
        i += ln
    elif wire == 5:
        i += 4
    else:
        raise ValueError(f"unsupported wire type {wire}")
    return i


def _parse_summary_value(buf):
    """parse Summary.Value submessage. returns (tag, simple_value or None)."""
    i = 0
    n = len(buf)
    tag = None
    simple = None
    while i < n:
        key, i = _read_varint(buf, i)
        field = key >> 3
        wire = key & 0x7
        if field == 7 and wire == 2:  # node_name (string)  -- unused
            ln, i = _read_varint(buf, i)
            i += ln
        elif field == 1 and wire == 2:  # tag (string)
            ln, i = _read_varint(buf, i)
            tag = buf[i:i + ln].decode("utf-8", errors="replace")
            i += ln
        elif field == 2 and wire == 5:  # simple_value (float)
            simple = struct.unpack_from("<f", buf, i)[0]
            i += 4
        else:
            i = _skip(buf, i, wire)
    return tag, simple


def _parse_summary(buf):
    """parse Summary message; returns list of (tag, simple_value)."""
    out = []
    i = 0
    n = len(buf)
    while i < n:
        key, i = _read_varint(buf, i)
        field = key >> 3
        wire = key & 0x7
        if field == 1 and wire == 2:  # value (Summary.Value)
            ln, i = _read_varint(buf, i)
            sub = buf[i:i + ln]
            i += ln
            tag, sv = _parse_summary_value(sub)
            if tag is not None and sv is not None:
                out.append((tag, sv))
        else:
            i = _skip(buf, i, wire)
    return out


def _parse_event(buf):
    """parse Event message; return (step, [(tag, value)])."""
    i = 0
    n = len(buf)
    step = None
    pairs = []
    while i < n:
        key, i = _read_varint(buf, i)
        field = key >> 3
        wire = key & 0x7
        if field == 1 and wire == 1:  # wall_time (double)
            i += 8
        elif field == 2 and wire == 0:  # step (int64)
            step, i = _read_varint(buf, i)
        elif field == 5 and wire == 2:  # summary (Summary)
            ln, i = _read_varint(buf, i)
            sub = buf[i:i + ln]
            i += ln
            pairs.extend(_parse_summary(sub))
        else:
            i = _skip(buf, i, wire)
    return step, pairs


def parse_tfevents(path):
    """yield (step, [(tag, value)]) per Event."""
    with open(path, "rb") as f:
        data = f.read()
    n = len(data)
    i = 0
    bad = 0
    total_records = 0
    total_events = 0
    while i + 12 <= n:
        ln = struct.unpack_from("<Q", data, i)[0]
        i += 8 + 4  # length + masked_crc(length)
        if i + ln + 4 > n:
            break
        payload = data[i:i + ln]
        i += ln + 4  # payload + masked_crc(payload)
        total_records += 1
        try:
            step, pairs = _parse_event(payload)
        except Exception as e:
            bad += 1
            continue
        total_events += 1
        if step is None or not pairs:
            continue
        yield step, pairs
    print(f"  [stat {os.path.basename(path)}] records={total_records} parsed={total_events} bad={bad}", flush=True)


# --------------------- plotting ---------------------
def smooth(y, k=11):
    if len(y) < 3 or k <= 1:
        return y
    k = min(k, max(3, len(y) // 5 * 2 + 1))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(yp, kernel, mode="valid")


GROUPS = [
    ("Loss / total & components", [
        ("train/loss/total", "total"),
        ("train/loss/main_ce", "main_ce"),
        ("train/nld/exit_token_loss", "exit"),
        ("train/nld/sw_srs_loss", "latent_kw (sw_srs)"),
    ]),
    ("CE / Acc", [
        ("train/loss/think_ce", "think_ce"),
        ("train/loss/answer_ce", "answer_ce"),
        ("train/acc/top1_answer", "top1_ans"),
    ]),
    ("Vision: vis_loss [left] & cos(h, r/v/t) [right]", [
        ("train/nld/vis_loss", "vis_loss"),
        ("train/nld/vis_cos_h_r_mean", "cos(h,r)"),
        ("train/nld/vis_cos_h_v_mean", "cos(h,v)"),
        ("train/nld/vis_cos_h_t_mean", "cos(h,t)"),
    ]),
    ("Vision per-role: cos(h,v)", [
        ("train/nld/vis_cos_h_v_abstract", "abs"),
        ("train/nld/vis_cos_h_v_bridge", "br"),
        ("train/nld/vis_cos_h_v_concrete", "con"),
        ("train/nld/vis_cos_h_v_unified", "uni"),
    ]),
    ("Hidden state geometry", [
        ("train/collapse/h_first_last_cos", "first_last_cos"),
        ("train/collapse/h_adj_cos_mean", "adj_mean"),
        ("train/collapse/h_norm_mean", "h_norm (mean)"),
    ]),
    ("Laser-DWAL: q-entropy & top-k", [
        ("train/collapse/sw_srs_q_entropy_mean", "q_ent (mean)"),
        ("train/collapse/sw_srs_q_entropy_first_stage", "q_ent[early]"),
        ("train/collapse/sw_srs_q_entropy_last_stage", "q_ent[late]"),
        ("train/collapse/sw_srs_topk_hit_ratio", "topk_hit"),
    ]),
    ("Stage shrinking", [
        ("train/collapse/h_stage_diag_score", "diag"),
        ("train/collapse/h_stage_monotonic", "mono"),
        ("train/collapse/h_stage_shift_kl", "shift_kl"),
    ]),
    ("Saturation / Steps used", [
        ("train/collapse/h_sat_step1", "sat[1]"),
        ("train/collapse/h_sat_step_last", "sat[L]"),
        ("train/collapse/h_sat_early_exit_ratio", "early_exit"),
        ("train/nld/num_thought_steps_mean", "n_steps (mean)"),
        ("train/nld/thought_count", "thought_count"),
    ]),
    ("Optim: lr & grad_norm", [
        ("train/learning_rate", "lr"),
        ("train/grad_norm", "grad_norm"),
    ]),
    ("TGVR: visual evidence", [
        ("train/collapse/tgvr_cos_h_v_mean", "tgvr cos(h,v)"),
        ("train/collapse/tgvr_topk_recall_mean", "tgvr topk_recall"),
        ("train/collapse/tgvr_v_pos_norm_mean", "tgvr v_pos_norm"),
    ]),
]


def _norm(name):
    return name.replace("/", "_").replace(".", "_").replace("-", "_").lower()


def find_tag(scalars, name):
    nname = _norm(name)
    for t in scalars:
        if _norm(t) == nname:
            return t
    for t in scalars:
        nt = _norm(t)
        if nt.endswith("_" + nname) or nt.startswith(nname + "_"):
            return t
    for t in scalars:
        if nname in _norm(t):
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth", type=int, default=11)
    ap.add_argument("--list", action="store_true", help="only list available tags")
    args = ap.parse_args()

    if not os.path.isdir(args.logdir):
        print(f"[err] logdir not found: {args.logdir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        [os.path.join(args.logdir, f) for f in os.listdir(args.logdir) if f.startswith("events")],
        key=lambda p: os.path.getmtime(p),
    )
    print(f"[load] {len(files)} event files", flush=True)

    raw = defaultdict(dict)  # tag -> {step: value}
    for f in files:
        cnt = 0
        for step, pairs in parse_tfevents(f):
            for tag, val in pairs:
                raw[tag][step] = val
            cnt += 1
        print(f"  {os.path.basename(f)}: {cnt} events parsed", flush=True)

    scalars = {}
    for t, d in raw.items():
        steps = sorted(d.keys())
        scalars[t] = (np.array(steps), np.array([d[s] for s in steps], dtype=np.float64))

    print(f"[parsed] {len(scalars)} scalar tags", flush=True)
    for t in sorted(scalars):
        s, v = scalars[t]
        if len(s):
            print(f"  {t:40s} n={len(s):4d}  step=[{s[0]},{s[-1]}]  v=[{v.min():.4f},{v.max():.4f}]  last={v[-1]:.4f}", flush=True)
    if args.list:
        return

    # ---- plot ----
    n = len(GROUPS)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.6 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    last_step = max((s[-1] for s, _ in scalars.values() if len(s)), default=0)

    # 哪些组用 log y / 双 y 轴
    LOG_Y_GROUPS = {"Loss / total & components", "Optim: lr & grad_norm"}
    DUAL_Y_GROUPS = {
        # group title -> 哪些 label 走右轴
        "Hidden state geometry": {"h_norm (mean)"},
        "TGVR: visual evidence": {"tgvr v_pos_norm"},
    }

    for i, (title, metrics) in enumerate(GROUPS):
        ax = axes[i]
        ax2 = None
        right_labels = DUAL_Y_GROUPS.get(title, set())
        if right_labels:
            ax2 = ax.twinx()
        plotted = 0
        for tag_pat, label in metrics:
            tag = find_tag(scalars, tag_pat)
            if tag is None:
                continue
            steps, vals = scalars[tag]
            if len(steps) < 2:
                continue
            ys = smooth(vals, k=args.smooth)
            target = ax2 if (label in right_labels) else ax
            target.plot(steps, vals, alpha=0.18, linewidth=0.8)
            line, = target.plot(steps, ys, label=f"{label} (n={len(steps)})", linewidth=1.6)
            plotted += 1
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if title in LOG_Y_GROUPS:
            try:
                ax.set_yscale("symlog", linthresh=1e-4)
            except Exception:
                ax.set_yscale("log")
        if plotted:
            # 合并图例
            lines, labels = ax.get_legend_handles_labels()
            if ax2 is not None:
                l2, lb2 = ax2.get_legend_handles_labels()
                lines += l2
                labels += lb2
                ax2.set_ylabel("right-axis", fontsize=8)
            ax.legend(lines, labels, loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")

    for j in range(len(GROUPS), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Stage-2 training curves — last step = {last_step}",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[done] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
