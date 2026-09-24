#!/usr/bin/env python3
"""
Fine-tune Whisper on your own voice, with Wispr Flow's text as the labels.

The shadow archive (shadow.py) pairs every Wispr Flow dictation's audio with
the text Wispr produced for it. Exported with

    python shadow.py --dir <data> dataset --out <dataset>

it is a speaker-specific training set. This script LoRA-fine-tunes a Whisper
checkpoint on the "train" split, scores the "test" split (the newest clips,
never trained on) before and after, merges the adapter and converts the
result to CTranslate2, the format faster-whisper - and so Vox - loads:

    python train/finetune_whisper.py --data <dataset> --out <run dir>
    set VOX_MODEL=<run dir>\\ct2          (then restart Vox)

and a shadow variant with env {"VOX_MODEL": "<run dir>/ct2"} measures the
tuned model against Wispr on the whole archive. That archive includes the
training clips, so trust the held-out test numbers printed here first.

It is worth running at a few hours of audio (shadow.py announces 5 h);
minutes of audio only overfit. Needs its own venv - see train/requirements.txt
(CUDA torch for real runs; the CPU build is enough for --base
openai/whisper-tiny smoke tests).
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
import wave

import numpy as np

SAMPLE_RATE = 16000
MAX_SEC = 30.0  # Whisper's encoder sees 30 s; longer clips cannot be aligned


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected 16-bit {SAMPLE_RATE} Hz")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        ch = w.getnchannels()
    audio = pcm.astype(np.float32) / 32768.0
    return audio.reshape(-1, ch).mean(axis=1) if ch > 1 else audio


def load_split(data_dir, label_field, max_items=None):
    items = {"train": [], "test": []}
    skipped = 0
    with open(os.path.join(data_dir, "metadata.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = (r.get(label_field) or "").strip()
            if not text:
                continue
            if (r.get("seconds") or 0) > MAX_SEC:
                skipped += 1
                continue
            items[r.get("split", "train")].append(
                (os.path.join(data_dir, r["audio"]), text))
    if max_items:
        items = {k: v[:max_items] for k, v in items.items()}
    return items, skipped


_NORM_RE = re.compile(r"[a-z0-9']+")


def words(text):
    return _NORM_RE.findall(text.replace("’", "'").lower())


def wer(refs, hyps):
    """Corpus word error rate (Levenshtein on normalized words)."""
    err = n = 0
    for r, h in zip(refs, hyps):
        a, b = words(r), words(h)
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        err += prev[-1]
        n += len(a)
    return err / max(1, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", required=True, help="dataset from shadow.py dataset")
    ap.add_argument("--out", required=True, help="run folder (adapter, merged, ct2)")
    ap.add_argument("--base", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--label", default="text", choices=("text", "asr"),
                    help="text = what Wispr pasted (formatted, fillers gone); "
                         "asr = Wispr's raw recognition")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--device", default=None, help="cuda or cpu (auto)")
    ap.add_argument("--max-items", type=int, help="cap per split (smoke tests)")
    ap.add_argument("--no-convert", action="store_true",
                    help="skip the CTranslate2 conversion")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = device == "cuda"
    os.makedirs(args.out, exist_ok=True)

    items, skipped = load_split(args.data, args.label, args.max_items)
    hours = sum(len(read_wav(p)) for p, _ in items["train"]) / SAMPLE_RATE / 3600
    log(f"train {len(items['train'])} clips ({hours:.2f} h), test "
        f"{len(items['test'])}; {skipped} clips over {MAX_SEC:.0f} s skipped")
    if not items["train"] or not items["test"]:
        sys.exit("need both a train and a test split")

    processor = WhisperProcessor.from_pretrained(args.base)
    processor.tokenizer.set_prefix_tokens(language="english", task="transcribe",
                                          predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.to(device)
    start_id = model.config.decoder_start_token_id

    def features(batch):
        audio = [read_wav(p) for p, _ in batch]
        feats = processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE,
                                            return_tensors="pt").input_features
        return feats.to(device)  # float32; autocast / .half() take it from here

    def labels(batch):
        ids = [processor.tokenizer(t).input_ids for _, t in batch]
        ids = [x[1:] if x and x[0] == start_id else x for x in ids]
        width = max(len(x) for x in ids)
        out = torch.full((len(ids), width), -100, dtype=torch.long)
        for i, x in enumerate(ids):
            out[i, :len(x)] = torch.tensor(x)
        return out.to(device)

    @torch.no_grad()
    def transcribe(m, batch_items):
        m.eval()
        hyps = []
        for i in range(0, len(batch_items), 4):
            chunk = batch_items[i:i + 4]
            feats = features(chunk).to(next(m.parameters()).dtype)
            out = m.generate(input_features=feats, max_new_tokens=225)
            hyps += processor.batch_decode(out, skip_special_tokens=True)
        return [h.strip() for h in hyps]

    refs = [t for _, t in items["test"]]
    if fp16:
        model.half()
    base_hyps = transcribe(model, items["test"])
    base_wer = wer(refs, base_hyps)
    log(f"held-out WER before tuning: {base_wer:.1%}")
    model.float()

    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                      bias="none")
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"LoRA r={args.rank}: {trainable / 1e6:.1f}M trainable parameters")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    steps_per_epoch = math.ceil(len(items["train"]) / (args.batch * args.accum))
    total_steps = max(1, int(round(steps_per_epoch * args.epochs)))
    warmup = max(1, total_steps // 10)
    def lr_factor(s):  # linear warm-up, then linear decay to zero
        if s < warmup:
            return (s + 1) / warmup
        return max(0.0, (total_steps - s) / max(1, total_steps - warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    step = 0
    model.train()
    t0 = time.time()
    while step < total_steps:
        order = items["train"][:]
        random.shuffle(order)
        for k in range(0, len(order), args.batch):
            batch = order[k:k + args.batch]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                loss = model(input_features=features(batch), labels=labels(batch)).loss
            scaler.scale(loss / args.accum).backward()
            if (k // args.batch + 1) % args.accum == 0 or k + args.batch >= len(order):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % max(1, total_steps // 10) == 0 or step == total_steps:
                    log(f"step {step}/{total_steps} loss {loss.item():.3f}")
                if step >= total_steps:
                    break
    log(f"trained in {time.time() - t0:.0f} s")

    model.save_pretrained(os.path.join(args.out, "adapter"))
    merged = model.merge_and_unload()
    if fp16:
        merged.half()
    tuned_hyps = transcribe(merged, items["test"])
    tuned_wer = wer(refs, tuned_hyps)
    log(f"held-out WER after tuning: {tuned_wer:.1%} (before: {base_wer:.1%})")

    merged_dir = os.path.join(args.out, "merged")
    merged.save_pretrained(merged_dir, safe_serialization=True)
    processor.save_pretrained(merged_dir)
    # faster-whisper reads the mel settings from preprocessor_config.json;
    # transformers 5 folds them into processor_config.json instead.
    pre = os.path.join(merged_dir, "preprocessor_config.json")
    if not os.path.exists(pre):
        with open(pre, "w", encoding="utf-8") as f:
            json.dump(processor.feature_extractor.to_dict(), f, indent=2)
    ct2_dir = os.path.join(args.out, "ct2")
    if not args.no_convert:
        from ctranslate2.converters import TransformersConverter
        TransformersConverter(merged_dir, copy_files=[
            "tokenizer.json", "preprocessor_config.json"]).convert(
            ct2_dir, quantization="float16", force=True)
        log(f"CTranslate2 model: {ct2_dir}  (VOX_MODEL={ct2_dir})")

    summary = {
        "base": args.base, "label": args.label, "epochs": args.epochs,
        "lr": args.lr, "rank": args.rank, "train_clips": len(items["train"]),
        "train_hours": round(hours, 3), "test_clips": len(items["test"]),
        "skipped_over_30s": skipped, "wer_before": round(base_wer, 4),
        "wer_after": round(tuned_wer, 4), "ct2": None if args.no_convert else ct2_dir,
        "examples": [{"ref": r, "before": b, "after": a}
                     for r, b, a in list(zip(refs, base_hyps, tuned_hyps))[:5]],
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"summary: {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
