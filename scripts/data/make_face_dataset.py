#!/usr/bin/env python
"""Build a cropped-and-aligned dog-face dataset from a Hub image dataset (e.g. yaryar78/Ray_dog).

    # 1. detect, align, crop -> a local folder + contact sheets to review
    python scripts/data/make_face_dataset.py build --out data/ray_dog_faces

    # 2. look at data/ray_dog_faces/contact_sheet.jpg; drop bad crops by index
    python scripts/data/make_face_dataset.py build --out data/ray_dog_faces --exclude 12,40

    # 3. upload (private by default)
    python scripts/data/make_face_dataset.py push --out data/ray_dog_faces --repo-id yaryar78/Ray_dog_faces

How faces are found and aligned
-------------------------------
1. Grounding DINO (an open-vocabulary detector) is prompted with
   "dog face. dog eye. dog nose." on a 1024-px copy of each photo.
2. A face box is accepted only if a small eye or nose detection lies inside it,
   which rejects photos where the face is turned away or hidden.
3. Eyes and nose are detected again on the face region alone, where they are
   larger and localised more precisely.
4. If two eyes are found at a plausible distance, the photo is rotated so the eyes
   are level ("aligned"); otherwise the crop is upright ("box-only").
5. A square around the face box (``--margin`` x its longer side) is cut from the
   *full-resolution* photo and resized to ``--size``.

Every kept image records in metadata.jsonl how it was produced (scores, whether it
was aligned, roll angle, source crop size); rejected photos and the reason are
listed too. Nothing is inferred silently.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

DETECTOR = "IDEA-Research/grounding-dino-base"
PROMPT_ALL = "dog face. dog eye. dog nose."
PROMPT_PARTS = "dog eye. dog nose."


# ---- detection --------------------------------------------------------------------------
class Detector:
    def __init__(self, model_id=DETECTOR, device=None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def __call__(self, image, prompt, threshold=0.25):
        """Returns [(label, score, (x0, y0, x1, y1))] in ``image`` pixel coordinates."""
        inp = self.proc(images=image, text=prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp)
        r = self.proc.post_process_grounded_object_detection(
            out, inp["input_ids"], threshold=threshold, text_threshold=threshold, target_sizes=[image.size[::-1]])[0]
        labels = r.get("text_labels", r.get("labels"))
        return [(lab, float(s), tuple(float(v) for v in b)) for lab, s, b in zip(labels, r["scores"], r["boxes"].tolist())]


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _inside(pt, b):
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def _parts_in(dets, face, label, max_frac=0.15):
    """Small detections of ``label`` whose centre lies inside the face box, best first."""
    fa = _area(face)
    keep = [(s, b) for lab, s, b in dets if lab == label and _inside(_center(b), face) and _area(b) <= max_frac * fa]
    return sorted(keep, reverse=True)


def _dedupe_eyes(eyes, min_dist):
    """Keep the best eye, then the best other eye at least ``min_dist`` away."""
    if not eyes:
        return []
    first = eyes[0]
    for s, b in eyes[1:]:
        if math.dist(_center(b), _center(first[1])) >= min_dist:
            return [first, (s, b)]
    return [first]


def find_face(det: Detector, image, face_thr=0.3, part_thr=0.25):
    """Detect the face and its landmarks. Returns a dict, or (None, reason)."""
    small = image.copy()
    small.thumbnail((1024, 1024))
    k = image.width / small.width  # small -> full-resolution coordinates
    dets = det(small, PROMPT_ALL, threshold=part_thr)

    faces = [(s, b) for lab, s, b in dets if lab == "dog face" and s >= face_thr]
    if not faces:
        return None, "no face detected"
    # Prefer face boxes that contain parts; among those, the highest-scoring.
    scored = []
    for s, b in faces:
        n_parts = len(_parts_in(dets, b, "dog eye")) + len(_parts_in(dets, b, "dog nose"))
        if n_parts:
            scored.append((s, b))
    if not scored:
        return None, "face box contains no eye or nose (face turned away or hidden)"
    face_score, face = max(scored)
    face = tuple(v * k for v in face)

    # Second pass: eyes and nose on the face region (with a margin) at higher resolution.
    w, h = face[2] - face[0], face[3] - face[1]
    region = (max(0, face[0] - 0.15 * w), max(0, face[1] - 0.15 * h),
              min(image.width, face[2] + 0.15 * w), min(image.height, face[3] + 0.15 * h))
    crop = image.crop(tuple(int(v) for v in region))
    crop.thumbnail((640, 640))
    kc = (region[2] - region[0]) / crop.width
    part_dets = [(lab, s, (b[0] * kc + region[0], b[1] * kc + region[1], b[2] * kc + region[0], b[3] * kc + region[1]))
                 for lab, s, b in det(crop, PROMPT_PARTS, threshold=part_thr)]
    eyes = _dedupe_eyes(_parts_in(part_dets, face, "dog eye", 0.08), min_dist=0.15 * max(w, h))
    noses = _parts_in(part_dets, face, "dog nose", 0.15)
    return {
        "face_box": face, "face_score": face_score,
        "eyes": [_center(b) for _, b in eyes], "nose": _center(noses[0][1]) if noses else None,
    }, None


# ---- alignment + crop ---------------------------------------------------------------------
def align_and_crop(image, f, size=512, margin=1.3, max_roll_without_nose=40.0):
    """Rotate so the eyes are level, then cut a square centred on the face.

    * Two eyes give the roll angle. Which way is "up" is ambiguous from the eyes alone
      (a head turned 80 deg could be rolled +80 or -100), so with a nose the rotation
      is chosen that puts the nose *below* the eyes -- any roll can then be corrected.
      Without a nose only rolls up to ``max_roll_without_nose`` are corrected.
    * The square is ``margin`` x the face box's longer side, centred on the face box.
      Where it reaches past the photo's edge (often after a large rotation) the photo is
      mirror-padded rather than left black; ``outside_fraction`` records how much of
      the crop is padding.
    """
    import numpy as np
    from PIL import Image

    face = f["face_box"]
    cx, cy = _center(face)
    side = margin * max(face[2] - face[0], face[3] - face[1])

    roll, aligned = 0.0, False
    if len(f["eyes"]) == 2:
        (x1, y1), (x2, y2) = sorted(f["eyes"])          # left-most eye first
        ex, ey = x2 - x1, y2 - y1
        roll = math.degrees(math.atan2(ey, ex))          # in (-90, 90]
        nose = f["nose"]
        eye_dist = math.hypot(ex, ey)
        if nose is not None:
            nx, ny = nose[0] - (x1 + x2) / 2, nose[1] - (y1 + y2) / 2
            if math.hypot(nx, ny) > 0.3 * eye_dist:      # nose clearly separate from the eyes
                r = math.radians(roll)
                # y of the eye->nose vector after rotating the image by `roll` (PIL: counter-clockwise)
                if -nx * math.sin(r) + ny * math.cos(r) < 0:
                    roll = roll + 180 if roll <= 0 else roll - 180   # nose would end up above: flip
                aligned = True
        if not aligned and abs(roll) <= max_roll_without_nose:
            aligned = True
    if not aligned:
        roll = 0.0

    # Mirror-pad so a centred (and possibly rotated) square never reaches outside the array.
    arr = np.asarray(image)
    pad = int(min(side, min(arr.shape[:2]) - 1))
    padded = Image.fromarray(np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect"))
    pcx, pcy = cx + pad, cy + pad
    if abs(roll) > 0.5:
        padded = padded.rotate(roll, resample=Image.BICUBIC, center=(pcx, pcy))
    side = min(side, padded.width, padded.height)
    x0 = min(max(pcx - side / 2, 0), padded.width - side)
    y0 = min(max(pcy - side / 2, 0), padded.height - side)
    crop = padded.crop((round(x0), round(y0), round(x0 + side), round(y0 + side))).resize((size, size), Image.LANCZOS)

    # How much of the crop came from padding: map a grid of crop points back to the photo.
    g = np.linspace(0, 1, 24)
    gx, gy = np.meshgrid(x0 + g * side - pcx, y0 + g * side - pcy)
    r = math.radians(roll)  # undo the counter-clockwise rotation
    ox = cx + gx * math.cos(r) - gy * math.sin(r)
    oy = cy + gx * math.sin(r) + gy * math.cos(r)
    inside = (ox >= 0) & (ox < image.width) & (oy >= 0) & (oy < image.height)
    return crop, {"aligned": aligned, "roll_degrees": round(roll, 2), "source_crop_px": int(side),
                  "outside_fraction": round(1 - float(inside.mean()), 3)}


def _thumb_key(im):
    """A tiny grayscale fingerprint for spotting duplicate photos."""
    import numpy as np

    return np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32) / 255


# ---- contact sheets -----------------------------------------------------------------------
def contact_sheet(items, path, cols=10, cell=160):
    """items: [(PIL image, caption)]; caption printed under each thumbnail."""
    from PIL import Image, ImageDraw

    if not items:
        return
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (im, cap) in enumerate(items):
        t = im.copy()
        t.thumbnail((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + 18)
        sheet.paste(t, (x + (cell - t.width) // 2, y))
        d.text((x + 4, y + cell + 2), cap, fill="black")
    sheet.save(path, quality=90)


# ---- commands -----------------------------------------------------------------------------
def build(args):
    from datasets import load_dataset

    out = Path(args.out)
    if out.exists() and not args.exclude and not args.overwrite:
        sys.exit(f"{out} exists. Pass --overwrite to rebuild, or --exclude to drop crops from it.")
    exclude = {int(i) for i in args.exclude.split(",")} if args.exclude else set()

    if args.exclude and (out / "metadata.jsonl").exists():
        # Re-use the detections already made; only drop the excluded crops.
        rows = [json.loads(line) for line in (out / "metadata.jsonl").read_text().splitlines()]
        kept = [r for r in rows if r["source_index"] not in exclude]
        for r in rows:
            if r["source_index"] in exclude:
                (out / r["file_name"]).unlink(missing_ok=True)
        _write_metadata(out, kept)
        info = json.loads((out / "build_info.json").read_text())
        info["excluded_by_review"] = sorted(set(info.get("excluded_by_review", [])) | exclude)
        info["n_kept"] = len(kept)
        (out / "build_info.json").write_text(json.dumps(info, indent=2))
        print(f"Excluded {len(rows) - len(kept)}; {len(kept)} crops remain.")
        _sheets(out, kept)
        return

    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    ds = load_dataset(args.source, split=args.split, cache_dir=args.cache_dir)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    det = Detector(device=args.device)
    print(f"{len(ds)} photos; detector {DETECTOR} on {det.device}")

    kept, rejected, seen = [], [], []
    for i, ex in enumerate(ds):
        img = ex["image"].convert("RGB")
        f, reason = find_face(det, img, args.face_threshold, args.part_threshold)
        if f is None:
            rejected.append({"source_index": i, "reason": reason})
            print(f"[{i:3d}] rejected: {reason}")
            continue
        crop, info = align_and_crop(img, f, args.size, args.margin)
        if info["source_crop_px"] < args.min_source_px:
            rejected.append({"source_index": i, "reason": f"face too small ({info['source_crop_px']} px < {args.min_source_px})"})
            print(f"[{i:3d}] rejected: face too small")
            continue
        key = _thumb_key(crop)
        dup = next((j for j, k in seen if abs(key - k).mean() < args.dup_threshold), None)
        if dup is not None and not args.keep_duplicates:
            rejected.append({"source_index": i, "reason": f"duplicate of #{dup}"})
            print(f"[{i:3d}] rejected: duplicate of #{dup}")
            continue
        seen.append((i, key))
        name = f"images/{i:04d}.jpg"
        crop.save(out / name, quality=95)
        row = {"file_name": name, "source_index": i, "face_score": round(f["face_score"], 3),
               "eyes_found": len(f["eyes"]), "nose_found": f["nose"] is not None, **info}
        kept.append(row)
        print(f"[{i:3d}] {'aligned' if info['aligned'] else 'box-only'} roll={info['roll_degrees']:+.1f} pad={info['outside_fraction']:.2f} "
              f"src={info['source_crop_px']}px face={f['face_score']:.2f} eyes={len(f['eyes'])}")

    _write_metadata(out, kept)
    (out / "rejected.json").write_text(json.dumps(rejected, indent=2))
    info = {
        "source_dataset": args.source, "source_split": args.split,
        "source_revision": _hub_revision(args.source),
        "detector": DETECTOR, "detector_revision": det.revision,
        "face_threshold": args.face_threshold, "part_threshold": args.part_threshold,
        "margin": args.margin, "size": args.size, "min_source_px": args.min_source_px,
        "dup_threshold": None if args.keep_duplicates else args.dup_threshold,
        "code_commit": _git_commit(), "n_source": len(ds), "n_kept": len(kept),
        "n_aligned": sum(r["aligned"] for r in kept), "n_rejected": len(rejected),
        "excluded_by_review": [],
    }
    (out / "build_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nkept {len(kept)} ({info['n_aligned']} aligned), rejected {len(rejected)} -> {out}")
    _sheets(out, kept, ds, rejected)


def _write_metadata(out, rows):
    with open(out / "metadata.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _sheets(out, kept, ds=None, rejected=None):
    from PIL import Image

    def cap(r):
        c = f"#{r['source_index']} {'A' if r['aligned'] else 'box'}"
        if r["aligned"] and abs(r["roll_degrees"]) >= 1:
            c += f" {r['roll_degrees']:+.0f}°"
        if r.get("outside_fraction", 0) > 0.05:
            c += f" pad{r['outside_fraction']:.0%}"
        return c

    contact_sheet([(Image.open(out / r["file_name"]), cap(r)) for r in kept],
                  out / "contact_sheet.jpg")
    if ds is not None and rejected:
        contact_sheet([(ds[r["source_index"]]["image"].convert("RGB"), f"#{r['source_index']}") for r in rejected],
                      out / "rejected_sheet.jpg")
    print(f"Review: {out / 'contact_sheet.jpg'}  (A = aligned on the eyes, box = upright crop)")


def push(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    out = Path(args.out)
    info = json.loads((out / "build_info.json").read_text())
    ds = load_dataset("imagefolder", data_dir=str(out), split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c == "label"])
    ds.push_to_hub(args.repo_id, private=not args.public, commit_message="Add cropped and aligned dog faces")
    api = HfApi()
    api.upload_file(path_or_fileobj=_card(args.repo_id, info, ds).encode(), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="dataset", commit_message="Dataset card")
    api.upload_file(path_or_fileobj=str(out / "build_info.json"), path_in_repo="build_info.json",
                    repo_id=args.repo_id, repo_type="dataset", commit_message="Build provenance")
    if (out / "contact_sheet.jpg").exists():
        api.upload_file(path_or_fileobj=str(out / "contact_sheet.jpg"), path_in_repo="contact_sheet.jpg",
                        repo_id=args.repo_id, repo_type="dataset", commit_message="Contact sheet")
    print(f"Pushed {len(ds)} images -> https://huggingface.co/datasets/{args.repo_id} "
          f"({'public' if args.public else 'private'})")


def _card(repo_id, info, ds):
    return f"""---
license: other
task_categories:
- unconditional-image-generation
tags:
- dogs
- faces
- aligned
source_datasets:
- {info['source_dataset']}
---

# {repo_id.split('/')[-1]}

Face crops of the dog in [{info['source_dataset']}](https://huggingface.co/datasets/{info['source_dataset']}),
centred and -- where both eyes were found -- rotated so the eyes are level, resized to
{info['size']}x{info['size']}.

![contact sheet](contact_sheet.jpg)

| | |
|---|---|
| Source photos | {info['n_source']} |
| Kept crops | {len(ds)} |
| Eye-aligned | {info['n_aligned']} (the rest are upright crops around the detected face box) |
| Rejected automatically | {info['n_rejected']} (no visible face, face too small, or a duplicate photo) |
| Removed on review | {len(info.get('excluded_by_review', []))} (`{info.get('excluded_by_review', [])}`) |

## How it was made

1. [{info['detector']}](https://huggingface.co/{info['detector']}) (Grounding DINO, zero-shot) prompted with
   `dog face. dog eye. dog nose.` (face threshold {info['face_threshold']}, parts {info['part_threshold']}).
2. A face box is kept only if an eye or nose detection lies inside it.
3. Eyes and nose are re-detected on the face region; with two eyes the photo is rotated so they are level
   (the nose decides which way is up, so sideways heads are corrected too).
4. A square of {info['margin']}x the face box's longer side, centred on the face, is cut from the full-resolution
   photo (mirror-padded where it passes the edge) and near-duplicate photos are dropped.

Code: [`scripts/data/make_face_dataset.py`](https://github.com/yar78/vision-generative-models-survey/blob/{info.get('code_commit') or 'main'}/scripts/data/make_face_dataset.py)
at commit `{info.get('code_commit')}`, reading `{info['source_dataset']}` at revision `{info.get('source_revision')}`.
Full parameters in `build_info.json`.

## Columns

| column | |
|---|---|
| `image` | the {info['size']}x{info['size']} face crop |
| `source_index` | row of the photo in `{info['source_dataset']}` |
| `face_score` | detector confidence for the face box |
| `eyes_found`, `nose_found` | landmarks found on the face |
| `aligned` | True if rotated to level the eyes |
| `roll_degrees` | rotation applied |
| `source_crop_px` | side of the square cut from the original photo (below {info['size']} means upscaled) |
| `outside_fraction` | share of the crop beyond the photo's edge, filled by mirror padding |

## Use

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", split="train")
```

In this repo's configs: `data: {{dataset: hf, hf_repo: {repo_id}}}`.
"""


def _hub_revision(repo_id):
    """The exact commit of the source dataset that was read."""
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo_id).sha
    except Exception:  # noqa: BLE001
        return None


def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="detect, align and crop faces into --out")
    b.add_argument("--source", default="yaryar78/Ray_dog")
    b.add_argument("--split", default="train")
    b.add_argument("--out", required=True)
    b.add_argument("--size", type=int, default=512)
    b.add_argument("--margin", type=float, default=1.3, help="crop side = margin x face box's longer side")
    b.add_argument("--face-threshold", type=float, default=0.3)
    b.add_argument("--part-threshold", type=float, default=0.25)
    b.add_argument("--min-source-px", type=int, default=128, help="reject faces smaller than this in the original")
    b.add_argument("--dup-threshold", type=float, default=0.02, help="mean abs difference of 32x32 crops below which two crops are duplicates")
    b.add_argument("--keep-duplicates", action="store_true")
    b.add_argument("--exclude", help="comma-separated source indices to drop after reviewing the contact sheet")
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--limit", type=int, help="only the first N photos (for a quick test)")
    b.add_argument("--device")
    b.add_argument("--cache-dir")
    b.set_defaults(func=build)
    q = sub.add_parser("push", help="upload --out to the Hub as a dataset")
    q.add_argument("--out", required=True)
    q.add_argument("--repo-id", required=True)
    q.add_argument("--public", action="store_true")
    q.set_defaults(func=push)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
