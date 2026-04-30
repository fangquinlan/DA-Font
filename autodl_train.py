#!/usr/bin/env python3
"""Prepare DA-Font data from local fonts/calligraphy images, then start training.

Default layout expected by this script:

    data_root/
      HK/ JP/ KR/ SC/ TC/        font files, already subsetted by cmap
      Shufa/<style_name>/<char>.jpg
      SourceHanSansHK-Regular.otf
      SourceHanSansJP-Regular.otf
      SourceHanSansKR-Regular.otf
      SourceHanSansSC-Regular.otf
      SourceHanSansTC-Regular.otf
      PlangothicP1-Regular.ttf
      PlangothicP2-Regular.ttf
      DA-Font/                   this repository

Run from the repository root:

    python autodl_train.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


FONT_EXTS = {".ttf", ".otf", ".ttc", ".otc"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
REGION_SOURCE_FONTS = {
    "HK": "SourceHanSansHK-Regular.otf",
    "JP": "SourceHanSansJP-Regular.otf",
    "KR": "SourceHanSansKR-Regular.otf",
    "SC": "SourceHanSansSC-Regular.otf",
    "TC": "SourceHanSansTC-Regular.otf",
}
DEFAULT_REGION_ORDER = ("HK", "JP", "KR", "SC", "TC")
UNSAFE_FILENAME_CHARS = set('/\\:*?"<>|')
_PIL_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


@dataclass
class StyleSource:
    name: str
    kind: str
    region: str
    path: Path
    chars: list[str] = field(default_factory=list)
    rendered: int = 0
    skipped: int = 0


class Progress:
    def __init__(self, iterable: Iterable, desc: str, total: int | None = None):
        try:
            from tqdm import tqdm

            self.iterable = tqdm(iterable, desc=desc, total=total)
        except Exception:
            self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_data_root(root: Path) -> Path:
    candidates = [root.parent, root]
    for candidate in candidates:
        if (candidate / "Shufa").exists() or any((candidate / r).exists() for r in REGION_SOURCE_FONTS):
            return candidate
    return root.parent


def safe_style_name(*parts: str) -> str:
    raw = "__".join(part for part in parts if part)
    raw = unicodedata.normalize("NFKC", raw)
    out = []
    for ch in raw:
        if ch in UNSAFE_FILENAME_CHARS or ord(ch) < 32:
            out.append("_")
        elif ch.isspace():
            out.append("_")
        else:
            out.append(ch)
    name = "".join(out).strip(" ._")
    return name or "style"


def unicode_hex(ch: str) -> str:
    return f"{ord(ch):04X}"


def is_safe_character(ch: str) -> bool:
    if len(ch) != 1:
        return False
    if ch in UNSAFE_FILENAME_CHARS or ch in {".", " "}:
        return False
    cat = unicodedata.category(ch)
    return not cat.startswith("C")


def font_files_in(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in FONT_EXTS)


def image_files_in(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def require_module(name: str, package_hint: str | None = None) -> None:
    try:
        __import__(name)
    except ImportError as exc:
        hint = package_hint or name
        raise SystemExit(
            f"Missing dependency `{name}`. Install it first, for example: "
            f"pip install {hint} or pip install -r requirements-autodl.txt"
        ) from exc


def check_dependencies() -> None:
    for module, package in [
        ("fontTools", "fonttools"),
        ("lmdb", "lmdb"),
        ("torch", None),
        ("torchvision", None),
        ("sconf", "sconf"),
        ("cv2", "opencv-python"),
        ("einops", "einops"),
    ]:
        require_module(module, package)


def open_ttfont(font_path: Path):
    from fontTools.ttLib import TTFont

    return TTFont(str(font_path), lazy=True)


def glyph_has_ink(ttfont, ch: str) -> bool:
    from fontTools.pens.boundsPen import BoundsPen

    try:
        cmap = ttfont.getBestCmap() or {}
        glyph_name = cmap.get(ord(ch))
        if not glyph_name or glyph_name == ".notdef":
            return False
        glyph_set = ttfont.getGlyphSet()
        glyph = glyph_set[glyph_name]
        pen = BoundsPen(glyph_set)
        glyph.draw(pen)
        return pen.bounds is not None
    except Exception:
        # Some font formats do not draw cleanly through fontTools. Rendering
        # below still catches blank glyphs, so do not drop the character here.
        return True


def chars_from_font(font_path: Path, limit: int | None = None) -> list[str]:
    ttfont = open_ttfont(font_path)
    try:
        cmap = ttfont.getBestCmap() or {}
        chars = []
        for codepoint in sorted(cmap):
            try:
                ch = chr(codepoint)
            except ValueError:
                continue
            if not is_safe_character(ch):
                continue
            if glyph_has_ink(ttfont, ch):
                chars.append(ch)
            if limit and len(chars) >= limit:
                break
        return chars
    finally:
        ttfont.close()


def cmap_chars_from_font(font_path: Path) -> set[str]:
    ttfont = open_ttfont(font_path)
    try:
        cmap = ttfont.getBestCmap() or {}
        return {chr(codepoint) for codepoint in cmap if is_safe_character(chr(codepoint))}
    finally:
        ttfont.close()


def foreground_bbox(img: Image.Image, bg: int) -> tuple[int, int, int, int] | None:
    arr = np.asarray(img.convert("L"))
    if bg >= 128:
        mask = arr < 245
    else:
        mask = arr > 10
    if int(mask.sum()) < 8:
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fit_to_square(img: Image.Image, size: int, pad: int, bg: int) -> Image.Image | None:
    bbox = foreground_bbox(img, bg)
    if bbox is None:
        return None
    img = img.crop(bbox)
    inner = max(1, size - 2 * pad)
    ratio = min(inner / img.width, inner / img.height)
    new_size = (max(1, int(round(img.width * ratio))), max(1, int(round(img.height * ratio))))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    out = Image.new("L", (size, size), color=bg)
    out.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return out


def get_pil_font(font_path: Path, font_size: int) -> ImageFont.FreeTypeFont:
    key = (str(font_path), font_size)
    font = _PIL_FONT_CACHE.get(key)
    if font is None:
        font = ImageFont.truetype(str(font_path), font_size)
        _PIL_FONT_CACHE[key] = font
    return font


def render_font_char(font_path: Path, ch: str, size: int, pad: int) -> Image.Image | None:
    canvas_size = size * 4
    font_size = int(size * 1.65)
    try:
        font = get_pil_font(font_path, font_size)
    except Exception as exc:
        raise RuntimeError(f"Cannot load font {font_path}") from exc

    canvas = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(canvas)
    try:
        bbox = draw.textbbox((0, 0), ch, font=font)
    except Exception:
        return None
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width <= 0 or height <= 0:
        return None
    x = (canvas_size - width) / 2 - bbox[0]
    y = (canvas_size - height) / 2 - bbox[1]
    draw.text((x, y), ch, font=font, fill=0)
    return fit_to_square(canvas, size=size, pad=pad, bg=255)


def process_shufa_image(src: Path, size: int, pad: int, invert: bool) -> Image.Image | None:
    try:
        img = Image.open(src).convert("L")
    except Exception:
        return None
    # Source images are expected to be black background with white characters.
    img = img.point(lambda p: 255 if p >= 128 else 0)
    out = fit_to_square(img, size=size, pad=pad, bg=0)
    if out is None:
        return None
    if invert:
        out = ImageOps.invert(out)
    return out


def run_parallel(jobs: list[tuple], worker_fn, desc: str, workers: int, chunksize: int):
    if workers <= 1 or len(jobs) <= 1:
        for job in Progress(jobs, desc, total=len(jobs)):
            yield worker_fn(job)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(worker_fn, jobs, chunksize=chunksize)
        for result in Progress(results, desc, total=len(jobs)):
            yield result


def render_worker_count(args) -> int:
    if args.render_workers is not None:
        return max(1, args.render_workers)
    cpu_count = os.cpu_count() or 2
    return max(1, min(12, cpu_count // 2))


def _render_content_job(job):
    ch, font_paths, out_dir, size, pad = job
    for raw_font_path in font_paths:
        img = render_font_char(Path(raw_font_path), ch, size, pad)
        if img is None:
            continue
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        img.save(Path(out_dir) / f"{ch}.png")
        return ch, True
    return ch, False


def _render_style_font_job(job):
    ch, font_path, out_dir, size, pad = job
    img = render_font_char(Path(font_path), ch, size, pad)
    if img is None:
        return ch, False
    img.save(Path(out_dir) / f"{ch}.png")
    return ch, True


def _render_style_shufa_job(job):
    ch, src_path, out_dir, size, pad, invert = job
    img = process_shufa_image(Path(src_path), size, pad, invert=invert)
    if img is None:
        return ch, False
    img.save(Path(out_dir) / f"{ch}.png")
    return ch, True


def choose_content_font(
    ch: str,
    sources_by_char: dict[str, set[str]],
    source_fonts: dict[str, Path],
    fallback_fonts: list[Path],
) -> Path | None:
    preferred = list(DEFAULT_REGION_ORDER)
    sources = sources_by_char.get(ch, set())
    if "Shufa" in sources:
        preferred = ["HK", *[r for r in DEFAULT_REGION_ORDER if r != "HK"]]
    elif sources:
        preferred = [r for r in DEFAULT_REGION_ORDER if r in sources] + [
            r for r in DEFAULT_REGION_ORDER if r not in sources
        ]

    for region in preferred:
        font_path = source_fonts.get(region)
        if font_path and font_path.exists():
            try:
                ttfont = open_ttfont(font_path)
                ok = glyph_has_ink(ttfont, ch)
                ttfont.close()
            except Exception:
                ok = True
            if ok:
                return font_path

    for font_path in fallback_fonts:
        if font_path.exists():
            try:
                ttfont = open_ttfont(font_path)
                ok = glyph_has_ink(ttfont, ch)
                ttfont.close()
            except Exception:
                ok = True
            if ok:
                return font_path
    return None


def discover_sources(args) -> tuple[list[StyleSource], dict[str, set[str]]]:
    data_root = Path(args.data_root).resolve()
    sources: list[StyleSource] = []
    sources_by_char: dict[str, set[str]] = {}

    region_dirs = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir() or child.name in {"DA-Font", ".git", "__pycache__", "Shufa"}:
            continue
        if font_files_in(child):
            region_dirs.append(child)

    for region_dir in region_dirs:
        region = region_dir.name.upper()
        for font_path in font_files_in(region_dir):
            chars = chars_from_font(font_path, limit=args.limit_chars_per_font)
            if not chars:
                continue
            name = safe_style_name(region, font_path.stem)
            source = StyleSource(name=name, kind="font", region=region, path=font_path, chars=chars)
            sources.append(source)
            for ch in chars:
                sources_by_char.setdefault(ch, set()).add(region)

    shufa_root = data_root / "Shufa"
    if shufa_root.exists():
        by_style: dict[Path, list[Path]] = {}
        for img_path in image_files_in(shufa_root):
            stem = img_path.stem
            if len(stem) != 1 or not is_safe_character(stem):
                continue
            by_style.setdefault(img_path.parent, []).append(img_path)

        for style_dir, files in sorted(by_style.items(), key=lambda item: str(item[0])):
            rel = style_dir.relative_to(shufa_root)
            name = safe_style_name("Shufa", *rel.parts)
            chars = sorted({p.stem for p in files})
            if args.limit_chars_per_font:
                chars = chars[: args.limit_chars_per_font]
            source = StyleSource(name=name, kind="shufa", region="HK", path=style_dir, chars=chars)
            sources.append(source)
            for ch in chars:
                sources_by_char.setdefault(ch, set()).add("Shufa")

    if args.limit_fonts:
        sources = sources[: args.limit_fonts]
        kept = {s.name for s in sources}
        sources_by_char = {}
        for source in sources:
            marker = "Shufa" if source.kind == "shufa" else source.region
            for ch in source.chars:
                sources_by_char.setdefault(ch, set()).add(marker)
        print(f"Limited sources to {len(kept)} style(s) for testing.")

    return sources, sources_by_char


def render_content_images(
    chars: list[str],
    sources_by_char: dict[str, set[str]],
    content_dir: Path,
    data_root: Path,
    args,
) -> set[str]:
    content_dir.mkdir(parents=True, exist_ok=True)
    source_fonts = {
        region: data_root / filename
        for region, filename in REGION_SOURCE_FONTS.items()
        if (data_root / filename).exists()
    }
    fallback_fonts = [
        data_root / "PlangothicP1-Regular.ttf",
        data_root / "PlangothicP2-Regular.ttf",
    ]
    fallback_fonts = [p for p in fallback_fonts if p.exists()]

    if "HK" not in source_fonts:
        raise SystemExit(f"Missing required content font: {data_root / REGION_SOURCE_FONTS['HK']}")
    if not fallback_fonts:
        print("Warning: no Plangothic fallback fonts found.")

    all_candidate_fonts = list(source_fonts.values()) + fallback_fonts
    coverage_by_font = {font_path: cmap_chars_from_font(font_path) for font_path in all_candidate_fonts}
    worker_count = render_worker_count(args)
    print(f"Rendering content glyphs with {worker_count} process(es).")

    jobs = []
    for ch in chars:
        preferred = list(DEFAULT_REGION_ORDER)
        sources = sources_by_char.get(ch, set())
        if "Shufa" in sources:
            preferred = ["HK", *[r for r in DEFAULT_REGION_ORDER if r != "HK"]]
        elif sources:
            preferred = [r for r in DEFAULT_REGION_ORDER if r in sources] + [
                r for r in DEFAULT_REGION_ORDER if r not in sources
            ]

        font_paths = []
        for region in preferred:
            font_path = source_fonts.get(region)
            if font_path and ch in coverage_by_font.get(font_path, set()):
                font_paths.append(str(font_path))
        for font_path in fallback_fonts:
            if ch in coverage_by_font.get(font_path, set()):
                font_paths.append(str(font_path))
        if font_paths:
            jobs.append((ch, font_paths, str(content_dir), args.size, args.padding))

    rendered: set[str] = set()
    for ch, ok in run_parallel(jobs, _render_content_job, "render content", worker_count, args.render_chunksize):
        if ok:
            rendered.add(ch)

    skipped = len(chars) - len(rendered)
    if skipped:
        print(f"Skipped {skipped} content glyph(s) because SourceHanSans/Plangothic rendered empty.")
    if len(rendered) < max(4, args.kshot):
        raise SystemExit(f"Only {len(rendered)} content glyphs were rendered; need at least {max(4, args.kshot)}.")
    return rendered


def render_style_images(
    sources: list[StyleSource],
    valid_chars: set[str],
    train_dir: Path,
    args,
) -> list[StyleSource]:
    train_dir.mkdir(parents=True, exist_ok=True)
    kept: list[StyleSource] = []
    worker_count = render_worker_count(args)
    print(f"Rendering style glyphs with {worker_count} process(es).")

    for source in Progress(sources, "render styles"):
        out_dir = train_dir / source.name
        out_dir.mkdir(parents=True, exist_ok=True)
        chars = [ch for ch in source.chars if ch in valid_chars]

        if source.kind == "font":
            jobs = [(ch, str(source.path), str(out_dir), args.size, args.padding) for ch in chars]
            results = run_parallel(jobs, _render_style_font_job, f"render {source.name}", worker_count, args.render_chunksize)
            for _ch, ok in results:
                if ok:
                    source.rendered += 1
                else:
                    source.skipped += 1
        else:
            image_by_char = {p.stem: p for p in image_files_in(source.path) if p.stem in chars}
            jobs = []
            for ch in chars:
                src = image_by_char.get(ch)
                if src is None:
                    source.skipped += 1
                    continue
                jobs.append((ch, str(src), str(out_dir), args.size, args.padding, args.invert_shufa))
            results = run_parallel(jobs, _render_style_shufa_job, f"render {source.name}", worker_count, args.render_chunksize)
            for _ch, ok in results:
                if ok:
                    source.rendered += 1
                else:
                    source.skipped += 1

        if source.rendered >= args.kshot:
            kept.append(source)
        else:
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"Skipped style `{source.name}` because it rendered only {source.rendered} glyph(s).")

    if not kept:
        raise SystemExit("No usable style fonts/images were rendered.")
    return kept


def make_validation_mirror(train_dir: Path, val_dir: Path, sources: list[StyleSource], args) -> None:
    val_dir.mkdir(parents=True, exist_ok=True)
    candidates = [s for s in sources if s.rendered >= args.kshot]
    if not candidates:
        raise SystemExit("No style has enough glyphs for validation.")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    count = min(args.max_val_fonts, len(candidates))
    selected = candidates[:count]

    for source in selected:
        src = train_dir / source.name
        dst = val_dir / safe_style_name("VAL", source.name)
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if args.copy_validation:
            shutil.copytree(src, dst)
        else:
            try:
                os.symlink(src, dst, target_is_directory=True)
            except OSError:
                shutil.copytree(src, dst)

    print(f"Validation mirrors: {count} style(s). All original styles remain in training.")


def split_unicodes(unicodes: list[str], holdout_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if holdout_ratio <= 0:
        return unicodes, unicodes
    rng = random.Random(seed)
    shuffled = list(unicodes)
    rng.shuffle(shuffled)
    n_val = max(1, int(math.ceil(len(shuffled) * holdout_ratio)))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    if len(train) < 4:
        raise SystemExit("Character holdout left fewer than 4 training glyphs; lower --holdout-char-ratio.")
    return train, val


def build_lmdb_and_meta(prepared_dir: Path, content_dir: Path, train_dir: Path, val_dir: Path, args) -> None:
    os.environ["DAFONT_LMDB_MAP_SIZE_GB"] = str(args.lmdb_map_size_gb)
    from build_dataset.build_meta4train import build_meta4train_lmdb, build_train_meta

    meta_dir = prepared_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    all_unicodes = sorted(unicode_hex(p.stem) for p in content_dir.glob("*.png") if len(p.stem) == 1)
    seen_unis, unseen_unis = split_unicodes(all_unicodes, args.holdout_char_ratio, args.seed)

    all_path = meta_dir / "all_content_unis.json"
    seen_path = meta_dir / "train_unis.json"
    unseen_path = meta_dir / "val_unis.json"
    all_path.write_text(json.dumps(all_unicodes, ensure_ascii=False, indent=2), encoding="utf-8")
    seen_path.write_text(json.dumps(seen_unis, ensure_ascii=False, indent=2), encoding="utf-8")
    unseen_path.write_text(json.dumps(unseen_unis, ensure_ascii=False, indent=2), encoding="utf-8")

    ns = SimpleNamespace(
        saving_dir=str(prepared_dir),
        content_font=str(content_dir),
        train_font_dir=str(train_dir),
        val_font_dir=str(val_dir),
        seen_unis_file=str(seen_path),
        unseen_unis_file=str(unseen_path),
    )
    build_meta4train_lmdb(ns)
    build_train_meta(ns)


def load_content_encoder(device: str, vae_path: Path):
    import torch
    from model import content_enc_builder

    encoder = content_enc_builder(1, 32, 256)
    state = torch.load(str(vae_path), map_location="cpu")
    encoder_state = {}
    for key, value in state.items():
        if key.startswith("_encoder."):
            encoder_state[key[len("_encoder.") :]] = value
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        print(f"Warning: unexpected encoder keys: {unexpected}")
    if missing:
        print(f"Warning: missing encoder keys: {missing}")
    encoder.to(device)
    encoder.eval()
    return encoder


def similarity_path(prepared_dir: Path, args) -> Path:
    if args.full_similarity_json:
        return (prepared_dir / "meta" / "all_char_similarity_unicode.json").resolve()
    return (prepared_dir / "meta" / "content_similarity_features.pt").resolve()


def write_similarity(content_dir: Path, sim_path: Path, vae_path: Path, args) -> None:
    import torch
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() and not args.cpu_similarity else "cpu"
    encoder = load_content_encoder(device, vae_path)
    transform = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor()])

    files = sorted(p for p in content_dir.glob("*.png") if len(p.stem) == 1)
    if not files:
        raise SystemExit("No content images found for similarity computation.")

    features = []
    names = []
    batch_imgs = []
    batch_names = []

    with torch.no_grad():
        for path in Progress(files, "encode content"):
            img = transform(Image.open(path).convert("L")) - 0.5
            batch_imgs.append(img)
            batch_names.append(unicode_hex(path.stem))
            if len(batch_imgs) >= args.sim_batch_size:
                tensor = torch.stack(batch_imgs).to(device)
                feat = encoder(tensor)
                if args.sim_pool_size > 0:
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat, (args.sim_pool_size, args.sim_pool_size))
                feat = feat.flatten(1)
                features.append(feat.detach().cpu())
                names.extend(batch_names)
                batch_imgs.clear()
                batch_names.clear()
        if batch_imgs:
            tensor = torch.stack(batch_imgs).to(device)
            feat = encoder(tensor)
            if args.sim_pool_size > 0:
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (args.sim_pool_size, args.sim_pool_size))
            feat = feat.flatten(1)
            features.append(feat.detach().cpu())
            names.extend(batch_names)

    matrix = torch.cat(features, dim=0).float()
    matrix = torch.nn.functional.normalize(matrix, dim=1)
    sim_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.full_similarity_json:
        print(f"Writing compact similarity features for {len(names)} glyphs...")
        torch.save({"names": names, "features": matrix.half()}, str(sim_path))
        return

    matrix_for_matmul = matrix.to(device) if device == "cuda" else matrix
    print(f"Writing similarity JSON for {len(names)} glyphs...")
    with sim_path.open("w", encoding="utf-8") as fout:
        fout.write("{")
        first_row = True
        for start in range(0, len(names), args.sim_row_chunk):
            chunk = matrix_for_matmul[start : start + args.sim_row_chunk] @ matrix_for_matmul.T
            chunk = chunk.clamp(min=-1.0, max=1.0).detach().cpu().numpy()
            for row_idx, row in enumerate(chunk):
                if not first_row:
                    fout.write(",")
                first_row = False
                name = names[start + row_idx]
                values = {names[col]: round(float(row[col]), 6) for col in range(len(names))}
                values[name] = 1.0
                fout.write("\n")
                fout.write(json.dumps(name, ensure_ascii=False))
                fout.write(":")
                fout.write(json.dumps(values, ensure_ascii=False, separators=(",", ":")))
        fout.write("\n}\n")


def write_config(config_path: Path, prepared_dir: Path, content_dir: Path, args) -> None:
    workers = args.workers
    if workers is None:
        workers = max(1, min((os.cpu_count() or 2) // 2, 16))

    vae_path = (repo_root() / "pretrained_weights" / "VQ-VAE_Parms_chn_.pth").resolve()
    sim_path = similarity_path(prepared_dir, args)
    work_dir = (repo_root() / "results" / args.task_name).resolve()
    data_path = (prepared_dir / "lmdb").resolve()
    data_meta = (prepared_dir / "meta" / "train.json").resolve()
    all_content_json = (prepared_dir / "meta" / "all_content_unis.json").resolve()

    text = f"""use_half: False
use_ddp: False

vae_pth: "{vae_path.as_posix()}"
sim_path: "{sim_path.as_posix()}"
work_dir: "{work_dir.as_posix()}"
data_path: "{data_path.as_posix()}"
data_meta: "{data_meta.as_posix()}"
all_content_json: "{all_content_json.as_posix()}"
content_font: "{content_dir.name}"

num_embeddings: 100
vae_batch_size: 256
vae_lr: 1e-3
vae_iter: 10000

input_size: {args.size}
num_heads: 8
kshot: {args.kshot}
num_positive_samples: 2

batch_size: {args.batch_size}
n_workers: {workers}
prefetch_factor: {args.prefetch_factor}
iter: {args.iterations}
g_lr: {args.g_lr}
d_lr: {args.d_lr}
step_size: {args.step_size}
gamma: {args.gamma}
overwrite: False
adam_betas: [0.0, 0.9]

cv_n_unis: {args.cv_n_unis}
cv_n_fonts: {min(args.cv_n_fonts, args.max_val_fonts)}

print_freq: {args.print_freq}
val_freq: {args.val_freq}
save_freq: {args.val_freq}
tb_freq: {args.tb_freq}
progress_freq: {args.progress_freq}
save: last-best
"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")


def prepare(args) -> tuple[Path, Path]:
    check_dependencies()

    root = repo_root()
    data_root = Path(args.data_root).resolve()
    prepared_dir = (root / args.prepared_dir).resolve()
    images_root = prepared_dir / "images"
    content_dir = images_root / "content" / "content_SourceHanSansComposite"
    train_dir = images_root / "train"
    val_dir = images_root / "val"
    config_path = prepared_dir / "autodl_config.yaml"
    sim_path = similarity_path(prepared_dir, args)

    if prepared_dir.exists() and not args.force:
        has_lmdb_meta = (prepared_dir / "lmdb").exists() and (prepared_dir / "meta" / "train.json").exists()
        has_rendered_content = content_dir.exists() and any(content_dir.glob("*.png"))
        if has_lmdb_meta and has_rendered_content:
            print(f"Reusing rendered images/LMDB/meta at {prepared_dir}. Use --force to rebuild.")
            vae_path = (root / "pretrained_weights" / "VQ-VAE_Parms_chn_.pth").resolve()
            if not sim_path.exists():
                write_similarity(content_dir, sim_path, vae_path, args)
            write_config(config_path, prepared_dir, content_dir, args)
            return prepared_dir, config_path

    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    images_root.mkdir(parents=True, exist_ok=True)

    sources, sources_by_char = discover_sources(args)
    print(f"Discovered {len(sources)} style source(s).")
    if not sources:
        raise SystemExit(f"No font files or Shufa images found under {data_root}")

    all_chars = sorted(sources_by_char.keys(), key=lambda ch: (ord(ch), ch))
    valid_chars = render_content_images(all_chars, sources_by_char, content_dir, data_root, args)
    kept_sources = render_style_images(sources, valid_chars, train_dir, args)
    make_validation_mirror(train_dir, val_dir, kept_sources, args)
    build_lmdb_and_meta(prepared_dir, content_dir, train_dir, val_dir, args)

    vae_path = (root / "pretrained_weights" / "VQ-VAE_Parms_chn_.pth").resolve()
    sim_path = similarity_path(prepared_dir, args)
    write_similarity(content_dir, sim_path, vae_path, args)
    write_config(config_path, prepared_dir, content_dir, args)

    manifest = {
        "data_root": str(data_root),
        "prepared_dir": str(prepared_dir),
        "styles": [
            {
                "name": s.name,
                "kind": s.kind,
                "region": s.region,
                "source": str(s.path),
                "rendered": s.rendered,
                "skipped": s.skipped,
            }
            for s in kept_sources
        ],
        "content_glyphs": len(valid_chars),
        "config": str(config_path),
    }
    (prepared_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return prepared_dir, config_path


def configure_runtime(args) -> None:
    workers = args.workers if args.workers is not None else max(1, min((os.cpu_count() or 2) // 2, 16))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, workers)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, workers)))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def start_training(config_path: Path, args) -> None:
    if args.batch_size < 6:
        raise SystemExit("DA-Font contrastive loss requires --batch-size >= 6.")

    root = repo_root()
    ckpt = root / "results" / args.task_name / "checkpoints" / args.task_name / "latest.pth"
    cmd = [sys.executable, "train.py", args.task_name, str(config_path)]
    if args.resume == "auto" and ckpt.exists():
        cmd.extend(["--resume", str(ckpt)])
    elif args.resume and args.resume != "auto":
        cmd.extend(["--resume", args.resume])

    configure_runtime(args)
    print("Starting training:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(root), check=True)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Prepare DA-Font data and start AutoDL training.")
    parser.add_argument("--data-root", default=str(default_data_root(root)), help="Root containing HK/JP/KR/SC/TC/Shufa and SourceHanSans fonts.")
    parser.add_argument("--prepared-dir", default="prepared_autodl", help="Output directory for rendered images, LMDB, meta, config.")
    parser.add_argument("--task-name", default="autodl_da_font", help="Training run name and result subdirectory.")
    parser.add_argument("--force", action="store_true", help="Rebuild rendered images/LMDB/meta even if they already exist.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip data preparation and use the existing generated config.")
    parser.add_argument("--no-train", action="store_true", help="Prepare data only; do not launch train.py.")
    parser.add_argument("--resume", default="auto", help="auto, a checkpoint path, or empty string.")

    parser.add_argument("--size", type=int, default=128, help="Glyph image size.")
    parser.add_argument("--padding", type=int, default=12, help="Glyph padding inside the square image.")
    parser.add_argument("--render-workers", type=int, default=None, help="Parallel processes for glyph/image rendering. Default uses up to 12.")
    parser.add_argument("--render-chunksize", type=int, default=16, help="Chunk size for parallel rendering jobs.")
    parser.add_argument("--invert-shufa", action="store_true", help="Invert Shufa images to black glyphs on white background.")
    parser.add_argument("--copy-validation", action="store_true", help="Copy validation mirrors instead of using directory symlinks.")
    parser.add_argument("--max-val-fonts", type=int, default=8, help="Number of mirrored validation style folders.")
    parser.add_argument("--holdout-char-ratio", type=float, default=0.0, help="Optional character holdout ratio. Default trains on all chars.")
    parser.add_argument("--lmdb-map-size-gb", type=float, default=64, help="LMDB map size in GiB. Raise this if LMDB reports MapFullError.")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers. Default uses up to half of CPU cores, capped at 16.")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=0, help="0 means train indefinitely.")
    parser.add_argument("--kshot", type=int, default=4)
    parser.add_argument("--g-lr", type=float, default=2e-4)
    parser.add_argument("--d-lr", type=float, default=4e-4)
    parser.add_argument("--step-size", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--print-freq", type=int, default=1000)
    parser.add_argument("--val-freq", type=int, default=5000)
    parser.add_argument("--progress-freq", type=int, default=50, help="Lightweight training heartbeat frequency in steps.")
    parser.add_argument("--tb-freq", type=int, default=100)
    parser.add_argument("--cv-n-unis", type=int, default=20)
    parser.add_argument("--cv-n-fonts", type=int, default=8)
    parser.add_argument("--cuda-visible-devices", default="0")

    parser.add_argument("--sim-batch-size", type=int, default=128)
    parser.add_argument("--sim-row-chunk", type=int, default=128)
    parser.add_argument("--sim-pool-size", type=int, default=4, help="Pool encoder features before similarity. 4 is compact; 0 keeps full features.")
    parser.add_argument("--cpu-similarity", action="store_true", help="Compute similarity on CPU even if CUDA is available.")
    parser.add_argument("--full-similarity-json", action="store_true", help="Write the original full NxN similarity JSON. Not recommended for large character sets.")

    parser.add_argument("--limit-fonts", type=int, default=None, help="Testing only: limit number of style sources.")
    parser.add_argument("--limit-chars-per-font", type=int, default=None, help="Testing only: limit glyphs per source.")
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    prepared_dir = (repo_root() / args.prepared_dir).resolve()
    config_path = prepared_dir / "autodl_config.yaml"
    if args.skip_prepare:
        if not config_path.exists():
            raise SystemExit(f"Generated config not found: {config_path}")
    else:
        prepared_dir, config_path = prepare(args)

    print(f"Prepared data: {prepared_dir}")
    print(f"Generated config: {config_path}")

    if not args.no_train:
        start_training(config_path, args)


if __name__ == "__main__":
    main()
