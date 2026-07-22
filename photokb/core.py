#!/usr/bin/env python3
"""PhotoKB analysis engine — CLIP tagging, EXIF, colours, notes, index, search.

Pipeline:  photos -> local CLIP analysis + EXIF -> Markdown notes -> Obsidian
           (+ CLIP embeddings for semantic search).  Everything runs locally.

This module holds the library functions; the ``pkb`` command (see cli.py) drives
them. Requires a Python environment with torch + open_clip installed.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration  (edit these, or override with --source / --vault / env vars)
# ---------------------------------------------------------------------------
DEFAULT_SOURCES = [Path.home() / "Documents" / "Photos"]
DEFAULT_VAULT = Path.home() / "Documents" / "PhotoKB"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}

MODEL_NAME = os.environ.get("PHOTOKB_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("PHOTOKB_PRETRAINED", "laion2b_s34b_b79k")

DISPLAY_MAX_EDGE = 2560     # web-sized copy longest edge (keeps Obsidian snappy)
JPEG_QUALITY = 87

# Zero-shot decision thresholds. Cosine similarities come from L2-normalised
# ViT-B-32/laion2b embeddings; matching concepts typically land ~0.22–0.32.
# If tags feel too sparse/noisy, run `pkb.py debug <name>` and adjust these.
OBJECT_MIN_COS = 0.215       # a subject must score at least this to be tagged
OBJECT_REL_DELTA = 0.045     # ...or be within this of the top subject
OBJECT_TOP_K = 6
SCENE_MIN_COS = 0.215
SCENE_TOP_K = 2
STYLE_MIN_COS = 0.225
STYLE_TOP_K = 2
TIME_MIN_COS = 0.210
WEATHER_MIN_COS = 0.225
WEATHER_MIN_MARGIN = 0.012   # weather is unreliable -> require confidence
SEASON_MIN_COS = 0.235
SEASON_MIN_MARGIN = 0.015    # season is least reliable -> strictest

PHOTOS_DIR, ASSETS_DIR, ALBUMS_DIR, DASH_DIR, DATA_DIR = (
    "Photos", "Assets", "Albums", "Dashboards", ".photokb")

# ---------------------------------------------------------------------------
# Taxonomy: display name -> CLIP prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = ["a photo of {}.", "a picture of {}.", "a snapshot of {}.", "{}."]

OBJECTS = {
    "mountain": "a mountain", "hill": "a hill", "valley": "a valley",
    "river": "a river", "lake": "a lake", "sea": "the sea, an ocean",
    "waterfall": "a waterfall", "beach": "a sandy beach",
    "coastline": "a coastline", "forest": "a forest of trees", "tree": "a tree",
    "flowers": "flowers", "grass": "a field of grass", "rocks": "rocks",
    "cliffs": "cliffs", "sand": "sand", "snow": "snow", "ice": "ice",
    "glacier": "a glacier", "sky": "the sky", "clouds": "clouds",
    "sun": "the bright sun", "moon": "the moon in the sky",
    "stars": "a sky full of stars", "rainbow": "a rainbow",
    "water reflection": "reflections on water", "building": "a building",
    "house": "a house", "skyscraper": "a skyscraper", "church": "a church",
    "temple": "a temple", "castle": "a castle", "bridge": "a bridge",
    "road": "a road", "tower": "a tower", "ruins": "old ruins",
    "monument": "a monument", "harbor": "a harbor", "pier": "a pier",
    "boat": "a boat", "ship": "a ship", "sailboat": "a sailboat", "car": "a car",
    "bicycle": "a bicycle", "motorcycle": "a motorcycle", "train": "a train",
    "airplane": "an airplane", "person": "a person",
    "people": "a group of people", "child": "a child",
    "crowd": "a large crowd of people", "dog": "a dog", "cat": "a cat",
    "bird": "a bird", "horse": "a horse", "animal": "an animal",
    "food": "a plate of food", "drink": "a drink", "coffee": "a cup of coffee",
    "bottle": "a bottle", "table": "a table", "furniture": "furniture",
    "chair": "a chair", "window": "a window", "door": "a door",
    "kitchen": "a kitchen", "sink": "a sink", "appliance": "a kitchen appliance",
    "tableware": "cups and dishes", "painting": "a painting on the wall",
    "book": "books", "plant": "a plant",
}
SCENES = {
    "landscape": "a wide landscape", "mountain landscape": "a mountain landscape",
    "seascape": "a seascape of open water", "coastal": "a coastal scene by the sea",
    "cityscape": "a cityscape", "urban": "an urban street scene",
    "countryside": "a rural countryside scene", "village": "a small village",
    "forest": "a forest scene", "garden": "a garden", "beach": "a beach scene",
    "harbor": "a harbor with boats", "indoor": "an indoor scene",
    "interior": "a room interior", "night scene": "a scene at night",
    "aerial": "an aerial view from above", "close-up": "a close-up photograph",
    "portrait": "a portrait of a person", "still life": "a still life arrangement",
}
SETTING = {"indoor": "an indoor scene inside a building",
           "outdoor": "an outdoor scene outside"}
TIME_OF_DAY = {
    "daytime": "a scene in bright daylight", "night": "a scene at night in the dark",
    "sunset": "a scene at sunset with warm colors", "sunrise": "a scene at sunrise",
    "golden hour": "a scene during golden hour", "dusk": "a scene at dusk after sunset",
}
WEATHER = {
    "sunny": "a sunny day with clear blue sky", "cloudy": "a cloudy day with clouds",
    "overcast": "an overcast grey day", "foggy": "a foggy misty scene",
    "rainy": "a rainy day", "snowy": "a snowy scene covered in snow",
}
SEASON = {
    "spring": "a spring scene with fresh green leaves and blossoms",
    "summer": "a bright green summer scene",
    "autumn": "an autumn scene with orange and red fall foliage",
    "winter": "a cold winter scene with bare trees or snow",
}
STYLES = {
    "landscape": "landscape photography", "portrait": "portrait photography",
    "street": "street photography", "macro": "macro close-up photography",
    "architecture": "architecture photography", "wildlife": "wildlife photography",
    "astro": "astrophotography of the night sky", "travel": "travel photography",
    "black and white": "a black and white photograph", "food": "food photography",
    "interior": "interior photography of a room",
}
CATEGORIES = {"objects": OBJECTS, "scene": SCENES, "setting": SETTING,
              "time_of_day": TIME_OF_DAY, "weather": WEATHER, "season": SEASON,
              "style": STYLES}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# ---------------------------------------------------------------------------
# CLIP engine
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, verbose: bool = True):
        import torch
        import open_clip
        self.torch = torch
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        if verbose:
            print(f"[clip] loading {MODEL_NAME}/{PRETRAINED} on {self.device} ...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self._banks: dict[str, tuple[list[str], object]] = {}
        self._build_banks()
        self.dim = next(iter(self._banks.values()))[1].shape[1]

    def _encode_text(self, prompts):
        with self.torch.no_grad():
            toks = self.tokenizer(prompts).to(self.device)
            feats = self.model.encode_text(toks)
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats

    def _build_banks(self):
        for cat, mapping in CATEGORIES.items():
            names, vecs = list(mapping.keys()), []
            for name in names:
                variants = [t.format(mapping[name]) for t in PROMPT_TEMPLATES]
                emb = self._encode_text(variants).mean(dim=0)
                emb /= emb.norm()
                vecs.append(emb)
            self._banks[cat] = (names, self.torch.stack(vecs))

    def encode_tensors(self, tensors):
        import numpy as np
        if not tensors:
            return np.zeros((0, self.dim), np.float32)
        with self.torch.no_grad():
            x = self.torch.stack(tensors).to(self.device)
            feats = self.model.encode_image(x)
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    def encode_query(self, text: str):
        return self._encode_text([text]).float().cpu().numpy()[0]

    def score(self, image_emb, category):
        import numpy as np
        names, bank = self._banks[category]
        vec = self.torch.from_numpy(np.ascontiguousarray(image_emb)).to(self.device)
        sims = (bank @ vec).float().cpu().numpy()
        return sorted(zip(names, sims.tolist()), key=lambda z: -z[1])


# ---------------------------------------------------------------------------
# Turn raw scores into conservative tags
# ---------------------------------------------------------------------------
def _argmax_gated(scored, floor, margin=0.0):
    if not scored:
        return None
    (n0, c0) = scored[0]
    c1 = scored[1][1] if len(scored) > 1 else -1.0
    return n0 if (c0 >= floor and (c0 - c1) >= margin) else None


def analyze(engine, emb) -> dict:
    s = {c: engine.score(emb, c) for c in CATEGORIES}
    top_obj = s["objects"][0][1] if s["objects"] else 0.0
    objects = [n for n, c in s["objects"][:OBJECT_TOP_K]
               if c >= OBJECT_MIN_COS or c >= top_obj - OBJECT_REL_DELTA][:OBJECT_TOP_K]
    return {
        "objects": objects,
        "scene": [n for n, c in s["scene"][:SCENE_TOP_K] if c >= SCENE_MIN_COS],
        "setting": _argmax_gated(s["setting"], floor=-1.0),
        "time_of_day": _argmax_gated(s["time_of_day"], TIME_MIN_COS),
        "weather": _argmax_gated(s["weather"], WEATHER_MIN_COS, WEATHER_MIN_MARGIN),
        "season": _argmax_gated(s["season"], SEASON_MIN_COS, SEASON_MIN_MARGIN),
        "style": [n for n, c in s["style"][:STYLE_TOP_K] if c >= STYLE_MIN_COS],
    }


# ---------------------------------------------------------------------------
# EXIF (Pillow only)
# ---------------------------------------------------------------------------
def extract_exif(path) -> dict:
    from PIL import Image, ExifTags
    tag = {v: k for k, v in ExifTags.TAGS.items()}
    gtag = {v: k for k, v in ExifTags.GPSTAGS.items()}
    out = {k: None for k in ("date", "time", "datetime", "camera", "lens",
                             "focal_length", "aperture", "shutter", "iso", "gps")}
    out["width"] = out["height"] = None

    def rat(x):
        try:
            if isinstance(x, (tuple, list)) and len(x) == 2:
                return x[0] / x[1] if x[1] else None
            return float(x)
        except (TypeError, ZeroDivisionError, ValueError):
            return None

    def clean(x):
        if isinstance(x, bytes):
            x = x.decode("utf-8", "ignore")
        if isinstance(x, str):
            x = x.strip().strip("\x00").strip()
        return x or None

    try:
        with Image.open(path) as im:
            w, h = im.size
            exif = im.getexif()
            # 274 = Orientation; values 5-8 mean the photo is rotated 90°, so the
            # upright width/height are swapped relative to the stored pixels.
            if exif and exif.get(274) in (5, 6, 7, 8):
                w, h = h, w
            out["width"], out["height"] = w, h
    except Exception:
        return out
    if not exif:
        return out

    def g(name):
        t = tag.get(name)
        return exif.get(t) if t is not None else None

    out["camera"] = " ".join(p for p in (clean(g("Make")), clean(g("Model"))) if p) or None
    try:
        ifd = exif.get_ifd(ExifTags.IFD.Exif)
    except Exception:
        ifd = {}

    def e(name):
        t = tag.get(name)
        return ifd.get(t) if t is not None else None

    out["lens"] = clean(e("LensModel"))
    fl = rat(e("FocalLength"))
    if fl:
        out["focal_length"] = f"{fl:g}mm"
    fn = rat(e("FNumber"))
    if fn:
        out["aperture"] = f"f/{fn:g}"
    exp = rat(e("ExposureTime"))
    if exp:
        out["shutter"] = f"{exp:g}s" if exp >= 1 else f"1/{round(1 / exp)}s"
    iso = e("ISOSpeedRatings") or e("PhotographicSensitivity")
    if isinstance(iso, (tuple, list)):
        iso = iso[0] if iso else None
    if iso:
        out["iso"] = int(iso)
    raw = clean(e("DateTimeOriginal")) or clean(g("DateTime"))
    if raw:
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(raw, fmt)
                out["date"] = dt.strftime("%Y-%m-%d")
                out["time"] = dt.strftime("%H:%M:%S")
                out["datetime"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                break
            except ValueError:
                continue
    try:
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:
        gps = {}
    if gps:
        def gp(name):
            t = gtag.get(name)
            return gps.get(t) if t is not None else None

        def coord(vals, ref):
            try:
                d = (rat(vals[0]) or 0) + (rat(vals[1]) or 0) / 60 + (rat(vals[2]) or 0) / 3600
                return round(-d if ref in ("S", "W") else d, 6)
            except (TypeError, IndexError):
                return None
        lat = coord(gp("GPSLatitude"), clean(gp("GPSLatitudeRef")))
        lon = coord(gp("GPSLongitude"), clean(gp("GPSLongitudeRef")))
        if lat is not None and lon is not None:
            out["gps"] = [lat, lon]
    return out


# ---------------------------------------------------------------------------
# Dominant colours (HSV naming)
# ---------------------------------------------------------------------------
def dominant_colors(img, k=5, top=3):
    small = img.convert("RGB").resize((128, 128))
    pal = small.quantize(colors=k)
    palette = pal.getpalette()
    names = []
    for _cnt, idx in sorted(pal.getcolors(), key=lambda c: -c[0]):
        r, g, b = palette[idx * 3: idx * 3 + 3]
        h, sat, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        hd = h * 360
        if sat < 0.12:
            name = "black" if v < 0.20 else "dark gray" if v < 0.45 else "gray" if v < 0.80 else "white"
        elif v < 0.16:
            name = "black"
        elif 15 <= hd < 45 and v < 0.65:
            name = "brown"
        elif 15 <= hd < 45 and sat < 0.45:
            name = "beige"
        elif hd < 15 or hd >= 345:
            name = "red"
        elif hd < 45:
            name = "orange"
        elif hd < 70:
            name = "yellow"
        elif hd < 160:
            name = "green"
        elif hd < 200:
            name = "teal"
        elif hd < 255:
            name = "blue"
        elif hd < 290:
            name = "purple"
        else:
            name = "pink"
        if name not in names:
            names.append(name)
        if len(names) >= top:
            break
    return names


# ---------------------------------------------------------------------------
# Imaging
# ---------------------------------------------------------------------------
def load_upright(path):
    from PIL import Image, ImageOps
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def make_display(img, dest: Path):
    from PIL import Image
    # Pillow >=9.1 uses Image.Resampling.LANCZOS; the old Image.LANCZOS alias is
    # gone in Pillow 12. This getattr works on every version.
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    w, h = img.size
    scale = min(1.0, DISPLAY_MAX_EDGE / max(w, h))
    out = img if scale >= 1.0 else img.resize((round(w * scale), round(h * scale)), resample)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)


# ---------------------------------------------------------------------------
# Embedding index + manifest  (<vault>/.photokb)
# ---------------------------------------------------------------------------
class Index:
    def __init__(self, vault: Path):
        import numpy as np
        self.np = np
        self.dir = vault / DATA_DIR
        self.emb_path = self.dir / "embeddings.npy"
        self.json_path = self.dir / "index.json"
        self.records, self.emb = [], None
        if self.json_path.exists():
            self.records = json.loads(self.json_path.read_text("utf-8"))
        if self.emb_path.exists():
            self.emb = np.load(self.emb_path)
        if self.emb is None or len(self.records) != len(self.emb):
            self.records, self.emb = [], None
        self._ix = {r["stem"]: i for i, r in enumerate(self.records)}

    def get(self, stem):
        i = self._ix.get(stem)
        return self.records[i] if i is not None else None

    def upsert(self, record, embedding):
        emb = embedding.astype(self.np.float32).reshape(1, -1)
        stem = record["stem"]
        if stem in self._ix:
            i = self._ix[stem]
            self.records[i] = record
            self.emb[i] = emb[0]
        else:
            self.records.append(record)
            self.emb = emb if self.emb is None else self.np.concatenate([self.emb, emb], 0)
            self._ix[stem] = len(self.records) - 1

    def prune(self, keep):
        idx = [i for i, r in enumerate(self.records) if r["stem"] in keep]
        removed = len(self.records) - len(idx)
        if removed and self.emb is not None:
            self.emb = self.emb[idx]
            self.records = [self.records[i] for i in idx]
            self._ix = {r["stem"]: i for i, r in enumerate(self.records)}
        return removed

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.emb is not None:
            self.np.save(self.emb_path, self.emb)
        self.json_path.write_text(json.dumps(self.records, ensure_ascii=False, indent=1), "utf-8")

    def rank(self, q, k=12, exclude=None):
        if self.emb is None or len(self.emb) == 0:
            return []
        q = q.astype(self.np.float32)
        q /= (self.np.linalg.norm(q) + 1e-9)
        sims = self.emb @ q
        out = []
        for i in self.np.argsort(-sims):
            r = self.records[int(i)]
            if exclude and r["stem"] == exclude:
                continue
            out.append((r, float(sims[int(i)])))
            if len(out) >= k:
                break
        return out


# ---------------------------------------------------------------------------
# Markdown note rendering (+ round-trip to preserve user edits)
# ---------------------------------------------------------------------------
USER_FIELDS = ("location", "rating", "people", "album")


def _q(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _scalar(k, v):
    if v is None or v == "":
        return f"{k}:"
    if isinstance(v, bool):
        return f"{k}: {str(v).lower()}"
    if isinstance(v, (int, float)):
        return f"{k}: {v}"
    return f"{k}: {_q(v)}"


def _list(k, values, inline_numeric=False):
    values = [v for v in (values or []) if v not in (None, "")]
    if not values:
        return f"{k}: []"
    if inline_numeric:
        return f"{k}: [{', '.join(str(v) for v in values)}]"
    return f"{k}:\n" + "\n".join(f"  - {_q(v)}" for v in values)


def frontmatter(m) -> str:
    fm = ["type: photo", _scalar("image", m["image"]), _scalar("original", m["original"]),
          _scalar("date", m.get("date")), _scalar("time", m.get("time")),
          _scalar("datetime", m.get("datetime")), _scalar("camera", m.get("camera")),
          _scalar("lens", m.get("lens")), _scalar("focal_length", m.get("focal_length")),
          _scalar("aperture", m.get("aperture")), _scalar("shutter", m.get("shutter")),
          _scalar("iso", m.get("iso"))]
    fm.append(_list("gps", m["gps"], inline_numeric=True) if m.get("gps") else "gps:")
    fm += [_scalar("location", m.get("location")), _scalar("collection", m.get("collection")),
           _scalar("setting", m.get("setting")),
           _list("scene", m.get("scene")), _list("objects", m.get("objects")),
           _scalar("time_of_day", m.get("time_of_day")), _scalar("weather", m.get("weather")),
           _scalar("season", m.get("season")), _list("style", m.get("style")),
           _list("colors", m.get("colors")), _list("tags", m.get("tags")),
           _list("keywords", m.get("keywords")), _scalar("rating", m.get("rating")),
           _list("people", m.get("people")), _scalar("album", m.get("album")),
           _scalar("width", m.get("width")), _scalar("height", m.get("height"))]
    return "---\n" + "\n".join(fm) + "\n---\n"


def body(m) -> str:
    bits = []
    if m.get("scene"):
        bits.append("**Scene:** " + ", ".join(m["scene"]))
    if m.get("objects"):
        bits.append("**Objects:** " + ", ".join(m["objects"]))
    if m.get("style"):
        bits.append("**Style:** " + ", ".join(m["style"]))
    if m.get("colors"):
        bits.append("**Colors:** " + ", ".join(m["colors"]))
    ctx = " · ".join(x for x in (m.get("time_of_day"), m.get("weather"), m.get("season")) if x)
    if ctx:
        bits.append("**Context:** " + ctx)
    block = "\n> ".join(bits) if bits else "_(no confident tags)_"
    return (f"![[{m['image']}]]\n\n> [!abstract] AI analysis (local CLIP)\n> {block}\n\n"
            "## Notes\n%% Your own description — what it is, where, why you kept it. %%\n\n"
            "## Related\n-\n")


def split_note(text):
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[4:end], text[end + 5:]
    return None, text


def parse_user_fields(fm_inner):
    out, lines, i = {}, fm_inner.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if ":" not in line or line.startswith(" "):
            i += 1
            continue
        key, _, rest = line.partition(":")
        key, rest = key.strip(), rest.strip()
        if key in USER_FIELDS:
            if rest and rest != "[]":
                out[key] = _unquote(rest)
            else:
                items, j = [], i + 1
                while j < len(lines) and lines[j].lstrip().startswith("- "):
                    items.append(_unquote(lines[j].lstrip()[2:].strip()))
                    j += 1
                if items:
                    out[key] = items
                i = j - 1
        i += 1
    return out


def _unquote(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if s.lstrip("-").isdigit():
        return int(s)
    return s


def write_note(path: Path, m: dict):
    """Create the note, or refresh frontmatter while keeping the user's body/edits."""
    if path.exists():
        old = path.read_text("utf-8")
        fm_inner, old_body = split_note(old)
        if fm_inner is not None:
            for k, v in parse_user_fields(fm_inner).items():
                m[k] = v
            path.write_text(frontmatter(m) + "\n" + old_body, "utf-8")
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter(m) + "\n" + body(m), "utf-8")


# ---------------------------------------------------------------------------
# Vault scaffold
# ---------------------------------------------------------------------------
def scaffold(vault: Path):
    for sub in (PHOTOS_DIR, ASSETS_DIR, ALBUMS_DIR, DASH_DIR, DATA_DIR):
        (vault / sub).mkdir(parents=True, exist_ok=True)
    d = vault / DASH_DIR

    def w(name, text):
        p = d / name
        if not p.exists():
            p.write_text(text, "utf-8")

    w("Library.md", """# 📚 Photo Library

> Needs the **Dataview** plugin (Settings → Community plugins → Dataview).

```dataview
TABLE WITHOUT ID ("![[" + image + "|180]]") AS "Photo", file.link AS "Note",
  date AS "Date", setting AS "Setting", join(scene, ", ") AS "Scene"
FROM "Photos"
SORT date DESC, file.name ASC
```
""")
    w("Browse by Scene.md", """# 🏷️ Browse by Scene

```dataview
TABLE rows.file.link AS "Photos", length(rows) AS "N"
FROM "Photos"
FLATTEN scene AS s
GROUP BY s
SORT length(rows) DESC
```

# Browse by Object

```dataview
TABLE rows.file.link AS "Photos", length(rows) AS "N"
FROM "Photos"
FLATTEN objects AS o
GROUP BY o
SORT length(rows) DESC
```
""")
    w("Favorites.md", """# ⭐ Favorites

Set `rating` (1–5) in a photo's properties; 4★+ show up here.

```dataview
TABLE WITHOUT ID ("![[" + image + "|220]]") AS "Photo", file.link AS "Note",
  rating AS "★", join(scene, ", ") AS "Scene"
FROM "Photos"
WHERE rating >= 4
SORT rating DESC, date DESC
```
""")
    w("Map.md", """# 🗺️ Map

```dataview
TABLE WITHOUT ID file.link AS "Note", date AS "Date", gps AS "Lat/Lon"
FROM "Photos"
WHERE gps
SORT date DESC
```

> Install the **Obsidian Leaflet** plugin to plot these on a real map.
""")
    w("Search Results.md", """# 🔎 Semantic Search

Rewritten each time you run `pkb.py search "<query>" --write`.

_(No search run yet.)_
""")
    tpl = vault / ALBUMS_DIR / "_Album Template.md"
    if not tpl.exists():
        tpl.write_text("""---
type: album
cover:
date:
tags: [album]
---

# {{title}}

## Highlights
-

## All photos

```dataview
TABLE WITHOUT ID ("![[" + image + "|180]]") AS "Photo", file.link AS "Note", date
FROM "Photos"
WHERE contains(album, this.file.name)
SORT date ASC
```
""", "utf-8")
    (vault / DATA_DIR / ".gitignore").write_text("*\n", "utf-8")
    (vault / "README.md").write_text(README_VAULT, "utf-8")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def iter_images(sources):
    for src in sources:
        if not src.exists():
            print(f"[warn] source not found: {src}", file=sys.stderr)
            continue
        for p in sorted(src.rglob("*")):
            # Skip hidden/AppleDouble files (e.g. ._DSC0001.JPG sidecars on
            # external drives) — they match the extension but aren't real images.
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
                yield p


def build(sources, vault, force=False, limit=None, batch=8, debug=False):
    scaffold(vault)
    index = Index(vault)
    files = list(iter_images(sources))
    if limit:
        files = files[:limit]
    stems_seen = set()
    todo = []
    for f in files:
        stem = f.stem
        stems_seen.add(stem)
        st = f.stat()
        rec = index.get(stem)
        note_ok = rec and (vault / rec.get("note", "")).exists()
        if (not force and rec and note_ok
                and rec.get("mtime") == int(st.st_mtime) and rec.get("size") == st.st_size):
            continue
        todo.append(f)

    print(f"[build] {len(files)} photos, {len(todo)} to (re)analyse -> {vault}")
    if not todo:
        removed = index.prune(stems_seen)
        index.save()
        print(f"[build] up to date. pruned {removed} stale entries.")
        return

    engine = Engine()
    done = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        tensors, staged = [], []
        for f in chunk:
            try:
                img = load_upright(f)
            except Exception as ex:
                print(f"  [skip] {f.name}: {ex}", file=sys.stderr)
                continue
            asset_name = f.stem + ".jpg"
            make_display(img, vault / ASSETS_DIR / asset_name)
            exif = extract_exif(f)
            colors = dominant_colors(img)
            tensors.append(engine.preprocess(img))
            staged.append((f, asset_name, exif, colors))
            img.close()
        embs = engine.encode_tensors(tensors)
        for (f, asset_name, exif, colors), emb in zip(staged, embs):
            tags = analyze(engine, emb)
            if debug:
                print(f"  {f.name}: {tags}")
            m = _assemble(f, asset_name, exif, colors, tags, _collection_of(f, sources))
            note_rel = f"{PHOTOS_DIR}/{m['note_name']}"
            write_note(vault / note_rel, m)
            index.upsert({"stem": f.stem, "note": note_rel, "image": asset_name,
                          "source": str(f), "date": exif.get("date"),
                          "mtime": int(f.stat().st_mtime), "size": f.stat().st_size}, emb)
            done += 1
            print(f"  [{done}/{len(todo)}] {m['note_name']}")
    removed = index.prune(stems_seen)
    index.save()
    print(f"[build] done. {done} notes written, {removed} stale pruned.")


GENERIC_DIRS = {"jpg", "jpeg", "png", "raw", "export", "exports", "out", "output"}


def _collection_of(f, sources):
    """Short label for which source folder a photo came from (for filtering)."""
    for root in sources:
        try:
            f.relative_to(root)
        except ValueError:
            continue
        name = root.name
        return root.parent.name if name.lower() in GENERIC_DIRS else name
    return None


def _assemble(f, asset_name, exif, colors, tags, collection=None):
    scene, objects, style = tags["scene"], tags["objects"], tags["style"]
    singles = [tags.get(k) for k in ("setting", "time_of_day", "weather", "season")]
    keywords = []
    for x in ([collection] if collection else []) + scene + objects + style + colors + [s for s in singles if s]:
        if x not in keywords:
            keywords.append(x)
    tag_src = (["photo"] + ([collection] if collection else [])
               + ([tags["setting"]] if tags["setting"] else []) + scene + objects[:4] + style)
    if tags["time_of_day"]:
        tag_src.append(tags["time_of_day"])
    tagslugs = []
    for t in tag_src:
        sl = slug(t)
        if sl and sl not in tagslugs:
            tagslugs.append(sl)
    note_name = (f"{exif['date']} {f.stem}.md" if exif.get("date") else f"{f.stem}.md")
    return {"image": asset_name, "original": str(f), "note_name": note_name,
            "collection": collection, "colors": colors, "tags": tagslugs,
            "keywords": keywords, **exif, **tags}


# ---------------------------------------------------------------------------
# Search / similar / stats / debug
# ---------------------------------------------------------------------------
def search(vault, query, k=12, write=False):
    index = Index(vault)
    if not index.records:
        print("No index yet. Run `build` first.")
        return
    engine = Engine(verbose=False)
    hits = index.rank(engine.encode_query(query), k=k)
    print(f'\nTop {len(hits)} for: "{query}"\n')
    for r, sc in hits:
        print(f"  {sc:.3f}  {r['stem']:16}  {r.get('note','')}")
    if write:
        lines = ["# 🔎 Semantic Search", "", f'> Query: **{query}**', ""]
        for r, sc in hits:
            lines += [f"### {sc:.3f} — {r['stem']}", f"![[{r['image']}|360]]",
                      f"[[{Path(r['note']).stem}]]", ""]
        (vault / DASH_DIR / "Search Results.md").write_text("\n".join(lines), "utf-8")
        print(f'\nWrote results to "{DASH_DIR}/Search Results.md" — open it in Obsidian.')


def similar(vault, stem, k=12):
    index = Index(vault)
    i = index._ix.get(stem)
    if i is None:
        cand = [s for s in index._ix if stem in s]
        if len(cand) == 1:
            stem = cand[0]
            i = index._ix[stem]
        else:
            print(f"'{stem}' not in index. Candidates: {cand[:8]}")
            return
    hits = index.rank(index.emb[i], k=k + 1, exclude=stem)
    print(f"\nMost similar to {stem}:\n")
    for r, sc in hits:
        print(f"  {sc:.3f}  {r['stem']:16}  {r.get('note','')}")


def stats(vault):
    index = Index(vault)
    n = len(index.records)
    print(f"\nPhotoKB @ {vault}")
    print(f"  photos indexed : {n}")
    if not n:
        return
    dated = [r for r in index.records if r.get("date")]
    if dated:
        ds = sorted(r["date"] for r in dated)
        print(f"  date range     : {ds[0]} … {ds[-1]}")
    print(f"  embeddings     : {None if index.emb is None else index.emb.shape}")


def debug_one(sources, name):
    files = [f for f in iter_images(sources) if name in f.stem or name in f.name]
    if not files:
        print(f"No photo matching '{name}'.")
        return
    f = files[0]
    print(f"[debug] {f}")
    engine = Engine()
    img = load_upright(f)
    emb = engine.encode_tensors([engine.preprocess(img)])[0]
    for cat in CATEGORIES:
        top = engine.score(emb, cat)[:6]
        print(f"  {cat:12} " + ", ".join(f"{n}={c:.3f}" for n, c in top))
    print(f"  colors       {dominant_colors(img)}")
    print(f"  -> tags      {analyze(engine, emb)}")


# ---------------------------------------------------------------------------
README_VAULT = """# PhotoKB Vault

A **local, Markdown-driven AI photo library**. Each photo is a note with
AI-generated tags (local CLIP) + EXIF; Obsidian + Dataview organise and query
it. Nothing is uploaded.

- `Photos/` — one note per photo (frontmatter + embedded image)
- `Assets/` — web-sized display copies (originals stay put; see each note's
  `original:` field)
- `Albums/` — curated collections (`_Album Template`)
- `Dashboards/` — Dataview views (Library, Browse by Scene, Favorites, Map)
- `.photokb/` — CLIP embeddings + index (hidden; don't edit)

## Setup
1. Open this folder as a vault in Obsidian.
2. Install & enable the **Dataview** community plugin.
3. Open `Dashboards/Library`.
"""


def main():
    ap = argparse.ArgumentParser(prog="pkb.py", description="Local AI photo library for Obsidian.")
    ap.add_argument("--source", action="append", help="photo source dir (repeatable)")
    ap.add_argument("--vault", help="output Obsidian vault dir")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create the empty vault scaffold")
    b = sub.add_parser("build", help="analyse photos and write notes")
    b.add_argument("--force", action="store_true", help="re-analyse everything")
    b.add_argument("--limit", type=int, help="process at most N photos")
    b.add_argument("--debug", action="store_true", help="print tags per photo")
    s = sub.add_parser("search", help="semantic text search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=12)
    s.add_argument("--write", action="store_true", help="render results into the vault")
    sm = sub.add_parser("similar", help="find visually similar photos")
    sm.add_argument("stem")
    sm.add_argument("-k", type=int, default=12)
    sub.add_parser("stats", help="library summary")
    d = sub.add_parser("debug", help="print raw CLIP scores for one photo")
    d.add_argument("name")
    args = ap.parse_args()

    if args.source:
        sources = [Path(p).expanduser() for p in args.source]
    elif os.environ.get("PHOTOKB_SOURCE"):
        sources = [Path(p).expanduser() for p in os.environ["PHOTOKB_SOURCE"].split(os.pathsep)]
    else:
        sources = DEFAULT_SOURCES
    if args.vault:
        vault = Path(args.vault).expanduser()
    elif os.environ.get("PHOTOKB_VAULT"):
        vault = Path(os.environ["PHOTOKB_VAULT"]).expanduser()
    else:
        vault = DEFAULT_VAULT

    if args.cmd == "init":
        scaffold(vault)
        print(f"Vault scaffolded at {vault}")
    elif args.cmd == "build":
        build(sources, vault, force=args.force, limit=args.limit, debug=args.debug)
    elif args.cmd == "search":
        search(vault, args.query, k=args.k, write=args.write)
    elif args.cmd == "similar":
        similar(vault, args.stem, k=args.k)
    elif args.cmd == "stats":
        stats(vault)
    elif args.cmd == "debug":
        debug_one(sources, args.name)


if __name__ == "__main__":
    main()
