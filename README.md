# PhotoKB

**A local, Markdown-driven AI photo library for Obsidian.**

Turn a folder of photos into a searchable knowledge base: every photo becomes a
Markdown note with AI-generated tags (local **CLIP**) and EXIF metadata, organised
and queried in Obsidian, with CLIP embeddings powering **semantic search** and
**“find similar”**, plus a fast, filterable **offline HTML gallery**. Everything
runs on your machine — nothing is uploaded.

## Features

- 🏷️ **Auto-tagging** — local OpenCLIP zero-shot across a 100+ label taxonomy
  (objects, scene, setting, time of day, weather, season, style), gated by
  conservative confidence thresholds.
- 🔎 **Semantic search** — natural-language queries (`"boats at night"`) via CLIP
  text embeddings.
- ✨ **Find similar** — nearest-neighbour by image embedding, client-side in the gallery.
- 🧭 **EXIF → frontmatter** — camera, lens, aperture, shutter, ISO, date, GPS.
- 🖼️ **Offline gallery** — self-contained `gallery.html`: justified / grid / masonry
  layouts, faceted filtering, lightbox, and (when served) one-click delete-to-Trash.
- 📓 **Obsidian-native** — notes + Dataview dashboards; images kept web-sized so
  Obsidian stays snappy while originals stay untouched on disk.
- 🔒 **100% local** — CLIP runs on Apple MPS / CUDA / CPU. No cloud, no accounts.

## How it works

```
photos ──► CLIP + EXIF analysis ──► Markdown notes (+ embeddings index) ──► Obsidian
                                             └──► gallery.html (browser)
```

Each photo gets a companion note:

```yaml
---
image: DSC06550.jpg
date: 2024-06-29
camera: SONY ILCE-7CM2
aperture: f/5.6
iso: 12800
setting: outdoor
scene: [seascape, harbor]
objects: [moon, sea, boat, water reflection]
time_of_day: night
colors: [teal, black]
tags: [photo, outdoor, seascape, harbor, moon, night]
rating:            # you fill (1–5)
---
![[DSC06550.jpg]]
```

## Install

Requires Python ≥ 3.9 and PyTorch. Install into an environment that already has
torch (recommended — avoids a multi-GB re-download):

```bash
pip install -e .
```

This adds a `pkb` command. No-install alternative: `PYTHONPATH=. python -m photokb <cmd>`.

> First run downloads the CLIP weights once (~600 MB, OpenCLIP ViT-B-32).

## Usage

```bash
pkb build                     # analyse new/changed photos → notes + embeddings (incremental)
pkb update                    # build + regenerate the gallery + open it
pkb search "quiet forest"     # semantic search   (--write renders results into the vault)
pkb similar DSC06550          # visually similar photos
pkb serve                     # gallery at http://127.0.0.1:8000 with one-click delete
pkb gallery --open            # (re)build just the gallery
pkb stats                     # library summary
pkb debug IMG_1234            # raw CLIP scores for one photo (for tuning thresholds)
```

`build` is incremental and preserves your manual edits (ratings, notes, albums).

## Configure

Resolved as: CLI flag → environment variable → default. Global flags go **before**
the subcommand (e.g. `pkb --vault ~/X serve`).

| What            | Flag                     | Env var          | Default              |
|-----------------|--------------------------|------------------|----------------------|
| Photo source    | `--source` (repeatable)  | `PHOTOKB_SOURCE` | `~/Documents/Photos` |
| Obsidian vault  | `--vault`                | `PHOTOKB_VAULT`  | `~/Documents/PhotoKB`|

Open a different folder as its own independent library:

```bash
pkb --source ~/Pictures/Trip --vault ~/Documents/TripKB build
```

## Obsidian

Open the vault folder in Obsidian and enable the **Dataview** community plugin for
the dashboards (Library, Browse by Scene/Object, Favorites, Map). The read-only
`gallery.html` opens in any browser; run `pkb serve` for the interactive version
with delete.

## Tuning tags & language

Thresholds and the label taxonomy live at the top of `photokb/core.py`. Run
`pkb debug <name>` to inspect raw cosine scores, adjust, then `pkb build --force`.
English works best; for other languages set a multilingual CLIP via
`PHOTOKB_MODEL` / `PHOTOKB_PRETRAINED`.

## License

[MIT](LICENSE)
