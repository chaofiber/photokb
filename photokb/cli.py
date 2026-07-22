"""Command-line interface for PhotoKB — the `pkb` command.

Dispatches to the analysis engine (`core`, a copy of the proven pkb.py) and the
gallery generator (`gallery`, a copy of make_gallery.py). Adds two conveniences
over the old scripts: `pkb gallery` and `pkb update` (build + gallery + open).

Folders resolve in this order: --source/--vault flags → PHOTOKB_SOURCE /
PHOTOKB_VAULT env vars → the defaults below.
"""
from __future__ import annotations

import argparse
import os
import webbrowser
from pathlib import Path

from . import core, gallery

HOME = Path.home()
DEFAULT_SOURCE = HOME / "Documents" / "Photos"
DEFAULT_VAULT = HOME / "Documents" / "PhotoKB"


def _sources(a):
    if a.source:
        return [Path(p).expanduser() for p in a.source]
    env = os.environ.get("PHOTOKB_SOURCE")
    if env:
        return [Path(p).expanduser() for p in env.split(os.pathsep)]
    return [DEFAULT_SOURCE]


def _vault(a):
    if a.vault:
        return Path(a.vault).expanduser()
    env = os.environ.get("PHOTOKB_VAULT")
    if env:
        return Path(env).expanduser()
    return DEFAULT_VAULT


def _build_gallery(vault, open_browser):
    photos, emb = gallery.collect(vault)
    out = vault / "gallery.html"
    out.write_text(
        gallery.render_html(photos, emb, f"{gallery.VAULT_NAME} · {len(photos)} photos"),
        "utf-8")
    print(f"Wrote {out}  ({len(photos)} photos, embeddings={'yes' if emb else 'no'})")
    if open_browser:
        webbrowser.open(out.as_uri())


def main():
    ap = argparse.ArgumentParser(
        prog="pkb", description="Local, Markdown-driven AI photo library for Obsidian.")
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
    m = sub.add_parser("similar", help="visually similar photos")
    m.add_argument("stem")
    m.add_argument("-k", type=int, default=12)
    sub.add_parser("stats", help="library summary")
    d = sub.add_parser("debug", help="raw CLIP scores for one photo")
    d.add_argument("name")
    g = sub.add_parser("gallery", help="(re)generate the offline HTML gallery")
    g.add_argument("--open", action="store_true", help="open it in the browser")
    u = sub.add_parser("update", help="build, regenerate the gallery, and open it")
    u.add_argument("--force", action="store_true")
    u.add_argument("--limit", type=int)
    u.add_argument("--no-open", action="store_true", help="don't open the browser")
    sv = sub.add_parser("serve", help="serve the gallery locally with one-click delete")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--no-open", action="store_true", help="don't open the browser")

    a = ap.parse_args()
    src, vault = _sources(a), _vault(a)

    if a.cmd == "init":
        core.scaffold(vault)
        print(f"Vault scaffolded at {vault}")
    elif a.cmd == "build":
        core.build(src, vault, force=a.force, limit=a.limit, debug=a.debug)
    elif a.cmd == "search":
        core.search(vault, a.query, k=a.k, write=a.write)
    elif a.cmd == "similar":
        core.similar(vault, a.stem, k=a.k)
    elif a.cmd == "stats":
        core.stats(vault)
    elif a.cmd == "debug":
        core.debug_one(src, a.name)
    elif a.cmd == "gallery":
        _build_gallery(vault, a.open)
    elif a.cmd == "update":
        core.build(src, vault, force=a.force, limit=a.limit)
        _build_gallery(vault, not a.no_open)
    elif a.cmd == "serve":
        from . import server
        server.serve(vault, port=a.port, open_browser=not a.no_open)


if __name__ == "__main__":
    main()
