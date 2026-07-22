#!/usr/bin/env python3
"""Local web server for the PhotoKB gallery — enables one-click delete.

    python -m photokb.server [--vault DIR] [--port 8000] [--no-open]
    # or, once the CLI knows it:  pkb serve

Serves the vault over http://127.0.0.1:<port> so gallery.html can call a small
POST /api/delete endpoint. "Delete" *moves* the original photo + its note +
display copy into <vault>/.trash/<stem>/ and prunes the index — recoverable, and
it won't reappear on the next build. Bound to localhost only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import core, gallery

DEFAULT_VAULT = Path.home() / "Documents" / "PhotoKB"


def regenerate(vault: Path) -> int:
    photos, emb = gallery.collect(vault)
    (vault / "gallery.html").write_text(
        gallery.render_html(photos, emb, f"{gallery.VAULT_NAME} · {len(photos)} photos"), "utf-8")
    return len(photos)


def delete_photo(vault: Path, stem: str) -> dict:
    idx = core.Index(vault)
    rec = idx.get(stem)
    if not rec:
        return {"ok": False, "error": f"'{stem}' not in index"}
    dest = vault / ".trash" / stem
    dest.mkdir(parents=True, exist_ok=True)
    # Move note, display copy, and the original into the trash (prefixed to avoid
    # case-insensitive collisions between DSC.jpg and DSC.JPG).
    note = vault / rec.get("note", "")
    if note.suffix == ".md" and note.exists():
        shutil.move(str(note), str(dest / note.name))
    if rec.get("image"):
        asset = vault / "Assets" / rec["image"]
        if asset.exists():
            shutil.move(str(asset), str(dest / ("display_" + rec["image"])))
    if rec.get("source"):
        src = Path(rec["source"])
        if src.exists():
            shutil.move(str(src), str(dest / ("original_" + src.name)))
    idx.prune({r["stem"] for r in idx.records if r["stem"] != stem})
    idx.save()
    remaining = regenerate(vault)
    return {"ok": True, "remaining": remaining, "trash": str(dest)}


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") == "/api/delete":
            try:
                n = int(self.headers.get("Content-Length", 0) or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                res = delete_photo(Path(self.directory), str(body.get("stem", "")))
                code = 200 if res.get("ok") else 404
            except Exception as e:  # noqa: BLE001
                res, code = {"ok": False, "error": str(e)}, 500
            payload = json.dumps(res).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass  # keep the console quiet


def serve(vault: Path, port: int = 8000, open_browser: bool = True):
    regenerate(vault)  # ensure gallery.html is fresh and in served (delete-enabled) mode
    httpd = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(vault)))
    url = f"http://127.0.0.1:{port}/gallery.html"
    print(f"PhotoKB gallery → {url}   (Ctrl-C to stop)")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


def main():
    ap = argparse.ArgumentParser(prog="pkb serve")
    ap.add_argument("--vault")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    vault = (Path(a.vault).expanduser() if a.vault
             else Path(os.environ["PHOTOKB_VAULT"]).expanduser() if os.environ.get("PHOTOKB_VAULT")
             else DEFAULT_VAULT)
    serve(vault, port=a.port, open_browser=not a.no_open)


if __name__ == "__main__":
    main()
