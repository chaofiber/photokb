#!/usr/bin/env python3
"""Generate a fancy, filterable, fully-offline HTML gallery for a PhotoKB vault.

Pure standard library — no torch / numpy / Pillow needed. Reads the photo notes'
YAML frontmatter + CLIP embeddings, writes ``<vault>/gallery.html`` (open in any
browser). Three layouts (justified rows / uniform grid / masonry), faceted
filtering, live search, lightbox with EXIF, and "find similar" (client-side
cosine on the embedded CLIP vectors).

Aspect ratios are read from the real display-JPEG pixels (parsed here), so
portrait vs landscape always lay out correctly regardless of EXIF orientation.

Usage:
    python make_gallery.py                          # default vault
    python make_gallery.py --vault ~/Documents/PhotoKB --open
"""
from __future__ import annotations

import argparse
import ast
import json
import struct
import sys
import webbrowser
from pathlib import Path

DEFAULT_VAULT = Path.home() / "Documents" / "PhotoKB"
ASSETS_DIR = "Assets"
DATA_DIR = ".photokb"
VAULT_NAME = "PhotoKB"


# ---- minimal YAML frontmatter parser (understands what pkb.py emits) ------
def _unquote(s: str):
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if s.lstrip("-").isdigit():
        return int(s)
    try:
        return float(s)
    except ValueError:
        return s


def parse_frontmatter(text: str) -> dict:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}
    lines = text[4:end].splitlines()
    data: dict = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or line[0] in " #" or ":" not in line:
            i += 1
            continue
        key, _, rest = line.partition(":")
        key, rest = key.strip(), rest.strip()
        if rest == "":
            items, j = [], i + 1
            while j < len(lines) and lines[j].startswith("  - "):
                items.append(_unquote(lines[j][4:]))
                j += 1
            if items:
                data[key] = items
                i = j
                continue
            data[key] = None
        elif rest == "[]":
            data[key] = []
        elif rest.startswith("[") and rest.endswith("]"):
            body = rest[1:-1].strip()
            data[key] = [_unquote(x) for x in body.split(",")] if body else []
        else:
            data[key] = _unquote(rest)
        i += 1
    return data


# ---- true pixel dims straight from the JPEG (no Pillow) -------------------
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_size(path: Path):
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                b = f.read(1)
                while b and b != b"\xff":
                    b = f.read(1)
                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                if not marker:
                    return None
                if marker[0] in _SOF:
                    f.read(3)
                    h = struct.unpack(">H", f.read(2))[0]
                    w = struct.unpack(">H", f.read(2))[0]
                    return (w, h)
                seg = f.read(2)
                if len(seg) < 2:
                    return None
                f.seek(struct.unpack(">H", seg)[0] - 2, 1)
    except Exception:
        return None


# ---- float32 .npy without numpy -------------------------------------------
def read_npy_f32(path: Path):
    try:
        with open(path, "rb") as f:
            if f.read(6) != b"\x93NUMPY":
                return None
            ver = f.read(2)
            hlen = struct.unpack("<H", f.read(2))[0] if ver[0] == 1 \
                else struct.unpack("<I", f.read(4))[0]
            header = ast.literal_eval(f.read(hlen).decode("latin1").strip())
            if "f4" not in header["descr"] or header.get("fortran_order"):
                return None
            rows, cols = header["shape"]
            floats = struct.unpack("<%df" % (rows * cols), f.read(rows * cols * 4))
            return [list(floats[r * cols:(r + 1) * cols]) for r in range(rows)]
    except Exception:
        return None


def collect(vault: Path):
    index_path = vault / DATA_DIR / "index.json"
    if not index_path.exists():
        sys.exit(f"No index at {index_path}. Run `pkb.py build` first.")
    records = json.loads(index_path.read_text("utf-8"))
    emb = read_npy_f32(vault / DATA_DIR / "embeddings.npy")
    photos = []
    for row, rec in enumerate(records):
        note_path = vault / rec["note"]
        fm = parse_frontmatter(note_path.read_text("utf-8")) if note_path.exists() else {}
        dims = jpeg_size(vault / ASSETS_DIR / rec["image"])
        w, h = dims if dims else (fm.get("width"), fm.get("height"))
        photos.append({
            "stem": rec["stem"], "src": f"{ASSETS_DIR}/{rec['image']}",
            "note": rec["note"][:-3] if rec["note"].endswith(".md") else rec["note"],
            "original": fm.get("original") or rec.get("source"),
            "date": fm.get("date") or rec.get("date"), "time": fm.get("time"),
            "setting": fm.get("setting"), "scene": fm.get("scene") or [],
            "objects": fm.get("objects") or [], "time_of_day": fm.get("time_of_day"),
            "weather": fm.get("weather"), "season": fm.get("season"),
            "style": fm.get("style") or [], "colors": fm.get("colors") or [],
            "collection": fm.get("collection"),
            "camera": fm.get("camera"), "lens": fm.get("lens"), "focal": fm.get("focal_length"),
            "aperture": fm.get("aperture"), "shutter": fm.get("shutter"), "iso": fm.get("iso"),
            "rating": fm.get("rating"), "gps": fm.get("gps") or None,
            "w": w, "h": h, "emb": row if emb else None,
        })
    emb_small = [[round(x, 4) for x in v] for v in emb] if emb else None
    return photos, emb_small


def render_html(photos, emb, title: str) -> str:
    data = {"photos": photos, "emb": emb, "vault": VAULT_NAME}
    return HTML.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False)).replace("__TITLE__", title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default=str(DEFAULT_VAULT))
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    vault = Path(args.vault).expanduser()
    photos, emb = collect(vault)
    out = vault / "gallery.html"
    out.write_text(render_html(photos, emb, f"{VAULT_NAME} · {len(photos)} photos"), "utf-8")
    print(f"Wrote {out}  ({len(photos)} photos, embeddings={'yes' if emb else 'no'})")
    if args.open:
        webbrowser.open(out.as_uri())


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#0e0f13;--panel:#16181f;--panel2:#1c1f28;--line:#2a2e3a;--fg:#e7e9ee;
  --muted:#9aa2b1;--accent:#5b8cff;--chip:#232734;--shadow:0 6px 24px rgba(0,0,0,.35)}
:root[data-theme="light"]{--bg:#f4f5f8;--panel:#fff;--panel2:#f0f2f6;--line:#e2e5ec;
  --fg:#1a1d24;--muted:#5b6472;--accent:#2f6bff;--chip:#eceff4;--shadow:0 6px 20px rgba(30,40,80,.12)}
*{box-sizing:border-box}html,body{margin:0}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
a{color:var(--accent);text-decoration:none}
header{position:sticky;top:0;z-index:20;display:flex;gap:10px;align-items:center;flex-wrap:wrap;
  padding:12px 18px;background:color-mix(in srgb,var(--panel) 88%,transparent);
  backdrop-filter:saturate(1.4) blur(10px);border-bottom:1px solid var(--line)}
header h1{font-size:16px;margin:0;font-weight:650;white-space:nowrap}
.count{color:var(--muted);font-size:12px}.grow{flex:1}
input,select,button{font:inherit;color:var(--fg)}
#q{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:9px 12px;width:min(320px,40vw);outline:none}
#q:focus{border-color:var(--accent)}
select,.btn{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:8px 11px;cursor:pointer}
.btn:hover,select:hover{border-color:var(--accent)}
main{display:grid;grid-template-columns:250px 1fr;align-items:start}
#facets{position:sticky;top:57px;max-height:calc(100vh - 57px);overflow:auto;padding:14px 14px 40px;
  border-right:1px solid var(--line);background:var(--panel)}
.facet{margin-bottom:14px}
.facet h3{margin:0 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);cursor:pointer}
.facet .opts{display:flex;flex-wrap:wrap;gap:6px}
.tag{display:inline-flex;align-items:center;gap:5px;background:var(--chip);border:1px solid transparent;
  border-radius:999px;padding:3px 9px;font-size:12px;cursor:pointer;user-select:none;transition:.12s}
.tag:hover{border-color:var(--accent)}.tag.on{background:var(--accent);color:#fff}
.tag .n{color:var(--muted);font-size:11px}.tag.on .n{color:#e8eeff}
.sw{width:11px;height:11px;border-radius:3px;border:1px solid rgba(128,128,128,.4)}
#active{display:flex;flex-wrap:wrap;gap:6px;padding:10px 18px 0}#active:empty{display:none}
#active .tag{background:var(--accent);color:#fff}
#grid{padding:14px 18px 60px}
figure{margin:0;position:relative;border-radius:12px;overflow:hidden;background:var(--panel2);box-shadow:var(--shadow);cursor:zoom-in}
figure img{display:block;width:100%;height:100%;object-fit:cover;transition:transform .35s ease}
figure:hover img{transform:scale(1.045)}
figure .meta{position:absolute;left:0;right:0;bottom:0;padding:18px 10px 8px;font-size:11px;color:#fff;
  background:linear-gradient(transparent,rgba(0,0,0,.72));opacity:0;transition:.2s;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
figure:hover .meta{opacity:1}
figure .rate{position:absolute;top:8px;right:9px;color:#ffd34d;font-size:12px;text-shadow:0 1px 3px rgba(0,0,0,.6)}
figure .del{position:absolute;top:8px;left:9px;width:30px;height:30px;border:none;border-radius:8px;font-size:14px;
  cursor:pointer;background:rgba(0,0,0,.5);color:#fff;opacity:0;transition:.15s;z-index:2}
figure:hover .del{opacity:1}
figure .del:hover{background:#e5484d}
.del-lb{background:transparent;border-color:#e5484d;color:#e5484d}
.del-lb:hover{background:#e5484d;color:#fff}
#grid.justified{display:flex;flex-wrap:wrap;gap:10px;align-content:flex-start}
#grid.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
#grid.grid figure{aspect-ratio:1}
#grid.masonry{column-count:4;column-gap:12px}
#grid.masonry figure{break-inside:avoid;margin:0 0 12px;display:inline-block;width:100%}
#grid.masonry figure img{height:auto}
@media(max-width:1500px){#grid.masonry{column-count:3}}
@media(max-width:1000px){main{grid-template-columns:1fr}
  #facets{position:static;max-height:none;border-right:none;border-bottom:1px solid var(--line)}}
@media(max-width:640px){#grid.masonry{column-count:2}#grid.grid{grid-template-columns:repeat(auto-fill,minmax(130px,1fr))}}
.empty{padding:60px;text-align:center;color:var(--muted);width:100%}
#lb{position:fixed;inset:0;z-index:50;display:none;background:rgba(7,8,12,.92);backdrop-filter:blur(6px)}
#lb.on{display:flex}
#lb .stage{flex:1;display:flex;align-items:center;justify-content:center;padding:24px;min-width:0}
#lb img{max-width:100%;max-height:calc(100vh - 48px);border-radius:8px;box-shadow:var(--shadow)}
#lb .info{width:330px;flex:none;background:var(--panel);border-left:1px solid var(--line);padding:20px;overflow:auto}
#lb .info h2{font-size:15px;margin:0 0 2px;word-break:break-all}
#lb .info .sub{color:var(--muted);font-size:12px;margin-bottom:14px}
.kv{display:grid;grid-template-columns:82px 1fr;gap:3px 10px;font-size:12.5px;margin-bottom:14px}
.kv .k{color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin:4px 0 12px}.chips .tag{cursor:default;font-size:11.5px}
.lb-actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}
#lb .nav{position:absolute;top:50%;transform:translateY(-50%);font-size:30px;color:#fff;background:rgba(0,0,0,.35);
  border:none;border-radius:50%;width:46px;height:46px;cursor:pointer}
#lb .nav:hover{background:rgba(0,0,0,.6)}#lb #prev{left:14px}#lb #next{right:344px}
#lb #close{position:absolute;top:14px;right:14px;font-size:18px;color:#fff;background:rgba(0,0,0,.4);
  border:none;border-radius:8px;width:38px;height:38px;cursor:pointer}
@media(max-width:820px){#lb{flex-direction:column}#lb .info{width:auto;border-left:none;
  border-top:1px solid var(--line);max-height:42vh}#lb #next{right:14px}}
.simbar{display:flex;align-items:center;gap:10px;padding:10px 18px 0;color:var(--muted);font-size:12.5px}
.simbar:empty{display:none}
</style>
</head>
<body>
<header>
  <h1>📷 __TITLE__</h1>
  <span class="count" id="count"></span>
  <span class="grow"></span>
  <input id="q" type="search" placeholder="Search tags, camera, filename…" autocomplete="off">
  <select id="layout" title="Layout">
    <option value="justified">Justified</option><option value="grid">Grid</option><option value="masonry">Masonry</option>
  </select>
  <select id="sort" title="Sort">
    <option value="date-desc">Newest</option><option value="date-asc">Oldest</option>
    <option value="name">Name</option><option value="rating">Rating</option>
  </select>
  <button class="btn" id="theme" title="Toggle theme">🌓</button>
</header>
<div id="active"></div>
<div class="simbar" id="simbar"></div>
<main>
  <aside id="facets"></aside>
  <section id="grid"></section>
</main>
<div id="lb">
  <button class="nav" id="prev">‹</button>
  <div class="stage"><img id="lbimg" alt=""></div>
  <button class="nav" id="next">›</button>
  <button id="close">✕</button>
  <div class="info" id="lbinfo"></div>
</div>
<script>
const DATA = /*__DATA__*/;
const PHOTOS = DATA.photos, EMB = DATA.emb, VAULT = DATA.vault;
const COLOR_HEX={black:'#111',"dark gray":'#555',gray:'#9aa0a6',white:'#f2f2f2',red:'#e5484d',
  orange:'#f5820b',yellow:'#f5d90a',green:'#46a758',teal:'#12a594',blue:'#3b82f6',purple:'#8e4ec6',
  pink:'#e93d82',brown:'#8b5a2b',beige:'#dcc79a'};
const FACETS=[["collection","Collection"],["scene","Scene"],["objects","Objects"],["setting","Setting"],["time_of_day","Time"],
  ["style","Style"],["colors","Colors"],["camera","Camera"]];
const asArr=v=>Array.isArray(v)?v:(v==null||v===""?[]:[v]);
const active={}; let sim=null;
PHOTOS.forEach(p=>{p._ar=(p.w&&p.h)?p.w/p.h:1.5;
  p._blob=[p.stem,p.collection,p.camera,p.lens,p.setting,p.time_of_day,p.weather,p.season,
    ...(p.scene||[]),...(p.objects||[]),...(p.style||[]),...(p.colors||[])].filter(Boolean).join(" ").toLowerCase();});
const $=id=>document.getElementById(id);
const SERVED=location.protocol.startsWith('http');   // delete only when served by `pkb serve`
async function deletePhoto(stem){
  if(!confirm(`Move "${stem}" to Trash?\n\nMoves the original photo + its note + display copy into the vault's .trash/ (recoverable).`))return;
  try{
    const r=await fetch('/api/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({stem})});
    const res=await r.json();
    if(!res.ok){alert('Delete failed: '+(res.error||'unknown'));return;}
    const i=PHOTOS.findIndex(p=>p.stem===stem);if(i>=0)PHOTOS.splice(i,1);
    $('lb').classList.remove('on');sim=null;render();
  }catch(e){alert('Delete error (is `pkb serve` running?): '+e);}
}
function buildFacets(){
  const box=$('facets');box.innerHTML="";
  for(const [key,label] of FACETS){
    const counts=new Map();
    for(const p of PHOTOS)for(const v of asArr(p[key]))counts.set(v,(counts.get(v)||0)+1);
    if(!counts.size)continue;
    const f=document.createElement('div');f.className='facet';
    const h=document.createElement('h3');h.textContent=label;
    const opts=document.createElement('div');opts.className='opts';
    for(const [v,n] of [...counts.entries()].sort((a,b)=>b[1]-a[1])){
      const t=document.createElement('span');t.className='tag';
      if(active[key]&&active[key].has(v))t.classList.add('on');
      if(key==='colors'){const s=document.createElement('span');s.className='sw';s.style.background=COLOR_HEX[v]||'#888';t.appendChild(s);}
      t.appendChild(document.createTextNode(v));
      const c=document.createElement('span');c.className='n';c.textContent=n;t.appendChild(c);
      t.onclick=()=>toggle(key,v);opts.appendChild(t);
    }
    h.onclick=()=>opts.style.display=opts.style.display==='none'?'flex':'none';
    f.appendChild(h);f.appendChild(opts);box.appendChild(f);
  }
}
function toggle(key,v){active[key]=active[key]||new Set();
  active[key].has(v)?active[key].delete(v):active[key].add(v);
  if(!active[key].size)delete active[key];sim=null;render();}
function matches(p){
  const q=$('q').value.trim().toLowerCase();
  if(q&&!p._blob.includes(q))return false;
  for(const key in active){const vals=asArr(p[key]);
    if(![...active[key]].some(x=>vals.includes(x)))return false;}
  return true;
}
function currentList(){
  let list=PHOTOS.filter(matches);const sort=$('sort').value;
  if(sim)list=list.filter(p=>sim.order.has(p.stem)).sort((a,b)=>sim.order.get(b.stem)-sim.order.get(a.stem));
  else if(sort==='date-desc')list.sort((a,b)=>(b.date||'').localeCompare(a.date||'')||a.stem.localeCompare(b.stem));
  else if(sort==='date-asc')list.sort((a,b)=>(a.date||'').localeCompare(b.date||'')||a.stem.localeCompare(b.stem));
  else if(sort==='name')list.sort((a,b)=>a.stem.localeCompare(b.stem));
  else if(sort==='rating')list.sort((a,b)=>(b.rating||0)-(a.rating||0));
  return list;
}
function render(){
  buildFacets();
  const layout=$('layout').value;
  const list=currentList();window._visible=list;
  const grid=$('grid');grid.className=layout;grid.innerHTML="";
  $('count').textContent=list.length+" / "+PHOTOS.length;
  if(!list.length){grid.innerHTML='<div class="empty">No photos match these filters.</div>';renderActive();return;}
  list.forEach((p,vi)=>{
    const fig=document.createElement('figure');fig.dataset.ar=p._ar;
    if(layout==='masonry')fig.style.aspectRatio=p._ar;
    fig.innerHTML=`<img loading="lazy" src="${p.src}" alt="${p.stem}">`+
      (SERVED?`<button class="del" title="Move to Trash" onclick="event.stopPropagation();deletePhoto('${p.stem}')">🗑</button>`:'')+
      (p.rating?`<span class="rate">${'★'.repeat(p.rating)}</span>`:'')+
      `<figcaption class="meta">${p.date||''} · ${(p.scene||[]).join(', ')}</figcaption>`;
    fig.onclick=()=>openLB(vi);grid.appendChild(fig);
  });
  if(layout==='justified')layoutJustified();
  renderActive();
}
function layoutJustified(){
  const grid=$('grid');if(grid.className!=='justified')return;
  const W=grid.clientWidth,gap=10,target=W<640?150:W<1000?190:240;
  let row=[],arsum=0;
  const place=(row,arsum,stretch)=>{const h=stretch?(W-gap*(row.length-1))/arsum:target;
    row.forEach(f=>{const ar=+f.dataset.ar;f.style.width=Math.floor(h*ar)+'px';f.style.height=Math.round(h)+'px';});};
  [...grid.querySelectorAll('figure')].forEach(f=>{
    const ar=+f.dataset.ar||1.5;row.push(f);arsum+=ar;
    if(target*arsum+gap*(row.length-1)>=W){place(row,arsum,true);row=[];arsum=0;}
  });
  if(row.length)place(row,arsum,false);
}
function renderActive(){
  const box=$('active');box.innerHTML="";
  for(const key in active)for(const v of active[key]){
    const t=document.createElement('span');t.className='tag on';t.textContent=v+' ✕';
    t.onclick=()=>toggle(key,v);box.appendChild(t);}
  if(Object.keys(active).length){const c=document.createElement('span');c.className='tag';c.textContent='clear all';
    c.onclick=()=>{for(const k in active)delete active[k];sim=null;render();};box.appendChild(c);}
}
let cur=0;
function openLB(vi){cur=vi;showLB();$('lb').classList.add('on');}
function showLB(){
  const p=window._visible[cur];$('lbimg').src=p.src;
  const kv=[["Date",(p.date||'')+' '+(p.time||'')],["Camera",p.camera],["Lens",p.lens],["Focal",p.focal],
    ["Aperture",p.aperture],["Shutter",p.shutter],["ISO",p.iso],["Setting",p.setting],["Time",p.time_of_day],
    ["Weather",p.weather],["Season",p.season],["Size",(p.w&&p.h)?p.w+"×"+p.h:null]]
    .filter(r=>r[1]&&(""+r[1]).trim()).map(r=>`<span class="k">${r[0]}</span><span>${r[1]}</span>`).join("");
  const chip=(arr,color)=>arr.map(v=>`<span class="tag">${color?`<span class="sw" style="background:${COLOR_HEX[v]||'#888'}"></span>`:''}${v}</span>`).join("");
  const noteURL=`obsidian://open?vault=${encodeURIComponent(VAULT)}&file=${encodeURIComponent(p.note)}`;
  $('lbinfo').innerHTML=`<h2>${p.stem}</h2><div class="sub">${(p.scene||[]).join(' · ')}</div>
    <div class="kv">${kv}</div>
    ${p.objects&&p.objects.length?`<div class="chips">${chip(p.objects)}</div>`:''}
    ${p.style&&p.style.length?`<div class="chips">${chip(p.style)}</div>`:''}
    ${p.colors&&p.colors.length?`<div class="chips">${chip(p.colors,true)}</div>`:''}
    <div class="lb-actions">
      ${EMB?`<button class="btn" onclick="findSimilar('${p.stem}')">✨ Find similar</button>`:''}
      <a class="btn" href="${noteURL}">Open in Obsidian</a>
      ${p.original?`<a class="btn" href="file://${encodeURI(p.original)}">Original</a>`:''}
      ${SERVED?`<button class="btn del-lb" onclick="deletePhoto('${p.stem}')">🗑 Delete</button>`:''}
    </div>`;
}
function nav(d){cur=(cur+d+window._visible.length)%window._visible.length;showLB();}
function findSimilar(stem){
  if(!EMB)return;
  const v=EMB[PHOTOS.find(p=>p.stem===stem).emb];
  const scored=PHOTOS.map(p=>{const w=EMB[p.emb];let d=0;for(let k=0;k<v.length;k++)d+=v[k]*w[k];return [p.stem,d];});
  scored.sort((a,b)=>b[1]-a[1]);
  sim={stem,order:new Map(scored.slice(0,30))};
  $('lb').classList.remove('on');for(const k in active)delete active[k];render();
  $('simbar').innerHTML=`✨ Similar to <b style="color:var(--fg)">&nbsp;${stem}</b> &nbsp;<span class="tag" onclick="sim=null;render();$('simbar').innerHTML=''">clear</span>`;
  window.scrollTo({top:0,behavior:'smooth'});
}
$('q').oninput=()=>{sim=null;render();};
$('sort').onchange=render;
$('layout').onchange=()=>{localStorage.setItem('pkblayout',$('layout').value);render();};
$('prev').onclick=e=>{e.stopPropagation();nav(-1);};
$('next').onclick=e=>{e.stopPropagation();nav(1);};
$('close').onclick=()=>$('lb').classList.remove('on');
$('lb').onclick=e=>{if(e.target.id==='lb')$('lb').classList.remove('on');};
document.addEventListener('keydown',e=>{if(!$('lb').classList.contains('on'))return;
  if(e.key==='Escape')$('lb').classList.remove('on');if(e.key==='ArrowLeft')nav(-1);if(e.key==='ArrowRight')nav(1);});
let rz;window.addEventListener('resize',()=>{clearTimeout(rz);rz=setTimeout(layoutJustified,120);});
function setTheme(t){document.documentElement.setAttribute('data-theme',t);localStorage.setItem('pkbtheme',t);}
$('theme').onclick=()=>setTheme(document.documentElement.getAttribute('data-theme')==='light'?'dark':'light');
setTheme(localStorage.getItem('pkbtheme')||'dark');
$('layout').value=localStorage.getItem('pkblayout')||'justified';
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
