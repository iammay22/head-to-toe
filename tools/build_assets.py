#!/usr/bin/env python3
"""《從頭相到腳》素材處理腳本

用法：python3 tools/build_assets.py
輸入：raw/head/*.jpg|png（頭部特寫，跟 base 同一個畫面）
      raw/body/*.jpg|png（全身，跟 base 同一個畫面）
輸出：assets/*.webp、assets/*.svg（佔位圖）、assets/manifest.json、assets/pack.js

檔名規則（決定切法）：
  base                       標準人
  face_<臉型>                整顆頭，例如 face_square
  face_<臉型>_chin_<下巴>    整顆頭，例如 face_oval_chin_pointed
  eyes_ / brows_ / nose_ / mouth_ / chinmark_   只在臉內側的部件（inner 切法）
  ears_ / hair_ / top_ / bottom_ / acc_         會超出外框的部件（outer 切法）
沒有正式圖的部件會自動用 SVG 佔位圖補上；有了同名正式圖就會蓋掉佔位圖。
"""
import base64, io, json, math, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, OUT = os.path.join(ROOT, 'raw'), os.path.join(ROOT, 'assets')
HS = 1024                 # 頭部空間邊長
BW, BH = 1078, 1932       # 身體空間
SKIN = (246, 204, 180)    # 標準膚色（會用 base 實測值覆蓋）
LINE = '#5e3a36'
ITEMS, LOG = {}, []


# ---------- 基本工具 ----------
def load(path, size):
    im = Image.open(path).convert('RGB')
    if im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im).astype(np.float32)


def greenness(a):
    return a[..., 1] - np.maximum(a[..., 0], a[..., 2])


def magentaness(a):
    return np.minimum(a[..., 0], a[..., 2]) - a[..., 1]


def key(a):
    """去背：回傳 (去溢色後的 RGB, alpha 0..1)。背景可以是純綠或純洋紅。"""
    corner = a[:12, :12].reshape(-1, 3).mean(0)
    use_mag = corner[1] < 100 and corner[0] > 150 and corner[2] > 150
    g = magentaness(a) if use_mag else greenness(a)
    alpha = np.clip((70 - g) / 50, 0, 1)
    out = a.copy()
    edge = alpha < 1
    if use_mag:
        lim = out[..., 1] + 10
        out[..., 0] = np.where(edge, np.minimum(out[..., 0], lim), out[..., 0])
        out[..., 2] = np.where(edge, np.minimum(out[..., 2], lim), out[..., 2])
    else:
        lim = np.maximum(out[..., 0], out[..., 2]) + 10
        out[..., 1] = np.where(edge, np.minimum(out[..., 1], lim), out[..., 1])
    return out, alpha


def lum(a):
    return a.mean(-1)


def is_skin(a):
    return (a[..., 0] > a[..., 1]) & (a[..., 1] > a[..., 2]) & (a[..., 0] - a[..., 2] > 25) & (lum(a) > 125)


def normalize_skin(a, patch, ref):
    """把這張圖的膚色統一成標準人的顏色（Gemini 每張的膚色會偏一點）。"""
    y0, y1, x0, x1 = patch
    p = a[y0:y1, x0:x1].reshape(-1, 3)
    p = p[(p[:, 0] > p[:, 1]) & (p[:, 1] > p[:, 2]) & (p.mean(1) > 150)]
    if len(p) < 50:
        return a, None
    gain = np.array(ref) / np.median(p, 0)
    if np.all(np.abs(gain - 1) < 0.004):
        return a, gain
    out = a.copy()
    m = is_skin(a)
    out[m] = np.clip(a[m] * gain, 0, 255)
    return out, gain


def face_component(a, alpha, seed):
    """臉的內側（被外框線圍住的那塊皮膚）。"""
    m = (alpha > 0.9) & (lum(a) > 150) & (a[..., 0] - a[..., 2] > 25)
    lab, _ = ndi.label(m)
    k = lab[seed[1], seed[0]]
    if k == 0:
        raise SystemExit('找不到臉：種子點不在皮膚上')
    return lab == k


def head_mask(a, alpha, seed, reach):
    """整顆頭（臉＋外框線），不含脖子。"""
    face = ndi.binary_fill_holes(face_component(a, alpha, seed))
    dark = (lum(a) < 150) & (alpha > 0.3)
    near = dark & ndi.binary_dilation(face, iterations=reach)
    return ndi.binary_dilation(face | near, iterations=2), face


def face_geo(a, alpha, seed=(512, 250)):
    """量臉：頭頂、下巴、額頭那一列的寬度和中心（額頭量不到耳朵，比較準）。"""
    f = face_component(a, alpha, seed)
    ys = np.where(f.any(1))[0]
    top, chin = int(ys.min()), int(ys.max())
    xx = np.where(f[top + 200])[0]
    return dict(top=top, chin=chin, w=float(xx.max() - xx.min()), cx=float((xx.max() + xx.min()) / 2))


def align_to(a, alpha, g, ref, keep_ratio):
    """把這張圖的臉對齊到標準頭：寬度、中心、頭頂對齊。
    keep_ratio=False 時連下巴也對齊（五官照臉的比例放），True 時等比例（臉型、下巴要保留原本的長度）。"""
    sx = ref['w'] / g['w']
    sy = sx if keep_ratio else (ref['chin'] - ref['top']) / (g['chin'] - g['top'])
    ox, oy = g['cx'] - ref['cx'] / sx, g['top'] - ref['top'] / sy
    if abs(sx - 1) < .015 and abs(sy - 1) < .015 and abs(ox) < 4 and abs(oy) < 4:
        return a, alpha, False
    tf = lambda ch, cval: ndi.affine_transform(ch, (1 / sy, 1 / sx), offset=(oy, ox), order=1, mode='constant', cval=cval)
    out = np.dstack([tf(a[..., i], 0.0) for i in range(3)])
    return out, np.clip(tf(alpha, 0.0), 0, 1), (round(sx, 3), round(sy, 3))


def extract_neck(a, alpha, hm, chin, cx, fade=None):
    """從頭部特寫切出高解析度的脖子：下巴下面那塊皮膚，接近衣領的地方淡出，
    下巴底下再往上補到 y=480，換成任何下巴形狀都不會露出縫。"""
    skin = (alpha > .5) & (lum(a) > 95) & (a[..., 0] - a[..., 2] > 18) & ~hm
    lab, _ = ndi.label(skin)
    k = lab[chin + 40, int(cx)]
    if k == 0:
        return None
    neck = lab == k
    neck[:chin - 150] = False
    gray = (alpha > .5) & (np.abs(a[..., 0] - a[..., 2]) < 16) & (lum(a) > 140)
    neck = ndi.binary_fill_holes(neck)
    outl = ndi.binary_dilation(neck, iterations=6) & (alpha > .3) & ~gray & ~hm
    m = neck | outl
    m[:chin - 150] = False
    dist = ndi.distance_transform_edt(~gray)
    al = alpha * m * np.clip((dist - 4) / 36, 0, 1)
    col = a.copy()
    r0 = chin + 18
    row = np.where(m[r0])[0]
    x0, x1 = row.min(), row.max() + 1
    src = ndi.gaussian_filter1d(a[r0, x0:x1], 10, axis=0)
    src[:5], src[-5:] = a[r0, x0:x0 + 5], a[r0, x1 - 5:x1]   # 兩邊的外框線保留
    for y in range(480, r0):
        fill = ~m[y, x0:x1]
        col[y, x0:x1][fill] = src[fill]
        al[y, x0:x1] = np.where(fill, alpha[r0, x0:x1] * m[r0, x0:x1], al[y, x0:x1])
    if fade:   # 只負責下巴接縫，往下交給身體圖的脖子和衣領
        yy = np.arange(a.shape[0])[:, None]
        al = al * np.clip((chin + fade[1] - yy) / (fade[1] - fade[0]), 0, 1)
    return col, ndi.gaussian_filter(al, .6)


def save_layer(id_, rgb, alpha, space, extra=None, quality=92):
    a8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    ys, xs = np.where(a8 > 6)
    if len(ys) == 0:
        LOG.append(f'!! {id_}: 切不出東西，略過')
        return
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    im = Image.fromarray(np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), a8])[y0:y1, x0:x1], 'RGBA')
    path = os.path.join(OUT, id_ + '.webp')
    im.save(path, 'WEBP', quality=quality, method=6)
    buf = open(path, 'rb').read()
    it = dict(s=space, x=int(x0), y=int(y0), w=int(x1 - x0), h=int(y1 - y0), ph=False,
              src='data:image/webp;base64,' + base64.b64encode(buf).decode())
    if extra:
        it.update(extra)
    ITEMS[id_] = it
    LOG.append(f'   {id_:28s} {space}  ({x0},{y0}) {x1-x0}x{y1-y0}  {len(buf)//1024+1}KB')


FY = (lambda y: y, 1.0)


def add_face_svg(id_, x, y, w, h, body, extra=None):
    f, sy = FY
    add_svg(id_, 'head', x, f(y), w, h * sy, f'<g transform="translate(0,{f(0):.2f}) scale(1,{sy:.4f})">{body}</g>', extra)


def add_svg(id_, space, x, y, w, h, body, extra=None):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
           f'viewBox="{x:.1f} {y:.1f} {w:.0f} {h:.0f}">{body}</svg>')
    if id_ in ITEMS and not ITEMS[id_]['ph']:
        return  # 已經有正式圖
    open(os.path.join(OUT, id_ + '.svg'), 'w').write(svg)
    it = dict(s=space, x=round(x), y=round(y), w=round(w), h=round(h), ph=True,
              src='data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode())
    if extra:
        it.update(extra)
    ITEMS[id_] = it


# ---------- 切法 ----------
def cut_inner(new, base, inner, thr=25):
    """只在臉內側的部件：跟標準人比對，留下有變化的地方。"""
    d = np.abs(new - base).sum(-1)
    al = np.clip((d - 20) / 120, 0, 1) * inner
    lab, n = ndi.label(d * inner > thr + 15)
    if n == 0:
        return new, al * 0
    sizes = ndi.sum(al > 0, lab, range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s > 50])
    keep = ndi.binary_dilation(keep, iterations=3)
    al = al * keep
    # 半透明的邊緣把混進去的標準膚色扣掉，換深膚色時才不會有一圈淺色
    a3 = np.maximum(al, 0.3)[..., None]
    col = np.clip(base + (new - base) / a3, 0, 255)
    return col, al


def cut_outer(new, new_alpha, base, base_alpha, thr=40, grow=4):
    """會超出外框的部件（耳朵、頭髮、衣服）：外框外新長出來的東西整塊留下，
    外框內只留有變化的地方。"""
    d = np.abs(new * new_alpha[..., None] - base * base_alpha[..., None]).sum(-1)
    d = np.maximum(d, np.abs(new_alpha - base_alpha) * 255)
    ch = d > thr
    ch = ndi.binary_opening(ch, iterations=1)
    lab, n = ndi.label(ch)
    if n == 0:
        return new_alpha * 0
    sizes = ndi.sum(ch, lab, range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s > 80])
    keep = ndi.binary_fill_holes(ndi.binary_dilation(keep, iterations=grow))
    soft = ndi.gaussian_filter(keep.astype(np.float32), 1.0)
    return new_alpha * np.clip(soft * 1.6, 0, 1)


# ---------- 頭部 ----------
def process_head():
    global SKIN
    d = os.path.join(RAW, 'head')
    files = {os.path.splitext(f)[0]: os.path.join(d, f) for f in sorted(os.listdir(d))}
    base_raw = load(files.pop('base'), (HS, HS))
    base, base_alpha = key(base_raw)
    patch = (300, 380, 450, 574)
    SKIN = tuple(float(v) for v in np.median(base[300:380, 450:574].reshape(-1, 3), 0).round(1))
    seed = (HS // 2, 400)
    ref_geo = face_geo(base, base_alpha)
    hmask, face = head_mask(base, base_alpha, seed, 12)
    inner_base = ndi.binary_erosion(face, iterations=8)
    soft = lambda m: ndi.gaussian_filter(m.astype(np.float32), 0.8)
    save_layer('face_oval_base', base, base_alpha * soft(hmask), 'head')
    nk = extract_neck(base, base_alpha, hmask, int(np.where(face)[0].max()), HS / 2)
    if nk: save_layer('neck_f', nk[0], nk[1], 'head')

    ys, xs = np.where(face)
    geo = dict(top=int(ys.min()) - 7, chin=int(ys.max()) + 7, cx=float((xs.min() + xs.max()) / 2))
    rows = []
    for y in range(geo['top'], geo['chin'] + 1):
        xx = np.where(hmask[y])[0]
        rows.append((xx.max() - xx.min()) / 2 if len(xx) else 0)
    geo['hw'] = rows

    for name, path in files.items():
        new = load(path, (HS, HS))
        new, new_alpha = key(new)
        new, gain = normalize_skin(new, patch, SKIN)
        kind = name.split('_')[0]
        try:
            if kind == 'face' and '_chin_' not in name:
                raise SystemExit   # 各臉型本來就跟標準頭畫在一起，寬窄是臉型本身，不能拉
            ng = face_geo(new, new_alpha)
            new, new_alpha, moved = align_to(new, new_alpha, ng, ref_geo, kind not in ('eyes', 'brows', 'nose', 'mouth'))
        except SystemExit:
            ng, moved = None, False
        top_shift = int((np.abs(new_alpha[:500] - base_alpha[:500]) > 0.5).sum())
        note = (f'對齊 {moved}，' if moved else '') + f'膚色校正 {np.round(gain, 3).tolist() if gain is not None else "略過"}，上半外框差 {top_shift}px'
        if name == 'base_m':   # 男生的標準頭（鵝蛋臉）
            hm, fc = head_mask(new, new_alpha, seed, 12)
            save_layer('face_oval_base_m', new, new_alpha * soft(hm), 'head', dict(chinY=int(np.where(fc)[0].max()) + 7))
            nk = extract_neck(new, new_alpha, hm, int(np.where(fc)[0].max()), HS / 2, fade=(90, 150))
            if nk: save_layer('neck_m', nk[0], nk[1], 'head')
        elif kind == 'face':
            parts = name.split('_')
            id_ = f'face_{parts[1]}_{parts[3] if len(parts) > 3 else "base"}'
            hm, fc = head_mask(new, new_alpha, seed, 12)
            fy = np.where(fc)[0]
            save_layer(id_, new, new_alpha * soft(hm), 'head', dict(chinY=int(fy.max()) + 7))
        elif kind == 'chinmark':   # 雙下巴、美人溝：只取下巴附近比皮膚暗的線條和陰影，位置記成相對下巴
            _, fc = head_mask(new, new_alpha, seed, 12)
            ys = np.where(fc.any(1))[0]; chin = int(ys.max())
            band = np.zeros_like(fc); band[chin - 120:chin - 3, int(geo['cx']) - 150:int(geo['cx']) + 150] = True
            inner = ndi.binary_erosion(fc, iterations=5) & band
            al = np.clip((np.mean(SKIN) - 8 - lum(new)) / 55, 0, 1) * inner
            al = ndi.gaussian_filter(al, .6)
            col = np.clip(base + (new - base) / np.maximum(al, .3)[..., None], 0, 255)
            save_layer(name, new, al, 'head', dict(rel='chin'))
            ITEMS[name]['y'] -= chin + 7
        elif kind in ('eyes', 'brows', 'nose', 'mouth'):
            _, fc = head_mask(new, new_alpha, seed, 12)
            inner = inner_base & ndi.binary_erosion(fc, iterations=8)
            box = {'nose': (320, 610, 125), 'mouth': (525, 680, 135)}.get(kind)
            if box:   # 新一批的圖臉頰陰影跟標準頭不一樣，只在五官附近找差異
                lim = np.zeros_like(inner); cx = int(geo['cx'])
                lim[box[0]:box[1], cx - box[2]:cx + box[2]] = True
                inner = inner & lim
            col, al = cut_inner(new, base, inner)
            save_layer(name, col, al, 'head')
        elif kind == 'ears':   # 耳朵只留臉外側那兩塊，臉本身的差異不要
            al = cut_outer(new, new_alpha, base, base_alpha)
            _, fc = head_mask(new, new_alpha, seed, 12)
            side = np.zeros_like(fc); side[280:860] = True
            side[:, int(geo['cx']) - 200:int(geo['cx']) + 200] = False
            al = al * (side & ~ndi.binary_erosion(face, iterations=6))
            lab, n = ndi.label(al > .2)
            if n:
                sz = ndi.sum(al > .2, lab, range(1, n + 1)); big = [i + 1 for i in np.argsort(sz)[-2:]]
                al = al * ndi.binary_dilation(np.isin(lab, big), iterations=3)
            save_layer(name, new, al, 'head')
        elif kind in ('hair', 'acc'):
            save_layer(name if kind != 'hair' else name + '_front', new,
                       cut_outer(new, new_alpha, base, base_alpha), 'head')
        else:
            LOG.append(f'!! raw/head/{name}: 看不懂的檔名，略過')
            continue
        LOG.append(f'      └ {note}')
    eyes = [v for k, v in ITEMS.items() if k.startswith('eyes_') and not v['ph']]
    geo['eyeY'] = float(np.mean([v['y'] + v['h'] / 2 for v in eyes])) if eyes else 408.0
    return geo, base, base_alpha


# ---------- 身體 ----------
def prep_body(path, head_geo):
    base, alpha = key(load(path, (BW, BH)))
    hand_patch = (int(BH * .545), int(BH * .60), int(BW * .262), int(BW * .31))
    base, _ = normalize_skin(base, hand_patch, SKIN)   # 把身體的膚色對到頭部標準人
    seed = (BW // 2, int(BH * .145))
    hm, face = head_mask(base, alpha, seed, 6)
    ys, xs = np.where(face)
    top, chin, cx = int(ys.min()) - 3, int(ys.max()) + 3, (xs.min() + xs.max()) / 2
    full_base, full_alpha = base.copy(), alpha.copy()
    # 擦掉身體自己的頭，把脖子往上補，換上頭部特寫
    erase = ndi.binary_dilation(hm, iterations=3)
    yn = chin + 5
    run = np.where(alpha[yn] > 0.5)[0]
    run = run[(run > cx - BW * .12) & (run < cx + BW * .12)]
    xl, xr = int(run.min()), int(run.max())
    row = base[yn, xl:xr + 1]
    dark = np.where(lum(row) < 110)[0]
    ow = max(2, int((dark < 8).sum()))
    line_col = row[dark].mean(0) if len(dark) else np.array([94, 58, 54])
    shade = base[yn, int(cx)]
    alpha = np.where(erase, 0, alpha)
    y_up = int(top + (chin - top) * .55)
    for y in range(y_up, yn + 1):
        seg = slice(xl, xr + 1)
        hole = alpha[y, seg] < 0.99
        fill = np.tile(shade, (xr - xl + 1, 1))
        fill[:ow] = line_col
        fill[-ow:] = line_col
        base[y, seg] = np.where(hole[:, None], fill, base[y, seg])
        alpha[y, seg] = 1
    # 衣服上的暖色反光改成中性灰，遊戲才分得出「皮膚」和「衣服」：只有脖子、手、腳踝算皮膚
    yy, xx = np.mgrid[0:BH, 0:BW]
    lab, _ = ndi.label((alpha > .5) & (lum(base) > 100))   # 脖子：從下巴下方往外找，碰到領口的線就停
    neck = ndi.binary_dilation(lab == lab[chin + 25, int(cx)], iterations=2) & (yy < BH * .30) & (np.abs(xx - cx) < BW * .16)
    neck |= ndi.binary_dilation(hm, iterations=9) & (yy > chin - 12) & (alpha > .3) & (lum(base) > 100)   # 下巴底下那一圈反光
    skin_zone = neck | ((yy > BH * .525) & (yy < BH * .645) & (np.abs(xx - cx) > BW * .168)) | ((yy > BH * .868) & (yy < BH * .915))
    cloth = (~skin_zone) & (lum(base) > 105)
    g = lum(base)[..., None]
    base = np.where(cloth[..., None], np.clip(g * 1.0, 0, 255), base)
    # 皮膚區裡偏白的反光改成膚色，換深膚色時才不會留一圈白邊
    rb = base[..., 0] - base[..., 2]
    pale = ((skin_zone & (rb >= 5)) | neck) & (lum(base) > 200) & (rb < 45) & (alpha > 0.3)
    base = np.where(pale[..., None], np.array(SKIN) * np.clip(g / np.mean(SKIN), .9, 1.03), base)
    k = (chin - top) / (head_geo['chin'] - head_geo['top'])
    return base, alpha, dict(k=round(float(k), 5), cx=round(float(cx), 1), top=top, chin=chin), full_base, full_alpha


def process_body(head_geo):
    d = os.path.join(RAW, 'body')
    files = {os.path.splitext(f)[0]: os.path.join(d, f) for f in sorted(os.listdir(d))}
    base, alpha, anchor, full_base, full_alpha = prep_body(files.pop('base'), head_geo)
    save_layer('body_base', base, alpha, 'body')
    if 'base_m' in files:   # 男生身體，跟女生同一個畫面
        mb, ma, man, _, _ = prep_body(files.pop('base_m'), head_geo)
        save_layer('body_base_m', mb, ma, 'body')
        LOG.append(f'      └ 頭的位置：女 {anchor}，男 {man}（遊戲用女生的）')
    hand_patch = (int(BH * .545), int(BH * .60), int(BW * .262), int(BW * .31))
    for name, path in files.items():
        new, na = key(load(path, (BW, BH)))
        new, _ = normalize_skin(new, hand_patch, SKIN)
        kind = name.split('_')[0]
        if kind in ('top', 'bottom', 'acc', 'dress'):
            save_layer(name, new, cut_outer(new, na, full_base, full_alpha), 'body')
        else:
            LOG.append(f'!! raw/body/{name}: 看不懂的檔名，略過')
    zones = dict(neck=anchor['chin'], shoulder=int(BH * .235), hip=int(BH * .545), ankle=int(BH * .874), shoe=int(BH * .905), ground=int(BH * .972))
    return anchor, zones


def process_ui():
    """手部特寫：去背、裁切、縮小，只當縮圖用。"""
    d = os.path.join(RAW, 'ui')
    if not os.path.isdir(d):
        return
    for f in sorted(os.listdir(d)):
        name = os.path.splitext(f)[0]
        im = Image.open(os.path.join(d, f)).convert('RGB')
        im.thumbnail((480, 480), Image.LANCZOS)
        a, al = key(np.asarray(im).astype(np.float32))
        ys, xs = np.where(al > .5)
        rgba = np.dstack([np.clip(a, 0, 255), al * 255]).astype(np.uint8)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        path = os.path.join(OUT, name + '.webp')
        Image.fromarray(rgba, 'RGBA').save(path, 'WEBP', quality=90, method=6)
        buf = open(path, 'rb').read()
        ITEMS[name] = dict(s='ui', x=0, y=0, w=rgba.shape[1], h=rgba.shape[0], ph=False, src='data:image/webp;base64,' + base64.b64encode(buf).decode())
        LOG.append(f'   {name:28s} ui    {rgba.shape[1]}x{rgba.shape[0]}  {len(buf)//1024+1}KB')

# ---------- SVG 佔位圖 ----------
def sm(a, b, t):
    x = min(1, max(0, (t - a) / (b - a)))
    return x * x * (3 - 2 * x)


FACES = {  # 臉型：(臉頰加寬函式, 下顎起點 t0, p, q, 長度倍率)
    'oval':   (lambda t: 1.0, .70, 1.9, 1.6, 1.0),
    'square': (lambda t: 1 + .10 * sm(.4, .7, t), .74, 3.2, 2.2, 1.0),
    'long':   (lambda t: .90, .70, 1.9, 1.6, 1.07),
    'round':  (lambda t: 1 + .08 * sm(.3, .65, t), .62, 2.1, 2.0, .96),
    'fire':   (lambda t: 1.03, .55, 1.25, 1.15, 1.0),
    'earth':  (lambda t: 1 + .14 * sm(.35, .7, t), .72, 2.6, 2.0, 1.0),
}
CHINS = {'base': (0, 0, 0), 'round': (.8, .5, 0), 'pointed': None, 'protrude': (1.2, 0, .05)}


def face_outline(geo, face, chin, n=140):
    m, t0, p, q, ys = FACES[face]
    if CHINS[chin] is None:
        p, q = 1.35, 1.2
    else:
        dp, dq, dy = CHINS[chin]
        p, q, ys = p + dp, q + dq, ys + dy
    hw = geo['hw']
    H = geo['chin'] - geo['top']
    base_hw = lambda t: hw[min(len(hw) - 1, int(t * (len(hw) - 1)))] - 3.5
    w0 = base_hw(t0) * m(t0)
    pts = []
    for i in range(n + 1):
        t = i / n
        if t < t0:
            w = base_hw(t) * m(t)
        else:
            u = (t - t0) / (1 - t0)
            w = w0 * max(0, 1 - u ** p) ** (1 / q)
        pts.append([t, w])
    ws = np.array([w for _, w in pts])
    k = np.ones(7) / 7
    sm_ws = np.convolve(np.pad(ws, 3, mode='edge'), k, mode='valid')
    sm_ws[:6], sm_ws[-2:] = ws[:6], ws[-2:]
    Hn = (H - 7) * ys
    out = [(geo['cx'] - w, geo['top'] + 3.5 + t * Hn) for (t, _), w in zip(pts, sm_ws)]
    out += [(geo['cx'] + w, geo['top'] + 3.5 + t * Hn) for (t, _), w in reversed(list(zip(pts, sm_ws)))]
    return out, geo['top'] + 3.5 + Hn


def placeholder_faces(geo):
    skin = '#%02x%02x%02x' % tuple(int(round(v)) for v in SKIN)
    meta = {}
    for f in FACES:
        for c in CHINS:
            id_ = f'face_{f}_{c}'
            pts, chin_y = face_outline(geo, f, c)
            hw_ear = max(geo['cx'] - x for x, y in pts if 380 < y < 480)
            meta[id_] = dict(chinY=round(chin_y + 3.5), hwEar=round(hw_ear + 3.5))
            if id_ in ITEMS:
                ITEMS[id_].setdefault('chinY', meta[id_]['chinY'])
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            poly = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
            add_svg(id_, 'head', min(xs) - 6, min(ys) - 6, max(xs) - min(xs) + 12, max(ys) - min(ys) + 12,
                    f'<polygon points="{poly}" fill="{skin}" stroke="{LINE}" stroke-width="7" stroke-linejoin="round"/>',
                    dict(chinY=meta[id_]['chinY']))
    return meta


def mirror_pair(local, cxl, cy, cx=512.0):
    """local 是以「往外」為 +x 畫的一側，回傳左右兩側。"""
    dx = cx - cxl
    return (f'<g transform="translate({cx - dx:.1f},{cy}) scale(-1,1)">{local}</g>'
            f'<g transform="translate({cx + dx:.1f},{cy})">{local}</g>')


def placeholder_features(geo):
    cx = geo['cx']
    # 眼睛（參數：半寬、半高、外眼角上揚角度、虹膜半徑、虹膜 y、上眼線粗細）
    eyes = {'apricot': (50, 25, 2, 21, -3, 6), 'narrow': (58, 12, 4, 12, -1, 6), 'sanpaku': (52, 21, 3, 11, -3, 6),
            'droopy': (52, 19, -10, 17, -2, 6), 'upturned': (54, 18, 12, 16, -3, 7)}
    for name, (w, h, tilt, r, iy, lid) in eyes.items():
        cid = f'c_{name}'
        shape = f'M {-w},0 Q 0,{-2*h} {w},0 Q 0,{1.5*h} {-w},0 Z'
        local = (f'<g transform="rotate({-tilt})"><clipPath id="{cid}"><path d="{shape}"/></clipPath>'
                 f'<path d="{shape}" fill="#fff"/>'
                 f'<g clip-path="url(#{cid})"><circle cx="0" cy="{iy}" r="{r}" fill="#6b4640"/>'
                 f'<circle cx="0" cy="{iy}" r="{r*.5:.1f}" fill="#2e1a1c"/>'
                 f'<circle cx="{-r*.4:.1f}" cy="{iy - r*.4:.1f}" r="{r*.26:.1f}" fill="#fff"/></g>'
                 f'<path d="M {-w},0 Q 0,{1.5*h} {w},0" fill="none" stroke="#3b2224" stroke-width="2.5" stroke-linecap="round" opacity=".7"/>'
                 f'<path d="M {-w-2},1 Q 0,{-2*h} {w},0 L {w+11},-5" fill="none" stroke="#3b2224" stroke-width="{lid}" stroke-linecap="round" stroke-linejoin="round"/>'
                 f'<path d="M {-w*.6:.0f},{-h-12} Q 0,{-h-22} {w*.8:.0f},{-h-10}" fill="none" stroke="#8a5a52" stroke-width="2.5" stroke-linecap="round"/></g>')
        add_face_svg('eyes_' + name, cx - 230, 340, 460, 135, mirror_pair(local, 360, 408, cx))
    # 眉毛（路徑、粗細、顏色）
    brows = {'willow': ('M -66,8 Q 0,-16 72,4', 7, LINE), 'crescent': ('M -62,12 Q 6,-34 70,10', 5.5, LINE),
             'droop': ('M -64,-8 Q 0,-8 72,18', 9, LINE), 'thick': ('M -66,4 Q 0,-10 70,2', 19, LINE),
             'sparse': ('M -60,4 Q 0,-8 66,3', 6, '#b9958b')}
    for name, (dpath, sw, col) in brows.items():
        local = f'<path d="{dpath}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round"/>'
        add_face_svg('brows_' + name, cx - 240, 295, 480, 100, mirror_pair(local, 370, 346, cx))
    # 鼻子（都以 (cx,520) 為鼻頭附近）
    L = lambda d, sw=5: f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'
    c = cx
    noses = {
        'gall':   L(f'M {c-4},440 L {c-6},500') + L(f'M {c-30},522 Q {c-34},548 {c-14},546 Q {c},560 {c+14},546 Q {c+34},548 {c+30},522', 5.5),
        'garlic': L(f'M {c-4},455 L {c-6},500') + L(f'M {c-44},520 Q {c-48},552 {c-20},548 Q {c},566 {c+20},548 Q {c+48},552 {c+44},520', 6),
        'hook':   L(f'M {c-2},430 Q {c+16},480 {c+2},545 Q {c-4},556 {c-12},548') + L(f'M {c-28},530 Q {c-30},546 {c-16},544', 4.5) + L(f'M {c+26},530 Q {c+28},546 {c+14},544', 4.5),
        'straight': L(f'M {c-8},425 L {c-10},540 Q {c},552 {c+14},542', 5),
        'button': L(f'M {c-16},532 Q {c},548 {c+16},532', 5),
        'upturned': L(f'M {c-22},528 Q {c},510 {c+22},528', 5) + f'<ellipse cx="{c-11}" cy="537" rx="5" ry="3.5" fill="{LINE}"/><ellipse cx="{c+11}" cy="537" rx="5" ry="3.5" fill="{LINE}"/>',
    }
    for name, body in noses.items():
        add_face_svg('nose_' + name, cx - 70, 415, 140, 165, body)
    # 嘴（半寬、上唇厚、下唇厚、嘴角上揚、唇峰）
    mouths = {'smile': (58, 9, 13, 12, 3), 'full': (52, 17, 22, 0, 4), 'thin': (54, 5, 6, 0, 2),
              'wide': (78, 9, 13, 2, 3), 'cherry': (30, 11, 14, 0, 4), 'bow': (54, 12, 13, 0, 9)}
    for name, (w, up, lo, lift, bow) in mouths.items():
        y = 622
        upper = (f'M {c-w},{y-lift} Q {c-w*.5},{y-up-2} {c-w*.18},{y-up-bow} Q {c},{y-up+bow*.6} {c+w*.18},{y-up-bow} '
                 f'Q {c+w*.5},{y-up-2} {c+w},{y-lift} Q {c},{y+5} {c-w},{y-lift} Z')
        lower = f'M {c-w},{y-lift} Q {c},{y+5} {c+w},{y-lift} Q {c},{y+lo*2} {c-w},{y-lift} Z'
        body = (f'<path d="{upper}" fill="#d9857e"/><path d="{lower}" fill="#e69c93"/>'
                + L(f'M {c-w-3},{y-lift-1} Q {c},{y+6} {c+w+3},{y-lift-1}', 4.5))
        add_face_svg('mouth_' + name, cx - 100, 580, 200, 95, body)
    # 下巴疊加層（以下巴底為 y=0，遊戲會依臉型的 chinY 定位）
    add_svg('chinmark_double', 'head', cx - 110, -70, 220, 60,
            L(f'M {c-84},-58 Q {c},-6 {c+84},-58', 4.5).replace(LINE, '#b9776a'), dict(rel='chin'))
    add_svg('chinmark_cleft', 'head', cx - 15, -62, 30, 50, L(f'M {c},-54 L {c},-22', 4.5).replace(LINE, '#b9776a'), dict(rel='chin'))


def placeholder_ears(geo, hw_ear):
    skin = '#%02x%02x%02x' % tuple(int(round(v)) for v in SKIN)
    # (寬倍率, 高倍率, 耳垂大小, y 位移)
    ears = {'fortune': (1.15, 1.25, 1.5, 6), 'protruding': (1.7, 1.05, 1.0, 0), 'flat': (.72, 1.0, .9, 0),
            'high': (1.0, 1.0, 1.0, -62), 'small': (.78, .74, .8, 0)}
    for name, (sx, sy, lobe, dy) in ears.items():
        p = (f'M -30,{-52*sy:.0f} C {30*sx:.0f},{-74*sy:.0f} {62*sx:.0f},{-20*sy:.0f} {42*sx:.0f},{26*sy:.0f} '
             f'C {36*sx*lobe:.0f},{(60+18*(lobe-1))*sy:.0f} {4*sx:.0f},{(66+26*(lobe-1))*sy:.0f} -30,{44*sy:.0f} Z')
        inner = f'M {8*sx:.0f},{-34*sy:.0f} C {34*sx:.0f},{-34*sy:.0f} {36*sx:.0f},{4*sy:.0f} {14*sx:.0f},{22*sy:.0f}'
        local = (f'<path d="{p}" fill="{skin}" stroke="{LINE}" stroke-width="7" stroke-linejoin="round"/>'
                 f'<path d="{inner}" fill="none" stroke="#b9776a" stroke-width="4" stroke-linecap="round"/>')
        add_face_svg('ears_' + name, geo['cx'] - hw_ear - 110, 290, (hw_ear + 110) * 2, 300,
                mirror_pair(local, geo['cx'] - hw_ear + 14, 448 + dy, geo['cx']), dict(behind=True))


def placeholder_hair(geo):
    cx, top = geo['cx'], geo['top']
    hw = geo['hw']
    H = geo['chin'] - geo['top']
    fill, st = '#dcdce0', f'stroke="{LINE}" stroke-width="7" stroke-linejoin="round"'
    edge = lambda t, d: (hw[int(t * (len(hw) - 1))] + d, top + t * H)

    def cap(t_end, d=16, bang=None):
        ts = [i / 40 * t_end for i in range(41)]
        left = [(cx - edge(t, d)[0], edge(t, d)[1] - (d if t < .1 else d * (1 - t / t_end))) for t in ts][::-1]
        right = [(cx + edge(t, d)[0], edge(t, d)[1] - (d if t < .1 else d * (1 - t / t_end))) for t in ts]
        pts = left + right
        dpath = 'M ' + ' L '.join(f'{x:.0f},{y:.0f}' for x, y in pts)
        xr, yr = right[-1]
        xl, yl = left[0]
        dpath += ' ' + (bang or f'Q {cx+150},{top+250} {cx+60},{top+215} Q {cx-40},{top+150} {cx-150},{top+235} Q {cx-215},{top+270} {xl:.0f},{yl:.0f}') + ' Z'
        return f'<path d="{dpath}" fill="{fill}" {st}/>'

    styles = {}
    styles['short'] = (cap(.42), None)
    side = lambda y: (f'<path d="M {cx-292},{top+250} Q {cx-312},{y-120} {cx-272},{y} L {cx-208},{y} Q {cx-236},{top+420} {cx-252},{top+250} Z" fill="{fill}" {st}/>'
                      f'<path d="M {cx+292},{top+250} Q {cx+312},{y-120} {cx+272},{y} L {cx+208},{y} Q {cx+236},{top+420} {cx+252},{top+250} Z" fill="{fill}" {st}/>')
    back = lambda y, w: f'<path d="M {cx-280},{top+210} Q {cx-w},{y-200} {cx-w+30},{y} L {cx+w-30},{y} Q {cx+w},{y-200} {cx+280},{top+210} Q {cx},{top-60} {cx-280},{top+210} Z" fill="#cfcfd4" {st}/>'
    styles['bob'] = (side(top + 690) + cap(.40), back(top + 720, 330))
    styles['long'] = (side(top + 900) + cap(.40), back(top + 1230, 345))
    styles['ponytail'] = (cap(.42), f'<path d="M {cx+170},{top+60} Q {cx+420},{top+80} {cx+390},{top+420} Q {cx+375},{top+640} {cx+300},{top+760} Q {cx+290},{top+520} {cx+250},{top+330} Z" fill="#cfcfd4" {st}/>')
    styles['bun'] = (f'<circle cx="{cx}" cy="{top-58}" r="92" fill="{fill}" {st}/>' + cap(.42), None)
    for name, (front, bk) in styles.items():
        add_svg(f'hair_{name}_front', 'head', cx - 440, top - 170, 880, 1500, front)
        if bk:
            add_svg(f'hair_{name}_back', 'head', cx - 440, top - 170, 880, 1500, bk)


def placeholder_ui():
    skin = '#%02x%02x%02x' % tuple(int(round(v)) for v in SKIN)
    st = f'fill="{skin}" stroke="{LINE}" stroke-width="5" stroke-linejoin="round"'
    # 手部特寫（掌寬、掌高、指長、指寬）
    hands = {'soft': (112, 104, 62, 25), 'slender': (88, 92, 100, 18), 'square': (104, 100, 78, 22), 'small': (78, 74, 54, 17)}
    for name, (pw, ph, fl, fw) in hands.items():
        px, py = 120 - pw / 2, 222 - ph
        body = ''
        ratios = [.82, .96, 1.0, .9]
        gap = (pw - 4 * fw) / 3
        for i, r in enumerate(ratios):
            x = px + i * (fw + gap)
            body += f'<rect x="{x:.0f}" y="{py - fl*r + 12:.0f}" width="{fw}" height="{fl*r + 10:.0f}" rx="{fw/2}" {st}/>'
        body += (f'<rect x="{px + pw - 6:.0f}" y="{py + ph*.28:.0f}" width="{fl*.62:.0f}" height="{fw}" rx="{fw/2}" transform="rotate(-28 {px+pw:.0f} {py+ph*.4:.0f})" {st}/>'
                 f'<rect x="{px:.0f}" y="{py:.0f}" width="{pw}" height="{ph}" rx="{26 if name != "square" else 12}" {st}/>'
                 f'<path d="M {px+pw*.2:.0f},{py+ph*.42:.0f} Q {px+pw*.5:.0f},{py+ph*.3:.0f} {px+pw*.84:.0f},{py+ph*.5:.0f} M {px+pw*.22:.0f},{py+ph*.68:.0f} Q {px+pw*.5:.0f},{py+ph*.56:.0f} {px+pw*.7:.0f},{py+ph*.8:.0f}" fill="none" stroke="#b9776a" stroke-width="3" stroke-linecap="round"/>')
        add_svg('hand_' + name, 'ui', 0, 0, 240, 240, body)
    master = (f'<circle cx="60" cy="52" r="34" {st}/><path d="M 24,40 Q 60,-8 96,40 Q 60,26 24,40 Z" fill="#6f7fae" stroke="{LINE}" stroke-width="5" stroke-linejoin="round"/>'
              f'<path d="M 34,66 Q 60,84 86,66 Q 82,112 60,116 Q 38,112 34,66 Z" fill="#fff" stroke="{LINE}" stroke-width="5" stroke-linejoin="round"/>'
              f'<path d="M 40,50 q 8,-7 16,0 M 64,50 q 8,-7 16,0" fill="none" stroke="{LINE}" stroke-width="4" stroke-linecap="round"/>')
    add_svg('master', 'ui', 0, 0, 120, 120, master)


# ---------- 主程式 ----------
def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(('.webp', '.svg')):
            os.remove(os.path.join(OUT, f))
    geo, _, _ = process_head()
    anchor, zones = process_body(geo)
    process_ui()
    global FY
    sy = (geo['chin'] - geo['eyeY']) / (745 - 408)   # 佔位圖原本照舊標準人畫的，換算到新標準人的五官位置
    FY = (lambda y: geo['eyeY'] + (y - 408) * sy, sy)
    face_meta = placeholder_faces(geo)
    hw_ear = face_meta['face_oval_base']['hwEar']
    placeholder_features(geo)
    placeholder_ears(geo, hw_ear)
    placeholder_hair(geo)
    placeholder_ui()
    meta = dict(skin=list(SKIN), head=dict(size=HS, top=geo['top'], chin=geo['chin'], cx=geo['cx'], hwEar=hw_ear, eyeY=round(geo['eyeY'], 1)),
                body=dict(w=BW, h=BH, anchor=anchor, zones=zones), faces=face_meta)
    manifest = {k: {kk: vv for kk, vv in v.items() if kk != 'src'} for k, v in ITEMS.items()}
    json.dump(dict(meta=meta, items=manifest), open(os.path.join(OUT, 'manifest.json'), 'w'), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, 'pack.js'), 'w') as f:
        f.write('// 由 tools/build_assets.py 產生，不要手動改\nwindow.PACK=' + json.dumps(dict(meta=meta, items=ITEMS), ensure_ascii=False, separators=(',', ':')) + ';\n')
    real = [k for k, v in ITEMS.items() if not v['ph']]
    print('\n'.join(LOG))
    print(f'\n正式圖 {len(real)} 個、佔位圖 {len(ITEMS) - len(real)} 個；pack.js {os.path.getsize(os.path.join(OUT, "pack.js"))//1024}KB')
    print('meta:', json.dumps({k: v for k, v in meta.items() if k != 'faces'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
