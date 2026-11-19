# -*- coding: utf-8 -*-
"""
定位最终版 (v6): 以提示字符为主体的精确匹配
  1. 提示图 -> 连通域+x重叠 分组出 N 个字符 -> 删掉贯穿整行的装饰横杠
     -> clean(干净提示图) + 每个字符的干净 bbox
  2. 对每个提示字符 i:
       尺度由 提示字符高 / 背景候选高 直接决定 (只留 ±12% 浮动)
       在提示字符原位置附近小范围搜索, 用 IoU 找背景里最佳对应
  3. 全局一对一 (匈牙利), 输出按提示顺序的点击点
"""
import io, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import capsolver as cs


def _longest_run(row_mask):
    xs = np.where(row_mask)[0]
    if len(xs) == 0:
        return 0
    best = 0
    start = prev = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
        else:
            best = max(best, prev - start + 1)
            start = prev = x
    return max(best, prev - start + 1)


def clean_tip(tip_img, thresh=60, decor_ratio=0.55):
    """
    分组提示图里的字符 (保留原像素, 含素材自带横杠)
    thresh=60: 只取纯黑, 去掉抗锯齿灰边, 与背景的纯黑核心笔画粗细一致
    """
    g = cs.to_gray(tip_img)
    mask = (g < thresh).astype(np.uint8) * 255
    groups = cs.group_chars(mask, min_area=40)
    out = []
    for (x, y, w, h, piece) in groups:
        if piece.shape[0] < 6 or piece.shape[1] < 4:
            continue
        out.append((x, y, w, h, piece))
    return mask, out


def _iou_at(reg, tpl):
    inter = np.logical_and(reg, tpl).sum()
    union = np.logical_or(reg, tpl).sum()
    return float(inter) / float(union) if union else 0.0


def cover_ratio(tpl, cand, size: int = 64) -> float:
    """
    覆盖度: |候选 ∩ 模板| / |候选|
    提示条的字符带装饰横杠 (模板 = 干净字符 + 装饰), 背景里是干净字符,
    所以"候选 ⊆ 模板" 才是正确判据 —— 对多余笔画天然鲁棒, 比 IoU 强得多。
    """
    a = _to_canvas(tpl, size) > 0
    b = _to_canvas(cand, size) > 0
    inter = np.logical_and(a, b).sum()
    tot = b.sum()
    return float(inter) / float(tot) if tot else 0.0


def _to_canvas(mask, size=96):
    """等比缩放 + 居中放到统一画布, 让不同尺度的形状可比"""
    m = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        return np.zeros((size, size), np.uint8)
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = (size - 8) / float(max(h, w))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas


_HOG = None            # 惰性创建: 有的环境(如 CI 上的系统 cv2)没有 HOGDescriptor
_HOG_OK = None         # None=未探测, True/False=探测结果


def _get_hog():
    global _HOG, _HOG_OK
    if _HOG_OK is None:
        try:
            _HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
            _HOG_OK = True
        except Exception:
            _HOG_OK = False
    return _HOG if _HOG_OK else None


_SVM = None
_NORM = None
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captcha_svm.xml")
NORM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svm_norm.npz")


def load_svm():
    """加载 SVM 打分器 (没有模型就返回 None)"""
    global _SVM, _NORM
    if _SVM is not None:
        return _SVM
    if os.path.exists(MODEL_PATH) and os.path.exists(NORM_PATH):
        try:
            _SVM = cv2.ml.SVM_load(MODEL_PATH)
            _NORM = np.load(NORM_PATH)
            print("[SVM] 模型已加载")
        except Exception as e:
            print(f"[SVM] 加载失败: {e}")
            _SVM = None
    return _SVM


def pair_feature(tpl, cand):
    """
    (模板mask, 候选mask) -> 特征向量
    核心: 把两个形状的轮廓叠加成一张图(模板=白, 候选=灰), HOG 描述"重合程度",
          这比左右拼接直接得多, SVM 能直接学到"两个形状是否对得上"
    没有 HOGDescriptor 的环境降级: HOG 部分填 0, 只用几何标量
    (长度必须跟训练时一致, 所以用零占位, 不改向量长度)
    """
    ca = _to_canvas(tpl, 64)
    cb = _to_canvas(cand, 64)
    ea = cv2.Canny(ca, 50, 150)
    eb = cv2.Canny(cb, 50, 150)
    both = np.zeros((64, 64), np.uint8)
    both[ea > 0] = 255
    both[eb > 0] = np.maximum(both[eb > 0], 130)
    hog = _get_hog()
    if hog is not None:
        h = hog.compute(both).flatten()
    else:
        h = np.zeros(1764, np.float32)      # 7*7*2*2*9 = 1764, 跟有 HOG 时同长度
    ta = cv2.resize(ca, (32, 32), interpolation=cv2.INTER_AREA)
    tb = cv2.resize(cb, (32, 32), interpolation=cv2.INTER_AREA)
    iou = _iou_at(ta, tb)
    sv = cs.iou(cs.shape_feature(ta, 32, 0), cs.shape_feature(tb, 32, 0))
    ch = chamfer_score(ca, cb)
    cov = cover_ratio(tpl, cand)
    # 反向覆盖: 真配对时模板的绝大部分也在候选里(装饰只占一部分)
    a64 = _to_canvas(tpl, 64) > 0
    b64 = _to_canvas(cand, 64) > 0
    inter = np.logical_and(a64, b64).sum()
    covb = float(inter) / float(a64.sum()) if a64.sum() else 0.0
    # 尺度先验: 正确配对时 背景字符高 / 提示字符高 集中在 ~1.5
    h_t = float(np.count_nonzero(tpl.any(axis=1)))
    h_c = float(np.count_nonzero(cand.any(axis=1)))
    scale = h_c / max(1.0, h_t)
    scal = np.array([iou, sv, ch, scale, cov, covb], np.float32)
    return np.concatenate([h, scal]).astype(np.float32)


def svm_score(tpl, cand):
    """SVM 对这一对 (模板, 候选) 的匹配打分 0~1"""
    svm = load_svm()
    if svm is None:
        return None
    try:
        f = pair_feature(tpl, cand)
        n_hog = int(_NORM["n_hog"])
        H = (f[:n_hog] - _NORM["mu_h"]) / _NORM["sd_h"]
        S = (f[n_hog:] - _NORM["mu_s"]) / _NORM["sd_s"]
        pca = _NORM["pca"]; center = _NORM["center"]
        proj = (H - center) @ pca
        feat = np.concatenate([proj, S * 5.0]).reshape(1, -1).astype(np.float32)
        r = svm.predict(feat)          # OpenCV 返回 (labels, values)
        vals = r[1] if isinstance(r, tuple) else r
        vals = np.asarray(vals)
        if vals.dtype == object or vals.size == 0:
            return 0.0
        return float(np.ravel(vals)[0])
    except Exception as e:
        if not getattr(svm_score, "_warned", False):
            svm_score._warned = True
            print(f"[SVM] 打分异常: {type(e).__name__}: {str(e)[:80]}")
        return None


def _edge(m):
    e = cv2.Canny((m > 0).astype(np.uint8) * 255, 50, 150)
    return e


def chamfer_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    轮廓 Chamfer 相似度: 对笔画粗细/轻微形变/缩放鲁棒, 比 IoU 更适合合成字符。
    输入二值图, 返回 0~1
    """
    ea = _edge(_to_canvas(a))
    eb = _edge(_to_canvas(b))
    if ea.sum() == 0 or eb.sum() == 0:
        return 0.0
    da = cv2.distanceTransform((eb == 0).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((ea == 0).astype(np.uint8), cv2.DIST_L2, 3)
    fwd = float(da[ea > 0].mean())
    bwd = float(db[eb > 0].mean())
    return 1.0 - min(1.0, (fwd + bwd) / 12.0)


ROT_ANGLES = (0, 8, -8, 16, -16, 24, -24, 30, -30)


def rot_mask(m, ang):
    """旋转掩码(保持画布尺寸), 用于旋转不变匹配"""
    if not ang:
        return m
    if m.dtype == bool:
        m = m.astype(np.uint8) * 255
    h, w = m.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
    return cv2.warpAffine(m, M, (w, h))


def best_rot_match(tf, tpl_mask, cand_mask, angles=ROT_ANGLES):
    """
    旋转不变匹配: 提示里的字符可能是旋转的, 而背景里的同一个字符不旋转(或反之)。
    扫若干角度取形状 IoU 与覆盖度的最优值。
    返回 (best_shape_iou, best_cover)
    """
    best_sv = 0.0
    best_cov = 0.0
    for ang in angles:
        m2 = rot_mask(cand_mask, ang)
        f2 = cs.shape_feature(m2, 32, 0)
        v = cs.iou(tf, f2)
        if v > best_sv:
            best_sv = v
        c = cover_ratio(tpl_mask, m2)      # 候选⊆模板 的覆盖度
        if c > best_cov:
            best_cov = c
    return best_sv, best_cov


# ------------------------------------------------------------------
# sprite 模板定位法 (思路同 chizw/Rainyun-Qiandao 的 ICR.py)
#   提示条里的图案集合(sprite)按 x 排序 = 点击顺序;
#   两边都做"膨胀->降采样->二值->放大"归一化, 笔宽被拉平;
#   每个图案扫 -45~45 度旋转, 在背景黑块内滑窗匹配, 最后贪心消冲突。
# ------------------------------------------------------------------

def _black_mask(img, thr=25):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.inRange(img, (0, 0, 0), (thr, thr, thr))


def _preprocess_mask(img, scale_factor=4, kernel_size=2, iterations=1):
    k = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.dilate(img, k, iterations=iterations)
    h, w = img.shape[:2]
    img = cv2.resize(img, (max(1, w // scale_factor), max(1, h // scale_factor)),
                     interpolation=cv2.INTER_AREA)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)


def _rot_expand(img, angle):
    """旋转并放大画布(避免裁掉内容)"""
    if not angle:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    nh = int(w * abs(np.sin(np.radians(angle))) + h * abs(np.cos(np.radians(angle))))
    nw = int(h * abs(np.sin(np.radians(angle))) + w * abs(np.cos(np.radians(angle))))
    M[0, 2] += (nw - w) / 2.0
    M[1, 2] += (nh - h) / 2.0
    return cv2.warpAffine(img, M, (nw, nh), borderValue=(0, 0, 0))


def _rects(binary, min_area=100):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            out.append((x, y, w, h))
    return out


def _merge_rects(rects, max_distance=0):
    """合并重叠或距离相近的矩形"""
    rects = [list(r) for r in rects]
    changed = True
    while changed and len(rects) > 1:
        changed = False
        new = []
        used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]:
                continue
            cur = rects[i]
            used[i] = True
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                r = rects[j]
                ix = max(0, min(cur[0] + cur[2], r[0] + r[2]) - max(cur[0], r[0]))
                iy = max(0, min(cur[1] + cur[3], r[1] + r[3]) - max(cur[1], r[1]))
                inter = ix * iy
                dx = max(0, max(cur[0], r[0]) - min(cur[0] + cur[2], r[0] + r[2]))
                dy = max(0, max(cur[1], r[1]) - min(cur[1] + cur[3], r[1] + r[3]))
                near = (dx * dx + dy * dy) ** 0.5 <= max_distance
                if inter > 0 or near:
                    x0 = min(cur[0], r[0]); y0 = min(cur[1], r[1])
                    x1 = max(cur[0] + cur[2], r[0] + r[2]); y1 = max(cur[1] + cur[3], r[1] + r[3])
                    cur = [x0, y0, x1 - x0, y1 - y0]
                    used[j] = True
                    changed = True
            new.append(cur)
        rects = new
    return [tuple(r) for r in rects]


def _black_mask_auto(img, max_thr=140):
    """
    背景图的合成字符常是"深灰描边"而非纯黑, 固定阈值会把笔画丢掉。
    只在暗部(0~max_thr)做 Otsu, 自适应出阈值。
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = g[g < max_thr]
    if dark.size == 0:
        return cv2.inRange(img, (0, 0, 0), (0, 0, 0))
    thr, _ = cv2.threshold(dark.reshape(-1, 1), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(max(20, min(thr, max_thr)))
    return cv2.inRange(img, (0, 0, 0), (thr, thr, thr))


def _bg_mask(img):
    """
    背景掩码: 近黑 **且中性**(R≈G≈B)。
    合成字符是纯黑描边; 照片的暗部是偏色的(深棕/深蓝), 用中性判据就能干净分开。
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mx = img.max(axis=2).astype(np.int16)
    mn = img.min(axis=2).astype(np.int16)
    m = ((mx < 62) & (mx > 3) & ((mx - mn) < 38)).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def _sprite_patterns(sp_mask):
    """
    切出 sprite 图里的每个图案(提示字符):
    先删掉贯穿全宽的装饰横杠, 再用列投影在空白列处切开(字符之间必有空白列)。
    连通域分组只作为兜底。
    """
    W = sp_mask.shape[1]
    m = sp_mask.copy()
    if W:
        cov = (m > 0).sum(axis=1) / float(W)
        m[cov >= 0.92] = 0                     # 贯穿全宽的行 = 装饰横杠
    col = (m > 0).sum(axis=0)
    segs = []
    x = 0
    while x < W:
        if col[x] == 0:
            x += 1
            continue
        s = x
        while x < W and col[x] > 0:
            x += 1
        segs.append((s, x - s))
    out = []
    for (x0, w) in segs:
        sub = m[:, x0:x0 + w]
        ys = np.where(sub.any(axis=1))[0]
        if len(ys) == 0:
            continue
        out.append((x0, int(ys.min()), w, int(ys.max() - ys.min() + 1)))
    if not out:
        out = _group_by_x(_rects(sp_mask, 40), gap=12)
    return out


def locate_by_edge(bg_img, sprite_img, angles=None, scales=None, verbose=False):
    """
    直接在整张背景的边缘图上滑窗匹配(不做背景分割 -> 不会漏掉淡描边字符)。
    1. sprite 图按 x 分组得到每个提示图案
    2. 每个图案扫角度 x 尺度, 用 Canny 边缘图做 matchTemplate
    返回 [(x, y, sim, idx), ...] 顺序 = 提示顺序
    """
    if angles is None:
        angles = list(range(-45, 46, 3))
    if scales is None:
        scales = (1.1, 1.35, 1.6, 1.9, 2.2)
    g = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY) if bg_img.ndim == 3 else bg_img
    bg_edge = cv2.Canny(g, 40, 120)
    sp = sprite_img if sprite_img.ndim == 3 else cv2.cvtColor(sprite_img, cv2.COLOR_GRAY2BGR)
    sp_mask = _black_mask(sp, 25)
    sp_regions = _sprite_patterns(sp_mask)
    sp_regions.sort(key=lambda r: r[0])
    if verbose:
        print(f"    [edge] sprite 图案 {len(sp_regions)} 个: {sp_regions[:6]}")

    hits = []
    for pi, (sx, sy, sw, sh) in enumerate(sp_regions):
        pat = sp_mask[sy:sy + sh, sx:sx + sw]
        pe = cv2.Canny((pat > 0).astype(np.uint8) * 255, 40, 120)
        if pe.sum() == 0:
            continue
        for ang in angles:
            pe_r = _rot_expand(pe, ang)
            for sc in scales:
                tw = int(round(pe_r.shape[1] * sc))
                th = int(round(pe_r.shape[0] * sc))
                if tw < 8 or th < 8 or tw >= bg_edge.shape[1] or th >= bg_edge.shape[0]:
                    continue
                t = cv2.resize(pe_r, (tw, th), interpolation=cv2.INTER_NEAREST)
                try:
                    res = cv2.matchTemplate(bg_edge, t, cv2.TM_CCOEFF_NORMED)
                except cv2.error:
                    continue
                _, mx, _, pos = cv2.minMaxLoc(res)
                hits.append({"pi": pi, "sim": float(mx), "ang": ang, "sc": sc,
                             "x": pos[0], "y": pos[1], "w": tw, "h": th})
    if not hits:
        return []
    hits.sort(key=lambda h: -h["sim"])
    used_p, used_xy = set(), set()
    final = []
    for h in hits:
        if h["pi"] in used_p:
            continue
        k = (h["x"] // 30, h["y"] // 30)
        if k in used_xy:
            continue
        used_p.add(h["pi"]); used_xy.add(k)
        final.append(h)
        if len(used_p) >= len(sp_regions):
            break
    final.sort(key=lambda h: h["pi"])
    if verbose:
        for h in final:
            print(f"    [edge] 图案{h['pi']+1} ang={h['ang']} sc={h['sc']} "
                  f"sim={h['sim']*100:.1f}% @({h['x']},{h['y']},{h['w']},{h['h']})")
    return [(h["x"] + h["w"] / 2.0, h["y"] + h["h"] / 2.0, h["sim"], h["pi"]) for h in final]


def _group_by_x(rects, gap=12):
    """
    把属于同一个字符的笔画碎片合并: x 方向近 且 y 方向也近才算同一字符。
    (只看 x 会把上下错开的不同字符也粘在一起)
    返回 [(x, y, w, h)]
    """
    def gap_of(a0, a1, b0, b1):
        if a1 < b0:
            return b0 - a1
        if b1 < a0:
            return a0 - b1
        return 0

    rects = sorted(rects, key=lambda r: r[0])
    groups = []
    for (x, y, w, h) in rects:
        x1, y1 = x + w - 1, y + h - 1
        placed = False
        for g in groups:
            if (gap_of(g[0], g[1], x, x1) <= gap
                    and gap_of(g[2], g[3], y, y1) <= gap):
                g[0] = min(g[0], x); g[1] = max(g[1], x1)
                g[2] = min(g[2], y); g[3] = max(g[3], y1)
                placed = True
                break
        if not placed:
            groups.append([x, x1, y, y1])
    return [(g[0], g[2], g[1] - g[0] + 1, g[3] - g[2] + 1) for g in groups]


def locate_by_sprite(bg_img, sprite_img, max_bg=10, angles=None,
                     max_cand=10, verbose=False):
    """
    用 sprite 图(提示条图案集合)在背景图里定位每个图案的位置。
    返回 [(x, y), ...] —— 顺序 = sprite 里图案从左到右 = 点击顺序
    """
    if angles is None:
        angles = list(range(-45, 46, 1))
    # 两边都用原分辨率, 尺度在匹配时按"背景块高 / 图案高"对齐
    bg_mask = _bg_mask(bg_img)
    sp = sprite_img
    if sp.ndim == 2:
        sp = cv2.cvtColor(sp, cv2.COLOR_GRAY2BGR)
    sp_mask = _black_mask(sp, 25)

    # 字符由多个笔画组成 -> 必须按 x 区间分组, 不能直接用连通域
    bg_regions = _group_by_x(_rects(bg_mask, 40), gap=12)
    bg_regions.sort(key=lambda r: r[2] * r[3], reverse=True)
    bg_regions = bg_regions[:max_bg]
    sp_regions = _group_by_x(_rects(sp_mask, 40), gap=12)
    sp_regions.sort(key=lambda r: r[0])          # 从左到右 = 提示顺序
    # 去掉贯穿整个 sprite 宽度的扁横条(装饰线, 不是字符)
    if sp_regions:
        sp_w = sp_mask.shape[1]
        hs = sorted(r[3] for r in sp_regions)
        med_h = hs[len(hs) // 2]
        keep = [r for r in sp_regions
                if r[3] >= 0.45 * med_h
                and r[2] < 0.92 * sp_w
                and not (r[2] > 1.4 * r[3] and r[3] < 0.75 * med_h)]
        if keep:
            sp_regions = keep
    if verbose:
        print(f"    [sprite] 背景黑块 {len(bg_regions)} 个, sprite 图案 {len(sp_regions)} 个")
        print(f"    [sprite] 背景块: {bg_regions[:6]}")
        print(f"    [sprite] 图案: {sp_regions[:6]}")

    matches = []
    for si, (sx, sy, sw, sh) in enumerate(sp_regions):
        roi = sp_mask[sy:sy + sh, sx:sx + sw]
        for ang in angles:
            rimg = _rot_expand(roi, -ang)
            rr = _rects(rimg, 0)
            if not rr:
                continue
            x_r, y_r, w_r, h_r = max(rr, key=lambda r: r[2] * r[3])
            troi = rimg[y_r:y_r + h_r, x_r:x_r + w_r]
            if troi.size == 0:
                continue
            for bi, (bx, by, bw, bh) in enumerate(bg_regions):
                broi = bg_mask[by:by + bh, bx:bx + bw]
                if broi.size == 0:
                    continue
                # 尺度对齐: 以"背景块高 / 图案高"为基准, 上下浮动
                base = bh / float(max(4, h_r))
                for f in (0.70, 0.84, 1.0, 1.18, 1.40):
                    s = base * f
                    tw2, th2 = int(round(w_r * s)), int(round(h_r * s))
                    if tw2 < 4 or th2 < 4 or th2 > bh or tw2 > bw:
                        continue
                    t = cv2.resize(troi, (tw2, th2), interpolation=cv2.INTER_NEAREST)
                    try:
                        res = cv2.matchTemplate(broi, t, cv2.TM_CCOEFF_NORMED)
                    except cv2.error:
                        continue
                    _, mx, _, pos = cv2.minMaxLoc(res)
                    matches.append({"si": si, "bi": bi, "ang": ang, "sim": float(mx),
                                    "x": bx + pos[0], "y": by + pos[1],
                                    "w": tw2, "h": th2})
    if not matches:
        return []
    matches.sort(key=lambda m: -m["sim"])
    used_s, used_b = set(), set()
    final = []
    for m in matches:
        if m["si"] in used_s or m["bi"] in used_b:
            continue
        used_s.add(m["si"]); used_b.add(m["bi"])
        final.append(m)
        if len(used_s) >= len(sp_regions) or len(used_b) >= len(bg_regions):
            break
    final.sort(key=lambda m: m["si"])
    if verbose:
        for m in final:
            print(f"    [sprite] 图案{m['si']+1} -> 背景块{m['bi']} ang={m['ang']} "
                  f"sim={m['sim']*100:.1f}% @({m['x']},{m['y']},{m['w']},{m['h']})")
    return [(m["x"] + m["w"] / 2.0, m["y"] + m["h"] / 2.0, m["sim"], m["si"])
            for m in final]


def locate_targets(tip_img, bg_img, scales=None, verbose=False,
                   bg_thr=42, min_scale=1.0, max_scale=2.8, topk=20,
                   scale_lo=1.10, scale_hi=2.40, ratio_max=1.75, svm_w=0.65):
    """
    定位: 提示字符模板 -> 背景掩码多尺度匹配
      1. clean_tip 分组出提示条里的 N 个字符 (保留原像素, 含素材自带横杠)
      2. 背景二值掩码 (近纯黑) 上多尺度 matchTemplate, IoU 复核
      3. 所有峰按 bbox 重叠聚成"位置", 匈牙利保证不同字符不占同一位置
    """
    tip_mask, groups = clean_tip(tip_img)
    N = len(groups)
    if N == 0:
        return {"points": [], "N": 0, "clean": tip_mask, "groups": groups}

    bgr = cs.to_bgr(bg_img)
    mx = bgr.max(axis=2) if bgr.ndim == 3 else bgr
    bg_mask = ((mx < bg_thr) & (mx > 1)).astype(np.uint8) * 255
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    if scales is None:
        scales = np.arange(min_scale, max_scale, 0.05)

    # --- 位置池: 背景纯黑连通域(天然互斥) ---
    pool = []
    for (bbox, m) in cs.segment_bg(bg_img, max_gray=bg_thr):
        x, y, w, h = bbox
        if w < 8 or h < 8:
            continue
        pool.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                     "mask": (m > 0)})
    if not pool:
        return {"points": [], "N": N, "clean": tip_mask, "groups": groups,
                "bg_mask": bg_mask}

    # --- 模板 ---
    tpls = []
    tpl_h = []
    for (gx, gy, gw, gh, piece) in groups:
        ys = np.where(piece.any(axis=1))[0]
        xs = np.where(piece.any(axis=0))[0]
        if len(ys) == 0 or len(xs) == 0:
            tpls.append(None)
            tpl_h.append(max(4, gh))
            continue
        t = (piece[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0)
        tpls.append(t)
        tpl_h.append(t.shape[0])

    # 候选尺寸过滤: 合成字符尺寸必是提示字符的 1.0~2.9 倍。
    # (浅色背景时照片里的深色色块也会被抠成候选, 用尺寸把它们滤掉)
    _hs = [h for h in tpl_h if h]
    if _hs and pool:
        _med = sorted(_hs)[len(_hs) // 2]
        _keep = [p for p in pool
                 if 1.0 <= p["h"] / float(max(4.0, _med)) <= 2.9
                 and 0.12 <= p["w"] / float(max(4.0, _med)) <= 3.0]
        if _keep:
            pool = _keep

    # --- 每个模板 x 每个位置: 直接做形状比对(多尺度 IoU + Chamfer + 宽高比) ---
    S = np.zeros((N, len(pool)))
    detail = {}
    for gi, tpl in enumerate(tpls):
        if tpl is None:
            continue
        th0, tw0 = tpl.shape
        tf = cs.shape_feature(tpl, 32, 0)
        te = _edge(tpl)
        ratio_t = tw0 / float(th0)
        for pj, p in enumerate(pool):
            best = None
            ph, pw = p["h"], p["w"]
            base = th0 / float(max(6, ph))
            for ds in (0.78, 0.86, 0.94, 1.0, 1.07, 1.15, 1.25):
                s = base * ds
                th, tw = int(round(ph * s)), int(round(pw * s))
                if th < 6 or tw < 6:
                    continue
                t = cv2.resize(p["mask"].astype(np.uint8) * 255, (tw, th),
                               interpolation=cv2.INTER_AREA)
                # 两边各自等比归一化到统一画布后再比形状
                ca = _to_canvas(tpl)
                cb = _to_canvas(t)
                iv = _iou_at(ca, cb)
                # 旋转不变: 提示里的字符可能被随机旋转, 背景里的不转
                sv, cov = best_rot_match(tf, tpl, p["mask"])
                ch = chamfer_score(ca, cb)
                ratio_b = pw / float(ph)
                ar = 1.0 - min(1.0, abs(ratio_t - ratio_b) / max(ratio_t, ratio_b) / 0.45)
                sc = 0.45 * cov + 0.20 * ch + 0.10 * iv + 0.10 * sv + 0.05 * ar
                if best is None or sc > best[0]:
                    best = (sc, iv, sv, ch, ar, ds, (t > 0))
            if best:
                sc, iv, sv, ch, ar, ds, tbest = best
                # SVM 二次打分 (学过"什么样的配对是真的")
                ms = svm_score(tpl, tbest)
                if ms is not None:
                    # decision 值 -> 温和加权 (几何分为主, SVM 只做微调)
                    w = svm_w + (1.0 - svm_w) / (1.0 + np.exp(-3.0 * ms))
                    sc = sc * w
                S[gi, pj] = sc
                detail[(gi, pj)] = (sc, iv, sv, ch, ar, ds)

    from scipy.optimize import linear_sum_assignment
    if len(pool) >= N:
        rr, cc2 = linear_sum_assignment(-S)
        pairs = list(zip(rr.tolist(), cc2.tolist()))
    else:
        pairs = [(gi, int(np.argmax(S[gi]))) for gi in range(N) if S[gi].max() > 0]

    # --- 带约束的组合枚举 ---
    # 强先验(实测): 同一题里 3 个目标字符的合成高度比例 <= ~1.6, 且都是提示字符的 1.2~2.3 倍
    cand_lists = []
    for gi in range(N):
        lst = [(S[gi, pj], pj) for pj in range(len(pool)) if S[gi, pj] > 0]
        lst.sort(key=lambda z: -z[0])
        cand_lists.append(lst[:topk])

    def combo_ok(sel):
        used = []
        hs = []
        for (gi, pj) in sel:
            p = pool[pj]
            # 几何互斥: 与已选位置重叠过大 -> 视为同一处
            clash = False
            for q in used:
                ix = max(0, min(p["x"] + p["w"], q["x"] + q["w"]) - max(p["x"], q["x"]))
                iy = max(0, min(p["y"] + p["h"], q["y"] + q["h"]) - max(p["y"], q["y"]))
                inter = ix * iy
                if inter > 0:
                    u = p["w"] * p["h"] + q["w"] * q["h"] - inter
                    if u > 0 and inter / float(u) > 0.30:
                        clash = True
                        break
            if clash:
                return False
            used.append(p)
            # 候选高度 / 提示字符高度 的合理倍数
            r = p["h"] / float(max(4, tpl_h[gi]))
            if r < scale_lo or r > scale_hi:
                return False
            hs.append(p["h"])
        if len(hs) >= 2:
            if max(hs) / float(min(hs)) > ratio_max:
                return False
        return True

    best_combo = None
    best_total = -1.0
    import itertools
    if all(cand_lists):
        for combo in itertools.product(*cand_lists):
            sel = [(gi, combo[gi][1]) for gi in range(N)]
            if not combo_ok(sel):
                continue
            total = sum(S[gi, combo[gi][1]] for gi in range(N))
            if total > best_total:
                best_total = total
                best_combo = sel
    if best_combo is not None:
        pairs = best_combo

    points = []
    for (gi, pj) in pairs:
        if S[gi, pj] <= 0:
            continue
        sc, iv, sv, ch, ar, ds = detail[(gi, pj)]
        p = pool[pj]
        points.append({"tpl": gi, "score": float(sc), "cc": 0.0, "iou": float(iv),
                       "iou_shape": float(sv), "chamfer": float(ch), "scale": float(ds),
                       "bbox": [int(p["x"]), int(p["y"]), int(p["w"]), int(p["h"])],
                       "cx": p["x"] + p["w"] / 2.0, "cy": p["y"] + p["h"] / 2.0,
                       "tip_bbox": [int(groups[gi][0]), int(groups[gi][1]),
                                    int(groups[gi][2]), int(groups[gi][3])]})
    points.sort(key=lambda z: z["tpl"])

    if verbose:
        print(f"  N={N} 字符组 {groups}  位置池 {len(pool)}")
        for p in points:
            print(f"    tpl{p['tpl']} score={p['score']:.3f} iou={p['iou']:.2f} "
                  f"shape={p['iou_shape']:.2f} cham={p['chamfer']:.2f} "
                  f"s={p['scale']:.2f} bbox={p['bbox']}")
    return {"points": points, "N": N, "clean": tip_mask, "groups": groups,
            "bg_mask": bg_mask, "n_cand": len(pool)}
