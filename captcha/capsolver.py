# -*- coding: utf-8 -*-
"""
腾讯云验证码 (turing.captcha.qcloud.com / gtimg) 图形点选 - 纯 OpenCV 解题核心

结构:
  提示图 (img_index=0): 深色底 + 浅色字符序列,  需要按顺序点出
  背景图 (img_index=1): 自然照片 + 合成上去的黑色描边字符, 需要定位

方案:
  1. 分割: 提示图用亮度阈值提亮字符, 背景图用"近纯黑像素"提取描边
  2. 归一化: 每个字符 mask 裁剪 -> 等比 pad 到 32x32 -> 二值
  3. 匹配: 逐对算 IoU 相似度, 匈牙利算法做最优一对一指派
  4. 输出: 背景图原图像素坐标 -> 页面坐标缩放
"""
import re
import numpy as np
import cv2

# ---------------------------------------------------------------- 工具

def load_url_bytes(url: str, referer: str = "https://cloud.tencent.com/") -> bytes:
    import requests
    h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
         "Referer": referer}
    r = requests.get(url, headers=h, timeout=15)
    r.raise_for_status()
    return r.content


def decode(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("decode failed")
    return img


def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def ascii_art(mask: np.ndarray, w: int = 34, h: int = 16, on: str = "#", off: str = ".") -> str:
    """把二值 mask 打成字符画, 方便终端里肉眼校验分割效果"""
    m = (mask > 0).astype(np.uint8)
    ys = np.where(m.any(axis=1))[0]
    xs = np.where(m.any(axis=0))[0]
    if len(ys) == 0:
        return "(empty)"
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    small = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    return "\n".join("".join(on if v > 0.35 else off for v in row) for row in small)


# ---------------------------------------------------------------- 分割

def _char_boxes(mask: np.ndarray, min_area: int, min_w: int, min_h: int,
                merge_gap: int = 0):
    """连通域 -> 字符候选框 (x, y, w, h)"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or w < min_w or h < min_h:
            continue
        boxes.append([int(x), int(y), int(w), int(h)])
    return boxes


def split_by_projection(mask: np.ndarray, gap: int = 2):
    """按垂直投影的空白列切分, 处理相邻但不相连的字符"""
    col = (mask > 0).sum(axis=0)
    solid = col > 0
    boxes = []
    x = 0
    W = len(solid)
    while x < W:
        if not solid[x]:
            x += 1
            continue
        start = x
        blank = 0
        while x < W:
            if solid[x]:
                blank = 0
            else:
                blank += 1
                if blank > gap:
                    break
            x += 1
        end = x - blank
        boxes.append((start, end))
    return boxes


def drop_stray_bars(piece: np.ndarray, seg_w: int) -> np.ndarray:
    """
    删掉独立的扁长连通域 (干扰横线/删除线)。
    判据: 宽 >= 0.3*段宽 且 宽 > 2.5*高 —— 字符本体的横杠与竖笔相连, 不会被误删。
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(piece, 8)
    out = piece.copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w >= 0.3 * seg_w and w > h * 2.5:
            out[lab == i] = 0
    return out


def remove_full_rows(mask: np.ndarray, ratio: float = 0.78) -> np.ndarray:
    """
    删除干扰横线 —— 判据: 该行是"贯穿行"(覆盖率>=ratio) 且 宽度明显超出该段字符主体宽度。
    U 的底横宽度 == 主体宽度 -> 保留; A 上横穿的干扰线比主体宽 1.5 倍 -> 删除。
    """
    if mask.size == 0:
        return mask
    W = mask.shape[1]
    out = mask.copy()
    for _ in range(3):
        rows = (out > 0).sum(axis=1).astype(np.float32) / float(W)
        full = rows >= ratio
        if not full.any():
            break
        body_cols = np.where((out > 0) & (~full[:, None]))[1]
        if len(body_cols) == 0:
            break
        core_w = body_cols.max() - body_cols.min() + 1
        for y in np.where(full)[0]:
            xs = np.where(out[y] > 0)[0]
            if len(xs) == 0:
                continue
            if (xs.max() - xs.min() + 1) > core_w * 1.2:
                out[y] = 0
        if not ((out > 0) & (~full[:, None])).any():
            break
    return out


def remove_bars_by_connectivity(mask: np.ndarray, min_v: int = 6) -> np.ndarray:
    """
    删掉"独立"的贯穿横条: 与任何竖笔画相连的连通域才是字符主体,
    纯水平、与竖笔画不相连的连通域 = 干扰线 (如 U 的底横与两竖相连, 会保留)。
    """
    v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((min_v, 1), np.uint8))
    if v.max() == 0:
        return mask
    n, lab = cv2.connectedComponents((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return mask
    stroke_labels = set(np.unique(lab[v > 0]).tolist())
    stroke_labels.discard(0)
    if not stroke_labels:
        return mask
    keep = np.zeros(n, bool)
    for i in stroke_labels:
        keep[i] = True
    out = np.where(keep[lab], mask, 0).astype(np.uint8)
    return out if out.max() > 0 else mask


def remove_dominant_segments(mask: np.ndarray) -> np.ndarray:
    """
    逐行去掉"最长水平段"。
    干扰横条穿过字符时会与字符笔画分成 >=2 段 -> 最长的那段就是横条, 删掉;
    独占一行的字符横杠(如 U 的底横)只有 1 段 -> 不受影响。
    提示图与背景图用同一规则, 规范化后两边可比。
    """
    out = mask.copy()
    H = mask.shape[0]
    for y in range(H):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) < 2:
            continue
        segs = []
        start = prev = xs[0]
        for x in xs[1:]:
            if x == prev + 1:
                prev = x
            else:
                segs.append((start, prev))
                start = prev = x
        segs.append((start, prev))
        if len(segs) >= 2:
            j = max(range(len(segs)), key=lambda i: segs[i][1] - segs[i][0])
            out[y, segs[j][0]:segs[j][1] + 1] = 0
    return out


def strip_bars(mask: np.ndarray, min_v: int = 9) -> np.ndarray:
    """
    去掉字符上下方贯穿的干扰横条。
    原理: 横条不产生竖笔画 -> 用竖笔画的行范围确定字符主体, 范围外的像素全部裁掉。
    """
    v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((min_v, 1), np.uint8))
    if v.max() == 0:
        return mask
    ys = np.where((v > 0).any(axis=1))[0]
    if len(ys) == 0:
        return mask
    out = mask.copy()
    out[:ys.min()] = 0
    out[ys.max() + 1:] = 0
    return out


def group_chars(mask: np.ndarray, min_area: int = 40, min_overlap: int = 3) -> list:
    """
    把二值掩码按"字符"分组。
    做法: 连通域 -> 按 x 区间重叠合并。
    横杠既会把字符切开(碎片靠 x 重叠合回来), 也会横跨到邻字(靠 min_overlap 区分)。
    返回 list[(x0, y0, w, h, mask)]
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comps.append([int(x), int(y), int(w), int(h)])
    if not comps:
        return []
    comps.sort(key=lambda c: c[0])
    groups = []
    for c in comps:
        cx0, cx1 = c[0], c[0] + c[2] - 1
        placed = False
        for g in groups:
            if min(g["x1"], cx1) - max(g["x0"], cx0) >= min_overlap:
                g["x0"] = min(g["x0"], cx0)
                g["x1"] = max(g["x1"], cx1)
                g["y0"] = min(g["y0"], c[1])
                g["y1"] = max(g["y1"], c[1] + c[3] - 1)
                placed = True
                break
        if not placed:
            groups.append({"x0": cx0, "x1": cx1, "y0": c[1], "y1": c[1] + c[3] - 1})
    # 合并因范围扩张而新产生重叠的组
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                if min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]) >= min_overlap:
                    a["x0"] = min(a["x0"], b["x0"]); a["x1"] = max(a["x1"], b["x1"])
                    a["y0"] = min(a["y0"], b["y0"]); a["y1"] = max(a["y1"], b["y1"])
                    groups.pop(j); changed = True; break
            if changed:
                break
    out = []
    for g in groups:
        x0, x1 = g["x0"], g["x1"]
        y0, y1 = g["y0"], g["y1"]
        piece = mask[y0:y1 + 1, x0:x1 + 1]
        out.append((int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1), piece))
    return out


def remove_wide_rows(mask: np.ndarray, ratio: float = 0.55) -> np.ndarray:
    """
    删掉"装饰横杠": 某行的最长水平连续段占比超过 ratio -> 整行删除。
    装饰横杠只画在提示条上(背景里的合成字符没有), 所以只需对提示图单边处理;
    阈值 0.55 让字符自身的横杠(A 的横杠占比 ~0.26)保留, 而贯穿的装饰杠被去掉。
    """
    W = mask.shape[1]
    if W == 0:
        return mask
    out = mask.copy()
    for y in range(mask.shape[0]):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) == 0:
            continue
        best = 0
        start = prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x == prev + 1:
                prev = x
            else:
                best = max(best, prev - start + 1)
                start = prev = x
        best = max(best, prev - start + 1)
        if best >= ratio * W:
            out[y] = 0
    return out


def segment_tip(tip_img, thresh: int = 90, min_h: int = 8, strip_decor: bool = True):
    """
    提示图: 白底黑字(腾讯字体自带横杠, 提示条上还有额外画的装饰横杠)。
    连通域 + x 重叠分组出字符 -> 去掉装饰横杠 -> 保留干净字符本体。
    返回 list[(bbox, mask)]  (bbox 的 x 顺序 = 点击顺序)
    """
    g = to_gray(tip_img)
    mask = (g < thresh).astype(np.uint8) * 255
    if mask.sum() == 0:
        return []
    out = []
    for (x, y, w, h, piece) in group_chars(mask, min_area=40):
        if strip_decor:
            piece = remove_wide_rows(piece, 0.55)
            ys = np.where(piece.any(axis=1))[0]
            xs = np.where(piece.any(axis=0))[0]
            if len(ys) == 0 or len(xs) == 0:
                continue
            piece = piece[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if piece.shape[0] < min_h or piece.shape[1] < 4:
            continue
        out.append(((x, y, piece.shape[1], piece.shape[0]), piece))
    return out


def merge_fragments(cands, ox=0.5, oy=1.1, max_extra=12):
    """
    把被横杠切断的同一个字符的碎片合并回完整字符。
    判据: x 范围重叠 > ox*min宽  且  中心 y 距离 < oy*max高 (上下相邻的两块)。
    水平排列的不同字符 x 不重叠, 不会被误合。
    """
    n = len(cands)
    if n < 2:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        ax, ay, aw, ah = cands[i][0]
        for j in range(i + 1, n):
            bx, by, bw, bh = cands[j][0]
            ix = min(ax + aw, bx + bw) - max(ax, bx)
            if ix <= 0.5 * min(aw, bw):
                continue
            if abs((ay + ah / 2.0) - (by + bh / 2.0)) > oy * max(ah, bh):
                continue
            union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    extra = []
    for _, idxs in groups.items():
        if len(idxs) < 2:
            continue
        x0 = min(cands[i][0][0] for i in idxs)
        y0 = min(cands[i][0][1] for i in idxs)
        x1 = max(cands[i][0][0] + cands[i][0][2] for i in idxs)
        y1 = max(cands[i][0][1] + cands[i][0][3] for i in idxs)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0 or w > 200 or h > 200:
            continue
        extra.append((x0, y0, w, h))
        if len(extra) >= max_extra:
            break
    return extra


def segment_bg(bg_img, max_gray: int = 45, min_area: int = 150, min_side: int = 10,
               max_fill: float = 0.82, merge: bool = True):
    """
    背景图: 自然照片 + 合成上去的近纯黑字符。
    合成笔画灰度 10~30 (近纯黑), 照片暗部很少这么黑 -> mx<45 就能干净分离;
    再按 x 重叠把被横杠切断的碎片合并成完整字符。
    """
    bgr = to_bgr(bg_img)
    mx = bgr.max(axis=2) if bgr.ndim == 3 else bgr
    core = ((mx < max_gray) & (mx > 1)).astype(np.uint8) * 255
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # 背景里字符是随机摆放的, 不能按 x 合并(会把相邻的不同字符粘一起) -> 用纯连通域
    boxes = _char_boxes(core, min_area, min_side, min_side)
    out = []
    for (x, y, w, h) in boxes:
        sub = core[y:y + h, x:x + w]
        fill = (sub > 0).sum() / float(w * h)
        if fill > max_fill:        # 实心块
            continue
        if h > w * 8 or w > h * 8:  # 细长条
            continue
        out.append(((x, y, w, h), sub))

    if merge:
        for (x, y, w, h) in merge_fragments(out):
            sub = core[y:y + h, x:x + w]
            if sub.sum() == 0:
                continue
            out.append(((x, y, w, h), sub))
    return out


def _holes(mask: np.ndarray) -> int:
    """连通域内部的空洞数 —— 描边字符(0/A/O)有空洞, 照片实心纹理没有"""
    h, w = mask.shape
    inv = (mask == 0).astype(np.uint8)
    ff = np.zeros((h + 2, w + 2), np.uint8)
    tmp = inv.copy()
    cv2.floodFill(tmp, ff, (0, 0), 2)
    outside = (tmp == 1)
    holes = inv.copy()
    holes[outside] = 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(holes, 4)
    cnt = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 3:
            cnt += 1
    return cnt


# ---------------------------------------------------------------- 归一化 / 匹配

def shape_feature(mask: np.ndarray, size: int = 32, erode: int = 0) -> np.ndarray:
    """
    形状特征: 裁剪 -> (可选)腐蚀去细线 -> 等比缩放到 size 画布。
    腐蚀在原始分辨率下做, 因此提示与背景需按各自尺度传 erode。
    """
    m = (mask > 0).astype(np.uint8) * 255
    ys = np.where((m > 0).any(axis=1))[0]
    xs = np.where((m > 0).any(axis=0))[0]
    if len(ys) == 0 or len(xs) == 0:
        return np.zeros((size, size), np.float32)
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
        crop = cv2.erode(crop, k)
        if crop.max() == 0:
            crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = (size - 4) / float(max(h, w))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = cv2.resize((crop > 0).astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.float32)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas


def norm_mask(mask: np.ndarray, size: int = 32) -> np.ndarray:
    """裁剪内容 -> 保持比例 pad 到 size x size, 返回 0/1 float32"""
    ys = np.where((mask > 0).any(axis=1))[0]
    xs = np.where((mask > 0).any(axis=0))[0]
    if len(ys) == 0 or len(xs) == 0:
        return np.zeros((size, size), np.float32)
    crop = (mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0).astype(np.float32)
    h, w = crop.shape
    s = (size - 4) / float(max(h, w))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.float32)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.minimum(a, b).sum()
    union = np.maximum(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def match_order(templates, candidates, size: int = 32, require_distinct: bool = True):
    """
    templates:   list[(bbox, mask)]  提示图字符, 顺序即点击顺序
    candidates: list[(bbox, mask)]  背景图候选
    返回 list of (tpl_index, cand_index, score) —— 按模板顺序
    """
    if not templates or not candidates:
        return []
    T = [norm_mask(m, size) for _, m in templates]
    C = [norm_mask(m, size) for _, m in candidates]
    S = np.zeros((len(T), len(C)), np.float32)
    for i, t in enumerate(T):
        for j, c in enumerate(C):
            S[i, j] = iou(t, c)

    from scipy.optimize import linear_sum_assignment
    if require_distinct and len(C) >= len(T):
        # 每个模板必须对应不同的候选
        cost = -S
        r, c = linear_sum_assignment(cost)
        pairs = sorted(zip(r.tolist(), c.tolist()))
        return [(i, j, float(S[i, j])) for i, j in pairs]
    # 候选不够: 允许复用 (带惩罚)
    best = []
    for i in range(len(T)):
        j = int(np.argmax(S[i]))
        best.append((i, j, float(S[i, j])))
    return best


# ---------------------------------------------------------------- 定位(模板匹配版)

def drop_long_bars(mask: np.ndarray, min_ratio: float = 0.28) -> np.ndarray:
    """删掉背景里贯穿很长的干扰条 (独立扁长连通域)"""
    W = mask.shape[1]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = mask.copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w >= min_ratio * W and w > h * 3:
            out[lab == i] = 0
    return out


def build_resp(bg_img, mode: str = "dark", thr: int = 110, ksize: int = 21,
               thin: bool = True) -> np.ndarray:
    """背景图 -> 合成笔画响应图 (二值, 笔画=白)"""
    gray = to_gray(bg_img)
    bgr = to_bgr(bg_img)
    if mode == "dark":
        mx = bgr.max(axis=2)
        r = ((mx < thr) & (mx > 3)).astype(np.uint8) * 255
    elif mode == "blackhat":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
        r = (bh > thr).astype(np.uint8) * 255
    elif mode == "diff":
        blur = cv2.GaussianBlur(gray, (0, 0), 11)
        d = blur.astype(np.int16) - gray.astype(np.int16)
        r = ((d > thr) & (gray < 190)).astype(np.uint8) * 255
    else:
        raise ValueError(mode)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, k3)
    r = drop_long_bars(r)
    if thin:
        r = thin_lines(r)
    return r


def _crop_tpl(mask: np.ndarray) -> np.ndarray:
    ys = np.where((mask > 0).any(axis=1))[0]
    xs = np.where((mask > 0).any(axis=0))[0]
    return (mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0).astype(np.uint8) * 255


def _peaks(res: np.ndarray, topn: int = 6, rad: int = 40):
    out = []
    work = res.copy()
    for _ in range(topn):
        _, mx, _, loc = cv2.minMaxLoc(work)
        if mx < 0.05:
            break
        out.append((float(mx), loc))
        x, y = loc
        work[max(0, y - rad):y + rad, max(0, x - rad):x + rad] = -1
    return out


def skeleton_filter(skel: np.ndarray, min_len: int = 22) -> np.ndarray:
    """
    骨架图里只保留"长笔画"(合成字符), 丢掉短碎线(照片纹理/麦穗边缘)。
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(skel, 8)
    out = skel.copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if max(w, h) < min_len:
            out[lab == i] = 0
    return out


def dt_feature(mask: np.ndarray, gain: float = 100.0) -> np.ndarray:
    """
    距离变换 -> 灰度特征图 (绝对映射, 不归一化, 保证提示图与背景图完全可比)。
      dt=0.5(细线/边缘) -> 亮 ; dt>=2.5(块状暗部内部) -> 黑
    结构完整保留, 块状噪声被压掉。
    """
    m = (mask > 0).astype(np.uint8)
    if m.max() == 0:
        return np.zeros(m.shape, np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    feat = np.clip(255.0 - dt.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    feat[m == 0] = 0
    return feat


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """
    距离变换脊线 -> 1px 细线骨架。
    任何粗细的笔画都被抽成单像素线, 粗细统一, 提示图与背景图因此可比。
    (thin_lines 那种硬阈值会把 1px 横线留下、4px 竖笔削薄, 形状失真)
    """
    m = (mask > 0).astype(np.uint8)
    if m.max() == 0:
        return m * 255
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dil = cv2.dilate(dt, np.ones((3, 3), np.float32))
    skel = ((dt >= dil - 1e-4) & (m > 0)).astype(np.uint8) * 255
    return skel


def thin_lines(mask: np.ndarray, max_dt: float = 2.6) -> np.ndarray:
    """
    只保留"细线结构": 距离变换值 <= max_dt 的像素。
    合成字符笔画宽 1~5px (dt 小), 照片暗块/麦穗阴影是块状 (dt 大) -> 被剔除。
    提示图与背景图用同一套规范化, 保证形状可比。
    """
    m = (mask > 0).astype(np.uint8) * 255
    if m.max() == 0:
        return m
    dt = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 3)
    out = ((dt <= max_dt) & (m > 0)).astype(np.uint8) * 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return out


def erode_mask(mask: np.ndarray, k: int = 1) -> np.ndarray:
    """腐蚀 k 像素: 去掉 1~2px 的细干扰线, 保留 4~5px 的字符笔画"""
    if k <= 0:
        return mask
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    out = cv2.erode(mask, ker)
    if out.max() == 0:
        return mask
    return out


def chamfer_score(a: np.ndarray, b: np.ndarray) -> float:
    """基于距离变换的形状相似度 —— 对多余的细干扰线比 IoU 宽容"""
    da = cv2.distanceTransform((a <= 0).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((b <= 0).astype(np.uint8), cv2.DIST_L2, 3)
    fa = float(db[a > 0].mean()) if (a > 0).any() else 20.0
    fb = float(da[b > 0].mean()) if (b > 0).any() else 20.0
    return 1.0 - min(1.0, (fa + fb) / 20.0)


def locate_targets(tip_img, bg_img, scales=None, topn=8, verbose=False):
    """
    定位提示字符在背景图中的位置。

    1. 提示图 -> 干净模板 (黑字白底, 阈值 90)
    2. 背景图 -> 近纯黑核心 (mx<45) -> 连通域候选 (合成字符只有几个, 非常干净)
    3. 逐对算归一化形状 IoU (32x32 画布, 保持宽高比)
    4. 匈牙利算法做最优一对一指派

    返回 {"points": [...], "tips": [...], "cands": [...], "resp": core_mask}
    """
    tips = segment_tip(tip_img)
    if not tips:
        return {"points": [], "tips": [], "cands": [], "resp": None}
    cands = segment_bg(bg_img)

    bgr = to_bgr(bg_img)
    mx = bgr.max(axis=2) if bgr.ndim == 3 else bgr
    core = ((mx < 45) & (mx > 1)).astype(np.uint8) * 255

    if not cands:
        return {"points": [], "tips": tips, "cands": [], "resp": core}

    T = [shape_feature(m, 32, erode=0) for _, m in tips]
    C = [shape_feature(m, 32, erode=0) for _, m in cands]
    S = np.zeros((len(T), len(C)), np.float32)
    for i, t in enumerate(T):
        for j, c in enumerate(C):
            S[i, j] = iou(t, c)

    from scipy.optimize import linear_sum_assignment
    if len(C) >= len(T):
        r, cc = linear_sum_assignment(-S)
        pairs = list(zip(r.tolist(), cc.tolist()))
    else:
        pairs = [(i, int(np.argmax(S[i]))) for i in range(len(T))]

    points = []
    for (ti, ci) in pairs:
        x, y, w, h = cands[ci][0]
        points.append({"tpl": ti, "cand": ci,
                       "cx": x + w / 2.0, "cy": y + h / 2.0,
                       "score": float(S[ti, ci]), "iou": float(S[ti, ci]),
                       "scale": 1.0, "bbox": [int(x), int(y), int(w), int(h)],
                       "tip_bbox": list(tips[ti][0]), "cand_bbox": list(cands[ci][0])})
    points.sort(key=lambda q: q["tpl"])

    if verbose:
        print("  候选: " + ", ".join(f"{c[0]}" for c in cands))
        for p in points:
            print(f"  tpl{p['tpl']} -> cand{p['cand']} iou={p['iou']:.3f} bbox={p['bbox']}")
    return {"points": points, "tips": tips, "cands": cands, "resp": core}



# ---------------------------------------------------------------- 端到端

class TencentCaptchaSolver:
    IMG_AREA = ".tencent-captcha-dy__image-area"
    BG_IMG = ".tencent-captcha-dy__verify-bg-img"
    TIP_IMG = ".tencent-captcha-dy__header-answer img"
    HEADER = ".tencent-captcha-dy__header-text"

    def __init__(self, page, referer="https://cloud.tencent.com/"):
        self.page = page
        self.referer = referer
        self.net_imgs = {}

    # ---------- 网络层取图 (最稳: 直接读浏览器已加载的响应体)
    async def _on_response(self, resp):
        u = resp.url
        if "cap_union_new_getcapbysig" not in u:
            return
        try:
            body = await resp.body()
        except Exception:
            return
        if not body:
            return
        self.net_imgs[u] = body

    def watch_network(self):
        self.page.on("response", self._on_response)

    @staticmethod
    def _norm_url(u: str) -> str:
        """去掉签名字段, 只保留 img_index 用于配对"""
        return "img_index=" + (u.split("img_index=")[1].split("&")[0] if "img_index=" in u else "?")

    async def fetch(self, url: str) -> bytes:
        """
        取图: 优先精确匹配浏览器已加载过的那个 URL (换题后 URL 会变, 必须按 URL 对齐),
        退化时按 img_index 取最新一条。
        """
        if not url:
            return b""
        body = self.net_imgs.get(url)
        if body:
            return body
        key = self._norm_url(url)
        best = None
        for u, b in self.net_imgs.items():
            if self._norm_url(u) == key:
                best = b          # dict 保序 -> 最后一条就是最新的
        if best:
            return best
        try:
            r = await self.page.request.get(url, headers={
                "Referer": self.referer,
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            })
            body = await r.body()
            if body:
                return body
        except Exception:
            pass
        # 兜底: 在页面上下文里 fetch (自带 Cookie / Referer / 签名)
        try:
            import base64
            b64 = await self.page.evaluate("""async (u) => {
                const r = await fetch(u, {credentials: 'include'});
                if (!r.ok) return null;
                const b = await r.arrayBuffer();
                const arr = new Uint8Array(b);
                let s = '';
                const CH = 0x8000;
                for (let i = 0; i < arr.length; i += CH)
                    s += String.fromCharCode.apply(null, arr.subarray(i, i + CH));
                return btoa(s);
            }""", url)
            if b64:
                return base64.b64decode(b64)
        except Exception:
            pass
        return load_url_bytes(url, self.referer)

    async def collect(self):
        """返回 dict: tip_url, bg_url, area_rect, tip_text"""
        return await self.page.evaluate("""(sels) => {
            const q = s => document.querySelector(s);
            const area = q(sels.area);
            const bg = q(sels.bg);
            const tip = q(sels.tip);
            const ht = q(sels.header);
            if (!area || !bg) return null;
            const st = bg.getAttribute('style') || '';
            const m = st.match(/background-image:\\s*url\\(["']?(.*?)["']?\\)/);
            const r = area.getBoundingClientRect();
            return {
                bg_url: m ? m[1] : null,
                tip_url: tip ? tip.src : null,
                tip_text: ht ? ht.innerText : '',
                area_rect: {x: r.x, y: r.y, w: r.width, h: r.height},
                bg_size: (bg.style.width || '') + 'x' + (bg.style.height || ''),
            };
        }""", {"area": self.IMG_AREA, "bg": self.BG_IMG, "tip": self.TIP_IMG,
               "header": self.HEADER})

    # ---------- 求解
    def solve_images(self, tip_img, bg_img):
        tips = segment_tip(tip_img)
        cands = segment_bg(bg_img)
        pairs = match_order(tips, cands)
        # 像素坐标 -> 目标点
        pts = []
        for (ti, ci, score) in pairs:
            x, y, w, h = cands[ci][0]
            pts.append({"tpl": ti, "cand": ci, "score": score,
                        "cx": x + w / 2.0, "cy": y + h / 2.0,
                        "bbox": [x, y, w, h]})
        return {"tips": tips, "cands": cands, "pairs": pairs, "points": pts}

    # ---------- 点击
    async def click_points(self, points, bg_shape, area_rect, humanize=True):
        """bg_shape = 原图 (h, w); area_rect = 页面 CSS 像素矩形"""
        sx = area_rect["w"] / float(bg_shape[1])
        sy = area_rect["h"] / float(bg_shape[0])
        seq = []
        for p in points:
            seq.append((area_rect["x"] + p["cx"] * sx, area_rect["y"] + p["cy"] * sy))
        for (px, py) in seq:
            if humanize:
                await self._human_move(px, py)
            await self.page.mouse.click(px, py)
            await self.page.wait_for_timeout(random_ms(280, 620))
        return seq

    async def _human_move(self, x, y, steps=None):
        import random
        cur = await self.page.evaluate("() => { const e=document.querySelector('.tencent-captcha-dy__image-area'); const r=e.getBoundingClientRect(); return {x:r.x, y:r.y}; }")
        sx, sy = cur["x"], cur["y"]
        steps = steps or random.randint(14, 26)
        for i in range(1, steps + 1):
            t = i / steps
            # 缓动 + 抖动
            ex = sx + (x - sx) * (t ** 2.2)
            ey = sy + (y - sy) * (t ** 2.2)
            jx = random.uniform(-2.2, 2.2)
            jy = random.uniform(-2.2, 2.2)
            await self.page.mouse.move(ex + jx, ey + jy)
            await self.page.wait_for_timeout(random_ms(8, 22))
        await self.page.mouse.move(x, y)
        await self.page.wait_for_timeout(random_ms(60, 140))


def random_ms(a, b):
    import random
    return random.uniform(a, b) / 1000.0
