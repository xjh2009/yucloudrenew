# -*- coding: utf-8 -*-
"""
腾讯 TCaptcha 滑动拼图验证码求解 (纯 OpenCV)

两种求解路径:
  1. 掩码模板匹配: 有拼图块图(BGRA)时, 用它的非透明区域做 mask,
     在背景图上做 matchTemplate —— 精度最高
  2. 轮廓/缺口检测: 没有拼图块图时, 用边缘检测找"被挖走的那一块"(近似方形凹槽)

返回的是缺口在**背景图原始像素坐标系**里的 x, 由调用方换算成页面坐标。
"""
import io, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import cv2


def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _tile_mask(tile):
    """从拼图块图里取形状掩码 (alpha 或非黑区域)"""
    if tile is None:
        return None
    if tile.ndim == 3 and tile.shape[2] == 4:
        alpha = tile[:, :, 3]
        if alpha.max() > 10:
            return (alpha > 40).astype(np.uint8) * 255
    # 没有 alpha: 用非黑区域当形状
    gray = _to_gray(tile)
    m = (gray > 30).astype(np.uint8) * 255
    if m.sum() < 50:
        return None
    # 只保留最大的连通块
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8) * 255


def _match_masked(bg_g, tile_g, mask, angles):
    """在 bg_g 上用 tile 掩码做归一化相关匹配 (含旋转扫描)"""
    best = (None, -1.0)
    for ang in angles:
        if ang:
            M = cv2.getRotationMatrix2D((tile_g.shape[1] / 2, tile_g.shape[0] / 2), ang, 1.0)
            tg = cv2.warpAffine(tile_g, M, (tile_g.shape[1], tile_g.shape[0]),
                                flags=cv2.INTER_LINEAR, borderValue=0)
            mk = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                                flags=cv2.INTER_NEAREST, borderValue=0)
        else:
            tg, mk = tile_g, mask
        if tg.shape[0] >= bg_g.shape[0] or tg.shape[1] >= bg_g.shape[1]:
            continue
        try:
            res = cv2.matchTemplate(bg_g, tg, cv2.TM_CCORR_NORMED, mask=mk)
        except cv2.error:
            continue
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[1]:
            best = (loc[0] + tg.shape[1] / 2.0, float(mx))
    return best


def _dark_map(bg):
    """暗区图: 比周围暗的地方亮 (缺口/凹区)"""
    g = _to_gray(bg).astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 9)
    d = np.clip((blur - g) * 8.0, 0, 255).astype(np.uint8)
    return d


def find_gap_by_mask(bg, tile, angles=(0, -8, 8, -15, 15, -22, 22)):
    """
    路径1: 用拼图块掩码在背景图上做归一化相关匹配。
    拼图块可能被随机旋转, 所以扫几个角度取最优。
    亮度匹配分数低时(缺口被移走的情况), 退到"暗区图 + 形状匹配"。
    返回 (x_center, score)
    """
    mask = _tile_mask(tile)
    if mask is None:
        return None, 0.0
    bg_g = _to_gray(bg)
    tile_bgr = tile[:, :, :3] if (tile.ndim == 3 and tile.shape[2] == 4) else tile
    tile_g = _to_gray(tile_bgr)
    if tile_g.shape != mask.shape[:2]:
        mask = cv2.resize(mask, (tile_g.shape[1], tile_g.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    best = _match_masked(bg_g, tile_g, mask, angles)
    if best[1] < 0.55:
        # 暗区图上做"纯形状"匹配: 模板只用掩码填充
        dm = _dark_map(bg)
        shape_tpl = np.where(mask > 0, 200, 0).astype(np.uint8)
        alt = _match_masked(dm, shape_tpl, mask, angles)
        if alt[1] > best[1]:
            best = alt
    return best


def find_gap_by_contour(bg, min_ratio=0.05, max_ratio=0.45):
    """
    路径2: 背景图上的缺口通常是"比周围暗"的凹区(原图内容被移走后留下的填充/阴影)。
    用局部亮度差找出这些暗块, 再挑形状规则的。
    返回 (x_center, score)
    """
    g = _to_gray(bg)
    H, W = g.shape[:2]
    blur = cv2.GaussianBlur(g, (0, 0), 9)
    diff = blur.astype(np.float32) - g.astype(np.float32)
    dark = (diff > 16).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    dark = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = (None, 0.0)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h < H * min_ratio or h > H * max_ratio:
            continue
        if w < h * 0.55 or w > h * 2.4:
            continue
        if w < W * min_ratio:
            continue
        if cv2.contourArea(c) < 40:
            continue
        squareness = min(w, h) / max(w, h)
        # 越方正、越"实心"(暗块填充率高)越可信
        fill = (dark[y:y + h, x:x + w] > 0).mean()
        score = squareness * (0.4 + 0.6 * min(1.0, fill / 0.55))
        if score > best[1]:
            best = (x + w / 2.0, float(score))
    return best


def solve_gap_x(bg, tile=None):
    """
    求解缺口 x (bg 原图坐标系)。
    先走掩码匹配, 分数不够再退回轮廓法。
    """
    x, score = find_gap_by_mask(bg, tile) if tile is not None else (None, 0.0)
    if x is not None and score > 0.35:
        return x, score, "mask"
    x2, s2 = find_gap_by_contour(bg)
    if x2 is not None and (x is None or s2 > 0.5):
        return x2, s2, "contour"
    return x, score, ("mask" if x is not None else "none")


# ------------------------------------------------------------------ 自测

def _make_bg(w, h, seed=7):
    """造一张有结构的底图 (渐变 + 纹理), 别用纯噪声"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (120 + 40 * np.sin(xx / 47.0) + 35 * np.cos(yy / 39.0)
            + 25 * np.sin((xx + yy) / 23.0))
    bg = np.clip(base, 40, 230).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (3, 3), 0)
    for _ in range(40):                        # 加点块状纹理
        x0, y0 = rng.integers(0, w - 60), rng.integers(0, h - 60)
        c = int(rng.integers(60, 170))
        bg[y0:y0 + 40, x0:x0 + 60] = c
    return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)


def _synth(bg_w=680, bg_h=480, gap_x=420, gap_y=120, tile_w=90, tile_h=90,
           cut_out=False):
    """
    模拟腾讯滑块 (gap_x/gap_y = 缺口**中心**):
      - 背景图在缺口处**内容保留**(腾讯只是画了缺口),
        拼图块就是从那儿裁下来的 -> 亮度+掩码匹配可精确命中
      - cut_out=True: 模拟"缺口被挖走"的变体(此时只能靠轮廓法)
    """
    bg = _make_bg(bg_w, bg_h)
    x0 = gap_x - tile_w // 2
    y0 = gap_y - tile_h // 2
    tm = np.zeros((tile_h, tile_w), np.uint8)
    cv2.ellipse(tm, (tile_w // 2, tile_h // 2), (tile_w // 2 - 2, tile_h // 2 - 2),
                0, 0, 360, 255, -1)
    patch = bg[y0:y0 + tile_h, x0:x0 + tile_w].copy()
    if cut_out:
        bg[y0:y0 + tile_h, x0:x0 + tile_w] = 60   # 挖走
    tile = np.zeros((tile_h, tile_w, 4), np.uint8)
    tile[:, :, :3] = patch
    tile[:, :, 3] = tm
    return bg, tile


def _self_test():
    ok = True
    bg, tile = _synth()
    x, score, how = solve_gap_x(bg, tile)
    print(f"[内容保留 + 掩码匹配] 缺口x={x} score={score:.3f} via={how}  (真值 420)")
    ok = ok and x is not None and abs(x - 420) < 20

    bg2, tile2 = _synth(gap_x=300, gap_y=200, cut_out=True)
    x2, s2, how2 = solve_gap_x(bg2, tile2)
    print(f"[缺口挖走 + 形状匹配] 缺口x={x2} score={s2:.3f} via={how2}  (真值 300)")
    ok = ok and x2 is not None and abs(x2 - 300) < 40

    bg3, tile3 = _synth(gap_x=250)
    x3, s3 = find_gap_by_mask(bg3, tile3)
    print(f"[另一个位置 + 掩码]   缺口x={x3} score={s3:.3f}  (真值 250)")
    ok = ok and x3 is not None and abs(x3 - 250) < 20

    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _self_test() else 1)
