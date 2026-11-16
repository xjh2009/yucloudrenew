# -*- coding: utf-8 -*-
"""
CI 回归测试 —— 不需要浏览器、不需要网络。
用仓库内 captcha/fixtures/ 的样本跑定位算法, 检查能不能复现腾讯确认过的正确位置。
  - fixtures 里 ec=0 的样本: 其 boxes 是腾讯官方确认过的正确位置(真值)
  - 断言单点准确率 / 整题全对率 达到阈值, 否则退出码非 0 -> CI 红
用法: python captcha/ci_test.py [--min-point 0.90] [--min-full 0.75]
"""
import io, sys, os, glob, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, cv2
import capsolver as cs
import locator as v4

FIXTURES = os.path.join(HERE, "fixtures")


def load_fixtures():
    data = []
    for f in sorted(glob.glob(os.path.join(FIXTURES, "*.npz"))):
        try:
            z = np.load(f, allow_pickle=True)
        except Exception as e:
            print(f"  [skip] {os.path.basename(f)}: {e}")
            continue
        data.append({
            "name": os.path.basename(f),
            "bg": cv2.cvtColor(z["bg"], cv2.COLOR_GRAY2BGR),
            "tip": cv2.cvtColor(z["tip"], cv2.COLOR_GRAY2BGR),
            "boxes": z["boxes"],
            "ec": str(z["ec"][0]),
        })
    return data


def match(a, b, thr=0.30):
    """两个 bbox 的 IoU"""
    ax, ay, aw, ah = [int(v) for v in a]
    bx, by, bw, bh = [int(v) for v in b]
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / float(u) if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-point", type=float, default=0.90)
    ap.add_argument("--min-full", type=float, default=0.75)
    ap.add_argument("--min-candidates", type=int, default=3,
                    help="每个样本里背景候选个数下限, 太少说明分割退化")
    args = ap.parse_args()

    data = load_fixtures()
    if not data:
        print(f"[FAIL] 没有找到 fixtures: {FIXTURES}")
        return 1
    print(f"fixtures: {len(data)} 个样本  ({FIXTURES})")

    model_ok = os.path.exists(os.path.join(HERE, "captcha_svm.xml"))
    print(f"SVM 模型: {'已加载' if model_ok else '缺失 (只用几何特征)'}")

    hit = tot = full = rounds = 0
    ncand_min = 10 ** 9
    crashed = 0
    for s in data:
        try:
            res = v4.locate_targets(s["tip"], s["bg"])
        except Exception as e:
            crashed += 1
            print(f"  [ERROR] {s['name']}: {type(e).__name__}: {e}")
            continue
        ncand_min = min(ncand_min, res.get("n_cand", 0))
        if s["ec"] != "0":
            continue                     # 只有腾讯确认过的成功轮次才有真值
        rounds += 1
        pts = res["points"]
        if len(pts) != len(s["boxes"]):
            print(f"  [MISS] {s['name']}: 定位 {len(pts)} 个, 应为 {len(s['boxes'])} 个")
            tot += len(s["boxes"])
            continue
        ok_all = True
        for k, tb in enumerate(s["boxes"]):
            good = match(tb, pts[k]["bbox"]) > 0.30
            hit += 1 if good else 0
            tot += 1
            if not good:
                ok_all = False
                print(f"  [MISS] {s['name']} 第{k+1}个字符没对上 "
                      f"(真值{tb} vs 定位{pts[k]['bbox']})")
        full += 1 if ok_all else 0

    if tot == 0:
        print("[FAIL] 没有可评估的样本")
        return 1
    pr = hit / float(tot)
    fr = full / float(rounds) if rounds else 0.0
    print("\n===== 回归结果 =====")
    print(f"  单点准确率 : {hit}/{tot} = {pr*100:.1f}%   (阈值 {args.min_point*100:.0f}%)")
    print(f"  整题全对率 : {full}/{rounds} = {fr*100:.1f}%   (阈值 {args.min_full*100:.0f}%)")
    print(f"  候选数下限 : {ncand_min}  崩溃: {crashed}")
    if crashed:
        print("[FAIL] 有样本处理时崩溃")
        return 1
    if ncand_min < args.min_candidates:
        print(f"[FAIL] 背景候选数过少 ({ncand_min} < {args.min_candidates}), 分割可能退化")
        return 1
    if pr < args.min_point:
        print("[FAIL] 单点准确率低于阈值")
        return 1
    if fr < args.min_full:
        print("[FAIL] 整题全对率低于阈值")
        return 1
    print("[PASS] 回归通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
