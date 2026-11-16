# -*- coding: utf-8 -*-
"""
训练 HOG + PCA + SVM 匹配打分器 (OpenCV 全栈)
  数据: 每轮 (bg, tip, 我选的boxes, errorCode) —— errorCode 是腾讯给的真值
  正例: errorCode==0 的轮次里, 提示模板 与 我选中的候选 的配对
  负例: 所有其它配对
  训练: 第1轮全负例 -> 难负例挖掘 -> 第2轮加入难负例
  验证: 按轮次留出 1/4, 报告成对级 precision/recall
"""
import io, sys, os, glob, tempfile
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import capsolver as cs
import locator as v4

_TMP = tempfile.gettempdir()
DATASETS = [os.path.join(_TMP, "tencent-cap-dataset"),
            os.path.join(_TMP, "tencent-cap-dataset-old")]
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "captcha_svm.xml")
NORM = os.path.join(HERE, "svm_norm.npz")
W = 5.0            # 几何标量权重 (相对 HOG)


def bg3(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def tip3(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def build_pairs(bg, tip, boxes, ec):
    """(提示模板, 候选mask, 标签) 列表; label=-1 表示标签不可靠, 应丢弃"""
    _, groups = v4.clean_tip(tip3(tip))
    cands = cs.segment_bg(bg3(bg), max_gray=42)
    out = []
    if not groups or not cands:
        return out
    tpls = []
    for (gx, gy, gw, gh, piece) in groups:
        ys = np.where(piece.any(axis=1))[0]
        xs = np.where(piece.any(axis=0))[0]
        if len(ys) == 0 or len(xs) == 0:
            tpls.append(None)
            continue
        tpls.append((piece[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0))
    for gi, tpl in enumerate(tpls):
        if tpl is None:
            continue
        # 每个候选与"我选的位置(真值框)"的重叠度
        scored = []
        for (bbox, m) in cands:
            if m.sum() == 0:
                continue
            ov = 0.0
            if ec == "0" and gi < len(boxes):
                bx, by, bw, bh = [int(v) for v in boxes[gi]]
                ix = max(0, min(bx + bw, bbox[0] + bbox[2]) - max(bx, bbox[0]))
                iy = max(0, min(by + bh, bbox[1] + bbox[3]) - max(by, bbox[1]))
                inter = ix * iy
                u = bw * bh + bbox[2] * bbox[3] - inter
                ov = inter / float(u) if u > 0 else 0.0
            scored.append((ov, bbox, m))
        # 重叠的一组里只有面积最大的(最完整的字符)当正例, 其余(碎片)丢弃
        best_area = 0
        for (ov, bbox, m) in scored:
            if ov > 0.25:
                best_area = max(best_area, bbox[2] * bbox[3])
        for (ov, bbox, m) in scored:
            if ov > 0.50 and bbox[2] * bbox[3] >= best_area:
                label = 1
            elif ov > 0.25:
                label = -1
            else:
                label = 0
            out.append((tpl, (m > 0), label))
    return out


def train_round(Xt, Yt, sel):
    """在 Xt[sel] 上训练, 返回 (svm, 归一化参数)"""
    Xs = Xt[sel]; Ys = Yt[sel]
    n_h = Xs.shape[1] - 6
    mh = Xs[:, :n_h].mean(axis=0); sh = Xs[:, :n_h].std(axis=0) + 1e-6
    ms = Xs[:, n_h:].mean(axis=0); ss = Xs[:, n_h:].std(axis=0) + 1e-6
    H = (Xs[:, :n_h] - mh) / sh
    Sc = (Xs[:, n_h:] - ms) / ss
    ncomp = int(min(30, H.shape[0] - 1, H.shape[1]))
    ctr = H.mean(axis=0).astype(np.float32)
    Hc = H - ctr
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    p = Vt[:ncomp].T.astype(np.float32)
    proj = Hc @ p
    feat = np.concatenate([proj, Sc * W], axis=1).astype(np.float32)
    s = cv2.ml.SVM_create()
    s.setKernel(cv2.ml.SVM_RBF)
    s.setC(20.0)
    s.setGamma(1.0 / max(1, feat.shape[1]))
    s.train(feat, cv2.ml.ROW_SAMPLE, Ys)
    return s, (mh, sh, ms, ss, p, ctr, n_h)


def decision(svm, par, Xsrc, idx):
    mh, sh, ms, ss, p, ctr, n_h = par
    f = Xsrc[idx]
    H = (f[:, :n_h] - mh) / sh
    Sc = (f[:, n_h:] - ms) / ss
    proj = (H - ctr) @ p
    feat = np.concatenate([proj, Sc * W], axis=1).astype(np.float32)
    _, pv = svm.predict(feat)
    return np.ravel(pv).astype(np.float32)


def main():
    files = []
    for d in DATASETS:
        if os.path.isdir(d):
            files += sorted(glob.glob(os.path.join(d, "*.npz")))
    print(f"数据集: {len(files)} 个样本  {DATASETS}")
    if not files:
        print("没有数据, 先跑 solve_run.py N --collect")
        return

    X, Y, groups = [], [], []
    npos = nneg = 0
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        ec = str(d["ec"][0])
        for (tpl, cand, lab) in build_pairs(d["bg"], d["tip"], d["boxes"], ec):
            if lab < 0:
                continue                      # 标签不可靠, 丢弃
            X.append(v4.pair_feature(tpl, cand))
            Y.append(lab); groups.append(f)
            if lab:
                npos += 1
                for k in range(4):
                    sh = np.roll(np.roll(cand, (k % 3) - 1, axis=0), (k % 2), axis=1)
                    X.append(v4.pair_feature(tpl, sh))
                    Y.append(1); groups.append(f)
                    npos += 1
            else:
                nneg += 1
    if not X:
        print("没有可用样本"); return
    X = np.array(X, np.float32); Y = np.array(Y, np.int32)
    groups = np.array(groups)
    print(f"原始样本: 正 {npos} / 负 {nneg}")

    # 留出划分(按轮次)
    uniq = sorted(set(groups.tolist()))
    val_files = set(uniq[::4])
    is_val = np.array([g in val_files for g in groups])
    tr_idx = np.where(~is_val)[0]; va_idx = np.where(is_val)[0]
    Xt, Yt = X[tr_idx], Y[tr_idx]
    print(f"划分: 训练 {len(tr_idx)} / 验证 {len(va_idx)} (验证轮次 {len(val_files)})")

    pos = np.where(Yt == 1)[0]; neg = np.where(Yt == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        print("正例或负例不足"); return
    reps = max(1, min(50, len(neg) // max(1, len(pos))))

    # 第1轮: 全部负例
    sel = np.concatenate([neg, np.tile(pos, reps)])
    svm, par = train_round(Xt, Yt, sel)
    # 难负例挖掘
    dv = decision(svm, par, Xt, np.arange(len(Xt)))
    hard = neg[np.argsort(-dv[neg])[:max(1, len(neg) // 3)]]
    sel2 = np.concatenate([neg, hard, np.tile(pos, reps)])
    svm, par = train_round(Xt, Yt, sel2)
    mh, sh, ms, ss, p, ctr, n_h = par
    print(f"难负例 {len(hard)}/{len(neg)}   PCA {n_h}->{p.shape[1]}   特征 {p.shape[1]+6} 维")

    svm.save(MODEL)
    np.savez(NORM, mu_h=mh, sd_h=sh, mu_s=ms, sd_s=ss, pca=p, center=ctr, n_hog=n_h)
    print(f"模型已保存: {MODEL}")

    # 验证集(成对级)
    dvv = decision(svm, par, X, va_idx)
    yv = Y[va_idx]
    pred = (dvv > 0).astype(np.int32)
    tp = int(((pred == 1) & (yv == 1)).sum()); fp = int(((pred == 1) & (yv == 0)).sum())
    fn = int(((pred == 0) & (yv == 1)).sum()); tn = int(((pred == 0) & (yv == 0)).sum())
    print(f"验证集: TP={tp} FP={fp} FN={fn} TN={tn}  正例={int((yv==1).sum())}")
    if tp + fp:
        print(f"  precision={tp/(tp+fp):.3f}  recall={tp/max(1,tp+fn):.3f}")
    dt = decision(svm, par, X, tr_idx)
    yt = Y[tr_idx]
    pt = (dt > 0).astype(np.int32)
    print(f"训练集: TP={int(((pt==1)&(yt==1)).sum())} FP={int(((pt==1)&(yt==0)).sum())} "
          f"(正例 {int((yt==1).sum())})")


if __name__ == "__main__":
    main()
