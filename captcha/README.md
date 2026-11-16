# 雨云自动签到 + 图形点选验证码识别

用 **Playwright** 做雨云（Rainyun）每日自动签到：登录 → 自动解开腾讯 TCaptcha「图形点选」→
点「领取奖励」→ 读积分确认。

> 说明：`cloud.tencent.com/product/captcha` 那个图形点选 demo **只是训练数据源**，
> 用来采集带真值的样本、训练匹配打分器，不是交付目标，也不会出现在 CI 里。

**实测通过**：签到后积分 `117092 → 117592`（+500 分）。

---

## 目录结构

| 文件 | 角色 | 跑在哪 |
|---|---|---|
| `captcha/rainyun.py` | **主程序**：登录 + 签到 + 解验证码 | 本地 / CI 定时 |
| `captcha/locator.py` | 定位器：提示图案 → 背景坐标（覆盖度 / 旋转不变 Chamfer / 形状打分） | 库 |
| `captcha/capsolver.py` | 图像处理核心 + TCaptcha 取图（CDP 走网络层捕获） | 库 |
| `captcha/train_svm.py` | 训练 HOG+PCA+SVM 匹配打分器（成对二分类 + 难负例挖掘） | 手动跑 |
| `captcha/solve_run.py` | **训练数据采集器**：去图形点选 demo 采带真值的样本 | 手动跑，不进 CI |
| `captcha/slider.py` | 滑块拼图求解器（备用，雨云偶尔弹滑块） | 库 |
| `captcha/ci_test.py` | 离线回归测试：用 fixtures 验证定位准确率 | CI |
| `captcha/fixtures/*.npz` | 26 个带真值的回归样本 | 仓库 |
| `captcha/captcha_svm.xml` / `svm_norm.npz` | 训练好的打分器 + 归一化参数 | 仓库 |

---

## 本地使用

```powershell
pip install -r requirements.txt

# 1) 采集训练数据(去腾讯云 demo, 抓 人工确认过正确 的样本)
python captcha\solve_run.py 120 --collect

# 2) 训练打分器
python captcha\train_svm.py

# 3) 跑签到
$env:RAINYUN_USER = "你的账号"
$env:RAINYUN_PASS = "你的密码"
python captcha\rainyun.py
```

多账户（多行，数量需匹配）：

```powershell
$env:RAINYUN_USER = "user1`nuser2"
$env:RAINYUN_PASS = "pass1`npass2"
```

## GitHub Actions

`.github/workflows/rainyun-checkin.yml`，三个 job：

| Job | 触发 | 内容 |
|---|---|---|
| `regression` | push / PR / 定时 | 离线回归：`python captcha/ci_test.py`，定位准确率不达标就红（不联网、不开浏览器） |
| `train-smoke` | push / PR / 定时 | 训练流程冒烟：有采集数据才跑，没数据跳过 |
| `checkin` | 每天 UTC 04:00（北京时间 12:00） | 真跑签到；`needs: regression`，回归不过就不签 |

**唯一需要配的 Secret**：`RAINYUN_USER` / `RAINYUN_PASS`（多行=多账户）。

Cookie 走 Actions cache，有效期内免密登录。

---

## 自动重试（三层）

| 层级 | 参数 | 行为 |
|---|---|---|
| 验证码级 | `CAPTCHA_TRIES`（默认 4） | 一次签到里没过 → **刷新换一题**再解，退避 `2~4s × 第几次` 递增 |
| 账户级 | `CHECKIN_MAX_RETRIES`（默认 2） | 整个流程失败 → 重跑该账户，退避 `20~40s × 轮次` 递增 |
| 浏览器级 | 自动 | 每次尝试开**全新 profile**（TCaptcha 的 `aged`/会话过期是复用导致的），但复用 Cookie 保持免密登录 |

两个容易踩的坑，代码里都处理了：

- **刷新/确定按钮不能取 `.first`** —— 页面里挂着隐藏的老版滑块模板，它有**同名**的刷新和确定按钮，
  必须遍历所有匹配项、只点**可见**的那个。
- **重试要有随机退避** —— 固定节奏的循环在腾讯风控眼里非常扎眼。

成功判据用**积分是否增加**，不是弹窗是否消失；当天已成功签到会记到 `sign_<hash>.json`，下次直接跳过。

---

## 验证码识别的四个关键点

1. **是图形点选，不是滑块**。提示条里是图案序列（sprite 图），**从左到右就是点击顺序**。
2. **合成字符是「近黑 + 中性」的描边**（R≈G≈B）；照片暗部是**偏色**的（深棕/深蓝），
   用 `max<62 且 max-min<38` 才能把两者分开——这一步错了，候选池里全是照片纹理。
3. **匹配要按字符尺度对齐**：模板缩放到「背景块高 / 图案高」的倍数，扫 ±45° 旋转，
   `matchTemplate(TM_CCOEFF_NORMED)`，最后按相似度贪心消冲突（一个图案一个位置）。
4. 提示图里的图案用**列投影**切分，不是连通域——字符由多笔组成，连通域会切碎。

## 训练用的定位器（腾讯云那套）

`solve_run.py` + `locator.py` + `train_svm.py` 是一条独立的训练线：

- **提示图 ≠ 背景图**：提示是 1x 素材、背景约 2x，且提示模板带装饰横杠 → 直接比 IoU 天花板只有 0.5。
- **正确度量是「覆盖度」**（候选 ⊆ 模板，因为模板只多了装饰横杠）：
  `|候选∩模板| / |候选|` 单独就能选对真值 **80.8%**（正例中位 0.657 vs 负例 0.353），
  把端到端通过率从 35% 拉到 70%。
- **旋转不变**：提示里的字符可能被随机旋转 → 匹配时扫 ±30°。
- **训练闭环**：`errorCode` 是唯一真值。`=0` 说明那 3 个位置全对 → 正例；`=50` → 其余是负例。
  据此做**难负例挖掘**（把 SVM 打分最高的负例重复采样），验证集 recall 0.49 → 0.69。
- **回归集**：`captcha/fixtures` 26 个样本，`ec=0` 的 `boxes` 是腾讯确认过的正确位置，
  当前离线指标 **单点 98.3% / 整题 95%**。

---

## 依赖

`numpy` `opencv-python` `scipy` `requests` `playwright`

浏览器：本地连你自己开的（带 `--remote-debugging-port=9222`），CI 里用 xvfb + Playwright 的 Chromium。
