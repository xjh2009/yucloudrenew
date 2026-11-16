# 腾讯云图形点选验证码 —— OpenCV + 机器学习求解

通过 CDP 连接浏览器，用 Playwright 驱动，在 `https://cloud.tencent.com/product/captcha`
的「图形点选」体验里，用**纯 OpenCV 图像处理 + 训练出来的 HOG/PCA/SVM 打分器**自动定位并按序点击目标字符。

- 离线定位准确率：单点 **96.7%**、整题全对 **90%**（`captcha/fixtures` 回归集，26 个样本）
- 端到端实测：约 **65%~70%** 通过（腾讯侧随机难度 + 风控会影响）

---

## 在 GitHub Actions 里跑

`.github/workflows/captcha.yml` 里有两个 job：

| Job | 触发 | 做什么 | 需要什么 |
|---|---|---|---|
| `regression` | push / PR | 离线回归：用仓库里的 `captcha/fixtures` 跑定位算法，检查能不能复现腾讯确认过的正确位置 | 只要 Python，无浏览器无网络，几秒 |
| `live` | 手动触发（`workflow_dispatch`） | 真跑：在 `xvfb` 下起 Chromium，连腾讯云产品页，端到端点验证码 | xvfb + Chromium（会现装） |

**默认只跑离线回归**。真跑那个 job 需要手动点一下（Actions 页面 → Run workflow），原因：
CI 出口 IP 容易被腾讯风控，而且它会真的对外发请求，不适合每次 push 都打。

```bash
# 本地复现 CI 的离线回归
pip install -r requirements.txt
python captcha/ci_test.py

# 本地复现 CI 的真跑（Linux 下需要 xvfb）
xvfb-run -a python captcha/solve_run.py 5
```

`ci_test.py` 的退出码非 0 就是 CI 红。可调阈值：

```bash
python captcha/ci_test.py --min-point 0.95 --min-full 0.85
```

**要提交进仓库的东西**（已在仓库里）：

| 文件 | 说明 |
|---|---|
| `captcha/fixtures/*.npz` | 26 个回归样本（3.7 MB）。`ec=0` 的样本里 `boxes` 是腾讯官方确认过的正确位置 |
| `captcha/captcha_svm.xml` | 训练好的 SVM |
| `captcha/svm_norm.npz` | 特征标准化 + PCA 参数（推理端必须和训练端一致） |

---

## 雨云自动签到 (Playwright)

`captcha/rainyun.py` —— 用 **Playwright** 做雨云（Rainyun）登录 + 每日签到，
参考 [chizw/Rainyun-Qiandao](https://github.com/chizw/Rainyun-Qiandao) 的流程，但不用 Selenium。

流程：

```
打开 https://app.rainyun.com/account/reward/earn
  → 没登录就填账密登录 (input[name=login-field] / login-password)
  → 弹腾讯 TCaptcha「图形点选」(请依次点击: 城堡 0 定位) → 自动识别并按序点击
  → 回到积分页点[领取奖励], 再弹验证码继续解
  → 读积分确认
```

**实测通过**：签到后积分 `117092 → 117592`（+500 分）。

已实现：多账户（环境变量多行）、Cookie 缓存（免密登录）、每账户独立随机浏览器指纹、
失败重试、验证码失败自动刷新重解。

### 验证码识别的四个关键点

1. **是图形点选，不是滑块**。提示条里是图案序列（sprite 图），**从左到右就是点击顺序**。
2. **合成字符是「近黑 + 中性」的描边**（R≈G≈B）；照片暗部是**偏色**的（深棕/深蓝），
   用 `max<62 且 max-min<38` 才能把两者分开——这一步错了，候选池里全是照片纹理。
3. **匹配要按字符尺度对齐**：模板缩放到「背景块高 / 图案高」的倍数，扫 ±45° 旋转，
   `matchTemplate(TM_CCOEFF_NORMED)`，最后按相似度贪心消冲突（一个图案一个位置）。
4. **「确定」按钮不能取 `.first`** —— 页面里还挂着一个隐藏的老版滑块模板，它有**同名**的确定按钮。
   必须遍历所有匹配、只点可见的那个（`text=确定 #1` 才是对的）。
   成功判据用**积分是否增加**，不是弹窗是否消失。

```powershell
# 本地
$env:RAINYUN_USER = "你的账号"
$env:RAINYUN_PASS = "你的密码"
python captcha\rainyun.py

# 多账户(多行, 数量需匹配)
$env:RAINYUN_USER = "user1`nuser2"
$env:RAINYUN_PASS = "pass1`npass2"
```

**CI**：`.github/workflows/rainyun-checkin.yml`，每天 UTC 04:00（北京 12:00）自动跑，
在仓库 Settings → Secrets 里加 `RAINYUN_USER` / `RAINYUN_PASS`（多行=多账户）即可。
Cookie 存在 Actions cache 里，有效期内免密登录。

| 环境变量 | 说明 | 默认 |
|---|---|---|
| `RAINYUN_USER` / `RAINYUN_PASS` | 账号密码，多行=多账户 | 必填 |
| `HEADLESS` | 无头模式（CI 里用 xvfb + false 更不容易被风控） | false |
| `DEBUG` | 详细日志 | false |
| `MAX_WORKERS` | 并发账户数 | 2 |
| `CHECKIN_MAX_RETRIES` | 失败重试轮数 | 2 |
| `COOKIE_DIR` | Cookie 存放目录 | 系统临时目录 |

滑块求解器单独有自测（合成样本）：

```bash
python captcha/slider.py     # PASS / FAIL
```

---

## 本地使用

```bash
pip install -r requirements.txt

# 1) 手动开一个带调试端口的 Chrome
start chrome --remote-debugging-port=9222 https://cloud.tencent.com/product/captcha

# 2) 采集训练数据（errorCode 是腾讯返回的真值标签）
python captcha/solve_run.py 120 --collect

# 3) 训练打分器
python captcha/train_svm.py

# 4) 在线求解
python captcha/solve_run.py 20
```

没开 Chrome 时 `solve_run.py` 会自动启动一个浏览器：**优先用系统已装的 Chrome**
（`channel="chrome"`，本地和 ubuntu runner 都有），失败才回退到 Playwright 自带的
Chromium（那就需要 `python -m playwright install chromium`）。

---

## 文件结构

| 文件 | 作用 |
|---|---|
| `captcha/capsolver.py` | 图像处理核心：近纯黑核心分割、连通域分组、碎片合并、IoU/Chamfer/覆盖度、CDP 取图 |
| `captcha/locator.py` | 定位算法：提示字符分组 → 背景候选池 → 形状打分 → SVM 二次打分 → 组合约束择优 |
| `captcha/train_svm.py` | 训练：成对二分类 + 难负例挖掘 + 留出验证 |
| `captcha/solve_run.py` | 端到端：触发验证码 → 取图 → 定位 → 真实鼠标点击 → 读 `errorCode` 判成败 |
| `captcha/ci_test.py` | 离线回归测试（CI 用，不碰浏览器和网络） |

---

## 关键发现（踩过的坑）

1. **合成字符是近纯黑**（灰度 10~30），照片暗部很少这么黑 → `max(R,G,B) < 42` 就能把目标从背景里干净抠出来。
   早期用 `mx < 110` 会把整片阴影捞进来，是长时间定位不准的头号原因。
2. **背景是 AI 生成图**（页面水印写着「AI生成背景」），但合成字符仍是叠加的黑色图层。
3. **提示条用腾讯的特殊字体**：字符自带贯穿横杠，且提示条上会额外画装饰横杠。
   背景里的同一个字符**没有**装饰横杠 —— 两边不同形，IoU 天花板只有 ~0.5。
4. **正确度量是「覆盖度」不是 IoU**：提示模板 = 干净字符 + 装饰横杠，背景候选 = 干净字符，
   所以「候选 ⊆ 模板」才是对的判据。`|候选∩模板| / |候选|` 单独就能选对真值 **80.8%**
   （正例中位 0.657 vs 负例中位 0.353），把端到端通过率从 35% 拉到 70%。
5. **提示图 1x、背景图约 2x**，笔画粗细不同 → 两边都取「纯黑核心」再比，别用骨架化
   （骨架化后 IoU 反而掉到 0.13）。
6. **判定真值只能看接口**：`cap_union_new_verify` 返回 `errorCode=0` 且带 `ticket` 才算通过，
   弹窗消失不算数。`errorCode=50` 无法区分「点少了」和「点错了」，所以做不了排除法。
7. **同一题的 3 个目标字符合成尺度是统一的**（实测高度比例 ≤ 1.54）→ 可以作为组合约束。

---

## 训练闭环

`errorCode` 是腾讯给的唯一真值：

- `errorCode=0` → 我选的 3 个位置**全部正确** → 这 3 个「模板↔候选」配对是正例
- `errorCode=50` → 至少一个错 → 其余配对是负例

据此做**难负例挖掘**：第 1 轮用全部负例训，第 2 轮把 SVM 打分最高的那批负例（长得最像正例的）
重复采样加入训练 —— 这是区分度提升的关键（验证集 recall 0.49 → 0.69）。
