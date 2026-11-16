# -*- coding: utf-8 -*-
"""
【训练数据采集器】—— 不是签到主程序, 也不进 CI。

去 cloud.tencent.com/product/captcha 的「图形点选」demo 反复抓题, 把每次的
(背景图, 提示图, 我选的位置, 腾讯返回的 errorCode) 存成 npz 样本:
    errorCode=0  ->  腾讯确认过这次点对了 -> 其中的位置是真值(正例)
    errorCode=50 ->  至少有一个点错了      -> 用来做负例/难负例

样本落在系统临时目录 tencent-cap-dataset, 用 captcha/train_svm.py 训练。

    python captcha\solve_run.py 120 --collect     # 采集
    python captcha\train_svm.py                  # 训练

日常签到请用 captcha/rainyun.py。
"""
import asyncio, io, sys, os, random, time, json, tempfile
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from playwright.async_api import async_playwright
import numpy as np, cv2
import capsolver as cs
import locator as v4

DATASET = os.path.join(tempfile.gettempdir(), "tencent-cap-dataset")
COLLECT = "--collect" in sys.argv
ONLY = None
if "--only" in sys.argv:
    ONLY = int(sys.argv[sys.argv.index("--only") + 1])
if COLLECT:
    os.makedirs(DATASET, exist_ok=True)

CDP_URL = os.environ.get("CAPTCHA_CDP", "http://127.0.0.1:9222")
TARGET_URL = "https://cloud.tencent.com/product/captcha"
HEADLESS = os.environ.get("CAPTCHA_HEADLESS", "0") == "1"


async def get_page(p):
    """返回 (page, 自己启动的 context 或 None)
    本地连已开好的 Chrome(CDP); CI 里连不上就自己起一个 Chromium"""
    try:
        browser = await p.chromium.connect_over_cdp(CDP_URL, timeout=6000)
        ctx = browser.contexts[0]
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        return page, None
    except Exception:
        pass
    profile = os.path.join(tempfile.gettempdir(), "captcha-chrome-profile")
    args = ["--remote-debugging-port=9222", "--no-sandbox",
            "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled",
            "--window-size=1400,1000"]
    viewport = {"width": 1400, "height": 1000}
    ctx = None
    # 优先用系统已装的 Chrome (本地/ubuntu runner 都有), 退回 playwright 自带的 chromium
    for kw in ({"channel": "chrome"}, {}):
        try:
            ctx = await p.chromium.launch_persistent_context(
                profile, headless=HEADLESS, args=args, viewport=viewport, **kw)
            break
        except Exception as e:
            last = e
    if ctx is None:
        print(f"[启动浏览器失败] {last}\n"
              f"请先执行: python -m playwright install chromium")
        raise last
    page = ctx.pages[0] if ctx.pages else await ctx.new_page()
    try:
        await page.goto(TARGET_URL, timeout=45000)
    except Exception:
        pass
    return page, ctx


async def trigger_click_captcha(page):
    """在产品页点'图形点选'卡片里的'立即体验'"""
    ok = await page.evaluate("""() => {
        const els = [...document.querySelectorAll('*')].filter(e =>
            e.children.length === 0 && e.innerText && e.innerText.includes('图形点选'));
        if (!els.length) return false;
        let card = els[0];
        for (let i = 0; i < 12 && card; i++) {
            card = card.parentElement;
            if (card && card.innerText && card.innerText.includes('立即体验')
                && card.innerText.includes('图形点选')) break;
        }
        if (!card) return false;
        const btn = [...card.querySelectorAll('a,button,div,span')].find(
            e => e.innerText && e.innerText.trim() === '立即体验');
        if (!btn) return false;
        btn.click();
        return true;
    }""")
    if ok:
        await page.wait_for_timeout(2200)
    return ok


async def popup_state(page):
    return await page.evaluate("""() => {
        const wrap = document.querySelector('#tcaptcha_transform_dy');
        if (!wrap) return {exists:false, visible:false};
        const st = wrap.getAttribute('style') || '';
        const visible = st.includes('display: block') || st.includes('display:block');
        const err = document.querySelector('.tencent-captcha-dy__verify-error-text');
        const suc = document.querySelector('.tencent-captcha-dy__verify-status-img--suc');
        const ht = document.querySelector('.tencent-captcha-dy__header-text');
        return {exists:true, visible,
                error: err ? (err.innerText||'').trim() : '',
                success: suc ? (suc.offsetParent !== null) : false,
                header: ht ? (ht.innerText||'').trim() : ''};
    }""")


async def wait_for_images(solver, timeout=6.0):
    """等浏览器把 提示图(img_index=0) 和 背景图(img_index=1) 都加载出来"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        keys = {solver._norm_url(u) for u in solver.net_imgs}
        if "img_index=0" in keys and "img_index=1" in keys:
            return True
        await asyncio.sleep(0.1)
    return False


async def click_marks(page):
    """点击后图片区里出现的标记元素(腾讯会显示点击序号)"""
    return await page.evaluate("""() => {
        const area = document.querySelector('.tencent-captcha-dy__image-area');
        if (!area) return [];
        const out = [];
        for (const e of area.querySelectorAll('*')) {
            const cls = (e.className||'').toString();
            if (/point|mark|dot|verify-.*item|number/i.test(cls)) {
                const r = e.getBoundingClientRect();
                if (r.width > 0) out.push({cls: cls.slice(0,60),
                    x: Math.round(r.x+r.width/2), y: Math.round(r.y+r.height/2),
                    t: (e.innerText||'').trim().slice(0,6)});
            }
        }
        return out;
    }""")


def snap_click_point(mask, bbox, cx, cy):
    """把点击位置吸附到最近的笔画像素(字符可能是空心的, 中心是洞)"""
    x, y, w, h = bbox
    sub = mask[y:y + h, x:x + w]
    ys, xs = np.where(sub > 0)
    if len(xs) == 0:
        return cx, cy
    d = (xs + x - cx) ** 2 + (ys + y - cy) ** 2
    i = int(np.argmin(d))
    return float(xs[i] + x), float(ys[i] + y)


def dump_bg(mask, pts, w=112, h=54):
    """背景掩码 + 命中点标记(1/2/3 = 提示顺序), 肉眼核对用"""
    if mask is None:
        return
    canvas = (mask > 0).astype(np.uint8)
    small = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    grid = [["#" if v > 0.30 else ("+" if v > 0.10 else ".") for v in row] for row in small]
    for i, p in enumerate(pts):
        mx = int(p["cx"] / mask.shape[1] * w)
        my = int(p["cy"] / mask.shape[0] * h)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                yy, xx = my + dy, mx + dx
                if 0 <= yy < h and 0 <= xx < w:
                    grid[yy][xx] = str(i + 1)
    print("    +" + "-" * w + "+")
    for row in grid:
        print("    |" + "".join(row) + "|")
    print("    +" + "-" * w + "+")


async def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    async with async_playwright() as p:
        page, own_ctx = await get_page(p)
        await page.bring_to_front()
        solver = cs.TencentCaptchaSolver(page)
        solver.watch_network()

        # 监听验证请求 —— errorCode==0 且有 ticket 才是真通过
        verify_result = {}

        async def on_resp(resp):
            if "turing.captcha.qcloud.com" not in resp.url:
                return
            if "cap_union_new_verify" in resp.url:
                try:
                    raw = (await resp.body()).decode("utf-8", "replace")
                    data = json.loads(raw)
                    verify_result.clear()
                    verify_result.update(data)
                    print(f"  [VERIFY] errorCode={data.get('errorCode')} "
                          f"ticket={'有' if data.get('ticket') else '无'} "
                          f"randstr={data.get('randstr')}")
                except Exception as e:
                    print(f"  [VERIFY] 解析失败: {e}")
                return
            if "getcapbysig" in resp.url:
                return
            body = b""
            try:
                body = await resp.body()
            except Exception:
                pass
            print(f"  [NET] {resp.status} {resp.url[:100]}")
            if body:
                try:
                    print(f"        resp: {body[:200].decode('utf-8', 'replace')}")
                except Exception:
                    print(f"        resp bytes: {len(body)}")
        page.on("response", on_resp)

        ok_cnt = fail_cnt = 0
        for rd in range(rounds):
            st = await popup_state(page)
            if st.get("visible"):
                # 旧题目: 关掉重开, 保证拿到新图
                try:
                    await page.click(".tencent-captcha-dy__header-close", timeout=2500)
                    await page.wait_for_timeout(600)
                except Exception:
                    pass
            solver.net_imgs.clear()
            verify_result.clear()
            if not await trigger_click_captcha(page):
                print("无法触发验证码"); break
            await page.wait_for_timeout(700)
            if not await wait_for_images(solver, 6.0):
                print("图片未就绪, 跳过本轮"); continue

            info = await solver.collect()
            if not info:
                print("collect 失败"); break
            ar = info["area_rect"]
            if ar["w"] < 50 or ar["h"] < 50:
                print("图片区异常, 跳过本轮"); await asyncio.sleep(1); continue

            bg_bytes = await solver.fetch(info["bg_url"])
            tip_bytes = await solver.fetch(info["tip_url"]) if info["tip_url"] else b""
            if not bg_bytes or not tip_bytes:
                print("取图失败"); continue
            bg = cs.decode(bg_bytes)
            tip = cs.decode(tip_bytes)

            res = v4.locate_targets(tip, bg, verbose=True)
            resp = res["bg_mask"]
            pts = sorted(res["points"], key=lambda q: q["tpl"])
            print(f"\n--- round {rd+1} --- 提示 {res['N']} 字符, 定位 {len(pts)}")
            if len(pts) != res["N"]:
                print("  定位数不足, 跳过")
                continue
            for pt in pts:
                print(f"  tpl{pt['tpl']} score={pt['score']:.3f} iou={pt['iou']:.2f} "
                      f"bbox={pt['bbox']}")

            sx = ar["w"] / float(bg.shape[1])
            dump_bg(resp, pts)
            click_list = pts[:ONLY] if ONLY else pts
            for pt in click_list:
                px, py = snap_click_point(resp, pt["bbox"], pt["cx"], pt["cy"])
                cx_page = ar["x"] + px * sx
                cy_page = ar["y"] + py * sx
                print(f"    -> 点 tpl{pt['tpl']} page({cx_page:.0f},{cy_page:.0f})")
                await solver._human_move(cx_page, cy_page)
                await page.mouse.click(cx_page, cy_page)
                await page.wait_for_timeout(random.uniform(0.45, 0.9))
                marks = await click_marks(page)
                print(f"       点击后标记: {marks}")

            try:
                await page.click(".tencent-captcha-dy__verify-confirm-btn", timeout=3000)
            except Exception:
                pass
            await page.wait_for_timeout(1900)

            st2 = await popup_state(page)
            ec = verify_result.get("errorCode")
            passed = (ec == "0" and bool(verify_result.get("ticket"))) or \
                     st2.get("success") or (not st2.get("visible"))
            if passed:
                ok_cnt += 1
                print(f"  ==> 通过! (errorCode={ec}, "
                      f"ticket={'有' if verify_result.get('ticket') else '无'})")
            else:
                fail_cnt += 1
                print(f"  ==> 失败: errorCode={ec} {st2.get('error') or st2}")

            if COLLECT:
                idx = len([f for f in os.listdir(DATASET) if f.endswith(".npz")])
                bx = (np.array([p["bbox"] for p in pts], np.int32) if pts
                      else np.zeros((0, 4), np.int32))
                np.savez_compressed(os.path.join(DATASET, f"s{idx:04d}.npz"),
                                    bg=cs.to_gray(bg), tip=cs.to_gray(tip),
                                    boxes=bx, ec=np.array([str(ec)]))
                print(f"  [采集] s{idx:04d}.npz ec={ec}")

        print(f"\n===== 结果: 通过 {ok_cnt} / 失败 {fail_cnt} =====")
        if own_ctx is not None:
            await own_ctx.close()


if __name__ == "__main__":
    asyncio.run(main())
