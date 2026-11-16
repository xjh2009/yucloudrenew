# -*- coding: utf-8 -*-
"""
雨云 (Rainyun) 自动签到 —— Playwright 版

流程:
  1. 打开积分页 https://app.rainyun.com/account/reward/earn
  2. 没登录就填账密登录; 弹腾讯滑块验证码就自动解 (OpenCV 掩码匹配)
  3. 回到积分页, 点"领取奖励"; 若再弹验证码继续解
  4. 读剩余积分, 推送通知

配置 (环境变量):
  RAINYUN_USER / RAINYUN_PASS   必填, 支持多行(多账户)
  HEADLESS=true                 无头模式
  DEBUG=true                    打印更多日志
  MAX_WORKERS=2                 并发账户数
  CHECKIN_MAX_RETRIES=2         失败重试次数
  NOTIFY_ONLY_FAILURE=true      只在有失败时推送
  CAPTCHA_CDP=http://127.0.0.1:9222   连已开好的浏览器(CI 里不用)
"""
import asyncio, io, os, sys, json, random, re, time, hashlib, tempfile, base64
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from playwright.async_api import async_playwright
import numpy as np, cv2
import slider as slider_mod

LOGIN_URL = "https://app.rainyun.com/auth/login"
EARN_URL = "https://app.rainyun.com/account/reward/earn"
COOKIE_DIR = os.environ.get("COOKIE_DIR",
                           os.path.join(tempfile.gettempdir(), "rainyun-cookies"))
SHOT_DIR = os.path.join(tempfile.gettempdir(), "ry_shots")


async def shot(page, name):
    """截图存盘, 便于事后核对页面/验证码长什么样"""
    try:
        os.makedirs(SHOT_DIR, exist_ok=True)
        path = os.path.join(SHOT_DIR, f"{name}.png")
        await page.screenshot(path=path)
        log(f"  [截图] {path}")
        return path
    except Exception as e:
        log(f"  [截图失败] {name}: {str(e)[:60]}")
        return None

DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
HEADLESS = os.environ.get("HEADLESS", "false").lower() == "true"
CDP_URL = os.environ.get("CAPTCHA_CDP", "")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def mask_user(u):
    return f"{u[:3]}***{u[-3:] if len(u) > 6 else u}"


# ------------------------------------------------------------------ 指纹

def make_fingerprint(seed_str):
    """每个账户一套稳定的随机指纹"""
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    ua_ver = rng.choice([124, 125, 126, 127, 128])
    ua = (f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          f"(KHTML, like Gecko) Chrome/{ua_ver}.0.0.0 Safari/537.36")
    lang = rng.choice([["zh-CN", "zh"], ["zh-CN", "zh", "en-US"]])
    vendors = [("Intel Inc.", "Intel Iris Xe Graphics"),
               ("NVIDIA Corporation", "NVIDIA GeForce RTX 4070/PCIe/SSE2"),
               ("AMD", "AMD Radeon RX 7800 XT")]
    vendor, renderer = rng.choice(vendors)
    js = f"""
    (() => {{
      'use strict';
      const mk = (name, val) => Object.defineProperty(navigator, name,
        {{get: () => val, configurable: true}});
      mk('webdriver', undefined);
      mk('hardwareConcurrency', {rng.choice([4, 8, 12, 16])});
      mk('deviceMemory', {rng.choice([8, 16, 32])});
      mk('languages', {lang!r});
      mk('language', {lang[0]!r});
      mk('platform', 'Win32');
      if (!window.chrome) window.chrome = {{runtime: {{}}}};
      const gp = WebGLRenderingContext.prototype.getParameter;
      WebGLRenderingContext.prototype.getParameter = function(p) {{
        if (p === 37445) return {vendor!r};
        if (p === 37446) return {renderer!r};
        return gp.apply(this, arguments);
      }};
      const orig = HTMLCanvasElement.prototype.toDataURL;
      HTMLCanvasElement.prototype.toDataURL = function() {{
        const c = this, ctx = c.getContext('2d');
        if (ctx) {{
          const im = ctx.getImageData(0, 0, c.width, c.height), d = im.data;
          for (let i = 0; i < d.length; i += 4) {{
            if (Math.random() < 0.006) {{ d[i] ^= 1; d[i+1] ^= 1; }}
          }}
          ctx.putImageData(im, 0, 0);
        }}
        return orig.apply(this, arguments);
      }};
    }})();
    """
    return ua, js


# ------------------------------------------------------------------ 验证码

SLIDER_JS = r"""() => {
    const bg = document.querySelector('#slideBg') || document.querySelector('[class*=bg-img]');
    if (!bg) return {ok: false};
    const cs = getComputedStyle(bg);
    const m = (cs.backgroundImage || '').match(/url\(["']?(.*?)["']?\)/);
    const r = bg.getBoundingClientRect();
    const out = {ok: true, bgUrl: m ? m[1] : null,
                 bgRect: {x: r.x, y: r.y, w: r.width, h: r.height},
                 tile: null, btn: null,
                 status: (document.querySelector('#tcStatus') || {}).className || ''};
    // 拼图块: 全文档搜(叠在背景图上), 排除图标/按钮类
    const cands = [];
    for (const e of document.querySelectorAll('*')) {
        const bi = getComputedStyle(e).backgroundImage || '';
        if (!bi || bi === 'none' || e.id === 'slideBg') continue;
        const u = bi.match(/url\(["']?(.*?)["']?\)/);
        if (!u) continue;
        const rr = e.getBoundingClientRect();
        if (rr.width < 15 || rr.width > 200 || rr.height < 15 || rr.height > 200) continue;
        const cls = (e.className || '').toString();
        if (/icon|btn|button|close|reload|error|fail|feedback|watermark|cover|status/i.test(cls)) continue;
        if (/icon|btn|button|close|reload|error|fail/i.test(e.id || '')) continue;
        cands.push({url: u[1], w: rr.width, h: rr.height, x: rr.x, y: rr.y,
                    id: e.id, cls: cls.slice(0, 60),
                    pos: (e.getAttribute('style') || '').slice(0, 200),
                    isData: u[1].startsWith('data:')});
    }
    out.tile = cands[0] || null;
    out.tileCands = cands.slice(0, 6);
    // 诊断: 整个文档里带 id 的元素 + 所有带背景图的元素
    out.tree = [];
    for (const e of document.querySelectorAll('[id], [class*="drag"], [class*="slide"]')) {
        const rr = e.getBoundingClientRect();
        const st = getComputedStyle(e);
        out.tree.push({tag: e.tagName, id: e.id, cls: (e.className || '').toString().slice(0, 40),
                       w: Math.round(rr.width), h: Math.round(rr.height),
                       x: Math.round(rr.x), y: Math.round(rr.y),
                       bi: /background-image/.test(st.backgroundImage || '') ? 'Y' : '',
                       pos: (e.getAttribute('style') || '').slice(0, 110)});
    }
    out.hostId = host.id
    out.hostCls = (host.className || '').toString();
    // 滑块按钮
    for (const sel of ['.tc-drag-icon', '.tc-drag-button', '.tc-drag-bar',
                       '.tc-drag-slider', '#tcStatus i', '#tcStatus span']) {
        const e = document.querySelector(sel);
        if (e) {
            const rr = e.getBoundingClientRect();
            if (rr.width > 4 && rr.height > 4) {
                out.btn = {x: rr.x, y: rr.y, w: rr.width, h: rr.height, sel};
                break;
            }
        }
    }
    return out;
}"""

SLIDER_STATUS_JS = r"""() => {
    const st = document.querySelector('#tcStatus');
    const cls = (st ? st.className : '') || '';
    const suc = document.querySelector('#statusSuccess');
    const err = document.querySelector('#statusError');
    return {cls: cls, success: cls.indexOf('show-success') >= 0,
            fail: cls.indexOf('show-fail') >= 0 || cls.indexOf('show-error') >= 0,
            sucVisible: !!(suc && suc.offsetParent !== null),
            errText: err ? (err.innerText || '').trim() : ''};
}"""


async def fetch_img(page, url, frame=None):
    """
    取图。验证码图在 turing.captcha.qcloud.com, 页面是 gtimg.com 的 iframe,
    浏览器里 fetch 会跨域失败 —— 用 Playwright 的 request 上下文(共享 cookie, 不吃 CORS)。
    """
    if not url:
        return None
    if url.startswith("data:"):
        try:
            import base64 as _b64
            head, _, payload = url.partition(",")
            return _b64.b64decode(payload)
        except Exception:
            return None
    # 1) Playwright request(带浏览器 cookie, 无 CORS 限制)
    try:
        r = await page.request.get(url, timeout=15000)
        if r.ok:
            body = await r.body()
            if body:
                return body
    except Exception as e:
        if DEBUG:
            log(f"    request 取图失败 {url[:70]}: {str(e)[:70]}")
    # 2) 退回页面里 fetch(部分域带 CORS 头时可用)
    js = """async (u) => {
        try {
            const r = await fetch(u, {credentials: 'include'});
            if (!r.ok) return null;
            const b = await r.arrayBuffer();
            const arr = new Uint8Array(b);
            let s = '';
            for (let i = 0; i < arr.length; i += 0x8000)
                s += String.fromCharCode.apply(null, arr.subarray(i, i + 0x8000));
            return btoa(s);
        } catch (e) { return null; }
    }"""
    for target in (page, frame):
        if target is None:
            continue
        try:
            b64 = await target.evaluate(js, url)
            if b64:
                return base64.b64decode(b64)
        except Exception as e:
            if DEBUG:
                log(f"    evaluate 取图失败 {url[:60]}: {str(e)[:60]}")
    return None


async def find_slider_frame(page, timeout=25):
    """找到含 #slideBg 的 frame (TCaptcha 可能在 iframe 里, 也可能内嵌)"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        for f in page.frames:
            try:
                d = await f.evaluate(SLIDER_JS)
            except Exception:
                continue
            if d.get("ok") and d.get("bgUrl"):
                return f, d
        await page.wait_for_timeout(500)
    return None, None


async def drag_mouse(page, x0, y0, dist, steps=28):
    """带缓动的拖动, 别用瞬间跳跃(会被判定为机器人)"""
    await page.mouse.move(x0, y0)
    await page.mouse.down()
    await page.wait_for_timeout(random.uniform(60, 160))
    for i in range(1, steps + 1):
        t = i / steps
        ease = 1 - (1 - t) ** 2.2
        y = y0 + random.uniform(-1.2, 1.2)
        await page.mouse.move(x0 + dist * ease, y)
        await page.wait_for_timeout(random.uniform(10, 26))
    await page.mouse.move(x0 + dist, y0)
    await page.wait_for_timeout(random.uniform(40, 110))
    await page.mouse.up()


CLICK_JS = r"""() => {
    const out = {ok: false, diag: [], html: ''};
    const vis = e => e && e.offsetParent !== null;
    const q = s => document.querySelector(s);

    // 1) 先定位弹窗: 找含"请依次点击"的文本, 向上找到弹窗容器
    let root = null;
    for (const e of document.querySelectorAll('*')) {
        const t = (e.innerText || '').trim();
        if (t.indexOf('请依次点击') >= 0 && e.children.length <= 4) {
            let n = e;
            for (let i = 0; i < 8 && n; i++) {
                const r = n.getBoundingClientRect();
                if (r.width > 200 && r.width < 600 && r.height > 150) { root = n; break; }
                n = n.parentElement;
            }
            if (!root) root = e;
            break;
        }
    }
    if (!root) root = q('#tcaptcha_transform_dy') || q('[id*=captcha_transform]') || document.body;
    out.rootCls = (root.className || '').toString().slice(0, 80);
    out.rootId = root.id || '';

    // 2) 弹窗内的背景图(带 background-image, 尺寸大)
    const bgC = [];
    for (const e of root.querySelectorAll('*')) {
        const bi = getComputedStyle(e).backgroundImage || '';
        if (bi === 'none') continue;
        const r = e.getBoundingClientRect();
        if (!vis(e) || r.width < 120 || r.height < 80) continue;
        const m = bi.match(/url\(["']?(.*?)["']?\)/);
        if (!m) continue;
        bgC.push({url: m[1], w: r.width, h: r.height, x: r.x, y: r.y,
                  id: e.id, cls: (e.className || '').toString().slice(0, 50)});
    }
    bgC.sort((a, b) => b.w * b.h - a.w * a.h);
    const bg = bgC[0] || null;

    // 3) 弹窗内的提示图(img, 宽大于高的横条)
    const tipC = [];
    for (const e of root.querySelectorAll('img')) {
        if (!vis(e)) continue;
        const r = e.getBoundingClientRect();
        if (r.width < 20 || r.width > 400 || r.height < 8 || r.height > 160) continue;
        tipC.push({url: e.src, w: r.width, h: r.height, x: r.x, y: r.y});
    }
    tipC.sort((a, b) => (b.w / Math.max(1, b.h)) - (a.w / Math.max(1, a.h)));
    const tip = tipC[0] || null;

    out.diag = bgC.slice(0, 4).concat(tipC.slice(0, 3));
    out.hasTip = !!tip;
    out.hasBg = !!bg;
    if (!tip || !bg) return out;
    out.ok = true;
    out.tipUrl = tip.url;
    out.bgUrl = bg.url;
    out.bgRect = {x: bg.x, y: bg.y, w: bg.w, h: bg.h};
    out.tipRect = {x: tip.x, y: tip.y, w: tip.w, h: tip.h};
    // 确定按钮
    let cf = null;
    for (const e of root.querySelectorAll('*')) {
        if ((e.innerText || '').trim() === '确定' && e.children.length <= 3) { cf = e; break; }
    }
    out.confirm = cf ? {x: cf.getBoundingClientRect().x, y: cf.getBoundingClientRect().y,
                       w: cf.getBoundingClientRect().width} : null;
    return out;
}"""


async def solve_click_captcha(page, tag=""):
    """图形点选验证码: 用 OpenCV 求解器定位, 再按提示顺序点击"""
    import locator as loc
    target = None
    info = None
    for cand in [page] + list(page.frames):
        try:
            d = await cand.evaluate(CLICK_JS)
        except Exception as e:
            if DEBUG:
                log(f"  {tag} 点选探测异常({type(e).__name__}): {str(e)[:70]}")
            continue
        if d.get("ok") and d.get("bgUrl"):
            target, info = cand, d
            break
        if DEBUG:
            log(f"  {tag} 点选探测: hasTip={d.get('hasTip')} hasBg={d.get('hasBg')} "
                f"tipCls={d.get('tipCls')} bgCls={d.get('bgCls')}")
            for t in (d.get("diag") or []):
                if t.get("bi") == "Y" or t.get("im"):
                    log(f"      - {t}")
            log("  html: " + (d.get("html") or "")[:1500])
    if not info:
        return False
    log(f"  {tag} 图形点选验证码, 提示: {info.get('header')}")

    tip_bytes = await fetch_img(page, info["tipUrl"], target)
    bg_bytes = await fetch_img(page, info["bgUrl"], target)
    if not tip_bytes or not bg_bytes:
        log(f"  {tag} 验证码图片下载失败 tip={bool(tip_bytes)} bg={bool(bg_bytes)}")
        return False
    tip = cv2.imdecode(np.frombuffer(tip_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    bg = cv2.imdecode(np.frombuffer(bg_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if tip is None or bg is None:
        log(f"  {tag} 验证码图片解码失败")
        return False
    # 存图便于核对(提示图是否完整/背景里的字符长什么样)
    try:
        import tempfile
        _sd = os.path.join(tempfile.gettempdir(), "ry_shots")
        os.makedirs(_sd, exist_ok=True)
        cv2.imwrite(os.path.join(_sd, f"tip_{int(time.time())}.png"), tip)
        cv2.imwrite(os.path.join(_sd, f"bg_{int(time.time())}.png"), bg)
    except Exception:
        pass

    # 定位: 优先"边缘图整图匹配"(不依赖背景分割, 不会漏掉淡描边字符), 不足时退回其它方法
    sp = []
    try:
        sp = loc.locate_by_sprite(bg, tip, verbose=DEBUG)
    except Exception as e:
        if DEBUG:
            log(f"  {tag} sprite 匹配异常: {str(e)[:70]}")
    if len(sp) < 2:
        try:
            sp2 = loc.locate_by_edge(bg, tip, verbose=DEBUG)
        except Exception:
            sp2 = []
        if len(sp2) > len(sp):
            sp = sp2
    if len(sp) < 2:
        try:
            r = loc.locate_targets(tip, bg)
            sp = [(p["cx"], p["cy"], p["iou"], p["tpl"])
                  for p in sorted(r["points"], key=lambda q: q["tpl"])]
        except Exception as e:
            if DEBUG:
                log(f"  {tag} 模板匹配异常: {str(e)[:70]}")
    log(f"  {tag} 定位 {len(sp)} 个目标")
    if not sp:
        return False

    # iframe 内的坐标要加上 iframe 自身偏移
    off = {"x": 0.0, "y": 0.0}
    if target is not page:
        try:
            fe = await target.frame_element()
            bb = await fe.bounding_box()
            if bb:
                off = {"x": bb["x"], "y": bb["y"]}
        except Exception as e:
            if DEBUG:
                log(f"  {tag} frame 偏移计算失败: {str(e)[:70]}")
    br = info["bgRect"]
    if br.get("w", 0) < 50 or br.get("h", 0) < 50:
        log(f"  {tag} 弹窗已关闭/尺寸异常, 跳过")
        return False
    sx = br["w"] / float(bg.shape[1])
    sy = br["h"] / float(bg.shape[0])
    # 合成字符是近纯黑的, 把点击点吸附到笔画上(别点进空心图案的洞里)
    bgg = bg if bg.ndim == 3 else cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    core = (bgg.max(axis=2) < 60)
    for (cx0, cy0, sim, si) in sp:
        cxi, cyi = int(cx0), int(cy0)
        if 0 <= cxi < core.shape[1] and 0 <= cyi < core.shape[0]:
            win = core[max(0, cyi - 12):cyi + 13, max(0, cxi - 12):cxi + 13]
            ys, xs = np.where(win)
            if len(xs):
                d = (xs + max(0, cxi - 12) - cx0) ** 2 + (ys + max(0, cyi - 12) - cy0) ** 2
                i = int(np.argmin(d))
                cxi = int(xs[i] + max(0, cxi - 12))
                cyi = int(ys[i] + max(0, cyi - 12))
        px = off["x"] + br["x"] + cxi * sx
        py = off["y"] + br["y"] + cyi * sy
        log(f"    {tag} 点第{si+1}个 sim={sim*100:.1f}% -> ({px:.0f},{py:.0f})")
        await page.mouse.move(px, py, steps=random.randint(12, 22))
        await page.wait_for_timeout(random.uniform(80, 220))
        await page.mouse.click(px, py)
        await page.wait_for_timeout(random.uniform(350, 700))

    await shot(page, f"05_clicked_{int(time.time())}")
    # 点确定(遍历 主文档 + 所有 frame, 因为"确定"可能在嵌套 iframe 里)
    # 注意: 不能用 .first —— 隐藏的老版滑块模板里也有同名按钮
    hit_btn = False
    sels = ["text=确定", ".verify-btn", ".verify-btn-text",
            ".tencent-captcha-dy__verify-confirm-btn", "button:has-text('确定')",
            "[class*=confirm-btn]", "[class*=verify-btn]"]
    for t in [page] + list(page.frames):
        if hit_btn:
            break
        for sel in sels:
            try:
                els = t.locator(sel)
                n = await els.count()
            except Exception:
                continue
            for i in range(n):
                try:
                    el = els.nth(i)
                    if await el.is_visible():
                        await el.click(timeout=2500)
                        hit_btn = True
                        log(f"    {tag} 已点确定 (sel={sel} #{i})")
                        break
                except Exception:
                    continue
            if hit_btn:
                break
    if not hit_btn:
        # 兜底1: 坐标点(confirm 坐标是相对 target 的, 要加 iframe 偏移)
        if info.get("confirm"):
            cx = off["x"] + info["confirm"]["x"] + info["confirm"]["w"] / 2.0
            cy = off["y"] + info["confirm"]["y"] + 14
            log(f"    {tag} 用坐标点确定 ({cx:.0f},{cy:.0f})")
            await page.mouse.click(cx, cy)
            hit_btn = True
        else:
            # 兜底2: 回车
            log(f"    {tag} 找不到确定按钮, 尝试回车")
            await page.keyboard.press("Enter")
    await page.wait_for_timeout(1500)
    # 看结果
    try:
        st = await target.evaluate("""() => {
            const suc = document.querySelector('.tencent-captcha-dy__verify-status-img--suc');
            const err = document.querySelector('.tencent-captcha-dy__verify-error-text');
            return {ok: !!(suc && suc.offsetParent !== null),
                    err: err ? (err.innerText||'').trim() : ''};
        }""")
    except Exception:
        st = {}
    log(f"  {tag} 点选结果 {st}")
    return bool(st.get("ok"))


async def refresh_captcha(page):
    """
    让验证码换一题(清掉 aged 过期态)。
    注意: 不能用 .first —— 页面里可能挂着多个(隐藏的老模板也有刷新按钮), 要遍历取可见的。
    """
    sels = ["#e_reload", "#reload", ".tencent-captcha-dy__footer-icon--refresh",
            "[class*=refresh]", "[class*=reload]"]
    for sel in sels:
        for target in [page] + list(page.frames):
            try:
                els = target.locator(sel)
                n = await els.count()
            except Exception:
                continue
            for i in range(n):
                try:
                    el = els.nth(i)
                    if await el.is_visible():
                        await el.click(timeout=2500)
                        await page.wait_for_timeout(2600)
                        return True
                except Exception:
                    continue
    return False


async def solve_any_captcha(page, tag="", tries=None):
    """
    自动判断验证码类型(图形点选 / 滑块)并处理。
    失败就刷新换一题重试, 带随机退避(避免固定节奏被风控)。
    """
    if tries is None:
        tries = int(os.environ.get("CAPTCHA_TRIES", "4"))
    last = "未知"
    for attempt in range(tries):
        await page.wait_for_timeout(random.uniform(0.8, 1.6))
        ok = False
        try:
            ok = await solve_click_captcha(page, tag=tag)
            if not ok:
                last = "点选未通过"
        except Exception as e:
            last = f"点选异常 {str(e)[:40]}"
            if DEBUG:
                log(f"  {tag} 点选验证码异常: {str(e)[:70]}")
        if ok:
            return True
        try:
            ok = await solve_slider(page, tag=tag, refresh_if_stale=False)
            if not ok:
                last = "滑块未通过"
        except Exception as e:
            last = f"滑块异常 {str(e)[:40]}"
            if DEBUG:
                log(f"  {tag} 滑块验证码异常: {str(e)[:70]}")
        if ok:
            return True
        # 换一题再来 + 递增退避
        await refresh_captcha(page)
        wait = random.uniform(2.0, 4.0) * (attempt + 1)
        log(f"  {tag} 第{attempt+1}/{tries} 次未通过({last}), 换题后 {wait:.1f}s 重试")
        await page.wait_for_timeout(wait * 1000)
    log(f"  {tag} 重试 {tries} 次仍未通过: {last}")
    return False


async def solve_slider(page, tag="", refresh_if_stale=True):
    """解腾讯滑块, 返回是否通过"""
    frame, info = await find_slider_frame(page)
    if not frame:
        log(f"  {tag} 没找到滑块验证码")
        return False
    # 状态 show-aged = 上一题的验证码已过期; 或缺拼图块/滑块按钮 = UI 还没渲染
    for _ in range(3):
        status = (info.get("status") or "")
        stale = ("aged" in status) or (not info.get("tile")) or (not info.get("btn"))
        if not stale or not refresh_if_stale:
            break
        log(f"  {tag} 验证码状态[{status or '无'}] 不可用, 点刷新换题")
        await refresh_captcha(page)
        frame, info = await find_slider_frame(page, timeout=12)
        if not frame:
            log(f"  {tag} 刷新后没找到滑块")
            return False
    log(f"  {tag} 检测到滑块验证码")
    await shot(page, f"04_slider_{tag}")
    if DEBUG:
        log(f"    bgUrl  = {str(info.get('bgUrl'))[:130]}")
        log(f"    bgRect = {info.get('bgRect')}")
        log(f"    tile   = {json.dumps(info.get('tile'), ensure_ascii=False)[:260]}")
        log(f"    host   = #{info.get('hostId')} {info.get('hostCls')}")
        for t in (info.get("tree") or []):
            blob = f"{t.get('id','')} {t.get('cls','')}".lower()
            if t.get("bi") == "Y" or "drag" in blob or "slide" in blob or "btn" in blob:
                log(f"      - {t}")
        log(f"    btn    = {info.get('btn')}")
        log(f"    status = {info.get('status')}")
    bg_bytes = await fetch_img(page, info["bgUrl"], frame)
    if not bg_bytes:
        log(f"  {tag} 背景图下载失败")
        return False
    bg = cv2.imdecode(np.frombuffer(bg_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if bg is None:
        log(f"  {tag} 背景图解码失败")
        return False
    tile = None
    if info.get("tile"):
        tb = await fetch_img(page, info["tile"]["url"], frame)
        if tb:
            tile = cv2.imdecode(np.frombuffer(tb, np.uint8), cv2.IMREAD_UNCHANGED)
            # sprite 大图: 按元素显示尺寸裁一块(通常是左上角那块)
            if tile is not None and info["tile"].get("w") and info["tile"].get("h"):
                dw, dh = info["tile"]["w"], info["tile"]["h"]
                th, tw = tile.shape[:2]
                if tw > dw * 1.5 or th > dh * 1.5:
                    k = min(tw / max(dw, 1), th / max(dh, 1))
                    tile = tile[:max(4, int(dh * k)), :max(4, int(dw * k))]
                    log(f"  {tag} 拼图块是 sprite 图, 已裁成 {tile.shape[1]}x{tile.shape[0]}")
    if tile is None:
        log(f"  {tag} 没拿到拼图块, 走轮廓兜底")
    gap_x, score, how = slider_mod.solve_gap_x(bg, tile)
    if gap_x is None:
        log(f"  {tag} 缺口定位失败")
        return False
    br = info["bgRect"]
    bw = bg.shape[1]
    disp_gap_x = br["x"] + gap_x * (br["w"] / float(bw))
    tile_x = info["tile"]["x"] if info.get("tile") else br["x"]
    tile_cx = tile_x + (info["tile"]["w"] / 2.0 if info.get("tile") else 0)
    gap_cx_disp = disp_gap_x
    if info.get("tile"):
        gap_cx_disp = disp_gap_x - (info["tile"]["w"] / 2.0) * (br["w"] / float(bw))
    dist = gap_cx_disp - tile_cx
    log(f"  {tag} 缺口x={gap_x:.0f}(原图) score={score:.2f} via={how} "
        f"-> 目标显示x={gap_cx_disp:.0f} 拼图块x={tile_cx:.0f} 拖动={dist:.0f}px")

    btn = info.get("btn")
    if not btn:
        log(f"  {tag} 找不到滑块按钮")
        return False
    sx = btn["x"] + btn["w"] / 2.0
    sy = btn["y"] + btn["h"] / 2.0
    # 拼图块通常挂在按钮上, 拖按钮即可
    await drag_mouse(page, sx, sy, dist)
    await page.wait_for_timeout(1200)
    try:
        st = await frame.evaluate(SLIDER_STATUS_JS)
    except Exception:
        st = {}
    ok = bool(st.get("success"))
    log(f"  {tag} 滑块{'通过' if ok else '未通过'} {st}")
    return ok


# ------------------------------------------------------------------ 主流程

async def dismiss_modal(page):
    """关掉可能挡住视线的确认弹窗"""
    for sel in ["button:has-text('确认')", ".modal-footer button.btn-primary",
                "button.btn-primary:has-text('确定')"]:
        try:
            el = page.locator(sel).first
            if await el.count() > 0 and await el.is_visible():
                await el.click(timeout=1500)
                await page.wait_for_timeout(400)
                return True
        except Exception:
            pass
    return False


async def read_points(page):
    """读积分 —— 页面里形如 "117,092 分" 的数字"""
    txt = await page.evaluate("() => document.body.innerText || ''")
    m = re.search(r'([\d,]{2,})\s*分', txt)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:
            pass
    m = re.search(r'积分[^\d\-]{0,14}(\d[\d,]*)', txt)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:
            pass
    return 0


async def checkin_state(page):
    """
    判断签到状态:
      need    -> 页面还显示[领取奖励], 说明还没签到
      done    -> 已签到(今日已签到/签到成功/明天再来...)
      unknown -> 没识别出来
    """
    txt = await page.evaluate("() => document.body.innerText || ''")
    if "领取奖励" in txt:
        return "need", txt
    for kw in ("已签到", "签到成功", "今日已", "明天再", "每日签到"):
        if kw in txt:
            return "done", txt
    return "unknown", txt


async def do_checkin(page, user, pwd):
    """返回 (ok, msg, points)"""
    await page.goto(EARN_URL, timeout=60000, wait_until="domcontentloaded")
    await page.wait_for_timeout(3500)
    await dismiss_modal(page)
    await shot(page, "01_earn_page")

    if "/auth/login" in page.url:
        log("  Cookie 失效, 走账号密码登录")
        try:
            await page.wait_for_selector("input[name='login-field']", timeout=20000)
            await page.fill("input[name='login-field']", user)
            await page.fill("input[name='login-password']", pwd)
            await page.click("button.btn-primary.btn-block")
        except Exception as e:
            return False, f"登录表单异常: {str(e)[:40]}", 0
        # 验证码(可能弹, 也可能不弹)
        await page.wait_for_timeout(2500)
        for attempt in range(3):
            solved = await solve_any_captcha(page, tag="登录")
            if not solved:
                # 没找到验证码就等等看是不是弹了
                await page.wait_for_timeout(2500)
                if "/auth/login" not in page.url:
                    break
                continue
            await page.wait_for_timeout(3000)
            if "/auth/login" not in page.url:
                break
            # 没过就刷新换题再来
            await refresh_captcha(page)
            await page.wait_for_timeout(random.uniform(2.0, 4.0) * 1000)
        await page.wait_for_timeout(2500)
        await dismiss_modal(page)
        if "/auth/login" in page.url:
            return False, "登录失败(验证码或账密)", 0
        log("  登录成功")
        await shot(page, "02_logged_in")
    else:
        log("  Cookie 有效, 免密登录")

    # 回到积分页
    if "/account/reward/earn" not in page.url:
        await page.goto(EARN_URL, timeout=60000, wait_until="domcontentloaded")
        await page.wait_for_timeout(3500)
    await dismiss_modal(page)

    # 点"领取奖励"
    state, txt = await checkin_state(page)
    pts_before = await read_points(page)
    if DEBUG:
        log(f"  签到前积分: {pts_before}")
    if DEBUG:
        log("  --- 签到前页面文本 ---")
        log("  " + txt.replace("\n", " | ")[:600])
    if state == "done":
        log("  今日已签到, 无需操作")
    else:
        clicked = False
        for sel in ["text=领取奖励", "a:has-text('领取奖励')", "button:has-text('领取奖励')",
                    "span:has-text('领取奖励')", "[class*=checkin]", "[class*=sign]"]:
            try:
                els = page.locator(sel)
                n = await els.count()
            except Exception:
                continue
            for i in range(n):
                try:
                    el = els.nth(i)
                    if await el.is_visible():
                        await el.click(timeout=4000)
                        clicked = True
                        log(f"  已点击[领取奖励] (sel={sel} #{i})")
                        await page.wait_for_timeout(2600)
                        await shot(page, "03_after_reward_click")
                        break
                except Exception:
                    continue
            if clicked:
                break
        if not clicked:
            log("  未找到[领取奖励]按钮")

        await page.wait_for_timeout(2500)
        for attempt in range(3):
            solved = await solve_any_captcha(page, tag="签到")
            if not solved:
                break
            await page.wait_for_timeout(2500)
            st2, _ = await checkin_state(page)
            if st2 != "need":
                break
            if attempt < 2:
                await refresh_captcha(page)
                await page.wait_for_timeout(random.uniform(2.0, 4.0) * 1000)
        await page.wait_for_timeout(1500)
        await dismiss_modal(page)
        state2, txt2 = await checkin_state(page)
        if DEBUG:
            log("  --- 签到后页面文本 ---")
            log("  " + txt2.replace("\n", " | ")[:600])
        points = await read_points(page)
        if state2 == "need" and points <= pts_before:
            return False, f"签到未完成(积分 {pts_before}->{points})", points
        return True, f"签到成功 (积分 {pts_before}->{points})", points

    points = await read_points(page)
    return True, "今日已签到", points


def _sign_state_file(user):
    h = hashlib.md5(("rainyun_sign_" + user).encode()).hexdigest()[:12]
    return os.path.join(COOKIE_DIR, "sign_%s.json" % h)


def load_sign_state(user):
    """今天是否已经**成功**签过 —— 成功的直接跳过, 失败的继续试"""
    try:
        with open(_sign_state_file(user), encoding="utf-8") as f:
            d = json.load(f)
        if d.get("date") == time.strftime("%Y-%m-%d") and d.get("signed"):
            return d
    except Exception:
        pass
    return None


def save_sign_state(user, data):
    try:
        os.makedirs(COOKIE_DIR, exist_ok=True)
        with open(_sign_state_file(user), "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


async def run_account(p, user, pwd):
    st = load_sign_state(user)
    if st:
        log(f"  今天已签到过(记录积分 {st.get('points')}), 跳过")
        return {"user": user, "ok": True, "msg": "今日已签到",
                "points": st.get("points", 0)}
    ua, fp_js = make_fingerprint(user)
    os.makedirs(COOKIE_DIR, exist_ok=True)
    ck = os.path.join(COOKIE_DIR, hashlib.md5(user.encode()).hexdigest()[:16] + ".json")
    ctx = None
    own = False
    try:
        if CDP_URL:
            try:
                browser = await p.chromium.connect_over_cdp(CDP_URL, timeout=6000)
                ctx = browser.contexts[0]
            except Exception:
                ctx = None
        if ctx is None:
            # 每次用全新 profile: TCaptcha 的本地状态(比如 aged/旧 sess)不能跨次复用,
            # 但 Cookie 会另外注入, 所以仍然免密登录
            prof = os.path.join(COOKIE_DIR, "prof_"
                                + hashlib.md5(user.encode()).hexdigest()[:8]
                                + "_" + str(int(time.time())))
            ctx = await p.chromium.launch_persistent_context(
                prof,
                headless=HEADLESS, channel="chrome", user_agent=ua,
                viewport={"width": 1440, "height": 900},
                args=["--no-sandbox", "--disable-dev-shm-usage",
                      "--disable-blink-features=AutomationControlled"])
            own = True
        await ctx.add_init_script(fp_js)
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        if os.path.exists(ck):
            try:
                await ctx.add_cookies(json.load(open(ck, "r", encoding="utf-8")))
                if DEBUG:
                    log("  已加载本地 Cookie")
            except Exception:
                pass
        ok, msg, points = await do_checkin(page, user, pwd)
        prev = load_sign_state(user) or {}
        save_sign_state(user, {
            "date": time.strftime("%Y-%m-%d"),
            "attempted": True,
            "signed": bool(ok and "签到成功" in msg),
            "points": points,
        })
        try:
            await ctx.storage_state(path=ck)
        except Exception:
            pass
        return {"user": user, "ok": ok, "msg": msg, "points": points}
    except Exception as e:
        import traceback
        if DEBUG:
            traceback.print_exc()
        return {"user": user, "ok": False, "msg": f"异常: {str(e)[:60]}", "points": 0}
    finally:
        if own and ctx is not None:
            try:
                await ctx.close()
            except Exception:
                pass


def parse_accounts():
    us = os.environ.get("RAINYUN_USER", "").replace("\r\n", "\n").replace("\r", "\n")
    ps = os.environ.get("RAINYUN_PASS", "").replace("\r\n", "\n").replace("\r", "\n")
    ul = [u.strip() for u in us.split("\n") if u.strip()]
    pl = [p.strip() for p in ps.split("\n") if p.strip()]
    n = min(len(ul), len(pl))
    if ul and pl and len(ul) != len(pl):
        log(f"用户/密码数量不一致({len(ul)} vs {len(pl)}), 取前 {n} 组")
    return list(zip(ul[:n], pl[:n]))


async def main(p):
    accounts = parse_accounts()
    if not accounts:
        log("没有配置 RAINYUN_USER / RAINYUN_PASS")
        return 1
    retries = int(os.environ.get("CHECKIN_MAX_RETRIES", "2"))
    workers = int(os.environ.get("MAX_WORKERS", "2"))
    results = {}
    pending = list(accounts)
    for attempt in range(retries + 1):
        if not pending:
            break
        log(f"===== 第 {attempt+1} 轮, 待处理 {len(pending)} 个账户 =====")
        failed = []
        sem = asyncio.Semaphore(max(1, workers))

        async def one(u, pw):
            async with sem:
                log(f"--- {mask_user(u)} 开始 ---")
                r = await run_account(p, u, pw)
                results[u] = r
                log(f"--- {mask_user(u)} {'成功' if r['ok'] else '失败'}: "
                    f"{r['msg']} 积分={r['points']} ---")
                return r

        for i, (u, pw) in enumerate(pending):
            if i:
                await asyncio.sleep(random.uniform(4, 10))
            await one(u, pw)
        pending = [(u, results[u]["msg"]) for u, _ in pending
                   if not results.get(u, {}).get("ok")]
        if pending:
            wait = random.uniform(20, 40) * (attempt + 1)
            log(f"等待 {wait:.0f}s 后重试 {len(pending)} 个失败账户"
                f" (原因: {[m for _, m in pending]})")
            await asyncio.sleep(wait)

    ok_n = sum(1 for r in results.values() if r["ok"])
    log(f"===== 汇总: {ok_n}/{len(accounts)} 成功 =====")
    for u, _ in accounts:
        r = results.get(u, {})
        log(f"  {mask_user(u)}: {'✅' if r.get('ok') else '❌'} {r.get('msg','')} "
            f"积分={r.get('points', 0)}")

    ok_n = sum(1 for r in results.values() if r["ok"])
    print("\n===== 最终汇总 =====")
    for u, _ in accounts:
        r = results.get(u, {})
        print(f"  {mask_user(u)}: {'✅ 成功' if r.get('ok') else '❌ 失败'} "
              f"积分={r.get('points', 0)} {r.get('msg','')}")
    return 0 if ok_n > 0 else 1


if __name__ == "__main__":
    async def _run():
        async with async_playwright() as p:
            return await main(p)
    sys.exit(asyncio.run(_run()))
