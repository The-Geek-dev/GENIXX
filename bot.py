# ============================================================
# Solana Meme Coin Trading Bot — Inline Button UI
# pip install python-telegram-bot solana solders aiohttp
#             python-dotenv base58 scikit-learn numpy joblib asyncpg
# ============================================================

import os, asyncio, aiohttp, base64, logging, time, traceback, json
import numpy as np
import joblib
import base58
import asyncpg
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TxOpts
from solana.rpc.commitment import Confirmed
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (Application, CommandHandler, CallbackQueryHandler,
                           MessageHandler, ContextTypes, ConversationHandler, filters)
from telegram.error import TelegramError

load_dotenv()

# ============================================================
# LOGGING
# ============================================================
def setup_logging():
    fmt  = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    if hasattr(ch.stream, "reconfigure"):
        ch.stream.reconfigure(encoding="utf-8", errors="replace")
    fh = RotatingFileHandler("bot.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    eh = RotatingFileHandler("bot_errors.log", maxBytes=2*1024*1024, backupCount=2, encoding="utf-8")
    eh.setLevel(logging.ERROR); eh.setFormatter(fmt)
    root.addHandler(ch); root.addHandler(fh); root.addHandler(eh)
    return logging.getLogger("SolanaBot")

log = setup_logging()

# ============================================================
# CONFIG
# ============================================================
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
WALLET_ADDRESS  = os.getenv("WALLET_ADDRESS")
PRIVATE_KEY_B58 = os.getenv("PRIVATE_KEY")
RPC_URL         = os.getenv("RPC_URL", "https://api.mainnet-beta.solana.com")
AUTHORIZED_USER = int(os.getenv("AUTHORIZED_USER_ID", 0))
_db_url = os.getenv("DATABASE_URL", "")
DATABASE_URL = _db_url
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_API  = "https://quote-api.jup.ag/v6/swap"
JUPITER_PRICE_API = "https://lite-api.jup.ag/price/v2"
DEXSCREENER_API   = "https://api.dexscreener.com/latest/dex/tokens/"
RUGCHECK_API      = "https://api.rugcheck.xyz/v1/tokens/{}/report/summary"
RUGCHECK_SCORE_MIN = int(os.getenv("RUGCHECK_SCORE_MIN", "500"))
HELIUS_RPC        = os.getenv("HELIUS_RPC", "")
USDC_MINT         = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SEEN_EXPIRY_SEC   = 1800

# Conversation states
WAITING_BUY_MINT    = 0
WAITING_BUY_SYMBOL  = 1
WAITING_SELL_MINT   = 2
WAITING_CONFIRM_BUY = 3
WAITING_SET_TP      = 10
WAITING_SET_TRAIL   = 11
WAITING_SET_STOP    = 12
WAITING_SET_AMOUNT  = 13
WAITING_SET_SLIP    = 14
WAITING_SET_SCORE   = 15
WAITING_SET_LIQ     = 16
WAITING_SET_RUGCHECK = 17
WAITING_SET_TRAIL_5X  = 20
WAITING_SET_TRAIL_10X = 21
WAITING_SET_TRAIL_20X = 22
WAITING_SET_TRAIL_50X = 23

def validate_config():
    missing = [k for k, v in {
        "TELEGRAM_TOKEN": TELEGRAM_TOKEN, "WALLET_ADDRESS": WALLET_ADDRESS,
        "PRIVATE_KEY": PRIVATE_KEY_B58, "AUTHORIZED_USER_ID": AUTHORIZED_USER,
        "DATABASE_URL": DATABASE_URL,
    }.items() if not v]
    if missing:
        raise SystemExit(f"Missing env vars: {', '.join(missing)}")
    log.info("Config validated")

keypair       = None
solana_client = None
db_pool       = None

# ============================================================
# STATE
# ============================================================
state = {
    "positions": {}, "demo_positions": {},
    "demo_total_pnl": 0.0, "demo_trades": [],
    "total_pnl": 0.0, "trades_history": [],
    "seen_pairs": {},
    "errors": [],
    "api_stats": {
        "price_ok": 0, "price_fail": 0,
        "quote_ok": 0, "quote_fail": 0,
        "swap_ok": 0,  "swap_fail": 0,
        "confirm_ok": 0, "confirm_timeout": 0,
    },
    "settings": {
        "take_profit": 2.0, "trailing_stop": 15, "stop_loss": 0.6,
        "trade_amount": 10.0, "demo_trade_amount": 100.0,
        "slippage_bps": 100, "priority_fee": 20000,
        "auto_snipe": False, "demo_mode": False,
        "min_liquidity": 10000, "min_score": 0.2,
        "min_rugcheck": 500,
        "max_retries": 3, "retry_delay": 1.5, "confirm_timeout": 45,
        # Tiered trailing stop — trail % applied after TP hit, based on peak multiplier
        "trail_5x":  4.0,
        "trail_10x": 3.0,
        "trail_20x": 2.0,
        "trail_50x": 1.5,
        # House money mode — sell initial capital at TP, let profit ride free
        "house_money_mode": True,
    },
    "ml_features": [], "ml_labels": [],
}

# ============================================================
# DATABASE
# ============================================================
async def init_db():
    global db_pool
    import ssl as _ssl
    _ssl_ctx = _ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = _ssl.CERT_NONE
    db_pool = await asyncpg.create_pool(
        DATABASE_URL, min_size=1, max_size=5,
        ssl=_ssl_ctx,
        server_settings={"application_name": "genixx_bot"},
        command_timeout=30,
    )
    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS positions (
                mint TEXT PRIMARY KEY, data JSONB NOT NULL,
                is_demo BOOLEAN DEFAULT FALSE, created_at TIMESTAMP DEFAULT NOW());
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY, symbol TEXT, mint TEXT,
                entry_price FLOAT, exit_price FLOAT, multiplier FLOAT,
                net_pnl FLOAT, fees_paid FLOAT, reason TEXT,
                is_demo BOOLEAN DEFAULT FALSE, tx_sig TEXT,
                features JSONB, created_at TIMESTAMP DEFAULT NOW());
            CREATE TABLE IF NOT EXISTS ml_data (
                id SERIAL PRIMARY KEY, features JSONB NOT NULL,
                label INTEGER NOT NULL, created_at TIMESTAMP DEFAULT NOW());
            CREATE TABLE IF NOT EXISTS bot_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
    log.info("Database ready")

async def db_save_settings():
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO settings (key,value) VALUES ('main',$1) ON CONFLICT (key) DO UPDATE SET value=$1",
            json.dumps(state["settings"]))

async def db_load_settings():
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT value FROM settings WHERE key='main'")
        if row: state["settings"].update(json.loads(row["value"]))

async def db_save_position(mint, pos, is_demo):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO positions (mint,data,is_demo) VALUES ($1,$2,$3) ON CONFLICT (mint) DO UPDATE SET data=$2,is_demo=$3",
            mint, json.dumps(pos), is_demo)

async def db_delete_position(mint):
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM positions WHERE mint=$1", mint)

async def db_load_positions():
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT mint,data,is_demo FROM positions")
        for r in rows:
            pos = json.loads(r["data"])
            (state["demo_positions"] if r["is_demo"] else state["positions"])[r["mint"]] = pos
    log.info(f"Loaded {len(state['positions'])} real + {len(state['demo_positions'])} demo positions")

async def db_save_trade(trade):
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO trades (symbol,mint,entry_price,exit_price,multiplier,
            net_pnl,fees_paid,reason,is_demo,tx_sig,features)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
        """, trade.get("symbol"), trade.get("mint"), trade.get("entry"),
            trade.get("exit"), trade.get("mult"), trade.get("net_pnl"),
            trade.get("fees_paid", 0), trade.get("reason"),
            trade.get("is_demo", False), trade.get("tx_sig"),
            json.dumps(trade.get("features", [])))

async def db_load_trades():
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM trades ORDER BY created_at DESC LIMIT 200")
        for r in rows:
            t = dict(r)
            t["features"] = json.loads(t["features"]) if t["features"] else []
            if t["is_demo"]:
                state["demo_trades"].append({
                    "symbol": t["symbol"], "mult": t["multiplier"],
                    "net_pnl": t["net_pnl"], "reason": t["reason"],
                    "projected_real": t["net_pnl"] * (
                        state["settings"]["trade_amount"] / state["settings"]["demo_trade_amount"])
                })
            else:
                state["trades_history"].append({
                    "symbol": t["symbol"], "mult": t["multiplier"],
                    "net_pnl": t["net_pnl"], "reason": t["reason"]})
        pnl_rows = await conn.fetch(
            "SELECT SUM(net_pnl) as total, is_demo FROM trades GROUP BY is_demo")
        for r in pnl_rows:
            if r["is_demo"]: state["demo_total_pnl"] = float(r["total"] or 0)
            else: state["total_pnl"] = float(r["total"] or 0)

async def db_save_ml_sample(features, label):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO ml_data (features,label) VALUES ($1,$2)",
            json.dumps(features), label)

async def db_load_ml_data():
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT features,label FROM ml_data ORDER BY id")
        for r in rows:
            feats = json.loads(r["features"])
            if len(feats) < 16:
                feats = feats + [0.0] * (16 - len(feats))
            state["ml_features"].append(feats)
            state["ml_labels"].append(r["label"])
    log.info(f"ML data loaded: {len(state['ml_features'])} samples")

async def load_all_from_db():
    await db_load_settings()
    await db_load_positions()
    await db_load_trades()
    await db_load_ml_data()
    if len(state["ml_features"]) >= 10:
        train_model()
    log.info("All state restored from DB")

# ============================================================
# ERROR TRACKING
# ============================================================
def log_error(ctx, e, extra=""):
    msg = f"[{ctx}] {type(e).__name__}: {e}" + (f" | {extra}" if extra else "")
    log.error(msg)
    log.debug(traceback.format_exc())
    state["errors"].append({"time": datetime.now().strftime("%H:%M:%S"), "context": ctx, "error": str(e)[:120]})
    if len(state["errors"]) > 20: state["errors"].pop(0)

async def notify_error(app, ctx, e):
    try:
        await app.bot.send_message(chat_id=AUTHORIZED_USER,
            text=f"⚠️ *Bot Error*\n`{ctx}`\n`{type(e).__name__}: {str(e)[:200]}`",
            parse_mode="Markdown")
    except TelegramError as te:
        log.error(f"Could not notify error: {te}")

async def with_retry(fn, retries=3, delay=1.5, label=""):
    last = None
    for i in range(1, retries + 1):
        try: return await fn()
        except Exception as e:
            last = e
            if i < retries: await asyncio.sleep(delay * (2 ** (i-1)))
    raise last

# ============================================================
# ML MODEL
# ============================================================
ml_model = None; ml_scaler = StandardScaler(); ml_ready = False

def extract_features(td):
    try:
        liq      = float(td.get("liquidity",{}).get("usd",0) or 0)
        vol24    = float(td.get("volume",{}).get("h24",0) or 0)
        vol5m    = float(td.get("volume",{}).get("m5",0) or 0)
        pc1      = float(td.get("priceChange",{}).get("h1",0) or 0)
        pc6      = float(td.get("priceChange",{}).get("h6",0) or 0)
        pc24     = float(td.get("priceChange",{}).get("h24",0) or 0)
        buys1h   = float(td.get("txns",{}).get("h1",{}).get("buys",0) or 0)
        sells1h  = float(td.get("txns",{}).get("h1",{}).get("sells",0) or 0)
        buys5m   = float(td.get("txns",{}).get("m5",{}).get("buys",0) or 0)
        sells5m  = float(td.get("txns",{}).get("m5",{}).get("sells",0) or 0)
        mcap     = float(td.get("marketCap",0) or 0)
        age      = (time.time()*1000-(td.get("pairCreatedAt") or time.time()*1000))/60000
        bs_ratio = buys1h/(sells1h+1)
        bs5m     = buys5m/(sells5m+1)
        liq_mcap = liq/(mcap+1)
        vol_liq  = vol24/(liq+1)
        return [liq, vol24, pc1, pc6, pc24, buys1h, sells1h, bs_ratio,
                age, mcap, vol5m, buys5m, sells5m, bs5m, liq_mcap, vol_liq]
    except Exception as e:
        log_error("extract_features", e); return [0.0]*16

def train_model():
    global ml_model, ml_scaler, ml_ready
    try:
        from sklearn.metrics import precision_score, recall_score
        X, y = np.array(state["ml_features"]), np.array(state["ml_labels"])
        if len(X) < 10: return None
        ml_scaler = StandardScaler(); Xs = ml_scaler.fit_transform(X)
        ml_model  = RandomForestClassifier(n_estimators=100, max_depth=6,
                                            min_samples_leaf=2, random_state=42,
                                            class_weight="balanced")
        ml_model.fit(Xs, y); ml_ready = True
        preds     = ml_model.predict(Xs)
        wins      = int(np.sum(y))
        precision = precision_score(y, preds, zero_division=0)
        recall    = recall_score(y, preds, zero_division=0)
        log.info(f"ML trained: {len(X)} samples | wins={wins} | precision={precision:.0%} recall={recall:.0%}")
        return precision
    except Exception as e:
        log_error("train_model", e); return None

def predict_score(features):
    try:
        if not ml_ready or ml_model is None: return 0.5
        if len(features) < 16:
            features = list(features) + [0.0] * (16 - len(features))
        return float(ml_model.predict_proba(ml_scaler.transform([features]))[0][1])
    except Exception as e:
        log_error("predict_score", e); return 0.5

async def record_trade_outcome(features, profitable):
    label = 1 if profitable else 0
    if len(features) < 16:
        features = list(features) + [0.0] * (16 - len(features))
    state["ml_features"].append(features)
    state["ml_labels"].append(label)
    await db_save_ml_sample(features, label)
    return train_model()

# ============================================================
# AUTH
# ============================================================
def auth(func):
    async def wrapper(update, ctx):
        uid = (update.effective_user.id if update.effective_user else None)
        if uid != AUTHORIZED_USER:
            await (update.message or update.callback_query.message).reply_text("Unauthorized.")
            return
        try: return await func(update, ctx)
        except Exception as e:
            log_error(f"cmd/{func.__name__}", e)
            msg = update.message or (update.callback_query.message if update.callback_query else None)
            if msg:
                await msg.reply_text(f"Error: `{type(e).__name__}: {str(e)[:100]}`", parse_mode="Markdown")
    return wrapper

def calc_fees(amount_usd, priority_lp=20000):
    dex = round(amount_usd*0.003, 4); pri = round((priority_lp/1e9)*150.0, 4)
    net = round(0.000005*150.0, 6)
    return {"dex_fee": dex, "priority_fee": pri, "network_fee": net, "total": round(dex+pri+net, 4)}

async def _safe_notify(app, text):
    if app is None:
        log.warning(f"_safe_notify called with app=None, dropping: {text[:80]}")
        return
    try:
        await app.bot.send_message(chat_id=AUTHORIZED_USER, text=text,
            parse_mode="Markdown", disable_web_page_preview=True)
    except TelegramError as e:
        log.error(f"Notify failed: {e}")

# ============================================================
# TOKEN DATA
# ============================================================
async def get_token_price(mint: str, pair_data: dict = None) -> float:
    if pair_data:
        try:
            price = float(pair_data.get("priceUsd", 0) or 0)
            if price > 0:
                state["api_stats"]["price_ok"] += 1
                return price
        except Exception:
            pass

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
            async with s.get(f"{DEXSCREENER_API}{mint}") as r:
                if r.status == 200:
                    pairs     = (await r.json()).get("pairs", [])
                    sol_pairs = [p for p in (pairs or []) if p.get("chainId") == "solana"]
                    if sol_pairs:
                        price = float(sol_pairs[0].get("priceUsd", 0) or 0)
                        if price > 0:
                            state["api_stats"]["price_ok"] += 1
                            return price
    except Exception as e:
        log.warning(f"DexScreener price failed: {e}")

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
            async with s.get(
                f"https://public-api.birdeye.so/defi/price?address={mint}",
                headers={"X-Chain": "solana"}
            ) as r:
                if r.status == 200:
                    data  = await r.json()
                    price = float(data.get("data", {}).get("value", 0) or 0)
                    if price > 0:
                        state["api_stats"]["price_ok"] += 1
                        return price
    except Exception as e:
        log.warning(f"Birdeye price failed: {e}")

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6)) as s:
            async with s.get(f"{JUPITER_PRICE_API}?ids={mint}") as r:
                if r.status == 200:
                    price = float((await r.json()).get("data", {}).get(mint, {}).get("price", 0) or 0)
                    if price > 0:
                        state["api_stats"]["price_ok"] += 1
                        return price
    except Exception as e:
        log.warning(f"Jupiter price failed: {e}")

    state["api_stats"]["price_fail"] += 1
    return 0.0

async def get_token_data(mint):
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.get(DEXSCREENER_API+mint) as r:
                if r.status != 200: return None
                pairs = [p for p in (await r.json()).get("pairs",[]) if p.get("chainId")=="solana"]
                return max(pairs, key=lambda p: float(p.get("liquidity",{}).get("usd",0) or 0)) if pairs else None
    except Exception as e:
        log_error("get_token_data", e); return None

async def check_token_safety(mint):
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
            async with s.get(RUGCHECK_API.format(mint)) as r:
                if r.status == 200:
                    d = await r.json()
                    return {"score": d.get("score",0), "risks": d.get("risks",[]), "rugged": d.get("rugged",False)}
    except Exception as e:
        log_error("check_token_safety", e)
    return {"score": 0, "risks": [{"name":"Safety unavailable"}], "rugged": False}

# ============================================================
# JUPITER
# ============================================================
async def get_quote(in_m, out_m, amt, slippage=100):
    async def _f():
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.get(JUPITER_QUOTE_API,
                params={"inputMint":in_m,"outputMint":out_m,"amount":amt,"slippageBps":slippage}) as r:
                if r.status != 200: raise ValueError(f"Quote {r.status}")
                return await r.json()
    try:
        q = await with_retry(_f, label="quote")
        state["api_stats"]["quote_ok"] += 1; return q
    except Exception as e:
        state["api_stats"]["quote_fail"] += 1; log_error("get_quote", e); return None

async def get_swap_tx(quote, priority_fee):
    async def _f():
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.post(JUPITER_SWAP_API, json={
                "quoteResponse": quote, "userPublicKey": WALLET_ADDRESS,
                "wrapAndUnwrapSol": True, "prioritizationFeeLamports": priority_fee,
                "dynamicComputeUnitLimit": True}) as r:
                if r.status != 200: raise ValueError(f"Swap {r.status}")
                d = await r.json()
                if not d.get("swapTransaction"): raise ValueError("Missing swapTransaction")
                return d["swapTransaction"]
    try:
        tx = await with_retry(_f, label="swap_tx")
        state["api_stats"]["swap_ok"] += 1; return tx
    except Exception as e:
        state["api_stats"]["swap_fail"] += 1; log_error("get_swap_tx", e); return None

async def sign_and_send(tx_b64):
    try:
        tx = VersionedTransaction.from_bytes(base64.b64decode(tx_b64))
        tx.sign([keypair])
        r  = await solana_client.send_raw_transaction(
            bytes(tx), opts=TxOpts(skip_preflight=False, preflight_commitment=Confirmed))
        return str(r.value)
    except Exception as e:
        log_error("sign_and_send", e); return None

async def confirm_tx(sig):
    for i in range(state["settings"]["confirm_timeout"]):
        try:
            st = (await solana_client.get_signature_statuses([sig])).value[0]
            if st and st.confirmation_status in ("confirmed","finalized"):
                state["api_stats"]["confirm_ok"] += 1; return True
            if st and st.err: return False
        except Exception as e:
            log.warning(f"Confirm poll {i+1}: {e}")
        await asyncio.sleep(1)
    state["api_stats"]["confirm_timeout"] += 1; return False

async def execute_buy(mint, amt):
    q = await get_quote(USDC_MINT, mint, int(amt*1e6), state["settings"]["slippage_bps"])
    if not q: return None
    tx = await get_swap_tx(q, state["settings"]["priority_fee"])
    if not tx: return None
    sig = await sign_and_send(tx)
    if not sig: return None
    return {"signature": sig, "confirmed": await confirm_tx(sig), "out_amount": int(q.get("outAmount",0))}

async def execute_sell(mint, token_amt):
    if token_amt <= 0: return None
    q = await get_quote(mint, USDC_MINT, token_amt, state["settings"]["slippage_bps"])
    if not q: return None
    tx = await get_swap_tx(q, state["settings"]["priority_fee"])
    if not tx: return None
    sig = await sign_and_send(tx)
    if not sig: return None
    return {"signature": sig, "confirmed": await confirm_tx(sig),
            "usdc_received": int(q.get("outAmount",0))/1e6}

# ============================================================
# KEYBOARDS
# ============================================================
def kb_main():
    s = state["settings"]
    sniper_btn = "🟢 Sniper ON" if s["auto_snipe"] else "🔴 Sniper OFF"
    demo_btn   = "📝 Demo ON"   if s["demo_mode"]  else "📝 Demo OFF"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💰 Positions",    callback_data="positions"),
         InlineKeyboardButton("📊 P&L",          callback_data="pnl")],
        [InlineKeyboardButton("🛒 Buy Token",    callback_data="buy_prompt"),
         InlineKeyboardButton("💸 Sell Token",   callback_data="sell_prompt")],
        [InlineKeyboardButton(sniper_btn,        callback_data="toggle_sniper"),
         InlineKeyboardButton(demo_btn,          callback_data="toggle_demo")],
        [InlineKeyboardButton("📝 Demo Trades",  callback_data="demo_menu"),
         InlineKeyboardButton("🧠 ML Stats",     callback_data="mlstats")],
        [InlineKeyboardButton("⚙️ Settings",     callback_data="settings_menu"),
         InlineKeyboardButton("🏥 Health",       callback_data="health")],
        [InlineKeyboardButton("📜 History",      callback_data="history")],
    ])

def kb_settings():
    s = state["settings"]
    hm_lbl = "🏠 House Money: ON" if s.get("house_money_mode") else "🏠 House Money: OFF"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🎯 Take Profit: {s['take_profit']}x",     callback_data="set_tp"),
         InlineKeyboardButton(f"📉 Trailing: {s['trailing_stop']}%",      callback_data="set_trail")],
        [InlineKeyboardButton(f"🛑 Stop Loss: {s['stop_loss']}x",         callback_data="set_stop"),
         InlineKeyboardButton(f"💵 Amount: ${s['trade_amount']}",         callback_data="set_amount")],
        [InlineKeyboardButton(f"⚡ Slippage: {s['slippage_bps']}bps",     callback_data="set_slip"),
         InlineKeyboardButton(f"🧠 Min Score: {s['min_score']:.0%}",      callback_data="set_score")],
        [InlineKeyboardButton(f"💧 Min Liq: ${s['min_liquidity']:,.0f}",  callback_data="set_liq"),
         InlineKeyboardButton(f"🛡️ Max RugScore: {s['min_rugcheck']}",   callback_data="set_rugcheck")],
        [InlineKeyboardButton(hm_lbl,                                      callback_data="toggle_house_money")],
        [InlineKeyboardButton("📐 Tiered Trail Settings",                  callback_data="tiered_trail_menu")],
        [InlineKeyboardButton("⬅️ Back to Menu",                           callback_data="main_menu")],
    ])

def kb_tiered_trail():
    s = state["settings"]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(
            f"🟢 ≥5x  → {s.get('trail_5x', 4.0)}% trail",  callback_data="set_trail_5x")],
        [InlineKeyboardButton(
            f"🟡 ≥10x → {s.get('trail_10x', 3.0)}% trail", callback_data="set_trail_10x")],
        [InlineKeyboardButton(
            f"🟠 ≥20x → {s.get('trail_20x', 2.0)}% trail", callback_data="set_trail_20x")],
        [InlineKeyboardButton(
            f"🔴 ≥50x → {s.get('trail_50x', 1.5)}% trail", callback_data="set_trail_50x")],
        [InlineKeyboardButton("⬅️ Back to Settings",                      callback_data="settings_menu")],
    ])

def kb_demo():
    s = state["settings"]
    demo_btn = "Turn OFF Demo" if s["demo_mode"] else "Turn ON Demo"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Demo Status",   callback_data="demostatus"),
         InlineKeyboardButton("📜 Demo History",  callback_data="demohistory")],
        [InlineKeyboardButton(f"{'🟢' if s['demo_mode'] else '🔴'} {demo_btn}",
                              callback_data="toggle_demo")],
        [InlineKeyboardButton("⬅️ Back to Menu",  callback_data="main_menu")],
    ])

def kb_back():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⬅️ Back to Menu", callback_data="main_menu")]
    ])

def kb_confirm_buy(mint, symbol):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Confirm Buy",  callback_data="confirm_buy_pending"),
         InlineKeyboardButton("❌ Cancel",       callback_data="main_menu")],
    ])

def kb_confirm_sell(mint):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Confirm Sell", callback_data=f"confirm_sell:{mint}"),
         InlineKeyboardButton("❌ Cancel",       callback_data="main_menu")],
    ])

def kb_positions(positions: dict, is_demo=False):
    rows = []
    for mint, pos in positions.items():
        if is_demo:
            rows.append([
                InlineKeyboardButton(f"💸 Close {pos['symbol']}", callback_data=f"dsell:{mint}"),
                InlineKeyboardButton("✂️ Take Profit Now", callback_data=f"dclose_now:{mint}")
            ])
        else:
            rows.append([InlineKeyboardButton(
                f"💸 Sell {pos['symbol']}",
                callback_data=f"sell_confirm:{mint}"
            )])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="main_menu")])
    return InlineKeyboardMarkup(rows)

# ============================================================
# DASHBOARD MESSAGE
# ============================================================
async def build_dashboard(show_positions=False) -> str:
    s    = state["settings"]
    n    = len(state["ml_features"])
    wins = sum(1 for x in state["trades_history"] if x["net_pnl"] > 0)
    total = len(state["trades_history"])
    sniper = "🟢 ON" if s["auto_snipe"] else "🔴 OFF"
    demo   = "🟢 ON" if s["demo_mode"]  else "🔴 OFF"
    ml_st  = f"✅ Ready ({n} samples)" if ml_ready else f"⏳ Training ({n}/10)"
    pnl_e  = "📈" if state["total_pnl"] >= 0 else "📉"
    hm_st  = "🏠 ON" if s.get("house_money_mode") else "OFF"
    msg = (
        f"🤖 *Solana Meme Coin Bot*\n"
        f"{'─'*28}\n\n"
        f"*💼 Portfolio*\n"
        f"├ Real P&L:      {pnl_e} *${state['total_pnl']:+.2f}*\n"
        f"├ Demo P&L:      📝 ${state['demo_total_pnl']:+.2f}\n"
        f"├ Trades:        {total} ({wins}W / {total-wins}L)\n\n"
        f"*🤖 Bot Status*\n"
        f"├ Auto-Sniper:   {sniper}\n"
        f"├ Demo Mode:     {demo}\n"
        f"├ ML Model:      {ml_st}\n\n"
        f"*⚙️ Active Settings*\n"
        f"├ Take Profit:   {s['take_profit']}x\n"
        f"├ Trailing Stop: {s['trailing_stop']}%\n"
        f"├ Stop Loss:     {s['stop_loss']}x\n"
        f"├ House Money:   {hm_st}\n"
        f"└ Trade Amount:  ${s['trade_amount']}\n"
    )
    if state["positions"]:
        msg += f"\n*📂 Open Positions ({len(state['positions'])})*\n"
        for mint, pos in state["positions"].items():
            p    = await get_token_price(mint)
            mult = p/pos["entry_price"] if p > 0 and pos["entry_price"] > 0 else 0
            pnl  = (mult-1)*pos["amount_usd"] - pos["fees_paid"]
            e    = "🟢" if pnl >= 0 else "🔴"
            hm   = " 🏠" if pos.get("capital_recovered") else ""
            msg += f"{e} {pos['symbol']}{hm} | {mult:.2f}x | ${pnl:+.2f}\n"
    return msg

# ============================================================
# HOUSE MONEY — PARTIAL SELL AT TP
# ============================================================
async def _partial_sell_for_capital_recovery(app, mint, pos, current_price, is_demo):
    """
    Sell the fraction of tokens needed to recover original capital (amount_usd + fees_paid).
    The remaining tokens ride for free with the tiered trailing stop.
    Updates pos in-place. Returns True on success.
    """
    total_tokens = pos.get("token_amount", 0)
    invested     = pos["amount_usd"] + pos["fees_paid"]

    if current_price <= 0 or total_tokens <= 0 or invested <= 0:
        return False

    # tokens_to_sell * current_price ≈ invested
    tokens_to_sell   = invested / current_price
    # Cap at 90% — always leave at least 10% riding even if math says otherwise
    tokens_to_sell   = min(tokens_to_sell, total_tokens * 0.90)
    tokens_remaining = total_tokens - tokens_to_sell
    sell_pct         = tokens_to_sell / total_tokens

    pfx = "📝 DEMO " if is_demo else ""

    if is_demo:
        usdc_recovered = tokens_to_sell * current_price
        sell_fee       = calc_fees(usdc_recovered)["dex_fee"]
        net_recovered  = usdc_recovered - sell_fee
        await _safe_notify(app,
            f"{pfx}🏠 *House Money — Capital Recovered!*\n\n"
            f"*{pos['symbol']}*\n"
            f"{'─'*20}\n"
            f"├ Sold:           {sell_pct:.0%} of position\n"
            f"├ USDC recovered: ${net_recovered:.2f}\n"
            f"├ Original cost:  ${invested:.2f}\n"
            f"├ Tokens riding:  {tokens_remaining:,.0f}\n"
            f"└ 🚀 Pure profit from here — trail active!")
    else:
        result = await execute_sell(mint, int(tokens_to_sell))
        if not result:
            await _safe_notify(app,
                f"⚠️ *House Money partial sell FAILED for {pos['symbol']}*\n"
                f"Falling back to normal trail.")
            return False
        usdc_recovered = result["usdc_received"]
        await _safe_notify(app,
            f"🏠 *House Money — Capital Recovered!*\n\n"
            f"*{pos['symbol']}*\n"
            f"{'─'*20}\n"
            f"├ Sold:           {sell_pct:.0%} of position\n"
            f"├ USDC recovered: ${usdc_recovered:.2f}\n"
            f"├ Original cost:  ${invested:.2f}\n"
            f"├ Tokens riding:  {tokens_remaining:,.0f}\n"
            f"└ 🚀 Pure profit from here — trail active!\n\n"
            f"🔗 [Solscan](https://solscan.io/tx/{result['signature']})")

    # Update position — remaining tokens have zero cost basis
    pos["token_amount"]      = int(tokens_remaining)
    pos["amount_usd"]        = 0.0
    pos["fees_paid"]         = 0.0
    pos["capital_recovered"] = True
    return True

# ============================================================
# CLOSE POSITION
# ============================================================
async def _close_position(app, mint, pos, price, reason, is_demo=False):
    entry    = pos["entry_price"]
    mult     = price / entry if entry > 0 else 1
    hm       = pos.get("capital_recovered", False)

    if hm:
        # House money path: remaining tokens have $0 cost basis — all exit value is profit
        token_amt = pos.get("token_amount", 0)
        gross     = token_amt * price
        sell_fee  = calc_fees(gross)["dex_fee"]
        if is_demo:
            net_pnl  = gross - sell_fee
            usdc_bk  = gross
            sig_link = ""
            proj     = net_pnl * (state["settings"]["trade_amount"] / state["settings"]["demo_trade_amount"])
            proj_txt = f"💡 Real projection: *${proj:+.2f}*\n"
            tx_sig   = None
        else:
            sell_r   = await execute_sell(mint, token_amt)
            usdc_bk  = sell_r["usdc_received"] if sell_r else gross * 0.997
            sell_fee = calc_fees(usdc_bk)["dex_fee"]
            net_pnl  = usdc_bk - sell_fee
            sig_link = f"\n🔗 [Solscan](https://solscan.io/tx/{sell_r['signature']})" if sell_r else ""
            proj_txt = ""; tx_sig = sell_r["signature"] if sell_r else None
    else:
        # Normal path
        sell_fee = calc_fees(pos["amount_usd"] * mult)["dex_fee"]
        if is_demo:
            net_pnl  = (mult-1)*pos["amount_usd"] - pos["fees_paid"] - sell_fee
            usdc_bk  = pos["amount_usd"]*mult
            sig_link = ""
            proj     = net_pnl*(state["settings"]["trade_amount"]/state["settings"]["demo_trade_amount"])
            proj_txt = f"💡 Real projection: *${proj:+.2f}*\n"
            tx_sig   = None
        else:
            sell_r   = await execute_sell(mint, pos.get("token_amount",0))
            usdc_bk  = sell_r["usdc_received"] if sell_r else pos["amount_usd"]*mult*0.997
            net_pnl  = usdc_bk - pos["amount_usd"] - pos["fees_paid"]
            sig_link = f"\n🔗 [Solscan](https://solscan.io/tx/{sell_r['signature']})" if sell_r else ""
            proj_txt = ""; tx_sig = sell_r["signature"] if sell_r else None

    ml_msg = ""
    if pos.get("features"):
        acc    = await record_trade_outcome(pos["features"], net_pnl > 0)
        ml_msg = f"\n🧠 ML updated ({len(state['ml_features'])} samples)"
        if acc: ml_msg += f" | Precision: {acc:.0%}"

    await db_save_trade({
        "symbol": pos["symbol"], "mint": mint, "entry": entry, "exit": price,
        "mult": mult, "net_pnl": net_pnl, "fees_paid": pos["fees_paid"]+sell_fee,
        "reason": reason, "is_demo": is_demo, "tx_sig": tx_sig,
        "features": pos.get("features",[])
    })
    await db_delete_position(mint)

    if is_demo:
        state["demo_total_pnl"] += net_pnl
        state["demo_trades"].append({"symbol": pos["symbol"], "mult": mult,
            "net_pnl": net_pnl, "reason": reason,
            "projected_real": net_pnl*(state["settings"]["trade_amount"]/state["settings"]["demo_trade_amount"])})
        del state["demo_positions"][mint]
    else:
        state["total_pnl"] += net_pnl
        state["trades_history"].append({"symbol": pos["symbol"], "mult": mult,
            "net_pnl": net_pnl, "reason": reason})
        del state["positions"][mint]

    pfx       = "📝 DEMO " if is_demo else ""
    e         = "✅" if net_pnl > 0 else "❌"
    pnl_total = state["demo_total_pnl"] if is_demo else state["total_pnl"]
    lbl       = "Demo" if is_demo else "Real"
    hm_tag    = "\n🏠 _(Capital already recovered — this was pure profit)_" if hm else ""
    await _safe_notify(app,
        f"{e} {pfx}{reason}\n\n"
        f"*{pos['symbol']}* | {mult:.2f}x\n"
        f"{'─'*20}\n"
        f"├ Entry:    ${entry:.6f}\n"
        f"├ Exit:     ${price:.6f}\n"
        f"├ Invested: ${pos['amount_usd']:.2f}\n"
        f"├ Back:     ${usdc_bk:.2f}\n"
        f"├ Fees:     -${pos['fees_paid']+sell_fee:.4f}\n"
        f"└ *Net P&L: ${net_pnl:+.2f}*\n\n"
        f"{proj_txt}💰 {lbl} P&L: ${pnl_total:+.2f}"
        f"{sig_link}{ml_msg}{hm_tag}"
    )

# ============================================================
# AUTO-SNIPER
# ============================================================
async def _fetch_full_pair(session: aiohttp.ClientSession, mint: str):
    try:
        async with session.get(f"{DEXSCREENER_API}{mint}") as r:
            if r.status != 200: return None
            pairs = [p for p in (await r.json()).get("pairs", []) or []
                     if p.get("chainId") == "solana"]
            if not pairs: return None
            return max(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0))
    except Exception as e:
        log_error("_fetch_full_pair", e)
        return None

async def fetch_new_pairs():
    now = time.time()
    expired = [m for m, t in state["seen_pairs"].items() if now - t > SEEN_EXPIRY_SEC]
    for m in expired:
        del state["seen_pairs"][m]
    if expired:
        log.info(f"fetch_new_pairs: expired {len(expired)} stale entries")

    new_mints = []
    connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=600, force_close=False)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=10, connect=4),
        headers={"Accept": "application/json"}
    ) as session:

        async def _discover(url):
            parts = [p for p in url.split("/") if p and p not in ("https:", "api.dexscreener.com")]
            label = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1] if parts else url[-20:]
            for attempt in range(2):
                try:
                    async with session.get(url) as r:
                        if r.status != 200:
                            log.warning(f"_discover [{label}]: HTTP {r.status}")
                            return []
                        data  = await r.json()
                        items = data if isinstance(data, list) else (data.get("pairs") or [])
                        found = []
                        for item in items:
                            if not isinstance(item, dict): continue
                            chain = item.get("chainId") or item.get("chain", "")
                            if chain and chain != "solana": continue
                            mint = (
                                (item.get("baseToken") or item.get("token") or {}).get("address")
                                or item.get("tokenAddress")
                                or item.get("mint")
                                or item.get("address")
                            )
                            if mint and mint not in state["seen_pairs"]:
                                found.append(mint)
                        log.info(f"_discover [{label}]: {len(found)} new mints")
                        return found
                except asyncio.TimeoutError:
                    log.warning(f"_discover [{label}]: timeout attempt {attempt+1}")
                    if attempt == 0: await asyncio.sleep(0.5)
                except Exception as e:
                    log_error(f"_discover [{label}]", e); return []
            return []

        discovered = await asyncio.gather(
            _discover("https://api.dexscreener.com/token-profiles/latest/v1"),
            _discover("https://api.dexscreener.com/token-boosts/latest/v1"),
            _discover("https://api.dexscreener.com/orders/v1/solana"),
            _discover("https://api.dexscreener.com/token-boosts/top/v1"),
        )
        for batch in discovered:
            for mint in batch:
                if mint not in new_mints:
                    new_mints.append(mint)

        if not new_mints:
            log.info("fetch_new_pairs: no new mints"); return []

        hydrated = await asyncio.gather(
            *[_fetch_full_pair(session, mint) for mint in new_mints[:40]]
        )

    pairs = [p for p in hydrated if p is not None]
    log.info(f"fetch_new_pairs: {len(pairs)} pairs ready")
    return pairs

async def evaluate_new_token(pair):
    base          = pair.get("baseToken") or pair.get("token") or {}
    mint          = base.get("address") or pair.get("tokenAddress", "")
    symbol        = base.get("symbol") or pair.get("symbol", "???")
    liq           = float(pair.get("liquidity", {}).get("usd", 0) or 0)
    features      = extract_features(pair)
    ml_score      = predict_score(features)
    safety        = await check_token_safety(mint)
    rugged        = safety.get("rugged", False)
    rc_score      = int(safety.get("score", 0) or 0)
    min_liq       = state["settings"]["min_liquidity"]
    min_score     = state["settings"]["min_score"]
    rc_too_risky  = rc_score > state["settings"].get("min_rugcheck", RUGCHECK_SCORE_MIN)
    passes        = liq > 0 and liq >= min_liq and ml_score >= min_score and not rugged and not rc_too_risky
    if not passes:
        reasons = []
        if liq <= 0:              reasons.append("liq=$0")
        elif liq < min_liq:       reasons.append(f"liq=${liq:,.0f} < min=${min_liq:,.0f}")
        if ml_score < min_score:  reasons.append(f"ml={ml_score:.0%} < min={min_score:.0%}")
        if rugged:                reasons.append("rugged=True")
        if rc_too_risky:          reasons.append(f"rugcheck={rc_score}")
        log.info(f"evaluate: {symbol} REJECTED — {', '.join(reasons)}")
    return {"mint": mint, "symbol": symbol, "liquidity": liq,
            "volume": float(pair.get("volume", {}).get("h24", 0) or 0),
            "market_cap": float(pair.get("marketCap", 0) or 0),
            "price_change": float(pair.get("priceChange", {}).get("h1", 0) or 0),
            "ml_score": ml_score, "safety": safety, "rc_score": rc_score,
            "features": features, "passes_rules": passes}

# ============================================================
# RAYDIUM WEBSOCKET SNIPER
# ============================================================
RAYDIUM_PROGRAM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

async def _handle_snipe(app, mint, pair, info):
    risks = ", ".join([r.get("name","") for r in info["safety"].get("risks",[])]) or "None"
    sym = info["symbol"]; liq = info["liquidity"]; ml = info["ml_score"]; rc = info["rc_score"]
    notif = (
        "⚡ *New Pool Detected*\n" + ("─" * 24) + "\n"
        + "💹 *" + sym + "*\n"
        + "├ Liquidity:  $" + f"{liq:,.0f}" + "\n"
        + "├ ML Score:   " + f"{ml:.0%}" + "\n"
        + "├ RugCheck:   " + str(rc) + " (lower=safer)\n"
        + "└ Risks:      " + risks + "\n"
    )
    if state["settings"]["demo_mode"]:
        price = await get_token_price(mint, pair_data=pair)
        if price <= 0: return
        amt  = state["settings"]["demo_trade_amount"]
        fees = calc_fees(amt)
        pos  = {"symbol": sym, "entry_price": price, "current_price": price,
                "peak_price": price, "amount_usd": amt-fees["total"], "fees_paid": fees["total"],
                "tp_hit": False, "features": info["features"], "ml_score": ml, "auto": True}
        state["demo_positions"][mint] = pos
        await db_save_position(mint, pos, True)
        await _safe_notify(app, notif + "\n📝 *DEMO Auto-bought @ $" + f"{price:.6f}" + "*")
    elif state["settings"]["auto_snipe"]:
        price = await get_token_price(mint, pair_data=pair)
        if price <= 0: return
        await _safe_notify(app, notif + "\n🤖 *Auto-sniping...*")
        result = await execute_buy(mint, state["settings"]["trade_amount"])
        if result:
            amt  = state["settings"]["trade_amount"]
            fees = calc_fees(amt)
            conf = "✅ Confirmed" if result["confirmed"] else "⏳ Pending"
            pos  = {"symbol": sym, "entry_price": price, "current_price": price,
                    "peak_price": price, "amount_usd": amt-fees["total"],
                    "token_amount": result["out_amount"], "fees_paid": fees["total"],
                    "tp_hit": False, "features": info["features"], "auto": True}
            state["positions"][mint] = pos
            await db_save_position(mint, pos, False)
            await _safe_notify(app,
                "✅ *Sniped " + sym + "!*\n"
                + "Entry: $" + f"{price:.6f}" + "\n" + conf + "\n"
                + "🔗 [Solscan](https://solscan.io/tx/" + result["signature"] + ")")
        else:
            await _safe_notify(app, "❌ Snipe failed for " + sym)

async def raydium_ws_sniper(app):
    try:
        import websockets
    except ImportError:
        log.warning("websockets not installed — run: pip install websockets")
        return
    wss_url = HELIUS_RPC
    if wss_url.startswith("https://"): wss_url = wss_url.replace("https://", "wss://")
    elif wss_url.startswith("http://"): wss_url = wss_url.replace("http://", "wss://")
    log.info("Raydium WS sniper connecting...")
    backoff = 2
    while True:
        try:
            async with websockets.connect(wss_url, ping_interval=20, ping_timeout=30) as ws:
                backoff = 2
                await ws.send(json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "logsSubscribe",
                    "params": [{"mentions": [RAYDIUM_PROGRAM]}, {"commitment": "confirmed"}]
                }))
                log.info("Raydium WS: subscribed")
                async for raw in ws:
                    try:
                        msg  = json.loads(raw)
                        val  = msg.get("params", {}).get("result", {}).get("value", {})
                        logs = val.get("logs", [])
                        if not any("initialize" in l.lower() for l in logs): continue
                        await asyncio.sleep(4)
                        for line in logs:
                            for part in line.split():
                                if len(part) in (43, 44) and part not in state["seen_pairs"]:
                                    state["seen_pairs"][part] = time.time()
                                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                                        pair_data = await _fetch_full_pair(session, part)
                                        if pair_data:
                                            info = await evaluate_new_token(pair_data)
                                            if info["passes_rules"]:
                                                await _handle_snipe(app, part, pair_data, info)
                    except Exception as e:
                        log_error("raydium_ws/msg", e)
        except Exception as e:
            log_error("raydium_ws/connect", e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

async def auto_sniper_loop(app):
    log.info("Auto-sniper started.")
    consecutive_errors = 0; loop_count = 0
    while True:
        try:
            if not state["settings"]["auto_snipe"] and not state["settings"]["demo_mode"]:
                await asyncio.sleep(2); continue
            loop_count += 1
            new_pairs = await fetch_new_pairs()
            consecutive_errors = 0
            if loop_count % 12 == 0:
                log.info(f"Sniper loop #{loop_count}: {len(new_pairs)} new pairs")

            new_tokens = []
            for pair in new_pairs:
                base = pair.get("baseToken") or pair.get("token") or {}
                mint = base.get("address") or pair.get("tokenAddress", "")
                if not mint or mint in state["seen_pairs"]: continue
                state["seen_pairs"][mint] = time.time()
                new_tokens.append((mint, pair))

            if not new_tokens:
                await asyncio.sleep(2); continue

            infos = await asyncio.gather(
                *[evaluate_new_token(pair) for _, pair in new_tokens],
                return_exceptions=True
            )

            for (mint, pair), info in zip(new_tokens, infos):
                if isinstance(info, Exception):
                    log.error(f"evaluate error {mint[:8]}: {info}"); continue
                if not info["passes_rules"]: continue
                if mint in state["positions"] or mint in state["demo_positions"]: continue

                risks = ", ".join([r.get("name","") for r in info["safety"].get("risks",[])]) or "None"
                notif = (
                    f"🔍 *New Token Detected*\n{'─'*24}\n"
                    f"🪙 *{info['symbol']}*\n"
                    f"├ Liquidity:  ${info['liquidity']:,.0f}\n"
                    f"├ Volume:     ${info['volume']:,.0f}\n"
                    f"├ Market Cap: ${info['market_cap']:,.0f}\n"
                    f"├ Price 1h:   {info['price_change']:+.1f}%\n"
                    f"├ ML Score:   {info['ml_score']:.0%} confidence\n"
                    f"├ RugCheck:   {info['rc_score']} (lower=safer)\n"
                    f"└ Risks:      {risks}\n"
                )
                if state["settings"]["demo_mode"]:
                    price = await get_token_price(mint, pair_data=pair)
                    if price <= 0: continue
                    amt  = state["settings"]["demo_trade_amount"]
                    fees = calc_fees(amt)
                    pos  = {"symbol": info["symbol"], "entry_price": price,
                            "current_price": price, "peak_price": price,
                            "amount_usd": amt-fees["total"], "fees_paid": fees["total"],
                            "tp_hit": False, "features": info["features"],
                            "ml_score": info["ml_score"], "auto": True}
                    state["demo_positions"][mint] = pos
                    await db_save_position(mint, pos, True)
                    await _safe_notify(app, notif + f"\n📝 *DEMO Auto-bought @ ${price:.6f}*")
                elif state["settings"]["auto_snipe"]:
                    if info["ml_score"] < state["settings"]["min_score"]:
                        await _safe_notify(app, notif + f"\n⚠️ *Skipped — ML score too low ({info['ml_score']:.0%})*")
                        continue
                    price = await get_token_price(mint, pair_data=pair)
                    if price <= 0: continue
                    amt  = state["settings"]["trade_amount"]
                    fees = calc_fees(amt)
                    await _safe_notify(app, notif + "\n🤖 *Auto-sniping...*")
                    result = await execute_buy(mint, amt)
                    if result:
                        pos = {"symbol": info["symbol"], "entry_price": price,
                               "current_price": price, "peak_price": price,
                               "amount_usd": amt-fees["total"], "token_amount": result["out_amount"],
                               "fees_paid": fees["total"], "tp_hit": False,
                               "features": info["features"], "auto": True}
                        state["positions"][mint] = pos
                        await db_save_position(mint, pos, False)
                        await _safe_notify(app,
                            f"✅ *Sniped {info['symbol']}!*\n"
                            f"Entry: ${price:.6f}\n"
                            f"{'✅ Confirmed' if result['confirmed'] else '⏳ Pending'}\n"
                            f"🔗 [Solscan](https://solscan.io/tx/{result['signature']})")
                    else:
                        await _safe_notify(app, f"❌ Snipe failed for {info['symbol']}")
        except Exception as e:
            consecutive_errors += 1
            log_error("auto_sniper_loop", e)
            if consecutive_errors >= 5:
                await notify_error(app, "auto_sniper_loop (5x)", e)
                consecutive_errors = 0
        await asyncio.sleep(2)

# ============================================================
# POSITION MONITOR
# ============================================================
def _tiered_trail(peak_mult: float) -> float:
    s = state["settings"]
    if peak_mult >= 50: return s.get("trail_50x", 1.5) / 100
    if peak_mult >= 20: return s.get("trail_20x", 2.0) / 100
    if peak_mult >= 10: return s.get("trail_10x", 3.0) / 100
    if peak_mult >= 5:  return s.get("trail_5x",  4.0) / 100
    return 0.05

async def monitor_positions(app):
    log.info("Position monitor started.")
    price_history: dict = {}
    while True:
        await asyncio.sleep(0.5)
        for is_demo, pool in [(False, state["positions"]), (True, state["demo_positions"])]:
            mints = list(pool.keys())
            if not mints: continue

            price_results = await asyncio.gather(
                *[get_token_price(m) for m in mints],
                return_exceptions=True
            )
            price_map = {
                m: (p if isinstance(p, float) and p > 0 else 0.0)
                for m, p in zip(mints, price_results)
            }

            for mint, pos in list(pool.items()):
                try:
                    price = price_map.get(mint, 0.0)
                    if price <= 0: continue
                    pos["current_price"] = price
                    entry = pos["entry_price"]; mult = price / entry
                    tp    = state["settings"]["take_profit"]
                    sl    = state["settings"]["stop_loss"]

                    if mint not in price_history: price_history[mint] = []
                    price_history[mint].append((time.time(), price))
                    if len(price_history[mint]) > 20: price_history[mint].pop(0)

                    if price > pos.get("peak_price", entry):
                        pos["peak_price"] = price

                    peak_mult        = pos["peak_price"] / entry if entry > 0 else 1.0
                    post_tp_trail    = _tiered_trail(peak_mult)
                    trailing_trigger = pos["peak_price"] * (1 - post_tp_trail)

                    reason = None

                    if mult <= sl:
                        reason = f"🔴 Stop Loss at {mult:.2f}x"

                    elif pos.get("tp_hit"):
                        hist = price_history.get(mint, [])
                        momentum_exit = False
                        if len(hist) >= 5:
                            old_price = hist[-5][1]
                            drop_pct  = (old_price - price) / old_price if old_price > 0 else 0
                            if drop_pct > 0.03:
                                momentum_exit = True
                                log.info(f"Momentum exit: {pos['symbol']} -{drop_pct:.1%} in ~2.5s")

                        if price <= trailing_trigger:
                            trail_pct = int(post_tp_trail * 100)
                            reason = f"🟡 Trailing Stop at {mult:.2f}x ({trail_pct}% trail from {peak_mult:.1f}x peak)"
                        elif momentum_exit:
                            reason = f"⚡ Momentum Exit at {mult:.2f}x"

                    elif mult >= tp and not pos.get("tp_hit"):
                        pos["tp_hit"] = True
                        trail_pct = int(_tiered_trail(mult) * 100)
                        pfx = "📝 DEMO " if is_demo else ""

                        if state["settings"].get("house_money_mode") and not pos.get("capital_recovered"):
                            ok = await _partial_sell_for_capital_recovery(app, mint, pos, price, is_demo)
                            if ok:
                                await db_save_position(mint, pos, is_demo)
                            else:
                                await db_save_position(mint, pos, is_demo)
                                await _safe_notify(app,
                                    f"{pfx}🎯 *TP Hit — {pos['symbol']}* {mult:.2f}x\n"
                                    f"Trailing stop now active! 🚀\n"
                                    f"_{trail_pct}% tight trail engaged (tiered)_")
                        else:
                            await db_save_position(mint, pos, is_demo)
                            await _safe_notify(app,
                                f"{pfx}🎯 *TP Hit — {pos['symbol']}* {mult:.2f}x\n"
                                f"Trailing stop now active! 🚀\n"
                                f"_{trail_pct}% tight trail engaged (tiered)_")

                    if reason:
                        price_history.pop(mint, None)
                        await _close_position(app, mint, pos, price, reason, is_demo)
                except Exception as e:
                    log_error(f"monitor/{mint[:8]}", e)

# ============================================================
# CALLBACK HANDLER
# ============================================================
@auth
async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q    = update.callback_query
    data = q.data
    await q.answer()
    try:
        result = await _button_handler_inner(update, ctx, q, data)
        return result
    except Exception as e:
        log.error(f"button_handler crash [{data}]: {e}\n{traceback.format_exc()}")
        try:
            msg = await build_dashboard()
            await q.edit_message_text(
                f"⚠️ Something went wrong. Returning to dashboard.\n\n{msg}",
                parse_mode="Markdown", reply_markup=kb_main())
        except Exception:
            try:
                await update.effective_chat.send_message(
                    "⚠️ An error occurred. Use /start to return to the dashboard.",
                    reply_markup=kb_main())
            except Exception:
                pass

async def _button_handler_inner(update: Update, ctx: ContextTypes.DEFAULT_TYPE, q, data):
    # ── Main Menu ──
    if data == "main_menu":
        msg = await build_dashboard()
        await q.edit_message_text(msg, parse_mode="Markdown", reply_markup=kb_main())

    # ── Positions ──
    elif data == "positions":
        if not state["positions"]:
            await q.edit_message_text("📭 *No open positions.*", parse_mode="Markdown",
                reply_markup=kb_main())
        else:
            lines = []
            for mint, pos in state["positions"].items():
                p    = await get_token_price(mint)
                mult = p/pos["entry_price"] if p > 0 else 0
                pnl  = (mult-1)*pos["amount_usd"] - pos["fees_paid"]
                e    = "🟢" if pnl >= 0 else "🔴"
                hm   = " 🏠" if pos.get("capital_recovered") else ""
                lines.append(f"{e} *{pos['symbol']}*{hm} | {mult:.2f}x | ${pnl:+.2f}")
            await q.edit_message_text(
                "*📂 Open Positions*\n\n" + "\n".join(lines),
                parse_mode="Markdown",
                reply_markup=kb_positions(state["positions"]))

    # ── P&L ──
    elif data == "pnl":
        t    = state["trades_history"]
        wins = sum(1 for x in t if x["net_pnl"] > 0)
        e    = "📈" if state["total_pnl"] >= 0 else "📉"
        await q.edit_message_text(
            f"{e} *P&L Summary*\n\n"
            f"├ Real P&L:     *${state['total_pnl']:+.2f}*\n"
            f"├ Demo P&L:     ${state['demo_total_pnl']:+.2f}\n"
            f"├ Total Trades: {len(t)}\n"
            f"├ Wins:         {wins}\n"
            f"└ Losses:       {len(t)-wins}",
            parse_mode="Markdown", reply_markup=kb_main())

    # ── Toggle Sniper ──
    elif data == "toggle_sniper":
        state["settings"]["auto_snipe"] = not state["settings"]["auto_snipe"]
        await db_save_settings()
        status = "🟢 ON" if state["settings"]["auto_snipe"] else "🔴 OFF"
        msg = await build_dashboard()
        await q.edit_message_text(msg + f"\n\n*Auto-Sniper turned {status}*",
            parse_mode="Markdown", reply_markup=kb_main())

    # ── Toggle Demo ──
    elif data == "toggle_demo":
        state["settings"]["demo_mode"] = not state["settings"]["demo_mode"]
        await db_save_settings()
        status = "🟢 ON" if state["settings"]["demo_mode"] else "🔴 OFF"
        msg = await build_dashboard()
        await q.edit_message_text(msg + f"\n\n*Demo Mode turned {status}*",
            parse_mode="Markdown", reply_markup=kb_main())

    # ── Toggle House Money ──
    elif data == "toggle_house_money":
        state["settings"]["house_money_mode"] = not state["settings"].get("house_money_mode", True)
        await db_save_settings()
        status = "🟢 ON" if state["settings"]["house_money_mode"] else "🔴 OFF"
        await q.edit_message_text(
            f"⚙️ *Settings*\n\n🏠 House Money Mode turned *{status}*\n\n"
            f"_When ON: at TP hit, the bot sells enough tokens to recover your\n"
            f"initial capital. The rest rides free with the trailing stop._",
            parse_mode="Markdown", reply_markup=kb_settings())

    # ── Demo Menu ──
    elif data == "demo_menu":
        await q.edit_message_text("📝 *Demo Trading*", parse_mode="Markdown", reply_markup=kb_demo())

    elif data == "demostatus":
        if not state["demo_positions"]:
            await q.edit_message_text("📭 *No demo positions.*", parse_mode="Markdown", reply_markup=kb_main())
        else:
            lines = []
            for mint, pos in state["demo_positions"].items():
                p    = await get_token_price(mint)
                mult = p/pos["entry_price"] if p > 0 else 0
                pnl  = (mult-1)*pos["amount_usd"] - pos["fees_paid"]
                proj = pnl*(state["settings"]["trade_amount"]/state["settings"]["demo_trade_amount"])
                e    = "🟢" if pnl >= 0 else "🔴"
                tp_tag = " 🎯" if pos.get("tp_hit") else ""
                hm_tag = " 🏠" if pos.get("capital_recovered") else ""
                lines.append(f"{e} *{pos['symbol']}*{tp_tag}{hm_tag} | {mult:.2f}x | Sim: ${pnl:+.2f} | Real: *${proj:+.2f}*")
            await q.edit_message_text(
                f"📝 *Demo Positions*\n\n" + "\n".join(lines) +
                f"\n\n📊 Total Demo P&L: ${state['demo_total_pnl']:+.2f}",
                parse_mode="Markdown",
                reply_markup=kb_positions(state["demo_positions"], is_demo=True))

    elif data == "demohistory":
        if not state["demo_trades"]:
            await q.edit_message_text("📭 *No demo history yet.*", parse_mode="Markdown", reply_markup=kb_main())
        else:
            lines = [
                f"{'✅' if t['net_pnl']>0 else '❌'} {t['symbol']} | {t['mult']:.2f}x | "
                f"${t['net_pnl']:+.2f} → Real: ${t.get('projected_real',0):+.2f}"
                for t in state["demo_trades"][-10:]
            ]
            await q.edit_message_text(
                f"📝 *Demo History*\n\n" + "\n".join(lines) +
                f"\n\n📊 Total: ${state['demo_total_pnl']:+.2f}",
                parse_mode="Markdown", reply_markup=kb_main())

    # ── ML Stats ──
    elif data == "mlstats":
        from sklearn.metrics import precision_score, recall_score
        n = len(state["ml_features"])
        if not ml_ready or n == 0:
            await q.edit_message_text(
                f"🧠 *ML Model*\n\nNot ready yet.\n{n}/10 samples collected.",
                parse_mode="Markdown", reply_markup=kb_main())
        else:
            wins  = int(sum(state["ml_labels"]))
            Xs    = ml_scaler.transform(state["ml_features"])
            preds = ml_model.predict(Xs)
            prec  = precision_score(state["ml_labels"], preds, zero_division=0)
            rec   = recall_score(state["ml_labels"], preds, zero_division=0)
            nms   = ["Liquidity","Vol24h","Δ1h","Δ6h","Δ24h","Buys1h","Sells1h","B/S","Age","Mcap","Vol5m","Buys5m","Sells5m","B/S5m","Liq/Mcap","Vol/Liq"]
            top   = sorted(zip(nms, ml_model.feature_importances_), key=lambda x: -x[1])[:5]
            await q.edit_message_text(
                f"🧠 *ML Model Stats*\n\n"
                f"├ Samples:   {n}\n├ Wins:      {wins} / {n}\n"
                f"├ Win rate:  {wins/n:.0%}\n├ Precision: {prec:.0%}\n└ Recall:    {rec:.0%}\n\n"
                f"*Top Features:*\n" + "\n".join([f"  {nm}: {v:.1%}" for nm,v in top]),
                parse_mode="Markdown", reply_markup=kb_main())

    # ── Settings Menu ──
    elif data == "settings_menu":
        await q.edit_message_text("⚙️ *Settings*\n\nTap any setting to change it:",
            parse_mode="Markdown", reply_markup=kb_settings())

    elif data == "set_tp":
        ctx.user_data["setting"] = "take_profit"
        await q.edit_message_text(
            f"🎯 *Take Profit*\nCurrent: {state['settings']['take_profit']}x\n\nSend a new multiplier:",
            parse_mode="Markdown", reply_markup=kb_main())
        return WAITING_SET_TP

    elif data == "set_trail":
        ctx.user_data["setting"] = "trailing_stop"
        await q.edit_message_text(
            f"📉 *Trailing Stop*\nCurrent: {state['settings']['trailing_stop']}%\n\nSend a new percentage:",
            parse_mode="Markdown", reply_markup=kb_main())
        return WAITING_SET_TRAIL

    elif data == "set_stop":
        ctx.user_data["setting"] = "stop_loss"
        await q.edit_message_text(
            f"🛑 *Stop Loss*\nCurrent: {state['settings']['stop_loss']}x\n\nSend a new multiplier (e.g. `0.5`):",
            parse_mode="Markdown", reply_markup=kb_main())
        return WAITING_SET_STOP

    elif data == "set_amount":
        ctx.user_data["setting"] = "trade_amount"
        await q.edit_message_text(
            f"💵 *Trade Amount*\nCurrent: ${state['settings']['trade_amount']}\n\nSend a new amount in USD:",
            parse_mode="Markdown", reply_markup=kb_main())
        return WAITING_SET_AMOUNT

    elif data == "set_slip":
        ctx.user_data["setting"] = "slippage_bps"
        await q.edit_message_text(
            f"⚡ *Slippage*\nCurrent: {state['settings']['slippage_bps']} bps\n\nSend new slippage in bps (100 = 1%):",
            parse_mode="Markdown", reply_markup=kb_main())
        return WAITING_SET_SLIP

    elif data == "set_score":
        ctx.user_data["setting"] = "min_score"
        await q.edit_message_text(
            f"🧠 *Min ML Score*\nCurrent: {state['settings']['min_score']:.0%}\n\nSend a value 0–1 (e.g. `0.7`):",
            parse_mode="Markdown", reply_markup=kb_main())
        return WAITING_SET_SCORE

    elif data == "set_liq":
        ctx.user_data["setting"] = "min_liquidity"
        await q.edit_message_text(
            f"💧 *Min Liquidity*\nCurrent: ${state['settings']['min_liquidity']:,}\n\nSend a new value in USD:",
            parse_mode="Markdown", reply_markup=kb_main())
        return WAITING_SET_LIQ

    elif data == "set_rugcheck":
        ctx.user_data["setting"] = "min_rugcheck"
        await q.edit_message_text(
            f"🛡️ *Max RugCheck Score*\nCurrent: {state['settings']['min_rugcheck']}\n\n"
            f"Lower = safer. Recommended: 500–1000\n\nSend a new value:",
            parse_mode="Markdown", reply_markup=kb_main())
        return WAITING_SET_RUGCHECK

    # ── Tiered Trail Menu ──
    elif data == "tiered_trail_menu":
        s = state["settings"]
        await q.edit_message_text(
            f"📐 *Tiered Trailing Stop*\n\n"
            f"Trail tightens as price pumps higher.\n"
            f"Smaller % = locks in more profit.\n\n"
            f"Current config:\n"
            f"├ Peak ≥ 5x  → {s.get('trail_5x',  4.0)}% trail\n"
            f"├ Peak ≥ 10x → {s.get('trail_10x', 3.0)}% trail\n"
            f"├ Peak ≥ 20x → {s.get('trail_20x', 2.0)}% trail\n"
            f"└ Peak ≥ 50x → {s.get('trail_50x', 1.5)}% trail\n\n"
            f"_(Below 5x, default 5% trail applies)_\n\nTap a tier to edit:",
            parse_mode="Markdown", reply_markup=kb_tiered_trail())

    elif data == "set_trail_5x":
        ctx.user_data["setting"] = "trail_5x"
        await q.edit_message_text(
            f"🟢 *Trail % at ≥5x peak*\nCurrent: {state['settings'].get('trail_5x', 4.0)}%\n\n"
            f"Send a new percentage (e.g. `4`):\n_Recommended: 3–6%_",
            parse_mode="Markdown", reply_markup=kb_back())
        return WAITING_SET_TRAIL_5X

    elif data == "set_trail_10x":
        ctx.user_data["setting"] = "trail_10x"
        await q.edit_message_text(
            f"🟡 *Trail % at ≥10x peak*\nCurrent: {state['settings'].get('trail_10x', 3.0)}%\n\n"
            f"Send a new percentage (e.g. `3`):\n_Recommended: 2–5%_",
            parse_mode="Markdown", reply_markup=kb_back())
        return WAITING_SET_TRAIL_10X

    elif data == "set_trail_20x":
        ctx.user_data["setting"] = "trail_20x"
        await q.edit_message_text(
            f"🟠 *Trail % at ≥20x peak*\nCurrent: {state['settings'].get('trail_20x', 2.0)}%\n\n"
            f"Send a new percentage (e.g. `2`):\n_Recommended: 1.5–3%_",
            parse_mode="Markdown", reply_markup=kb_back())
        return WAITING_SET_TRAIL_20X

    elif data == "set_trail_50x":
        ctx.user_data["setting"] = "trail_50x"
        await q.edit_message_text(
            f"🔴 *Trail % at ≥50x peak*\nCurrent: {state['settings'].get('trail_50x', 1.5)}%\n\n"
            f"Send a new percentage (e.g. `1.5`):\n_Recommended: 1–2%_",
            parse_mode="Markdown", reply_markup=kb_back())
        return WAITING_SET_TRAIL_50X

    # ── Buy prompt ──
    elif data == "buy_prompt":
        await q.edit_message_text("🛒 *Buy Token*\n\nSend the token mint address:",
            parse_mode="Markdown", reply_markup=kb_main())
        ctx.user_data["action"] = "buy"
        return WAITING_BUY_MINT

    # ── Sell prompt ──
    elif data == "sell_prompt":
        if not state["positions"]:
            await q.edit_message_text("📭 *No open positions to sell.",
                parse_mode="Markdown", reply_markup=kb_main())
        else:
            await q.edit_message_text("💸 *Sell Token*\n\nSelect a position:",
                parse_mode="Markdown", reply_markup=kb_positions(state["positions"]))

    # ── Confirm sell from positions list ──
    elif data.startswith("sell_confirm:"):
        mint = data.split(":")[1]
        if mint not in state["positions"]:
            await q.edit_message_text("❌ Position not found.", reply_markup=kb_main())
            return
        pos = state["positions"][mint]
        await q.edit_message_text(
            f"💸 *Confirm Sell*\n\nToken: *{pos['symbol']}*\n"
            f"Entry: ${pos['entry_price']:.6f}\nAmount: ${pos['amount_usd']:.2f}\n\nAre you sure?",
            parse_mode="Markdown", reply_markup=kb_confirm_sell(mint))

    elif data.startswith("confirm_sell:"):
        mint = data.split(":")[1]
        if mint not in state["positions"]:
            await q.edit_message_text("❌ Position not found.", reply_markup=kb_main())
            return
        pos = state["positions"][mint]
        await q.edit_message_text(f"⏳ Selling {pos['symbol']}...", parse_mode="Markdown")
        price = await get_token_price(mint) or pos["entry_price"]
        await _close_position(app, mint, pos, price, "Manual sell", False)
        msg = await build_dashboard()
        await q.edit_message_text(msg, parse_mode="Markdown", reply_markup=kb_main())

    # ── Demo: manual close ──
    elif data.startswith("dsell:"):
        mint = data.split(":")[1]
        if mint not in state["demo_positions"]:
            await q.edit_message_text("❌ Demo position not found.", reply_markup=kb_main())
            return
        pos   = state["demo_positions"][mint]
        price = await get_token_price(mint)
        if price <= 0:
            await q.edit_message_text("❌ Price unavailable, try again.", reply_markup=kb_main())
            return
        mult = price / pos["entry_price"]
        await q.edit_message_text(
            f"💸 *Close Demo Position*\n\nToken: *{pos['symbol']}*\n"
            f"Entry: ${pos['entry_price']:.6f}\nCurrent: ${price:.6f} ({mult:.2f}x)\n\nClose at current price?",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Confirm Close", callback_data=f"dclose_confirm:{mint}"),
                 InlineKeyboardButton("❌ Cancel", callback_data="demostatus")]
            ]))

    elif data.startswith("dclose_confirm:"):
        mint = data.split(":")[1]
        if mint not in state["demo_positions"]:
            await q.edit_message_text("❌ Demo position not found.", reply_markup=kb_main())
            return
        pos   = state["demo_positions"][mint]
        price = await get_token_price(mint) or pos["entry_price"]
        await q.edit_message_text(f"⏳ Closing {pos['symbol']}...", parse_mode="Markdown")
        await _close_position(app, mint, pos, price, "✂️ Manual Close", True)
        msg = await build_dashboard()
        await q.edit_message_text(msg, parse_mode="Markdown", reply_markup=kb_main())

    # ── Demo: take profit now ──
    elif data.startswith("dclose_now:"):
        mint = data.split(":")[1]
        if mint not in state["demo_positions"]:
            await q.edit_message_text("❌ Demo position not found.", reply_markup=kb_main())
            return
        pos   = state["demo_positions"][mint]
        price = await get_token_price(mint) or pos["entry_price"]
        await q.edit_message_text(f"✂️ Taking profit on {pos['symbol']}...", parse_mode="Markdown")
        await _close_position(app, mint, pos, price, "✂️ Manual Take Profit", True)
        msg = await build_dashboard()
        await q.edit_message_text(msg, parse_mode="Markdown", reply_markup=kb_main())

    # ── Confirm buy (legacy inline) ──
    elif data.startswith("confirm_buy:"):
        parts  = data.split(":")
        mint, symbol = parts[1], parts[2]
        amt    = state["settings"]["trade_amount"]
        fees   = calc_fees(amt)
        await q.edit_message_text(f"⏳ Buying {symbol}...", parse_mode="Markdown")
        price  = await get_token_price(mint)
        if price <= 0:
            await q.edit_message_text("❌ Price unavailable. Try again.", reply_markup=kb_main())
            return
        result = await execute_buy(mint, amt)
        if not result:
            await q.edit_message_text(
                "❌ Buy failed.\n• Check USDC balance\n• Try increasing slippage in ⚙️ Settings",
                reply_markup=kb_main())
            return
        td  = await get_token_data(mint)
        pos = {"symbol": symbol, "entry_price": price, "current_price": price,
               "peak_price": price, "amount_usd": amt-fees["total"],
               "token_amount": result["out_amount"], "fees_paid": fees["total"],
               "tp_hit": False, "features": extract_features(td) if td else [0.0]*16, "auto": False}
        state["positions"][mint] = pos
        await db_save_position(mint, pos, False)
        msg = await build_dashboard()
        await q.edit_message_text(
            msg + f"\n\n✅ *Bought {symbol} @ ${price:.6f}*\n"
            f"🔗 [Solscan](https://solscan.io/tx/{result['signature']})",
            parse_mode="Markdown", reply_markup=kb_main(), disable_web_page_preview=True)

    elif data == "confirm_buy_pending":
        await handle_confirm_buy(update, ctx)

    # ── Health ──
    elif data == "health":
        a  = state["api_stats"]
        pt = a["price_ok"]+a["price_fail"]; qt = a["quote_ok"]+a["quote_fail"]
        st = a["swap_ok"]+a["swap_fail"]
        err_txt = "\n".join([f"  [{e['time']}] {e['context']}: {e['error']}"
                             for e in state["errors"][-3:]]) or "  None ✅"
        await q.edit_message_text(
            f"🏥 *Health*\n\n*API Rates:*\n"
            f"├ Price: {a['price_ok']}/{pt} ({a['price_ok']/max(pt,1):.0%})\n"
            f"├ Quote: {a['quote_ok']}/{qt} ({a['quote_ok']/max(qt,1):.0%})\n"
            f"├ Swap:  {a['swap_ok']}/{st} ({a['swap_ok']/max(st,1):.0%})\n"
            f"├ Confirm: {a['confirm_ok']} ok | {a['confirm_timeout']} timeout\n\n"
            f"*State:*\n"
            f"├ Positions: {len(state['positions'])} real / {len(state['demo_positions'])} demo\n"
            f"├ ML: {len(state['ml_features'])} samples | {'Ready' if ml_ready else 'Training'}\n"
            f"└ DB: {'Connected' if db_pool else 'Disconnected'}\n\n"
            f"*Recent Errors:*\n{err_txt}",
            parse_mode="Markdown", reply_markup=kb_main())

    # ── History ──
    elif data == "history":
        if not state["trades_history"]:
            await q.edit_message_text("📭 *No history yet.*", parse_mode="Markdown", reply_markup=kb_main())
        else:
            lines = [
                f"{'✅' if t['net_pnl']>0 else '❌'} {t['symbol']} | {t['mult']:.2f}x | ${t['net_pnl']:+.2f}"
                for t in state["trades_history"][-10:]
            ]
            await q.edit_message_text(
                "*📜 Last 10 Trades*\n\n" + "\n".join(lines),
                parse_mode="Markdown", reply_markup=kb_main())

# ============================================================
# MESSAGE HANDLERS
# ============================================================
async def handle_setting_input(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    key = ctx.user_data.get("setting")
    txt = update.message.text.strip()
    try:
        val = float(txt)
        state["settings"][key] = int(val) if key == "slippage_bps" else val
        await db_save_settings()
        label = key.replace("_", " ").title()
        await update.message.reply_text(f"✅ *{label}* updated to `{txt}`",
            parse_mode="Markdown", reply_markup=kb_main())
    except ValueError:
        await update.message.reply_text("❌ Invalid value. Please send a number.", reply_markup=kb_main())
    return ConversationHandler.END

async def handle_buy_mint(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    mint = update.message.text.strip()
    ctx.user_data["buy_mint"] = mint
    await update.message.reply_text(
        "Now send the *token symbol* (e.g. `PEPE`, `BONK`):", parse_mode="Markdown")
    return WAITING_BUY_SYMBOL

async def handle_buy_symbol(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    symbol  = update.message.text.strip().upper()
    mint    = ctx.user_data.get("buy_mint", "")
    is_demo = state["settings"]["demo_mode"]
    amt     = state["settings"].get("demo_trade_amount", state["settings"]["trade_amount"]) if is_demo else state["settings"]["trade_amount"]
    fees    = calc_fees(amt)
    try:
        price = await get_token_price(mint)
    except Exception as e:
        log_error("handle_buy_symbol/price", e)
        price = 0.0
    ctx.user_data["pending_buy_mint"]   = mint
    ctx.user_data["pending_buy_symbol"] = symbol
    demo_label = "📝 *[DEMO MODE — No real USDC will be spent]*\n\n" if is_demo else ""
    price_str  = f"${price:.6f}" if price > 0 else "⚠️ Fetching at confirm…"
    await update.message.reply_text(
        f"{demo_label}🛒 *Confirm Buy*\n\n"
        f"Token:    *{symbol}*\nMint:     `{mint[:16]}...`\n"
        f"Price:    {price_str}\nInvest:   ${amt:.2f}\n"
        f"Fees:     -${fees['total']}\nPosition: ${amt-fees['total']:.2f}\n\nTap confirm:",
        parse_mode="Markdown", reply_markup=kb_confirm_buy(mint, symbol))
    return WAITING_CONFIRM_BUY

async def handle_confirm_buy(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q    = update.callback_query
    data = q.data
    try: await q.answer()
    except Exception: pass
    if data != "confirm_buy_pending":
        msg = await build_dashboard()
        await q.edit_message_text(msg, parse_mode="Markdown", reply_markup=kb_main())
        return ConversationHandler.END
    mint    = ctx.user_data.get("pending_buy_mint", "")
    symbol  = ctx.user_data.get("pending_buy_symbol", "?")
    if not mint:
        await q.edit_message_text("❌ Session expired. Tap Buy Token again.", reply_markup=kb_main())
        return ConversationHandler.END
    is_demo = state["settings"]["demo_mode"]
    if is_demo:
        amt   = state["settings"].get("demo_trade_amount", state["settings"]["trade_amount"])
        fees  = calc_fees(amt)
        await q.edit_message_text(f"⏳ *[DEMO]* Simulating buy of {symbol}...", parse_mode="Markdown")
        price = await get_token_price(mint)
        if price <= 0:
            await q.edit_message_text("❌ Price unavailable. Try again.", reply_markup=kb_main())
            return ConversationHandler.END
        td  = await get_token_data(mint)
        pos = {"symbol": symbol, "entry_price": price, "current_price": price,
               "peak_price": price, "amount_usd": amt - fees["total"],
               "token_amount": (amt - fees["total"]) / price if price else 0,
               "fees_paid": fees["total"],
               "tp_hit": False, "features": extract_features(td) if td else [0.0]*16, "auto": False}
        state["demo_positions"][mint] = pos
        await db_save_position(mint, pos, True)
        msg = await build_dashboard()
        await q.edit_message_text(
            msg + f"\n\n📝 *[DEMO] Simulated buy of {symbol} @ ${price:.6f}*",
            parse_mode="Markdown", reply_markup=kb_main())
        return ConversationHandler.END
    amt   = state["settings"]["trade_amount"]
    fees  = calc_fees(amt)
    await q.edit_message_text(f"⏳ Buying {symbol}...", parse_mode="Markdown")
    price = await get_token_price(mint)
    if price <= 0:
        await q.edit_message_text("❌ Price unavailable. Try again.", reply_markup=kb_main())
        return ConversationHandler.END
    result = await execute_buy(mint, amt)
    if not result:
        await q.edit_message_text(
            "❌ *Buy failed.*\n• Check USDC balance\n• Try increasing slippage in ⚙️ Settings",
            parse_mode="Markdown", reply_markup=kb_main())
        return ConversationHandler.END
    td  = await get_token_data(mint)
    pos = {"symbol": symbol, "entry_price": price, "current_price": price,
           "peak_price": price, "amount_usd": amt - fees["total"],
           "token_amount": result["out_amount"], "fees_paid": fees["total"],
           "tp_hit": False, "features": extract_features(td) if td else [0.0]*16, "auto": False}
    state["positions"][mint] = pos
    await db_save_position(mint, pos, False)
    msg = await build_dashboard()
    await q.edit_message_text(
        msg + f"\n\n✅ *Bought {symbol} @ ${price:.6f}*\n"
        f"🔗 [Solscan](https://solscan.io/tx/{result['signature']})",
        parse_mode="Markdown", reply_markup=kb_main(), disable_web_page_preview=True)
    return ConversationHandler.END

# ============================================================
# /start COMMAND
# ============================================================
@auth
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await build_dashboard()
    await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=kb_main())

# ============================================================
# MAIN
# ============================================================
async def post_init(app):
    await init_db()
    await load_all_from_db()
    asyncio.create_task(monitor_positions(app))
    asyncio.create_task(auto_sniper_loop(app))
    if HELIUS_RPC:
        asyncio.create_task(raydium_ws_sniper(app))
        log.info("Raydium WebSocket sniper started")
    else:
        log.info("HELIUS_RPC not set — Raydium WS sniper disabled")
    log.info("All systems go")
    await _safe_notify(app, "🚀 *Bot restarted — all state restored.*\n\nSend /start to open the dashboard.")

def main():
    validate_config()
    global keypair, solana_client
    keypair       = Keypair.from_bytes(base58.b58decode(PRIVATE_KEY_B58))
    solana_client = AsyncClient(RPC_URL, commitment=Confirmed)
    log.info(f"Wallet: {keypair.pubkey()}")
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(button_handler)],
        states={
            WAITING_BUY_MINT:      [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_buy_mint)],
            WAITING_BUY_SYMBOL:    [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_buy_symbol)],
            WAITING_CONFIRM_BUY:   [CallbackQueryHandler(handle_confirm_buy)],
            WAITING_SET_TP:        [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_TRAIL:     [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_STOP:      [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_AMOUNT:    [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_SLIP:      [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_SCORE:     [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_LIQ:       [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_RUGCHECK:  [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_TRAIL_5X:  [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_TRAIL_10X: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_TRAIL_20X: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_TRAIL_50X: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
        },
        fallbacks=[CommandHandler("start", cmd_start)],
        per_message=False,
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(conv)
    log.info("Bot polling...")
    app.run_polling()

if __name__ == "__main__":
    main()