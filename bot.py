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
from datetime import datetime, timezone
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
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
WALLET_ADDRESS   = os.getenv("WALLET_ADDRESS")
PRIVATE_KEY_B58  = os.getenv("PRIVATE_KEY")
RPC_URL          = os.getenv("RPC_URL", "https://api.mainnet-beta.solana.com")
AUTHORIZED_USER  = int(os.getenv("AUTHORIZED_USER_ID", 0))
DATABASE_URL     = os.getenv("DATABASE_URL", "")
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_API  = "https://quote-api.jup.ag/v6/swap"
JUPITER_PRICE_API = "https://lite-api.jup.ag/price/v2"
DEXSCREENER_API   = "https://api.dexscreener.com/latest/dex/tokens/"
RUGCHECK_API      = "https://api.rugcheck.xyz/v1/tokens/{}/report/summary"
RUGCHECK_SCORE_MIN = int(os.getenv("RUGCHECK_SCORE_MIN", "500"))
HELIUS_RPC        = os.getenv("HELIUS_RPC", "")
USDC_MINT         = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
TOKEN_PROGRAM_ID  = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

# Conversation states
WAITING_BUY_MINT       = 0
WAITING_BUY_SYMBOL     = 1
WAITING_SELL_MINT      = 2
WAITING_CONFIRM_BUY    = 3
WAITING_SET_TP         = 10
WAITING_SET_TRAIL      = 11
WAITING_SET_STOP       = 12
WAITING_SET_AMOUNT     = 13
WAITING_SET_SLIP       = 14
WAITING_SET_ENTRY_SLIP = 36
WAITING_SET_EXIT_SLIP  = 37
WAITING_SET_SCORE      = 15
WAITING_SET_LIQ        = 16
WAITING_SET_RUGCHECK   = 17
WAITING_SET_TRAIL_5X   = 20
WAITING_SET_TRAIL_10X  = 21
WAITING_SET_TRAIL_20X  = 22
WAITING_SET_TRAIL_50X  = 23
WAITING_SET_MIN_AGE    = 24
WAITING_SET_VOL5M      = 25
WAITING_SET_MAX_DEMO   = 26
WAITING_SET_MAX_REAL   = 27
WAITING_SET_PT_5X      = 30
WAITING_SET_PT_10X     = 31
WAITING_SET_PT_20X     = 32
WAITING_SET_DAILY_LOSS = 33
WAITING_SET_BE_MULT    = 35
WAITING_SET_MOMENTUM_PCT     = 38
WAITING_SET_SELL_RATIO       = 39
WAITING_SET_MAX_HOLD         = 40
# NEW: early profit tier
WAITING_SET_PT_EARLY         = 41
WAITING_SET_PT_EARLY_MULT    = 42
WAITING_SET_STAGNATION_PCT   = 43
WAITING_SET_STAGNATION_SECS  = 44
WAITING_SET_MULTI_SIGNAL_CNT = 45
WAITING_SET_PRE_TP_TRAIL     = 46  # pre-TP fast trailing stop %
WAITING_SET_PRE_TP_TRAIL_ACT = 47  # activation threshold (e.g. 1.1x)

def validate_config():
    missing = [k for k, v in {
        "TELEGRAM_TOKEN": TELEGRAM_TOKEN, "WALLET_ADDRESS": WALLET_ADDRESS,
        "PRIVATE_KEY": PRIVATE_KEY_B58, "AUTHORIZED_USER_ID": AUTHORIZED_USER,
        "DATABASE_URL": DATABASE_URL,
    }.items() if not v]
    if missing:
        raise SystemExit(f"Missing env vars: {', '.join(missing)}")
    log.info("Config validated")

keypair = solana_client = db_pool = None

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
        "rpc_reconnects": 0,
    },
    "settings": {
        # Core trading
        "take_profit": 3.0,
        "trailing_stop": 15,
        "stop_loss": 0.5,
        "trade_amount": 10.0,
        "demo_trade_amount": 100.0,
        "slippage_bps": 100,
        "entry_slippage_bps": 100,
        "exit_slippage_bps": 200,
        "priority_fee": 20000,
        # Bot modes
        "auto_snipe": False,
        "demo_mode": False,
        "house_money_mode": True,
        "ml_real_only": False,
        # Filters
        "min_liquidity": 50000,
        "min_score": 0.5,
        "min_rugcheck": 500,
        "min_token_age_sec": 120,
        "min_vol5m_pct": 10.0,
        # Position limits
        "max_demo_positions": 5,
        "max_real_positions": 3,
        # Timing
        "seen_expiry_sec": 7200,
        "max_retries": 3,
        "retry_delay": 1.5,
        "confirm_timeout": 45,
        # Tiered trailing stop
        "trail_5x":  4.0,
        "trail_10x": 3.0,
        "trail_20x": 2.0,
        "trail_50x": 1.5,
        # Tiered profit-take (% of position to sell at each milestone)
        "pt_5x_pct":  25.0,
        "pt_10x_pct": 25.0,
        "pt_20x_pct": 25.0,
        # ── NEW: Early profit-take tier (sub-TP, e.g. 1.5x) ──────────────
        # Sells a slice early to lock in gains before reversal.
        # Set pt_early_pct to 0 to disable entirely.
        "pt_early_mult": 1.5,    # multiplier that triggers early sell (e.g. 1.5x)
        "pt_early_pct":  30.0,   # % of position to sell at that point
        # Risk management
        "daily_loss_limit_pct": 20.0,
        "daily_loss_pause_hrs": 4.0,
        "breakeven_mult": 2.0,
        "conviction_sizing": True,
        "max_hold_minutes": 120,
        "house_money_max_retries": 3,
        # ── Pre-TP fast trailing stop ─────────────────────────────────────
        # Independent tight trail active from pre_tp_trail_act_mult (e.g. 1.1x)
        # up until TP is hit. Catches fast reversals before the slower
        # multi-signal system can accumulate enough signal votes.
        # Set pre_tp_trail_pct to 0 to disable.
        "pre_tp_trail_pct":      3.0,   # % drop from peak that triggers exit
        "pre_tp_trail_act_mult": 1.1,   # multiplier at which this trail activates
        # ── Multi-signal exit (NEW) ───────────────────────────────────────
        # How many of the 3 dump signals must fire together to force an exit.
        # 1 = any single signal exits  (aggressive)
        # 2 = 2-of-3 signals exit      (balanced — recommended)
        # 3 = all 3 signals must fire   (conservative)
        "multi_signal_exit_count": 2,
        # Individual dump-detection signals
        "sell_ratio_flip_threshold": 1.2,   # lowered: more sensitive sell-pressure
        "vol_exhaustion_pct": 50.0,
        "momentum_exit_pct": 1.5,           # lowered: catch earlier reversals
        # ── NEW: Price stagnation exit ────────────────────────────────────
        # Exit if price hasn't moved more than stagnation_pct% over
        # stagnation_secs seconds (only active after TP is hit).
        "stagnation_pct":  2.0,   # % movement threshold
        "stagnation_secs": 60,    # observation window in seconds
        # Wallet concentration filter
        "max_wallet_concentration": 40.0,
    },
    "ml_features": [], "ml_labels": [],
    "daily_pnl": 0.0,
    "daily_pnl_date": "",
    "sniper_paused_until": 0.0,
    "recent_losses": {},
}

# ============================================================
# DATABASE
# ============================================================
async def init_db():
    global db_pool
    import ssl as _ssl
    ctx = _ssl.create_default_context()
    ctx.check_hostname = False; ctx.verify_mode = _ssl.CERT_NONE
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5, ssl=ctx,
        server_settings={"application_name": "genixx_bot"}, command_timeout=30)
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
            "INSERT INTO settings(key,value) VALUES('main',$1) ON CONFLICT(key) DO UPDATE SET value=$1",
            json.dumps(state["settings"]))

async def db_load_settings():
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT value FROM settings WHERE key='main'")
        if row: state["settings"].update(json.loads(row["value"]))

async def db_save_position(mint, pos, is_demo):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO positions(mint,data,is_demo) VALUES($1,$2,$3) ON CONFLICT(mint) DO UPDATE SET data=$2,is_demo=$3",
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
            INSERT INTO trades(symbol,mint,entry_price,exit_price,multiplier,
            net_pnl,fees_paid,reason,is_demo,tx_sig,features)
            VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
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
            closed_ts = t["created_at"].timestamp() if t.get("created_at") else 0
            if t["is_demo"]:
                state["demo_trades"].append({
                    "symbol": t["symbol"], "mult": t["multiplier"],
                    "net_pnl": t["net_pnl"], "reason": t["reason"],
                    "closed_at": closed_ts,
                    "projected_real": t["net_pnl"] * (
                        state["settings"]["trade_amount"] / state["settings"]["demo_trade_amount"])
                })
            else:
                state["trades_history"].append({
                    "symbol": t["symbol"], "mult": t["multiplier"],
                    "net_pnl": t["net_pnl"], "reason": t["reason"],
                    "closed_at": closed_ts})
        for r in await conn.fetch("SELECT SUM(net_pnl) as total,is_demo FROM trades GROUP BY is_demo"):
            if r["is_demo"]: state["demo_total_pnl"] = float(r["total"] or 0)
            else:            state["total_pnl"]       = float(r["total"] or 0)

async def db_save_ml_sample(features, label):
    async with db_pool.acquire() as conn:
        await conn.execute("INSERT INTO ml_data(features,label) VALUES($1,$2)",
            json.dumps(features), label)

async def db_load_ml_data():
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT features,label FROM ml_data ORDER BY id")
        skipped = 0
        for r in rows:
            feats = json.loads(r["features"])
            if len(feats) < 18:
                skipped += 1
                continue
            state["ml_features"].append(feats[:18])
            state["ml_labels"].append(r["label"])
        if skipped:
            log.warning(f"ML load: skipped {skipped} legacy <18-feature samples.")
    log.info(f"ML data loaded: {len(state['ml_features'])} samples")

async def load_all_from_db():
    await db_load_settings(); await db_load_positions()
    await db_load_trades();   await db_load_ml_data()
    if len(state["ml_features"]) >= 10: train_model()
    log.info("All state restored from DB")

# ============================================================
# ERROR TRACKING
# ============================================================
def log_error(ctx, e, extra=""):
    msg = f"[{ctx}] {type(e).__name__}: {e}" + (f" | {extra}" if extra else "")
    log.error(msg); log.debug(traceback.format_exc())
    state["errors"].append({"time": datetime.now().strftime("%H:%M:%S"), "context": ctx, "error": str(e)[:120]})
    if len(state["errors"]) > 20: state["errors"].pop(0)

def escape_md(text: str) -> str:
    import re
    return re.sub(r'([_*`\[\]])', r'\\\1', str(text))

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
    """16 market features + buy-size anomaly + hour-of-day = 18 features."""
    try:
        liq    = float(td.get("liquidity",{}).get("usd",0) or 0)
        vol24  = float(td.get("volume",{}).get("h24",0) or 0)
        vol5m  = float(td.get("volume",{}).get("m5",0) or 0)
        pc1    = float(td.get("priceChange",{}).get("h1",0) or 0)
        pc6    = float(td.get("priceChange",{}).get("h6",0) or 0)
        pc24   = float(td.get("priceChange",{}).get("h24",0) or 0)
        b1h    = float(td.get("txns",{}).get("h1",{}).get("buys",0) or 0)
        s1h    = float(td.get("txns",{}).get("h1",{}).get("sells",0) or 0)
        b5m    = float(td.get("txns",{}).get("m5",{}).get("buys",0) or 0)
        s5m    = float(td.get("txns",{}).get("m5",{}).get("sells",0) or 0)
        mcap   = float(td.get("marketCap",0) or 0)
        age    = (time.time()*1000-(td.get("pairCreatedAt") or time.time()*1000))/60000
        avg_buy_usd  = (vol5m / b5m) if b5m > 0 else 0
        avg_sell_usd = (vol5m / s5m) if s5m > 0 else 0
        buy_size_ratio = avg_buy_usd / (avg_sell_usd + 1)
        hour_utc = datetime.now(timezone.utc).hour
        return [liq, vol24, pc1, pc6, pc24, b1h, s1h, b1h/(s1h+1),
                age, mcap, vol5m, b5m, s5m, b5m/(s5m+1), liq/(mcap+1), vol24/(liq+1),
                buy_size_ratio, float(hour_utc)]
    except Exception as e:
        log_error("extract_features", e); return [0.0]*18

def train_model():
    global ml_model, ml_scaler, ml_ready
    try:
        from sklearn.metrics import precision_score, recall_score
        X, y = np.array(state["ml_features"]), np.array(state["ml_labels"])
        if len(X) < 10:
            log.info(f"ML not trained: only {len(X)} samples (need 10)")
            return None
        unique_classes = set(y.tolist())
        if len(unique_classes) < 2:
            label_name = "all wins" if 1 in unique_classes else "all losses"
            log.warning(f"ML skipped: only one class in {len(y)} samples ({label_name}).")
            ml_ready = False
            return None
        ml_scaler = StandardScaler(); Xs = ml_scaler.fit_transform(X)
        ml_model = RandomForestClassifier(n_estimators=100, max_depth=6,
            min_samples_leaf=2, random_state=42, class_weight="balanced")
        ml_model.fit(Xs, y); ml_ready = True
        preds = ml_model.predict(Xs)
        precision = precision_score(y, preds, zero_division=0)
        recall    = recall_score(y, preds, zero_division=0)
        wins = int(sum(y)); losses = len(y) - wins
        log.info(f"ML trained: {len(X)} samples ({wins}W/{losses}L) | precision={precision:.0%} recall={recall:.0%}")
        return precision
    except Exception as e:
        ml_ready = False
        log_error("train_model", e); return None

def predict_score(features):
    try:
        if not ml_ready or ml_model is None: return 0.5
        f = list(features)
        if len(f) < 18: f += [0.0]*(18-len(f))
        return float(ml_model.predict_proba(ml_scaler.transform([f]))[0][1])
    except Exception as e:
        log_error("predict_score", e); return 0.5

async def record_trade_outcome(features, profitable, is_demo=False):
    if state["settings"].get("ml_real_only") and is_demo:
        log.info("ML real-only: skipping demo trade")
        return None
    label = 1 if profitable else 0
    f = list(features)
    if len(f) < 18: f += [0.0]*(18-len(f))
    state["ml_features"].append(f)
    state["ml_labels"].append(label)
    await db_save_ml_sample(f, label)
    if len(state["ml_features"]) >= 10: return train_model()
    return None

# ============================================================
# AUTH & UTILITIES
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
            if msg: await msg.reply_text(f"Error: `{type(e).__name__}: {str(e)[:100]}`", parse_mode="Markdown")
    return wrapper

def calc_fees(amount_usd, priority_lp=20000):
    dex = round(amount_usd*0.003, 4); pri = round((priority_lp/1e9)*150.0, 4)
    net = round(0.000005*150.0, 6)
    return {"dex_fee": dex, "priority_fee": pri, "network_fee": net, "total": round(dex+pri+net, 4)}

# ============================================================
# DAILY P&L TRACKER
# ============================================================
def _reset_daily_pnl_if_needed():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state["daily_pnl_date"] != today:
        state["daily_pnl"]      = 0.0
        state["daily_pnl_date"] = today
        log.info(f"Daily P&L reset for {today}")

def _check_daily_loss_limit():
    s = state["settings"]
    limit_pct  = s.get("daily_loss_limit_pct", 20.0)
    capital    = s.get("trade_amount", 10.0) * s.get("max_real_positions", 3)
    threshold  = -abs(capital * limit_pct / 100.0)
    if state["daily_pnl"] <= threshold and time.time() > state.get("sniper_paused_until", 0):
        pause_secs = s.get("daily_loss_pause_hrs", 4.0) * 3600
        state["sniper_paused_until"] = time.time() + pause_secs
        log.warning(f"Daily loss limit hit (${state['daily_pnl']:.2f}). Sniper paused {pause_secs/3600:.1f}h.")
        return True
    return time.time() < state.get("sniper_paused_until", 0)

async def _safe_notify(app, text):
    if app is None:
        log.warning(f"_safe_notify called with app=None: {text[:80]}"); return
    try:
        await app.bot.send_message(chat_id=AUTHORIZED_USER, text=text,
            parse_mode="Markdown", disable_web_page_preview=True)
    except TelegramError as e:
        log.error(f"Notify failed: {e}")

# ============================================================
# TOKEN DATA & PRICING
# ============================================================
_price_cache: dict = {}
_PRICE_CACHE_TTL      = 7.5
_PRICE_CACHE_FAIL_TTL = 3.0
_rugcheck_cache: dict = {}
_RUGCHECK_CACHE_TTL   = 300

async def get_token_price(mint: str, pair_data: dict = None) -> float:
    if pair_data:
        try:
            price = float(pair_data.get("priceUsd", 0) or 0)
            if price > 0:
                state["api_stats"]["price_ok"] += 1
                _price_cache[mint] = (time.time(), price)
                return price
        except Exception:
            pass
    now = time.time()
    if mint in _price_cache:
        ts, p = _price_cache[mint]
        if p > 0 and now - ts < _PRICE_CACHE_TTL: return p
        if p == 0.0 and now - ts < _PRICE_CACHE_FAIL_TTL: return 0.0
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
            async with s.get(f"{DEXSCREENER_API}{mint}") as r:
                if r.status == 200:
                    pairs = [p for p in ((await r.json()).get("pairs") or []) if p.get("chainId") == "solana"]
                    if pairs:
                        pairs.sort(key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0), reverse=True)
                        price = float(pairs[0].get("priceUsd", 0) or 0)
                        if price > 0:
                            state["api_stats"]["price_ok"] += 1
                            _price_cache[mint] = (now, price); return price
                elif r.status == 429:
                    log.warning("DexScreener rate-limited")
    except Exception as e: log.warning(f"DexScreener price: {e}")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6)) as s:
            async with s.get(f"{JUPITER_PRICE_API}?ids={mint}&showExtraInfo=false") as r:
                if r.status == 200:
                    raw = (await r.json()).get("data", {}).get(mint, {})
                    price = float(raw.get("price") or 0)
                    if price > 0:
                        state["api_stats"]["price_ok"] += 1
                        _price_cache[mint] = (now, price); return price
    except Exception as e: log.warning(f"Jupiter price API: {e}")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
            async with s.get(JUPITER_QUOTE_API, params={"inputMint": USDC_MINT, "outputMint": mint,
                    "amount": 1_000_000, "slippageBps": 500}) as r:
                if r.status == 200:
                    q = await r.json(); out = int(q.get("outAmount", 0)); decimals = int(q.get("outputDecimals", 6))
                    if out > 0:
                        price = 1.0 / (out / (10 ** decimals))
                        state["api_stats"]["price_ok"] += 1
                        _price_cache[mint] = (now, price); return price
    except Exception as e: log.warning(f"Jupiter quote fallback price: {e}")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6)) as s:
            async with s.get(f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{mint}",
                    headers={"Accept": "application/json"}) as r:
                if r.status == 200:
                    price = float((await r.json()).get("data", {}).get("attributes", {}).get("price_usd") or 0)
                    if price > 0:
                        state["api_stats"]["price_ok"] += 1
                        _price_cache[mint] = (now, price); return price
    except Exception as e: log.warning(f"GeckoTerminal price: {e}")
    state["api_stats"]["price_fail"] += 1
    _price_cache[mint] = (now, 0.0)
    log.warning(f"All price sources failed for {mint[:16]}")
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
    now = time.time()
    if mint in _rugcheck_cache:
        ts, result = _rugcheck_cache[mint]
        if now - ts < _RUGCHECK_CACHE_TTL: return result
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            async with s.get(RUGCHECK_API.format(mint)) as r:
                if r.status == 429:
                    log.warning("RugCheck rate limited — using safe default")
                    return {"score": 0, "risks": [], "rugged": False}
                if r.status == 200:
                    d = await r.json()
                    result = {"score": d.get("score",0), "risks": d.get("risks",[]), "rugged": d.get("rugged",False)}
                    _rugcheck_cache[mint] = (now, result); return result
    except asyncio.TimeoutError: log.warning(f"RugCheck timeout {mint[:8]}")
    except Exception as e: log.warning(f"RugCheck error {mint[:8]}: {e}")
    safe = {"score": 0, "risks": [], "rugged": False}
    _rugcheck_cache[mint] = (now, safe); return safe

# ============================================================
# WALLET BALANCE
# ============================================================
async def get_wallet_balance() -> dict:
    result = {"sol": 0.0, "usdc": 0.0, "tokens": [], "total_usd": 0.0, "sol_usd": 0.0}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
            async with s.post(RPC_URL, json={"jsonrpc": "2.0", "id": 1, "method": "getBalance",
                    "params": [WALLET_ADDRESS]}) as r:
                if r.status == 200:
                    d = await r.json()
                    result["sol"] = (d.get("result", {}).get("value", 0) or 0) / 1e9
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.post(RPC_URL, json={"jsonrpc": "2.0", "id": 2,
                    "method": "getTokenAccountsByOwner",
                    "params": [WALLET_ADDRESS, {"programId": TOKEN_PROGRAM_ID}, {"encoding": "jsonParsed"}]}) as r:
                if r.status == 200:
                    accounts = (await r.json()).get("result", {}).get("value") or []
                    for acct in accounts:
                        info = (acct.get("account", {}).get("data", {}).get("parsed", {}).get("info", {}))
                        mint = info.get("mint", "")
                        amt  = float(info.get("tokenAmount", {}).get("uiAmount") or 0)
                        if amt <= 0: continue
                        if mint == USDC_MINT: result["usdc"] = amt
                        else: result["tokens"].append({"mint": mint, "amount": amt, "symbol": mint[:6]+"…", "usd": 0.0})
        sol_price = 0.0
        try:
            SOL_MINT = "So11111111111111111111111111111111111111112"
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6)) as s:
                async with s.get(f"{JUPITER_PRICE_API}?ids={SOL_MINT}") as r:
                    if r.status == 200:
                        sol_price = float((await r.json()).get("data", {}).get(SOL_MINT, {}).get("price") or 0)
        except Exception: pass
        sol_usd = result["sol"] * sol_price
        token_usd_total = 0.0
        for tok in result["tokens"]:
            pos = state["positions"].get(tok["mint"])
            if pos:
                tok["usd"] = tok["amount"] * pos.get("current_price", 0)
                tok["symbol"] = pos["symbol"]
                token_usd_total += tok["usd"]
        result["sol_usd"] = sol_usd
        result["total_usd"] = sol_usd + result["usdc"] + token_usd_total
    except Exception as e: log_error("get_wallet_balance", e)
    return result

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
        q = await with_retry(_f, label="quote"); state["api_stats"]["quote_ok"] += 1; return q
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
        tx = await with_retry(_f, label="swap_tx"); state["api_stats"]["swap_ok"] += 1; return tx
    except Exception as e:
        state["api_stats"]["swap_fail"] += 1; log_error("get_swap_tx", e); return None

async def _ensure_rpc():
    global solana_client
    try:
        await solana_client.get_slot()
        return solana_client
    except Exception as e:
        log.warning(f"RPC health check failed ({e}), reconnecting…")
        try: await solana_client.close()
        except Exception: pass
        solana_client = AsyncClient(RPC_URL, commitment=Confirmed)
        state["api_stats"]["rpc_reconnects"] += 1
        return solana_client

async def sign_and_send(tx_b64):
    try:
        client = await _ensure_rpc()
        tx = VersionedTransaction.from_bytes(base64.b64decode(tx_b64))
        tx.sign([keypair])
        r = await client.send_raw_transaction(
            bytes(tx), opts=TxOpts(skip_preflight=False, preflight_commitment=Confirmed))
        return str(r.value)
    except Exception as e:
        log_error("sign_and_send", e); return None

async def confirm_tx(sig):
    for i in range(state["settings"]["confirm_timeout"]):
        try:
            client = await _ensure_rpc()
            st = (await client.get_signature_statuses([sig])).value[0]
            if st and st.confirmation_status in ("confirmed","finalized"):
                state["api_stats"]["confirm_ok"] += 1; return True
            if st and st.err: return False
        except Exception as e: log.warning(f"Confirm poll {i+1}: {e}")
        await asyncio.sleep(1)
    state["api_stats"]["confirm_timeout"] += 1; return False

async def execute_buy(mint, amt):
    q = await get_quote(USDC_MINT, mint, int(amt*1e6), state["settings"].get("entry_slippage_bps", state["settings"]["slippage_bps"]))
    if not q: return None
    tx = await get_swap_tx(q, state["settings"]["priority_fee"])
    if not tx: return None
    sig = await sign_and_send(tx)
    if not sig: return None
    return {"signature": sig, "confirmed": await confirm_tx(sig), "out_amount": int(q.get("outAmount",0))}

async def execute_sell(mint, token_amt):
    if token_amt <= 0: return None
    q = await get_quote(mint, USDC_MINT, token_amt, state["settings"].get("exit_slippage_bps", state["settings"]["slippage_bps"]))
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
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💰 Positions",   callback_data="positions"),
         InlineKeyboardButton("📊 P&L",         callback_data="pnl")],
        [InlineKeyboardButton("🛒 Buy Token",   callback_data="buy_prompt"),
         InlineKeyboardButton("💸 Sell Token",  callback_data="sell_prompt")],
        [InlineKeyboardButton("🟢 Sniper ON" if s["auto_snipe"] else "🔴 Sniper OFF", callback_data="toggle_sniper"),
         InlineKeyboardButton("📝 Demo ON"   if s["demo_mode"]  else "📝 Demo OFF",   callback_data="toggle_demo")],
        [InlineKeyboardButton("📝 Demo Trades", callback_data="demo_menu"),
         InlineKeyboardButton("🧠 ML Stats",    callback_data="mlstats")],
        [InlineKeyboardButton("⚙️ Settings",    callback_data="settings_menu"),
         InlineKeyboardButton("🏥 Health",      callback_data="health")],
        [InlineKeyboardButton("📜 History",       callback_data="history"),
         InlineKeyboardButton("📈 P&L Breakdown", callback_data="pnl_breakdown")],
        [InlineKeyboardButton("💼 My Wallet",   callback_data="wallet")],
    ])

def kb_settings():
    s   = state["settings"]
    hm  = "🏠 House Money: ON"  if s.get("house_money_mode") else "🏠 House Money: OFF"
    mlo = "🧠 ML Real Only: ON" if s.get("ml_real_only")     else "🧠 ML Real Only: OFF"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🎯 Take Profit: {s['take_profit']}x",          callback_data="set_tp"),
         InlineKeyboardButton(f"📉 Trailing: {s['trailing_stop']}%",           callback_data="set_trail")],
        [InlineKeyboardButton(f"🛑 Stop Loss: {s['stop_loss']}x",              callback_data="set_stop"),
         InlineKeyboardButton(f"💵 Amount: ${s['trade_amount']}",              callback_data="set_amount")],
        [InlineKeyboardButton(f"⚡ Entry Slip: {s.get('entry_slippage_bps',s['slippage_bps'])}bps", callback_data="set_entry_slip"),
         InlineKeyboardButton(f"⚡ Exit Slip: {s.get('exit_slippage_bps',200)}bps",  callback_data="set_exit_slip")],
        [InlineKeyboardButton(f"🧠 Min Score: {s['min_score']:.0%}",           callback_data="set_score")],
        [InlineKeyboardButton(f"💧 Min Liq: ${s['min_liquidity']:,.0f}",       callback_data="set_liq"),
         InlineKeyboardButton(f"🛡️ Max Rug: {s['min_rugcheck']}",             callback_data="set_rugcheck")],
        [InlineKeyboardButton(f"⏱ Min Age: {s.get('min_token_age_sec',120)}s", callback_data="set_min_age"),
         InlineKeyboardButton(f"📊 Vol5m≥: {s.get('min_vol5m_pct',10)}% liq", callback_data="set_vol5m")],
        [InlineKeyboardButton(f"📂 Max Demo: {s.get('max_demo_positions',5)}",  callback_data="set_max_demo"),
         InlineKeyboardButton(f"📂 Max Real: {s.get('max_real_positions',3)}",  callback_data="set_max_real")],
        [InlineKeyboardButton(hm,  callback_data="toggle_house_money"),
         InlineKeyboardButton(mlo, callback_data="toggle_ml_real_only")],
        [InlineKeyboardButton(f"⚖️ Breakeven: {s.get('breakeven_mult',2.0)}x", callback_data="set_be_mult"),
         InlineKeyboardButton(f"🚨 Daily Loss: {s.get('daily_loss_limit_pct',20)}%", callback_data="set_daily_loss")],
        [InlineKeyboardButton(f"📐 Conviction Sizing: {'ON' if s.get('conviction_sizing') else 'OFF'}", callback_data="toggle_conviction_sizing")],
        [InlineKeyboardButton("🚨 Dump Detection Settings", callback_data="dump_detection_menu")],
        [InlineKeyboardButton("📐 Tiered Trail Settings",   callback_data="tiered_trail_menu")],
        [InlineKeyboardButton("⬅️ Back to Menu",            callback_data="main_menu")],
    ])

def kb_tiered_trail():
    s = state["settings"]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🟢 ≥5x  → {s.get('trail_5x', 4.0)}% trail",  callback_data="set_trail_5x")],
        [InlineKeyboardButton(f"🟡 ≥10x → {s.get('trail_10x', 3.0)}% trail", callback_data="set_trail_10x")],
        [InlineKeyboardButton(f"🟠 ≥20x → {s.get('trail_20x', 2.0)}% trail", callback_data="set_trail_20x")],
        [InlineKeyboardButton(f"🔴 ≥50x → {s.get('trail_50x', 1.5)}% trail", callback_data="set_trail_50x")],
        [InlineKeyboardButton("── Profit-Take Milestones ──", callback_data="noop")],
        [InlineKeyboardButton(f"⚡ Early: @{s.get('pt_early_mult',1.5)}x sell {s.get('pt_early_pct',30.0):.0f}%", callback_data="set_pt_early")],
        [InlineKeyboardButton(f"💰 @5x  sell {s.get('pt_5x_pct', 25.0):.0f}%",  callback_data="set_pt_5x")],
        [InlineKeyboardButton(f"💰 @10x sell {s.get('pt_10x_pct', 25.0):.0f}%", callback_data="set_pt_10x")],
        [InlineKeyboardButton(f"💰 @20x sell {s.get('pt_20x_pct', 25.0):.0f}%", callback_data="set_pt_20x")],
        [InlineKeyboardButton("⬅️ Back to Settings", callback_data="settings_menu")],
    ])

def kb_dump_detection():
    s = state["settings"]
    pre_tp_pct  = s.get("pre_tp_trail_pct", 3.0)
    pre_tp_act  = s.get("pre_tp_trail_act_mult", 1.1)
    pre_tp_lbl  = f"OFF" if pre_tp_pct == 0 else f"{pre_tp_pct}% from peak"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(
            f"⚡ Pre-TP Trail: {pre_tp_lbl} (acts @{pre_tp_act}x)",
            callback_data="set_pre_tp_trail"),
         InlineKeyboardButton(
            f"🎚 Activate @{pre_tp_act}x",
            callback_data="set_pre_tp_trail_act")],
        [InlineKeyboardButton(
            f"🔢 Signals needed: {s.get('multi_signal_exit_count', 2)}/4",
            callback_data="set_multi_signal_cnt")],
        [InlineKeyboardButton(f"⚡ Momentum Exit: {s.get('momentum_exit_pct', 1.5)}%",   callback_data="set_momentum_pct")],
        [InlineKeyboardButton(f"📉 Vol Exhaust: {s.get('vol_exhaustion_pct', 50.0):.0f}%", callback_data="set_vol_exhaust")],
        [InlineKeyboardButton(f"🚨 Sell Ratio: {s.get('sell_ratio_flip_threshold',1.2)}x", callback_data="set_sell_ratio")],
        [InlineKeyboardButton(f"⏸ Stagnation %: {s.get('stagnation_pct',2.0)}%",      callback_data="set_stagnation_pct"),
         InlineKeyboardButton(f"⏱ Window: {s.get('stagnation_secs',60)}s",             callback_data="set_stagnation_secs")],
        [InlineKeyboardButton(f"⏰ Max Hold: {s.get('max_hold_minutes', 120)}min",         callback_data="set_max_hold")],
        [InlineKeyboardButton("⬅️ Back to Settings", callback_data="settings_menu")],
    ])

def kb_demo():
    s = state["settings"]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Demo Status",  callback_data="demostatus"),
         InlineKeyboardButton("📜 Demo History", callback_data="demohistory")],
        [InlineKeyboardButton(f"{'🟢' if s['demo_mode'] else '🔴'} {'Turn OFF' if s['demo_mode'] else 'Turn ON'} Demo",
                              callback_data="toggle_demo")],
        [InlineKeyboardButton("⬅️ Back to Menu", callback_data="main_menu")],
    ])

def kb_back():
    return InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back to Menu", callback_data="main_menu")]])

def kb_confirm_buy(mint, symbol):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Confirm Buy", callback_data="confirm_buy_pending"),
         InlineKeyboardButton("❌ Cancel",      callback_data="main_menu")],
    ])

def kb_confirm_sell(mint):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Confirm Sell", callback_data=f"confirm_sell:{mint}"),
         InlineKeyboardButton("❌ Cancel",       callback_data="main_menu")],
    ])

def kb_positions(positions: dict, is_demo=False):
    rows = []
    for mint, pos in positions.items():
        lbl = pos['symbol'] + (" 🏠" if pos.get("capital_recovered") else "")
        if is_demo:
            rows.append([
                InlineKeyboardButton(f"💸 Close {lbl}", callback_data=f"dsell:{mint}"),
                InlineKeyboardButton("✂️ TP Now",        callback_data=f"dclose_now:{mint}")
            ])
        else:
            rows.append([InlineKeyboardButton(f"💸 Sell {lbl}", callback_data=f"sell_confirm:{mint}")])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="main_menu")])
    return InlineKeyboardMarkup(rows)

# ============================================================
# DASHBOARD
# ============================================================
async def build_dashboard() -> str:
    s     = state["settings"]
    n     = len(state["ml_features"])
    wins  = sum(1 for x in state["trades_history"] if x["net_pnl"] > 0)
    total = len(state["trades_history"])
    paused = time.time() < state.get("sniper_paused_until", 0)
    sniper_status = "⏸ PAUSED" if paused else ("🟢 ON" if s["auto_snipe"] else "🔴 OFF")
    msg = (
        f"🤖 *Solana Meme Coin Bot*\n{'─'*28}\n\n"
        f"*💼 Portfolio*\n"
        f"├ Real P&L:  {'📈' if state['total_pnl']>=0 else '📉'} *${state['total_pnl']:+.2f}*\n"
        f"├ Today P&L: {'📈' if state['daily_pnl']>=0 else '📉'} ${state['daily_pnl']:+.2f}\n"
        f"├ Demo P&L:  📝 ${state['demo_total_pnl']:+.2f}\n"
        f"├ Trades:    {total} ({wins}W / {total-wins}L)\n\n"
        f"*🤖 Bot Status*\n"
        f"├ Auto-Sniper: {sniper_status}\n"
        f"├ Demo Mode:   {'🟢 ON' if s['demo_mode'] else '🔴 OFF'}\n"
        f"├ ML Model:    {'✅ Ready' if ml_ready else '⏳ Training'} ({n} samples)\n\n"
        f"*⚙️ Active Settings*\n"
        f"├ Take Profit:  {s['take_profit']}x\n"
        f"├ Stop Loss:    {s['stop_loss']}x\n"
        f"├ Early Exit:   @{s.get('pt_early_mult',1.5)}x → sell {s.get('pt_early_pct',30.0):.0f}%\n"
        f"├ Breakeven:    {s.get('breakeven_mult',2.0)}x → lock entry\n"
        f"├ House Money:  {'🏠 ON' if s.get('house_money_mode') else 'OFF'}\n"
        f"├ Conviction:   {'📐 ON' if s.get('conviction_sizing') else 'OFF'}\n"
        f"├ Dump Signals: {s.get('multi_signal_exit_count',2)}/3 needed\n"
        f"├ Daily Limit:  {s.get('daily_loss_limit_pct',20)}% capital\n"
        f"├ Min Liq:      ${s['min_liquidity']:,.0f}\n"
        f"├ Min Score:    {s['min_score']:.0%}\n"
        f"├ Max Pos:      {s.get('max_demo_positions',5)}D / {s.get('max_real_positions',3)}R\n"
        f"└ Trade Amount: ${s['trade_amount']}\n"
    )
    if state["positions"]:
        msg += f"\n*📂 Open Positions ({len(state['positions'])})* \n"
        for mint, pos in state["positions"].items():
            p    = await get_token_price(mint)
            mult = p/pos["entry_price"] if p > 0 and pos["entry_price"] > 0 else 0
            pnl  = (mult-1)*pos["amount_usd"] - pos["fees_paid"]
            hm   = " 🏠" if pos.get("capital_recovered") else ""
            be   = " 🔒" if pos.get("breakeven_active") else ""
            msg += f"{'🟢' if pnl>=0 else '🔴'} {pos['symbol']}{hm}{be} | {mult:.2f}x | ${pnl:+.2f}\n"
    return msg

# ============================================================
# HOUSE MONEY — PARTIAL SELL AT TP
# ============================================================
async def _partial_sell_for_capital_recovery(app, mint, pos, current_price, is_demo):
    total_tokens = pos.get("token_amount", 0)
    invested     = pos["amount_usd"] + pos["fees_paid"]
    if current_price <= 0 or total_tokens <= 0 or invested <= 0: return False
    tokens_to_sell   = min(invested / current_price, total_tokens * 0.90)
    tokens_remaining = total_tokens - tokens_to_sell
    sell_pct         = tokens_to_sell / total_tokens
    pfx = "📝 DEMO " if is_demo else ""
    if is_demo:
        usdc_recovered = tokens_to_sell * current_price
        net_recovered  = usdc_recovered - calc_fees(usdc_recovered)["dex_fee"]
        await _safe_notify(app,
            f"{pfx}🏠 *House Money — Capital Recovered!*\n\n*{pos['symbol']}*\n{'─'*20}\n"
            f"├ Sold:           {sell_pct:.0%} of position\n"
            f"├ USDC recovered: ${net_recovered:.2f}\n"
            f"├ Original cost:  ${invested:.2f}\n"
            f"├ Tokens riding:  {tokens_remaining:,.0f}\n"
            f"└ 🚀 Pure profit from here — trail active!")
    else:
        result = await execute_sell(mint, int(tokens_to_sell))
        if not result:
            await _safe_notify(app, f"⚠️ *House Money partial sell FAILED for {pos['symbol']}*\nFalling back to normal trail.")
            return False
        await _safe_notify(app,
            f"🏠 *House Money — Capital Recovered!*\n\n*{pos['symbol']}*\n{'─'*20}\n"
            f"├ Sold:           {sell_pct:.0%} of position\n"
            f"├ USDC recovered: ${result['usdc_received']:.2f}\n"
            f"├ Original cost:  ${invested:.2f}\n"
            f"├ Tokens riding:  {tokens_remaining:,.0f}\n"
            f"└ 🚀 Pure profit from here — trail active!\n\n"
            f"🔗 [Solscan](https://solscan.io/tx/{result['signature']})")
    pos["token_amount"] = int(tokens_remaining)
    pos["amount_usd"]   = 0.0
    pos["fees_paid"]    = 0.0
    pos["capital_recovered"] = True
    return True

# ============================================================
# TIERED PROFIT-TAKE — sell a slice at 5x / 10x / 20x
# ============================================================
async def _tiered_profit_take(app, mint, pos, current_price, milestone: str, is_demo: bool):
    s         = state["settings"]
    pct_key   = f"pt_{milestone}_pct"
    pct       = s.get(pct_key, 0.0) / 100.0
    if pct <= 0: return
    total_tokens = pos.get("token_amount", 0)
    if total_tokens <= 0 or current_price <= 0: return
    tokens_to_sell   = int(total_tokens * pct)
    tokens_remaining = total_tokens - tokens_to_sell
    usdc_estimate    = tokens_to_sell * current_price
    pfx = "📝 DEMO " if is_demo else ""
    result = None
    if is_demo:
        sell_fee      = calc_fees(usdc_estimate)["dex_fee"]
        usdc_received = usdc_estimate - sell_fee
    else:
        result = await execute_sell(mint, tokens_to_sell)
        if not result:
            await _safe_notify(app, f"⚠️ *{pfx}Profit-take FAILED at {milestone} for {pos['symbol']}*")
            return
        usdc_received = result["usdc_received"]
        sell_fee      = calc_fees(usdc_received)["dex_fee"]
    pos["token_amount"] = tokens_remaining
    pos[f"pt_{milestone}_done"] = True
    await db_save_position(mint, pos, is_demo)
    sig_link = f"\n🔗 [Solscan](https://solscan.io/tx/{result['signature']})" if (not is_demo and result) else ""
    await _safe_notify(app,
        f"💰 {pfx}*Profit-Take @ {milestone} — {pos['symbol']}*\n{'─'*22}\n"
        f"├ Sold:       {pct:.0%} of position ({tokens_to_sell:,} tokens)\n"
        f"├ USDC in:    ${usdc_received:.2f}\n"
        f"├ Remaining:  {tokens_remaining:,} tokens\n"
        f"└ Still riding with tiered trail 🚀{sig_link}")

# ============================================================
# NEW: EARLY PROFIT-TAKE (sub-TP tier, e.g. 1.5x)
# ============================================================
async def _early_profit_take(app, mint, pos, current_price, mult, is_demo: bool):
    """
    Sell a configurable slice of the position when price crosses pt_early_mult
    (default 1.5x), before the main take-profit level is reached.
    This locks in gains from small pumps that then reverse (like CRISIS).
    """
    s    = state["settings"]
    pct  = s.get("pt_early_pct", 30.0) / 100.0
    if pct <= 0: return  # disabled

    total_tokens = pos.get("token_amount", 0)
    if total_tokens <= 0 or current_price <= 0: return

    tokens_to_sell   = int(total_tokens * pct)
    tokens_remaining = total_tokens - tokens_to_sell
    usdc_estimate    = tokens_to_sell * current_price
    pfx = "📝 DEMO " if is_demo else ""
    result = None

    if is_demo:
        sell_fee      = calc_fees(usdc_estimate)["dex_fee"]
        usdc_received = usdc_estimate - sell_fee
    else:
        result = await execute_sell(mint, tokens_to_sell)
        if not result:
            await _safe_notify(app, f"⚠️ *{pfx}Early profit-take FAILED for {pos['symbol']}*")
            return
        usdc_received = result["usdc_received"]

    pos["token_amount"]   = tokens_remaining
    pos["pt_early_done"]  = True
    # Tighten the trailing stop after the early sell (use 5x-tier level as floor)
    pos["early_exit_trail"] = s.get("trail_5x", 4.0) / 100.0
    await db_save_position(mint, pos, is_demo)

    sig_link = f"\n🔗 [Solscan](https://solscan.io/tx/{result['signature']})" if (not is_demo and result) else ""
    await _safe_notify(app,
        f"⚡ {pfx}*Early Profit-Take @ {mult:.2f}x — {pos['symbol']}*\n{'─'*24}\n"
        f"├ Sold:      {pct:.0%} of position ({tokens_to_sell:,} tokens)\n"
        f"├ USDC in:   ${usdc_received:.2f}\n"
        f"├ Remaining: {tokens_remaining:,} tokens\n"
        f"└ Trail tightened to {s.get('trail_5x',4.0)}% 🎯{sig_link}")

# ============================================================
# CLOSE POSITION
# ============================================================
async def _close_position(app, mint, pos, price, reason, is_demo=False):
    entry = pos["entry_price"]
    mult  = price / entry if entry > 0 else 1
    hm    = pos.get("capital_recovered", False)
    if hm:
        token_amt = pos.get("token_amount", 0)
        gross     = token_amt * price
        sell_fee  = calc_fees(gross)["dex_fee"]
        if is_demo:
            net_pnl  = gross - sell_fee; usdc_bk = gross; sig_link = ""; tx_sig = None
            proj     = net_pnl * (state["settings"]["trade_amount"] / state["settings"]["demo_trade_amount"])
            proj_txt = f"💡 Real projection: *${proj:+.2f}*\n"
        else:
            sell_r   = await execute_sell(mint, token_amt)
            usdc_bk  = sell_r["usdc_received"] if sell_r else gross * 0.997
            sell_fee = calc_fees(usdc_bk)["dex_fee"]
            net_pnl  = usdc_bk - sell_fee
            sig_link = f"\n🔗 [Solscan](https://solscan.io/tx/{sell_r['signature']})" if sell_r else ""
            proj_txt = ""; tx_sig = sell_r["signature"] if sell_r else None
    else:
        sell_fee = calc_fees(pos["amount_usd"] * mult)["dex_fee"]
        if is_demo:
            net_pnl  = (mult-1)*pos["amount_usd"] - pos["fees_paid"] - sell_fee
            usdc_bk  = pos["amount_usd"] * mult; sig_link = ""; tx_sig = None
            proj     = net_pnl * (state["settings"]["trade_amount"] / state["settings"]["demo_trade_amount"])
            proj_txt = f"💡 Real projection: *${proj:+.2f}*\n"
        else:
            sell_r   = await execute_sell(mint, pos.get("token_amount", 0))
            usdc_bk  = sell_r["usdc_received"] if sell_r else pos["amount_usd"]*mult*0.997
            net_pnl  = usdc_bk - pos["amount_usd"] - pos["fees_paid"]
            sig_link = f"\n🔗 [Solscan](https://solscan.io/tx/{sell_r['signature']})" if sell_r else ""
            proj_txt = ""; tx_sig = sell_r["signature"] if sell_r else None
    ml_msg = ""
    if pos.get("features"):
        acc    = await record_trade_outcome(pos["features"], net_pnl > 0, is_demo=is_demo)
        ml_msg = f"\n🧠 ML updated ({len(state['ml_features'])} samples)"
        if acc: ml_msg += f" | Precision: {acc:.0%}"
    await db_save_trade({"symbol": pos["symbol"], "mint": mint, "entry": entry, "exit": price,
        "mult": mult, "net_pnl": net_pnl, "fees_paid": pos["fees_paid"]+sell_fee,
        "reason": reason, "is_demo": is_demo, "tx_sig": tx_sig, "features": pos.get("features",[])})
    await db_delete_position(mint)
    if is_demo:
        state["demo_total_pnl"] += net_pnl
        state["demo_trades"].append({"symbol": pos["symbol"], "mult": mult, "net_pnl": net_pnl,
            "reason": reason, "closed_at": time.time(),
            "projected_real": net_pnl*(state["settings"]["trade_amount"]/state["settings"]["demo_trade_amount"])})
        del state["demo_positions"][mint]
    else:
        state["total_pnl"] += net_pnl
        _reset_daily_pnl_if_needed()
        state["daily_pnl"] += net_pnl
        state["trades_history"].append({"symbol": pos["symbol"], "mult": mult, "net_pnl": net_pnl,
            "reason": reason, "closed_at": time.time()})
        del state["positions"][mint]
    pfx       = "📝 DEMO " if is_demo else ""
    pnl_total = state["demo_total_pnl"] if is_demo else state["total_pnl"]
    lbl       = "Demo" if is_demo else "Real"
    hm_tag    = "\n🏠 _(Capital already recovered — pure profit)_" if hm else ""
    await _safe_notify(app,
        f"{'✅' if net_pnl>0 else '❌'} {pfx}{reason}\n\n"
        f"*{pos['symbol']}* | {mult:.2f}x\n{'─'*20}\n"
        f"├ Entry:    ${entry:.6f}\n├ Exit:     ${price:.6f}\n"
        f"├ Invested: ${pos['amount_usd']:.2f}\n├ Back:     ${usdc_bk:.2f}\n"
        f"├ Fees:     -${pos['fees_paid']+sell_fee:.4f}\n"
        f"└ *Net P&L: ${net_pnl:+.2f}*\n\n"
        f"{proj_txt}💰 {lbl} P&L: ${pnl_total:+.2f}{sig_link}{ml_msg}{hm_tag}")

# ============================================================
# AUTO-SNIPER
# ============================================================
async def _fetch_full_pair(session, mint):
    try:
        async with session.get(f"{DEXSCREENER_API}{mint}") as r:
            if r.status != 200: return None
            pairs = [p for p in (await r.json()).get("pairs",[]) or [] if p.get("chainId")=="solana"]
            return max(pairs, key=lambda p: float(p.get("liquidity",{}).get("usd",0) or 0)) if pairs else None
    except Exception as e:
        log_error("_fetch_full_pair", e); return None

async def fetch_new_pairs():
    now    = time.time()
    expiry = state["settings"].get("seen_expiry_sec", 7200)
    for m in [m for m, t in state["seen_pairs"].items() if now - t > expiry]:
        del state["seen_pairs"][m]
    new_mints = []
    connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=600, force_close=False)
    async with aiohttp.ClientSession(connector=connector,
        timeout=aiohttp.ClientTimeout(total=10, connect=4),
        headers={"Accept": "application/json"}) as session:
        async def _discover(url):
            label = url.split("/")[-1]
            for attempt in range(2):
                try:
                    async with session.get(url) as r:
                        if r.status != 200: return []
                        data  = await r.json()
                        items = data if isinstance(data, list) else (data.get("pairs") or [])
                        found = []
                        for item in items:
                            if not isinstance(item, dict): continue
                            chain = item.get("chainId") or item.get("chain", "")
                            if chain and chain != "solana": continue
                            mint = ((item.get("baseToken") or item.get("token") or {}).get("address")
                                    or item.get("tokenAddress") or item.get("mint") or item.get("address"))
                            if mint and mint not in state["seen_pairs"]: found.append(mint)
                        log.info(f"_discover [{label}]: {len(found)} new mints")
                        return found
                except asyncio.TimeoutError:
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
                if mint not in new_mints: new_mints.append(mint)
        if not new_mints: return []
        hydrated = await asyncio.gather(*[_fetch_full_pair(session, m) for m in new_mints[:40]])
    return [p for p in hydrated if p is not None]

async def evaluate_new_token(pair):
    base   = pair.get("baseToken") or pair.get("token") or {}
    mint   = base.get("address") or pair.get("tokenAddress", "")
    symbol = base.get("symbol") or pair.get("symbol", "???")
    liq    = float(pair.get("liquidity",{}).get("usd",0) or 0)
    vol5m  = float(pair.get("volume",{}).get("m5",0) or 0)
    vol24  = float(pair.get("volume",{}).get("h24",0) or 0)
    features = extract_features(pair)
    ml_score = predict_score(features)
    safety   = await check_token_safety(mint)
    rugged   = safety.get("rugged", False)
    rc_score = int(safety.get("score", 0) or 0)
    s             = state["settings"]
    min_liq       = s["min_liquidity"]
    min_score     = s["min_score"]
    min_age_sec   = s.get("min_token_age_sec", 120)
    min_vol5m_pct = s.get("min_vol5m_pct", 10.0)
    rc_too_risky  = rc_score > s.get("min_rugcheck", RUGCHECK_SCORE_MIN)
    pair_created  = pair.get("pairCreatedAt")
    age_sec       = (time.time()*1000 - pair_created)/1000 if pair_created else 9999
    vol5m_pct     = (vol5m / liq * 100) if liq > 0 else 0
    passes = (liq >= min_liq and ml_score >= min_score and not rugged
              and not rc_too_risky and age_sec >= min_age_sec and vol5m_pct >= min_vol5m_pct)
    if not passes:
        reasons = []
        if liq < min_liq:             reasons.append(f"liq=${liq:,.0f}<${min_liq:,.0f}")
        if ml_score < min_score:      reasons.append(f"ml={ml_score:.0%}<{min_score:.0%}")
        if rugged:                    reasons.append("rugged")
        if rc_too_risky:              reasons.append(f"rug={rc_score}")
        if age_sec < min_age_sec:     reasons.append(f"age={age_sec:.0f}s<{min_age_sec}s")
        if vol5m_pct < min_vol5m_pct: reasons.append(f"vol5m={vol5m_pct:.1f}%<{min_vol5m_pct}%")
        log.info(f"evaluate: {symbol} REJECTED — {', '.join(reasons)}")
    return {"mint": mint, "symbol": symbol, "liquidity": liq, "volume": vol24,
            "vol5m": vol5m, "vol5m_pct": vol5m_pct, "age_sec": age_sec,
            "market_cap": float(pair.get("marketCap",0) or 0),
            "price_change": float(pair.get("priceChange",{}).get("h1",0) or 0),
            "ml_score": ml_score, "safety": safety, "rc_score": rc_score,
            "features": features, "passes_rules": passes}

RAYDIUM_PROGRAM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

async def auto_sniper_loop(app):
    log.info("Auto-sniper started.")
    consecutive_errors = 0; loop_count = 0
    while True:
        try:
            if not state["settings"]["auto_snipe"] and not state["settings"]["demo_mode"]:
                await asyncio.sleep(2); continue
            loop_count += 1
            new_pairs = await fetch_new_pairs(); consecutive_errors = 0
            if loop_count % 12 == 0:
                log.info(f"Sniper #{loop_count}: {len(new_pairs)} pairs | demo={len(state['demo_positions'])} real={len(state['positions'])}")
            new_tokens = []
            for pair in new_pairs:
                base = pair.get("baseToken") or pair.get("token") or {}
                mint = base.get("address") or pair.get("tokenAddress","")
                if not mint or mint in state["seen_pairs"]: continue
                state["seen_pairs"][mint] = time.time()
                new_tokens.append((mint, pair))
            if not new_tokens: await asyncio.sleep(2); continue
            infos = await asyncio.gather(*[evaluate_new_token(p) for _,p in new_tokens], return_exceptions=True)
            for (mint, pair), info in zip(new_tokens, infos):
                if isinstance(info, Exception): continue
                if not info["passes_rules"]: continue
                if mint in state["positions"] or mint in state["demo_positions"]: continue
                s = state["settings"]
                if s["demo_mode"] and len(state["demo_positions"]) >= s.get("max_demo_positions",5):
                    log.info(f"Demo position limit reached, skipping {info['symbol']}"); continue
                if s["auto_snipe"] and not s["demo_mode"] and len(state["positions"]) >= s.get("max_real_positions",3):
                    log.info(f"Real position limit reached, skipping {info['symbol']}"); continue
                if s["auto_snipe"] and not s["demo_mode"]:
                    _reset_daily_pnl_if_needed()
                    if _check_daily_loss_limit():
                        paused_until = datetime.fromtimestamp(state["sniper_paused_until"], tz=timezone.utc)
                        log.info(f"Sniper paused until {paused_until:%H:%M UTC} (daily loss limit)")
                        continue
                if s.get("conviction_sizing") and ml_ready:
                    score = info["ml_score"]; base_amt = s["trade_amount"]
                    if score >= 0.80:   trade_amt = base_amt
                    elif score >= 0.65: trade_amt = round(base_amt * 0.75, 2)
                    else:               trade_amt = round(base_amt * 0.50, 2)
                else:
                    trade_amt = s["trade_amount"]
                age_min = info.get("age_sec",0)/60
                risks   = ", ".join([r.get("name","") for r in info["safety"].get("risks",[])]) or "None"
                notif = (
                    f"🔍 *New Token Detected*\n{'─'*24}\n🪙 *{info['symbol']}*\n"
                    f"├ Liquidity:  ${info['liquidity']:,.0f}\n"
                    f"├ Volume 24h: ${info['volume']:,.0f}\n"
                    f"├ Vol5m:      ${info.get('vol5m',0):,.0f} ({info.get('vol5m_pct',0):.1f}% of liq)\n"
                    f"├ Age:        {age_min:.1f} min\n"
                    f"├ Market Cap: ${info['market_cap']:,.0f}\n"
                    f"├ Price 1h:   {info['price_change']:+.1f}%\n"
                    f"├ ML Score:   {info['ml_score']:.0%} confidence\n"
                    f"├ RugCheck:   {info['rc_score']} (lower=safer)\n"
                    f"└ Risks:      {risks}\n"
                )
                if s["demo_mode"]:
                    price = await get_token_price(mint, pair_data=pair)
                    if price <= 0: continue
                    amt  = s["demo_trade_amount"]; fees = calc_fees(amt)
                    pos  = {"symbol": info["symbol"], "entry_price": price, "current_price": price,
                            "peak_price": price, "amount_usd": amt-fees["total"], "fees_paid": fees["total"],
                            "token_amount": (amt-fees["total"])/price,
                            "tp_hit": False, "features": info["features"], "ml_score": info["ml_score"],
                            "auto": True, "entry_time": time.time(), "peak_vol5m": 0.0,
                            "pt_early_done": False}
                    state["demo_positions"][mint] = pos
                    await db_save_position(mint, pos, True)
                    await _safe_notify(app, notif + f"\n📝 *DEMO Auto-bought @ ${price:.6f}*")
                elif s["auto_snipe"]:
                    if info["ml_score"] < s["min_score"]:
                        await _safe_notify(app, notif + f"\n⚠️ *Skipped — ML {info['ml_score']:.0%} < {s['min_score']:.0%}*")
                        continue
                    price = await get_token_price(mint, pair_data=pair)
                    if price <= 0: continue
                    sizing_note = ""
                    if s.get("conviction_sizing") and trade_amt < s["trade_amount"]:
                        sizing_note = f" _(conviction: ${trade_amt:.2f} @ {info['ml_score']:.0%})_"
                    await _safe_notify(app, notif + f"\n🤖 *Auto-sniping...{sizing_note}*")
                    result = await execute_buy(mint, trade_amt)
                    if result:
                        fees = calc_fees(trade_amt)
                        pos  = {"symbol": info["symbol"], "entry_price": price, "current_price": price,
                                "peak_price": price, "amount_usd": trade_amt-fees["total"],
                                "token_amount": result["out_amount"], "fees_paid": fees["total"],
                                "tp_hit": False, "features": info["features"], "auto": True,
                                "entry_time": time.time(), "peak_vol5m": 0.0,
                                "pt_early_done": False}
                        state["positions"][mint] = pos
                        await db_save_position(mint, pos, False)
                        await _safe_notify(app, f"✅ *Sniped {info['symbol']}!*\nEntry: ${price:.6f}\n"
                            f"🔗 [Solscan](https://solscan.io/tx/{result['signature']})")
                    else:
                        await _safe_notify(app, f"❌ Snipe failed for {info['symbol']}")
        except Exception as e:
            consecutive_errors += 1; log_error("auto_sniper_loop", e)
            if consecutive_errors >= 5:
                await notify_error(app, "auto_sniper_loop (5x)", e); consecutive_errors = 0
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
    price_history: dict = {}  # mint -> [(timestamp, price), ...]
    vol5m_history: dict = {}  # mint -> [vol5m, ...]

    while True:
        await asyncio.sleep(0.5)
        for is_demo, pool in [(False, state["positions"]), (True, state["demo_positions"])]:
            mints = list(pool.keys())
            if not mints: continue
            price_results = await asyncio.gather(*[get_token_price(m) for m in mints], return_exceptions=True)
            price_map = {m: (p if isinstance(p, float) and p > 0 else 0.0) for m, p in zip(mints, price_results)}

            for mint, pos in list(pool.items()):
                try:
                    price = price_map.get(mint, 0.0)
                    if price <= 0: continue
                    pos["current_price"] = price
                    entry = pos["entry_price"]; mult = price / entry
                    tp = state["settings"]["take_profit"]
                    sl = state["settings"]["stop_loss"]
                    s  = state["settings"]

                    # ── Price history (rolling 20 ticks) ────────────────────
                    if mint not in price_history: price_history[mint] = []
                    price_history[mint].append((time.time(), price))
                    if len(price_history[mint]) > 20: price_history[mint].pop(0)

                    if price > pos.get("peak_price", entry): pos["peak_price"] = price
                    peak_mult     = pos["peak_price"] / entry if entry > 0 else 1.0
                    # Use tightened trail if early exit already fired
                    early_trail   = pos.get("early_exit_trail")
                    post_tp_trail = early_trail if early_trail else _tiered_trail(peak_mult)
                    trailing_trigger = pos["peak_price"] * (1 - post_tp_trail)

                    # ── Breakeven stop ───────────────────────────────────────
                    be_mult = s.get("breakeven_mult", 2.0)
                    if mult >= be_mult and not pos.get("breakeven_active"):
                        pos["breakeven_active"] = True
                        pos["stop_loss_floor"]  = entry
                        await db_save_position(mint, pos, is_demo)
                        log.info(f"Breakeven activated for {pos['symbol']} @ {mult:.2f}x")

                    # ── NEW: Early profit-take (e.g. 1.5x) ──────────────────
                    early_mult = s.get("pt_early_mult", 1.5)
                    if (mult >= early_mult
                            and not pos.get("pt_early_done")
                            and s.get("pt_early_pct", 0) > 0):
                        await _early_profit_take(app, mint, pos, price, mult, is_demo)

                    # ── Tiered profit-take milestones (5x / 10x / 20x) ──────
                    for milestone, threshold in [("5x", 5.0), ("10x", 10.0), ("20x", 20.0)]:
                        if mult >= threshold and not pos.get(f"pt_{milestone}_done"):
                            await _tiered_profit_take(app, mint, pos, price, milestone, is_demo)

                    # ── Live 5m volume tracking ──────────────────────────────
                    b5m_live = 0.0; s5m_live = 0.0; vol5m_live = 0.0
                    now_ts = time.time()
                    if now_ts - pos.get("_last_td_fetch", 0) >= 5.2:
                        td = await get_token_data(mint)
                        if td: pos["_last_td_fetch"] = now_ts
                    else:
                        td = None
                    if td:
                        vol5m_live = float(td.get("volume",{}).get("m5",0) or 0)
                        b5m_live   = float(td.get("txns",{}).get("m5",{}).get("buys",0) or 0)
                        s5m_live   = float(td.get("txns",{}).get("m5",{}).get("sells",0) or 0)
                        if vol5m_live > pos.get("peak_vol5m", 0): pos["peak_vol5m"] = vol5m_live
                        if mint not in vol5m_history: vol5m_history[mint] = []
                        vol5m_history[mint].append(vol5m_live)
                        if len(vol5m_history[mint]) > 10: vol5m_history[mint].pop(0)

                    reason = None

                    # ── Hard stops (always active, no signal counting) ───────
                    if mult <= sl:
                        reason = f"🔴 Stop Loss at {mult:.2f}x"
                    elif pos.get("breakeven_active") and price <= entry:
                        reason = f"🔒 Breakeven Stop at {mult:.2f}x"

                    # ── Max hold time (force-exit stagnant positions) ─────────
                    elif s.get("max_hold_minutes"):
                        hold_secs = now_ts - pos.get("entry_time", now_ts)
                        if hold_secs > s["max_hold_minutes"] * 60:
                            reason = f"⏰ Max Hold Exit at {mult:.2f}x ({hold_secs/60:.0f}min)"

                    # ── Pre-TP fast trailing stop ────────────────────────────
                    # Activates as soon as mult >= pre_tp_trail_act_mult (e.g.
                    # 1.1x) and fires if price then drops pre_tp_trail_pct%
                    # from its peak. Runs BEFORE TP so it can catch a 1.3x→1.0x
                    # reversal without waiting for slow multi-signal votes.
                    # Disabled once TP is hit (post-TP uses tiered trail instead).
                    elif (not pos.get("tp_hit")
                            and s.get("pre_tp_trail_pct", 3.0) > 0
                            and mult >= s.get("pre_tp_trail_act_mult", 1.1)):
                        pre_tp_pct     = s["pre_tp_trail_pct"] / 100.0
                        pre_tp_trigger = pos["peak_price"] * (1.0 - pre_tp_pct)
                        if price <= pre_tp_trigger:
                            reason = (
                                f"⚡ Pre-TP Trail Exit at {mult:.2f}x "
                                f"(dropped {pre_tp_pct:.0%} from {peak_mult:.2f}x peak)"
                            )

                    # ── TP hit for first time — register BEFORE multi-signal ──
                    # Must come first in the elif chain so tp_hit is set on the
                    # same tick that price crosses TP. Multi-signal then runs
                    # freely on this tick AND all subsequent ticks (via the
                    # `or pos.get("tp_hit")` condition below).
                    elif mult >= tp and not pos.get("tp_hit"):
                        pos["tp_hit"] = True
                        trail_pct = int(_tiered_trail(mult) * 100)
                        pfx = "📝 DEMO " if is_demo else ""
                        if s.get("house_money_mode") and not pos.get("capital_recovered"):
                            ok = await _partial_sell_for_capital_recovery(app, mint, pos, price, is_demo)
                            await db_save_position(mint, pos, is_demo)
                            if not ok:
                                await _safe_notify(app,
                                    f"{pfx}🎯 *TP Hit — {pos['symbol']}* {mult:.2f}x\n"
                                    f"Trailing stop active! 🚀\n_{trail_pct}% tiered trail engaged_")
                        else:
                            await db_save_position(mint, pos, is_demo)
                            await _safe_notify(app,
                                f"{pfx}🎯 *TP Hit — {pos['symbol']}* {mult:.2f}x\n"
                                f"Trailing stop active! 🚀\n_{trail_pct}% tiered trail engaged_")

                    # ── Multi-signal dump detection (active above 1.1x) ───────
                    # Runs any time the position is in profit above 1.1x, OR
                    # after TP has been registered (pos["tp_hit"] is True).
                    # Because the TP-hit branch is now ABOVE this one in the
                    # elif chain, TP is always recorded on the crossing tick —
                    # multi-signal then catches dumps on that same tick and all
                    # subsequent ticks without interfering with TP registration.
                    elif mult >= 1.1 or pos.get("tp_hit"):
                        # ── House Money — trigger as soon as capital is
                        # recoverable (price covers invested + fees).
                        # No need to wait for TP — if we can sell enough
                        # tokens to get our money back right now, do it.
                        invested = pos["amount_usd"] + pos["fees_paid"]
                        position_value = pos.get("token_amount", 0) * price
                        can_recover = position_value >= invested * 1.05  # 5% buffer for sell fees
                        if (s.get("house_money_mode")
                                and not pos.get("capital_recovered")
                                and can_recover):
                            ok = await _partial_sell_for_capital_recovery(app, mint, pos, price, is_demo)
                            await db_save_position(mint, pos, is_demo)
                            if not ok:
                                log.warning(f"House money recovery failed for {pos['symbol']} — allowing exit")
                            else:
                                continue

                        # ── Multi-signal dump detection ──────────────────────
                        # Each of the 4 signals contributes 1 point.
                        # Exit if total >= multi_signal_exit_count.
                        # This prevents premature exits on single noisy signals.
                        signal_count = 0
                        signal_reasons = []

                        # Signal 1: Buy/sell ratio flip
                        flip_thresh = s.get("sell_ratio_flip_threshold", 1.2)
                        if b5m_live > 0 and s5m_live > b5m_live * flip_thresh:
                            signal_count += 1
                            signal_reasons.append(f"sell pressure ({s5m_live:.0f}>{b5m_live:.0f}*{flip_thresh})")

                        # Signal 2: Volume exhaustion
                        peak_v = pos.get("peak_vol5m", 0)
                        exhaust_pct = s.get("vol_exhaustion_pct", 50.0) / 100
                        if peak_v > 0 and vol5m_live > 0 and vol5m_live < peak_v * exhaust_pct:
                            signal_count += 1
                            signal_reasons.append(f"vol exhaustion ({vol5m_live:.0f}<{peak_v*exhaust_pct:.0f})")

                        # Signal 3: Momentum drop (price drop over last 5 ticks)
                        hist = price_history.get(mint, [])
                        momentum_drop = 0.0
                        if len(hist) >= 5:
                            old_p = hist[-5][1]
                            momentum_drop = (old_p - price) / old_p if old_p > 0 else 0
                            mom_thresh = s.get("momentum_exit_pct", 1.5) / 100
                            if momentum_drop > mom_thresh:
                                signal_count += 1
                                signal_reasons.append(f"momentum -{momentum_drop:.1%}")

                        # Signal 4: Price stagnation (price barely moved over N seconds)
                        stag_pct  = s.get("stagnation_pct", 2.0) / 100
                        stag_secs = s.get("stagnation_secs", 180)
                        hist_full = price_history.get(mint, [])
                        if len(hist_full) >= 2:
                            cutoff_ts = now_ts - stag_secs
                            old_prices = [p for ts, p in hist_full if ts <= cutoff_ts]
                            if old_prices:
                                baseline = old_prices[-1]
                                movement = abs(price - baseline) / baseline if baseline > 0 else 1.0
                                if movement < stag_pct:
                                    signal_count += 1
                                    signal_reasons.append(f"stagnation ({movement:.1%} in {stag_secs}s)")

                        required = s.get("multi_signal_exit_count", 2)
                        if signal_count >= required:
                            reason = f"🚨 Multi-Signal Exit at {mult:.2f}x [{', '.join(signal_reasons)}]"
                        elif price <= trailing_trigger:
                            reason = f"🟡 Trailing Stop at {mult:.2f}x ({int(post_tp_trail*100)}% trail from {peak_mult:.1f}x peak)"

                    if reason:
                        price_history.pop(mint, None)
                        vol5m_history.pop(mint, None)
                        await _close_position(app, mint, pos, price, reason, is_demo)
                except Exception as e:
                    log_error(f"monitor/{mint[:8]}", e)

# ============================================================
# CALLBACK HANDLER
# ============================================================
@auth
async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; data = q.data
    await q.answer()
    try:
        return await _button_handler_inner(update, ctx, q, data)
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"button_handler crash [{data}]: {e}\n{tb}")
        err_msg = f"⚠️ Error on `{data}`\n`{type(e).__name__}: {str(e)[:200]}`\n\nUse /start to return."
        try: await q.edit_message_text(err_msg, parse_mode="Markdown", reply_markup=kb_main())
        except Exception:
            try: await update.effective_chat.send_message(err_msg, parse_mode="Markdown", reply_markup=kb_main())
            except Exception: pass

async def _button_handler_inner(update, ctx, q, data):
    app = ctx.application
    if data == "main_menu":
        await q.edit_message_text(await build_dashboard(), parse_mode="Markdown", reply_markup=kb_main())

    elif data == "positions":
        if not state["positions"]:
            await q.edit_message_text("📭 No open positions.", reply_markup=kb_main())
        else:
            lines = []
            for mint, pos in state["positions"].items():
                p    = await get_token_price(mint)
                mult = p/pos["entry_price"] if p > 0 else 0
                pnl  = (mult-1)*pos["amount_usd"] - pos["fees_paid"]
                hm   = " 🏠" if pos.get("capital_recovered") else ""
                lines.append(f"{'🟢' if pnl>=0 else '🔴'} *{pos['symbol']}*{hm} | {mult:.2f}x | ${pnl:+.2f}")
            await q.edit_message_text("*📂 Open Positions*\n\n" + "\n".join(lines),
                parse_mode="Markdown", reply_markup=kb_positions(state["positions"]))

    elif data == "pnl":
        t = state["trades_history"]; wins = sum(1 for x in t if x["net_pnl"] > 0)
        await q.edit_message_text(
            f"{'📈' if state['total_pnl']>=0 else '📉'} *P&L Summary*\n\n"
            f"├ Real P&L:     *${state['total_pnl']:+.2f}*\n"
            f"├ Demo P&L:     ${state['demo_total_pnl']:+.2f}\n"
            f"├ Total Trades: {len(t)}\n├ Wins: {wins}\n└ Losses: {len(t)-wins}",
            parse_mode="Markdown", reply_markup=kb_main())

    elif data == "toggle_sniper":
        state["settings"]["auto_snipe"] = not state["settings"]["auto_snipe"]
        await db_save_settings()
        await q.edit_message_text(await build_dashboard(), parse_mode="Markdown", reply_markup=kb_main())

    elif data == "toggle_demo":
        state["settings"]["demo_mode"] = not state["settings"]["demo_mode"]
        await db_save_settings()
        await q.edit_message_text(await build_dashboard(), parse_mode="Markdown", reply_markup=kb_main())

    elif data == "toggle_house_money":
        state["settings"]["house_money_mode"] = not state["settings"].get("house_money_mode", True)
        await db_save_settings()
        status = "🟢 ON" if state["settings"]["house_money_mode"] else "🔴 OFF"
        await q.edit_message_text(
            f"⚙️ *Settings*\n\n🏠 House Money Mode turned *{status}*",
            parse_mode="Markdown", reply_markup=kb_settings())

    elif data == "toggle_ml_real_only":
        state["settings"]["ml_real_only"] = not state["settings"].get("ml_real_only", False)
        await db_save_settings()
        status = "🟢 ON" if state["settings"]["ml_real_only"] else "🔴 OFF"
        await q.edit_message_text(
            f"⚙️ *Settings*\n\n🧠 ML Real Only turned *{status}*",
            parse_mode="Markdown", reply_markup=kb_settings())

    elif data == "demo_menu":
        await q.edit_message_text("📝 *Demo Trading*", parse_mode="Markdown", reply_markup=kb_demo())

    elif data == "demostatus":
        if not state["demo_positions"]:
            await q.edit_message_text("📭 No demo positions.", reply_markup=kb_main())
        else:
            lines = []
            for mint, pos in state["demo_positions"].items():
                p    = await get_token_price(mint)
                mult = p/pos["entry_price"] if p > 0 else 0
                pnl  = (mult-1)*pos["amount_usd"] - pos["fees_paid"]
                proj = pnl*(state["settings"]["trade_amount"]/state["settings"]["demo_trade_amount"])
                tags = ("🎯" if pos.get("tp_hit") else "") + (" 🏠" if pos.get("capital_recovered") else "")
                lines.append(f"{'🟢' if pnl>=0 else '🔴'} *{pos['symbol']}* {tags} | {mult:.2f}x | ${pnl:+.2f} → *${proj:+.2f}*")
            await q.edit_message_text(
                f"📝 *Demo Positions*\n\n" + "\n".join(lines) +
                f"\n\n📊 Total: ${state['demo_total_pnl']:+.2f}",
                parse_mode="Markdown", reply_markup=kb_positions(state["demo_positions"], is_demo=True))

    elif data == "demohistory":
        if not state["demo_trades"]:
            await q.edit_message_text("📭 No demo history yet.", reply_markup=kb_main())
        else:
            lines = [f"{'✅' if t['net_pnl']>0 else '❌'} {t['symbol']} | {t['mult']:.2f}x | ${t['net_pnl']:+.2f} → ${t.get('projected_real',0):+.2f}"
                     for t in state["demo_trades"][-10:]]
            await q.edit_message_text(f"📝 *Demo History*\n\n" + "\n".join(lines) +
                f"\n\n📊 Total: ${state['demo_total_pnl']:+.2f}", parse_mode="Markdown", reply_markup=kb_main())

    elif data == "mlstats":
        from sklearn.metrics import precision_score, recall_score
        n = len(state["ml_features"])
        if not ml_ready or n == 0:
            wins_so_far = sum(state["ml_labels"]) if state["ml_labels"] else 0
            losses_so_far = n - wins_so_far
            await q.edit_message_text(
                f"🧠 *ML Model*\n\n*Not ready yet*\n\n"
                f"Samples so far: {n} ({int(wins_so_far)}W / {int(losses_so_far)}L)\n\n"
                f"Need 10+ samples with both wins and losses.",
                parse_mode="Markdown", reply_markup=kb_main())
        else:
            wins  = int(sum(state["ml_labels"]))
            Xs    = ml_scaler.transform(state["ml_features"])
            preds = ml_model.predict(Xs)
            prec  = precision_score(state["ml_labels"], preds, zero_division=0)
            rec   = recall_score(state["ml_labels"], preds, zero_division=0)
            nms   = ["Liq","Vol24h","Δ1h","Δ6h","Δ24h","Buys1h","Sells1h","B/S","Age","Mcap",
                     "Vol5m","Buys5m","Sells5m","B/S5m","Liq/Mcap","Vol/Liq","BuySizeRatio","Hour"]
            top   = sorted(zip(nms, ml_model.feature_importances_), key=lambda x: -x[1])[:5]
            ml_mode = "Real only" if state["settings"].get("ml_real_only") else "Demo + Real"
            await q.edit_message_text(
                f"🧠 *ML Model Stats*\n\n"
                f"├ Samples:   {n} ({ml_mode})\n├ Wins:      {wins}/{n}\n"
                f"├ Win rate:  {wins/n:.0%}\n├ Precision: {prec:.0%}\n└ Recall:    {rec:.0%}\n\n"
                f"*Top Features:*\n" + "\n".join([f"  {nm}: {v:.1%}" for nm,v in top]),
                parse_mode="Markdown", reply_markup=kb_main())

    elif data == "settings_menu":
        await q.edit_message_text("⚙️ *Settings*\n\nTap any setting to change it:",
            parse_mode="Markdown", reply_markup=kb_settings())

    elif data == "dump_detection_menu":
        s = state["settings"]
        await q.edit_message_text(
            f"🚨 *Dump Detection Settings*\n\n"
            f"*How multi-signal exit works:*\n"
            f"Each signal fires independently and scores 1 point.\n"
            f"The bot exits when the total reaches the threshold.\n\n"
            f"*Signals:*\n"
            f"├ 1️⃣ Sell Ratio Flip — sells > buys × {s.get('sell_ratio_flip_threshold',1.2)}\n"
            f"├ 2️⃣ Volume Exhaustion — vol5m < {s.get('vol_exhaustion_pct',50):.0f}% of peak\n"
            f"├ 3️⃣ Momentum Drop — price ->{s.get('momentum_exit_pct',1.5)}% in 5 ticks\n"
            f"└ 4️⃣ Price Stagnation — <{s.get('stagnation_pct',2.0)}% move in {s.get('stagnation_secs',60)}s\n\n"
            f"*Exit threshold:* {s.get('multi_signal_exit_count',2)} of 4 signals\n"
            f"_(1=aggressive, 2=balanced, 3=conservative)_",
            parse_mode="Markdown", reply_markup=kb_dump_detection())

    elif data == "set_pre_tp_trail":
        ctx.user_data["setting"] = "pre_tp_trail_pct"
        s = state["settings"]
        await q.edit_message_text(
            f"⚡ *Pre-TP Fast Trailing Stop*\n\n"
            f"Current: {s.get('pre_tp_trail_pct', 3.0)}% drop from peak\n"
            f"Activates at: {s.get('pre_tp_trail_act_mult', 1.1)}x\n\n"
            f"Once price rises above the activation multiplier, this trail locks in.\n"
            f"If price then drops this % from its peak, the bot sells immediately —\n"
            f"*before* waiting for multi-signal votes.\n\n"
            f"Example: peak=1.3x, trail=3% → exits at 1.26x\n\n"
            f"Set to *0* to disable. Recommended: 2–5%\nSend a value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_PRE_TP_TRAIL

    elif data == "set_pre_tp_trail_act":
        ctx.user_data["setting"] = "pre_tp_trail_act_mult"
        s = state["settings"]
        await q.edit_message_text(
            f"⚡ *Pre-TP Trail Activation Multiplier*\n\n"
            f"Current: {s.get('pre_tp_trail_act_mult', 1.1)}x\n\n"
            f"The pre-TP trailing stop only arms itself once price reaches this multiple.\n"
            f"Prevents the trail from firing on normal early noise.\n\n"
            f"Recommended: 1.05–1.2x\nSend a value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_PRE_TP_TRAIL_ACT

    elif data == "set_multi_signal_cnt":
        ctx.user_data["setting"] = "multi_signal_exit_count"
        await q.edit_message_text(
            f"🔢 *Signals Needed to Exit*\nCurrent: {state['settings'].get('multi_signal_exit_count',2)}\n\n"
            f"How many dump signals must fire together before the bot exits.\n"
            f"1 = any single signal (aggressive)\n"
            f"2 = 2-of-4 signals (balanced — recommended)\n"
            f"3 = 3-of-4 signals (conservative)\n"
            f"4 = all 4 signals (very conservative)\n\nSend 1, 2, 3, or 4:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_MULTI_SIGNAL_CNT

    elif data == "set_momentum_pct":
        ctx.user_data["setting"] = "momentum_exit_pct"
        await q.edit_message_text(
            f"⚡ *Momentum Exit %*\nCurrent: {state['settings'].get('momentum_exit_pct',1.5)}%\n\n"
            f"Exit if price drops this % over 5 monitor ticks.\n"
            f"Lower = more sensitive. Recommended: 1.5–3.0%\nSend a value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_MOMENTUM_PCT

    elif data == "set_vol_exhaust":
        ctx.user_data["setting"] = "vol_exhaustion_pct"
        await q.edit_message_text(
            f"📉 *Volume Exhaustion %*\nCurrent: {state['settings'].get('vol_exhaustion_pct',50.0):.0f}%\n\n"
            f"Exit if 5m volume drops below this % of its peak.\nRecommended: 40–60%\nSend a value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_TRAIL  # reuse handler

    elif data == "set_sell_ratio":
        ctx.user_data["setting"] = "sell_ratio_flip_threshold"
        await q.edit_message_text(
            f"🚨 *Sell Ratio Flip*\nCurrent: {state['settings'].get('sell_ratio_flip_threshold',1.2)}\n\n"
            f"Exit if sells > buys × this threshold.\nLower = more sensitive. Recommended: 1.1–1.5\nSend a value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_SELL_RATIO

    elif data == "set_stagnation_pct":
        ctx.user_data["setting"] = "stagnation_pct"
        await q.edit_message_text(
            f"⏸ *Stagnation % Threshold*\nCurrent: {state['settings'].get('stagnation_pct',2.0)}%\n\n"
            f"If price hasn't moved more than this % in the observation window, count as a dump signal.\n"
            f"Recommended: 1.5–3.0%\nSend a value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_STAGNATION_PCT

    elif data == "set_stagnation_secs":
        ctx.user_data["setting"] = "stagnation_secs"
        await q.edit_message_text(
            f"⏱ *Stagnation Window (seconds)*\nCurrent: {state['settings'].get('stagnation_secs',60)}s\n\n"
            f"How long to observe price for stagnation before the signal fires.\n"
            f"Recommended: 45–120s for meme coins\nSend a value in seconds:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_STAGNATION_SECS

    elif data == "set_max_hold":
        ctx.user_data["setting"] = "max_hold_minutes"
        await q.edit_message_text(
            f"⏰ *Max Hold Minutes*\nCurrent: {state['settings'].get('max_hold_minutes',120)}min\n\n"
            f"Force-exit any position held longer than this.\nSet to 0 to disable.\nSend a value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_MAX_HOLD

    elif data == "tiered_trail_menu":
        s = state["settings"]
        await q.edit_message_text(
            f"📐 *Tiered Trailing Stop & Profit-Take*\n\n"
            f"*Early Profit-Take (NEW):*\n"
            f"└ @ {s.get('pt_early_mult',1.5)}x → sell {s.get('pt_early_pct',30.0):.0f}% then tighten trail\n\n"
            f"*Trailing Stop:*\n"
            f"├ Peak ≥ 5x  → {s.get('trail_5x',4.0)}% trail\n"
            f"├ Peak ≥ 10x → {s.get('trail_10x',3.0)}% trail\n"
            f"├ Peak ≥ 20x → {s.get('trail_20x',2.0)}% trail\n"
            f"└ Peak ≥ 50x → {s.get('trail_50x',1.5)}% trail\n\n"
            f"*Standard Profit-Take Milestones:*\n"
            f"├ @ 5x  → sell {s.get('pt_5x_pct',25.0):.0f}%\n"
            f"├ @ 10x → sell {s.get('pt_10x_pct',25.0):.0f}%\n"
            f"└ @ 20x → sell {s.get('pt_20x_pct',25.0):.0f}%",
            parse_mode="Markdown", reply_markup=kb_tiered_trail())

    elif data == "set_pt_early":
        ctx.user_data["setting"] = "pt_early_pct"
        s = state["settings"]
        await q.edit_message_text(
            f"⚡ *Early Profit-Take % (at {s.get('pt_early_mult',1.5)}x)*\n"
            f"Current: {s.get('pt_early_pct',30.0):.0f}%\n\n"
            f"Sell this % of position when price hits the early multiplier.\n"
            f"Set to 0 to disable. Recommended: 25–40%\nSend a value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_PT_EARLY

    elif data == "set_pt_early_mult":
        ctx.user_data["setting"] = "pt_early_mult"
        await q.edit_message_text(
            f"⚡ *Early Profit-Take Multiplier*\n"
            f"Current: {state['settings'].get('pt_early_mult',1.5)}x\n\n"
            f"Trigger the early sell at this multiplier.\nRecommended: 1.3–2.0x\nSend a value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_PT_EARLY_MULT

    elif data == "set_tp":
        ctx.user_data["setting"] = "take_profit"
        await q.edit_message_text(f"🎯 *Take Profit*\nCurrent: {state['settings']['take_profit']}x\n\nSend new multiplier:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_TP

    elif data == "set_trail":
        ctx.user_data["setting"] = "trailing_stop"
        await q.edit_message_text(f"📉 *Trailing Stop*\nCurrent: {state['settings']['trailing_stop']}%\n\nSend new %:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_TRAIL

    elif data == "set_stop":
        ctx.user_data["setting"] = "stop_loss"
        await q.edit_message_text(f"🛑 *Stop Loss*\nCurrent: {state['settings']['stop_loss']}x\n\nSend new multiplier:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_STOP

    elif data == "set_amount":
        ctx.user_data["setting"] = "trade_amount"
        await q.edit_message_text(f"💵 *Trade Amount*\nCurrent: ${state['settings']['trade_amount']}\n\nSend new USD amount:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_AMOUNT

    elif data == "set_entry_slip":
        ctx.user_data["setting"] = "entry_slippage_bps"
        cur = state["settings"].get("entry_slippage_bps", state["settings"]["slippage_bps"])
        await q.edit_message_text(f"⚡ *Entry Slippage*\nCurrent: {cur}bps\n\nSend new bps (100 = 1%):",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_ENTRY_SLIP

    elif data == "set_exit_slip":
        ctx.user_data["setting"] = "exit_slippage_bps"
        cur = state["settings"].get("exit_slippage_bps", 200)
        await q.edit_message_text(f"⚡ *Exit Slippage*\nCurrent: {cur}bps\n\nSend new bps (100 = 1%):",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_EXIT_SLIP

    elif data == "set_score":
        ctx.user_data["setting"] = "min_score"
        await q.edit_message_text(f"🧠 *Min ML Score*\nCurrent: {state['settings']['min_score']:.0%}\n\nSend 0–1 (e.g. 0.65):",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_SCORE

    elif data == "set_liq":
        ctx.user_data["setting"] = "min_liquidity"
        await q.edit_message_text(f"💧 *Min Liquidity*\nCurrent: ${state['settings']['min_liquidity']:,}\n\nSend new USD value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_LIQ

    elif data == "set_rugcheck":
        ctx.user_data["setting"] = "min_rugcheck"
        await q.edit_message_text(f"🛡️ *Max RugCheck Score*\nCurrent: {state['settings']['min_rugcheck']}\n\nLower=safer. Send new value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_RUGCHECK

    elif data == "set_min_age":
        ctx.user_data["setting"] = "min_token_age_sec"
        await q.edit_message_text(f"⏱ *Min Token Age*\nCurrent: {state['settings'].get('min_token_age_sec',120)}s\n\nSend value in seconds:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_MIN_AGE

    elif data == "set_vol5m":
        ctx.user_data["setting"] = "min_vol5m_pct"
        await q.edit_message_text(f"📊 *Min 5m Volume %*\nCurrent: {state['settings'].get('min_vol5m_pct',10)}%\n\nSend new value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_VOL5M

    elif data == "set_max_demo":
        ctx.user_data["setting"] = "max_demo_positions"
        await q.edit_message_text(f"📂 *Max Demo Positions*\nCurrent: {state['settings'].get('max_demo_positions',5)}\n\nSend new value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_MAX_DEMO

    elif data == "set_max_real":
        ctx.user_data["setting"] = "max_real_positions"
        await q.edit_message_text(f"📂 *Max Real Positions*\nCurrent: {state['settings'].get('max_real_positions',3)}\n\nSend new value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_MAX_REAL

    elif data == "set_trail_5x":
        ctx.user_data["setting"] = "trail_5x"
        await q.edit_message_text(f"🟢 *Trail % at ≥5x*\nCurrent: {state['settings'].get('trail_5x',4.0)}%\n\nSend new %:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_TRAIL_5X

    elif data == "set_trail_10x":
        ctx.user_data["setting"] = "trail_10x"
        await q.edit_message_text(f"🟡 *Trail % at ≥10x*\nCurrent: {state['settings'].get('trail_10x',3.0)}%\n\nSend new %:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_TRAIL_10X

    elif data == "set_trail_20x":
        ctx.user_data["setting"] = "trail_20x"
        await q.edit_message_text(f"🟠 *Trail % at ≥20x*\nCurrent: {state['settings'].get('trail_20x',2.0)}%\n\nSend new %:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_TRAIL_20X

    elif data == "set_trail_50x":
        ctx.user_data["setting"] = "trail_50x"
        await q.edit_message_text(f"🔴 *Trail % at ≥50x*\nCurrent: {state['settings'].get('trail_50x',1.5)}%\n\nSend new %:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_TRAIL_50X

    elif data == "set_pt_5x":
        ctx.user_data["setting"] = "pt_5x_pct"
        await q.edit_message_text(f"💰 *Profit-Take @ 5x*\nCurrent: {state['settings'].get('pt_5x_pct',25.0):.0f}%\n\nSend new %:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_PT_5X

    elif data == "set_pt_10x":
        ctx.user_data["setting"] = "pt_10x_pct"
        await q.edit_message_text(f"💰 *Profit-Take @ 10x*\nCurrent: {state['settings'].get('pt_10x_pct',25.0):.0f}%\n\nSend new %:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_PT_10X

    elif data == "set_pt_20x":
        ctx.user_data["setting"] = "pt_20x_pct"
        await q.edit_message_text(f"💰 *Profit-Take @ 20x*\nCurrent: {state['settings'].get('pt_20x_pct',25.0):.0f}%\n\nSend new %:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_PT_20X

    elif data == "set_be_mult":
        ctx.user_data["setting"] = "breakeven_mult"
        await q.edit_message_text(
            f"⚖️ *Breakeven Stop Multiplier*\nCurrent: {state['settings'].get('breakeven_mult',2.0)}x\n\n"
            f"Move stop to entry once price reaches this multiple.\nSend new value:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_BE_MULT

    elif data == "set_daily_loss":
        ctx.user_data["setting"] = "daily_loss_limit_pct"
        await q.edit_message_text(
            f"🚨 *Daily Loss Limit*\nCurrent: {state['settings'].get('daily_loss_limit_pct',20)}%\n\nSend new %:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_DAILY_LOSS

    elif data == "toggle_conviction_sizing":
        state["settings"]["conviction_sizing"] = not state["settings"].get("conviction_sizing", True)
        await db_save_settings()
        status = "🟢 ON" if state["settings"]["conviction_sizing"] else "🔴 OFF"
        await q.edit_message_text(
            f"⚙️ *Settings*\n\n📐 Conviction Sizing turned *{status}*",
            parse_mode="Markdown", reply_markup=kb_settings())

    elif data == "noop":
        pass

    elif data == "wallet":
        await q.edit_message_text("⏳ Fetching wallet...", parse_mode="Markdown")
        w = await get_wallet_balance()
        lines = [
            f"*💼 Wallet Balance*\n{'─'*24}",
            f"├ SOL:  {w['sol']:.4f} (${w['sol_usd']:.2f})",
            f"├ USDC: ${w['usdc']:.2f}",
        ]
        for tok in w["tokens"][:5]:
            lines.append(f"├ {tok['symbol']}: {tok['amount']:,.0f} (${tok['usd']:.2f})")
        lines.append(f"└ *Total: ${w['total_usd']:.2f}*")
        await q.edit_message_text("\n".join(lines), parse_mode="Markdown", reply_markup=kb_main())

    elif data == "buy_prompt":
        ctx.user_data["action"] = "buy"
        await q.edit_message_text("🛒 *Buy Token*\n\nSend the token mint address:",
            parse_mode="Markdown", reply_markup=kb_main()); return WAITING_BUY_MINT

    elif data == "sell_prompt":
        if not state["positions"]:
            await q.edit_message_text("📭 No open positions to sell.", reply_markup=kb_main())
        else:
            await q.edit_message_text("💸 *Sell Token*\n\nSelect a position:",
                parse_mode="Markdown", reply_markup=kb_positions(state["positions"]))

    elif data.startswith("sell_confirm:"):
        mint = data.split(":")[1]
        if mint not in state["positions"]:
            await q.edit_message_text("❌ Position not found.", reply_markup=kb_main()); return
        pos = state["positions"][mint]
        await q.edit_message_text(
            f"💸 *Confirm Sell*\n\nToken: *{pos['symbol']}*\nEntry: ${pos['entry_price']:.6f}\n\nAre you sure?",
            parse_mode="Markdown", reply_markup=kb_confirm_sell(mint))

    elif data.startswith("confirm_sell:"):
        mint = data.split(":")[1]
        if mint not in state["positions"]:
            await q.edit_message_text("❌ Position not found.", reply_markup=kb_main()); return
        pos = state["positions"][mint]
        await q.edit_message_text(f"⏳ Selling {pos['symbol']}...", parse_mode="Markdown")
        price = await get_token_price(mint) or pos["entry_price"]
        await _close_position(app, mint, pos, price, "✂️ Manual Sell", False)
        await q.edit_message_text(await build_dashboard(), parse_mode="Markdown", reply_markup=kb_main())

    elif data.startswith("dsell:"):
        mint = data.split(":")[1]
        if mint not in state["demo_positions"]:
            await q.edit_message_text("❌ Demo position not found.", reply_markup=kb_main()); return
        pos   = state["demo_positions"][mint]
        price = await get_token_price(mint)
        if price <= 0:
            await q.edit_message_text("❌ Price unavailable.", reply_markup=kb_main()); return
        mult = price / pos["entry_price"]
        await q.edit_message_text(
            f"💸 *Close Demo Position*\n\n*{pos['symbol']}*\nCurrent: ${price:.6f} ({mult:.2f}x)\n\nConfirm?",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Confirm", callback_data=f"dclose_confirm:{mint}"),
                 InlineKeyboardButton("❌ Cancel",  callback_data="demostatus")]]))

    elif data.startswith("dclose_confirm:"):
        mint = data.split(":")[1]
        if mint not in state["demo_positions"]:
            await q.edit_message_text("❌ Demo position not found.", reply_markup=kb_main()); return
        pos   = state["demo_positions"][mint]
        price = await get_token_price(mint) or pos["entry_price"]
        await q.edit_message_text(f"⏳ Closing {pos['symbol']}...", parse_mode="Markdown")
        await _close_position(app, mint, pos, price, "✂️ Manual Close", True)
        await q.edit_message_text(await build_dashboard(), parse_mode="Markdown", reply_markup=kb_main())

    elif data.startswith("dclose_now:"):
        mint = data.split(":")[1]
        if mint not in state["demo_positions"]:
            await q.edit_message_text("❌ Demo position not found.", reply_markup=kb_main()); return
        pos   = state["demo_positions"][mint]
        price = await get_token_price(mint) or pos["entry_price"]
        await q.edit_message_text(f"✂️ Taking profit on {pos['symbol']}...", parse_mode="Markdown")
        await _close_position(app, mint, pos, price, "✂️ Manual Take Profit", True)
        await q.edit_message_text(await build_dashboard(), parse_mode="Markdown", reply_markup=kb_main())

    elif data.startswith("confirm_buy:"):
        parts  = data.split(":"); mint, symbol = parts[1], parts[2]
        amt    = state["settings"]["trade_amount"]; fees = calc_fees(amt)
        await q.edit_message_text(f"⏳ Buying {symbol}...", parse_mode="Markdown")
        price  = await get_token_price(mint)
        if price <= 0:
            await q.edit_message_text("❌ Price unavailable.", reply_markup=kb_main()); return
        result = await execute_buy(mint, amt)
        if not result:
            await q.edit_message_text("❌ Buy failed.", reply_markup=kb_main()); return
        td  = await get_token_data(mint)
        pos = {"symbol": symbol, "entry_price": price, "current_price": price, "peak_price": price,
               "amount_usd": amt-fees["total"], "token_amount": result["out_amount"],
               "fees_paid": fees["total"], "tp_hit": False,
               "features": extract_features(td) if td else [0.0]*18, "auto": False,
               "entry_time": time.time(), "peak_vol5m": 0.0, "pt_early_done": False}
        state["positions"][mint] = pos; await db_save_position(mint, pos, False)
        await q.edit_message_text(await build_dashboard() + f"\n\n✅ *Bought {symbol} @ ${price:.6f}*\n"
            f"🔗 [Solscan](https://solscan.io/tx/{result['signature']})",
            parse_mode="Markdown", reply_markup=kb_main(), disable_web_page_preview=True)

    elif data == "confirm_buy_pending":
        await handle_confirm_buy(update, ctx)

    elif data == "health":
        a  = state["api_stats"]
        pt = a["price_ok"]+a["price_fail"]; qt = a["quote_ok"]+a["quote_fail"]; st = a["swap_ok"]+a["swap_fail"]
        err_txt = "\n".join([
            f"  [{escape_md(e['time'])}] {escape_md(e['context'])}: {escape_md(e['error'])}"
            for e in state["errors"][-3:]]) or "  None ✅"
        s = state["settings"]
        await q.edit_message_text(
            f"🏥 *Health*\n\n*API Rates:*\n"
            f"├ Price: {a['price_ok']}/{pt} ({a['price_ok']/max(pt,1):.0%})\n"
            f"├ Quote: {a['quote_ok']}/{qt} ({a['quote_ok']/max(qt,1):.0%})\n"
            f"├ Swap:  {a['swap_ok']}/{st} ({a['swap_ok']/max(st,1):.0%})\n"
            f"├ Confirm: {a['confirm_ok']} ok | {a['confirm_timeout']} timeout\n"
            f"├ RPC Reconnects: {a.get('rpc_reconnects',0)}\n\n"
            f"*State:*\n"
            f"├ Positions: {len(state['positions'])} real / {len(state['demo_positions'])} demo\n"
            f"├ ML: {len(state['ml_features'])} samples | {'Ready' if ml_ready else 'Training'}\n"
            f"└ DB: {'✅ Connected' if db_pool else '❌ Disconnected'}\n\n"
            f"*Recent Errors:*\n{err_txt}",
            parse_mode="Markdown", reply_markup=kb_main())

    elif data == "pnl_breakdown":
        now_ts = time.time()
        windows = [("1h", 3600), ("3h", 10800), ("6h", 21600), ("8h", 28800), ("24h", 86400)]
        def window_stats(trades, seconds, include_proj=False):
            cutoff = now_ts - seconds
            t = [x for x in trades if x.get("closed_at", 0) >= cutoff]
            if not t: return "None"
            pnl  = sum(x["net_pnl"] for x in t)
            wins = sum(1 for x in t if x["net_pnl"] > 0)
            base = f"{len(t)} trades ({wins}W/{len(t)-wins}L) *${pnl:+.2f}*"
            if include_proj:
                proj = sum(x.get("projected_real", 0) for x in t)
                base += f" _(real≈${proj:+.2f})_"
            return base
        real_lines = "\n".join([f"├ {lbl}: {window_stats(state['trades_history'], secs)}" for lbl, secs in windows])
        demo_lines = "\n".join([f"├ {lbl}: {window_stats(state['demo_trades'], secs, True)}" for lbl, secs in windows])
        await q.edit_message_text(
            f"📈 *P&L Breakdown*\n\n*💰 Real:*\n{real_lines}\n\n*📝 Demo:*\n{demo_lines}",
            parse_mode="Markdown", reply_markup=kb_main())

    elif data == "history":
        if not state["trades_history"]:
            await q.edit_message_text("📭 No history yet.", reply_markup=kb_main())
        else:
            lines = [f"{'✅' if t['net_pnl']>0 else '❌'} {t['symbol']} | {t['mult']:.2f}x | ${t['net_pnl']:+.2f}"
                     for t in state["trades_history"][-10:]]
            await q.edit_message_text("*📜 Last 10 Trades*\n\n" + "\n".join(lines),
                parse_mode="Markdown", reply_markup=kb_main())

# ============================================================
# MESSAGE HANDLERS
# ============================================================
async def handle_setting_input(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    key = ctx.user_data.get("setting"); txt = update.message.text.strip()
    try:
        val = float(txt)
        int_keys = ("slippage_bps","entry_slippage_bps","exit_slippage_bps",
                    "max_demo_positions","max_real_positions","min_token_age_sec",
                    "max_hold_minutes","multi_signal_exit_count","stagnation_secs")
        state["settings"][key] = int(val) if key in int_keys else val
        await db_save_settings()
        await update.message.reply_text(f"✅ *{key.replace('_',' ').title()}* updated to `{txt}`",
            parse_mode="Markdown", reply_markup=kb_main())
    except ValueError:
        await update.message.reply_text("❌ Invalid value. Send a number.", reply_markup=kb_main())
    return ConversationHandler.END

async def handle_buy_mint(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["buy_mint"] = update.message.text.strip()
    await update.message.reply_text("Now send the *token symbol* (e.g. `PEPE`):", parse_mode="Markdown")
    return WAITING_BUY_SYMBOL

async def handle_buy_symbol(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    symbol  = update.message.text.strip().upper()
    mint    = ctx.user_data.get("buy_mint","")
    is_demo = state["settings"]["demo_mode"]
    amt     = state["settings"]["demo_trade_amount"] if is_demo else state["settings"]["trade_amount"]
    fees    = calc_fees(amt)
    try:    price = await get_token_price(mint)
    except: price = 0.0
    ctx.user_data["pending_buy_mint"]   = mint
    ctx.user_data["pending_buy_symbol"] = symbol
    await update.message.reply_text(
        f"{'📝 *DEMO — No real USDC spent*\n\n' if is_demo else ''}🛒 *Confirm Buy*\n\n"
        f"Token:  *{symbol}*\nMint:   `{mint[:16]}...`\n"
        f"Price:  {'$'+f'{price:.6f}' if price > 0 else '⚠️ Fetching at confirm'}\n"
        f"Invest: ${amt:.2f}\nFees:   -${fees['total']}\n\nTap confirm to execute:",
        parse_mode="Markdown", reply_markup=kb_confirm_buy(mint, symbol))
    return WAITING_CONFIRM_BUY

async def handle_confirm_buy(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try: await q.answer()
    except Exception: pass
    if q.data != "confirm_buy_pending":
        await q.edit_message_text(await build_dashboard(), parse_mode="Markdown", reply_markup=kb_main())
        return ConversationHandler.END
    mint   = ctx.user_data.get("pending_buy_mint","")
    symbol = ctx.user_data.get("pending_buy_symbol","?")
    if not mint:
        await q.edit_message_text("❌ Session expired. Tap Buy Token again.", reply_markup=kb_main())
        return ConversationHandler.END
    is_demo = state["settings"]["demo_mode"]
    amt     = state["settings"]["demo_trade_amount"] if is_demo else state["settings"]["trade_amount"]
    fees    = calc_fees(amt)
    await q.edit_message_text(f"⏳ {'[DEMO] ' if is_demo else ''}Buying {symbol}...", parse_mode="Markdown")
    price = await get_token_price(mint)
    if price <= 0 and is_demo:
        await q.edit_message_text("❌ Price unavailable for demo trade.", reply_markup=kb_main())
        return ConversationHandler.END
    if is_demo:
        td  = await get_token_data(mint)
        pos = {"symbol": symbol, "entry_price": price, "current_price": price, "peak_price": price,
               "amount_usd": amt-fees["total"], "token_amount": (amt-fees["total"])/price,
               "fees_paid": fees["total"], "tp_hit": False,
               "features": extract_features(td) if td else [0.0]*18, "auto": False,
               "entry_time": time.time(), "peak_vol5m": 0.0, "pt_early_done": False}
        state["demo_positions"][mint] = pos; await db_save_position(mint, pos, True)
        await q.edit_message_text(await build_dashboard() + f"\n\n📝 *[DEMO] Bought {symbol} @ ${price:.6f}*",
            parse_mode="Markdown", reply_markup=kb_main())
    else:
        result = await execute_buy(mint, amt)
        if not result:
            await q.edit_message_text("❌ Buy failed. Check USDC balance or increase slippage.",
                parse_mode="Markdown", reply_markup=kb_main())
            return ConversationHandler.END
        td  = await get_token_data(mint)
        pos = {"symbol": symbol, "entry_price": price, "current_price": price, "peak_price": price,
               "amount_usd": amt-fees["total"], "token_amount": result["out_amount"],
               "fees_paid": fees["total"], "tp_hit": False,
               "features": extract_features(td) if td else [0.0]*18, "auto": False,
               "entry_time": time.time(), "peak_vol5m": 0.0, "pt_early_done": False}
        state["positions"][mint] = pos; await db_save_position(mint, pos, False)
        await q.edit_message_text(await build_dashboard() + f"\n\n✅ *Bought {symbol} @ ${price:.6f}*\n"
            f"🔗 [Solscan](https://solscan.io/tx/{result['signature']})",
            parse_mode="Markdown", reply_markup=kb_main(), disable_web_page_preview=True)
    return ConversationHandler.END

# ============================================================
# /start
# ============================================================
@auth
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(await build_dashboard(), parse_mode="Markdown", reply_markup=kb_main())

# ============================================================
# MAIN
# ============================================================
async def post_init(app):
    await init_db(); await load_all_from_db()
    asyncio.create_task(monitor_positions(app))
    asyncio.create_task(auto_sniper_loop(app))
    if HELIUS_RPC:
        asyncio.create_task(raydium_ws_sniper(app))
        log.info("Raydium WS sniper started")
    log.info("All systems go")
    await _safe_notify(app, "🚀 *Bot restarted — all state restored.*\n\nSend /start to open the dashboard.")

async def raydium_ws_sniper(app):
    try: import websockets
    except ImportError: log.warning("pip install websockets"); return
    wss_url = HELIUS_RPC.replace("https://","wss://").replace("http://","wss://")
    backoff = 2
    while True:
        try:
            async with websockets.connect(wss_url, ping_interval=20, ping_timeout=30) as ws:
                backoff = 2
                await ws.send(json.dumps({"jsonrpc":"2.0","id":1,"method":"logsSubscribe",
                    "params":[{"mentions":[RAYDIUM_PROGRAM]},{"commitment":"confirmed"}]}))
                async for raw in ws:
                    try:
                        msg  = json.loads(raw)
                        logs = msg.get("params",{}).get("result",{}).get("value",{}).get("logs",[])
                        if not any("initialize" in l.lower() for l in logs): continue
                        await asyncio.sleep(4)
                        for line in logs:
                            for part in line.split():
                                if len(part) in (43,44) and part not in state["seen_pairs"]:
                                    state["seen_pairs"][part] = time.time()
                                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as sess:
                                        pd = await _fetch_full_pair(sess, part)
                                        if pd:
                                            info = await evaluate_new_token(pd)
                                            if info["passes_rules"]:
                                                await _handle_snipe(app, part, pd, info)
                    except Exception as e: log_error("raydium_ws/msg", e)
        except Exception as e:
            log_error("raydium_ws/connect", e)
            await asyncio.sleep(backoff); backoff = min(backoff*2, 60)

async def _handle_snipe(app, mint, pair, info):
    risks = ", ".join([r.get("name","") for r in info["safety"].get("risks",[])]) or "None"
    age_min = info.get("age_sec", 0) / 60
    notif = (
        f"⚡ *New Pool Detected*\n{'─'*24}\n💹 *{info['symbol']}*\n"
        f"├ Liquidity:  ${info['liquidity']:,.0f}\n"
        f"├ Vol5m:      ${info.get('vol5m',0):,.0f} ({info.get('vol5m_pct',0):.1f}%)\n"
        f"├ Age:        {age_min:.1f} min\n"
        f"├ ML Score:   {info['ml_score']:.0%}\n"
        f"├ RugCheck:   {info['rc_score']}\n└ Risks:      {risks}\n"
    )
    if state["settings"]["demo_mode"]:
        price = await get_token_price(mint, pair_data=pair)
        if price <= 0: return
        amt  = state["settings"]["demo_trade_amount"]; fees = calc_fees(amt)
        pos  = {"symbol": info["symbol"], "entry_price": price, "current_price": price,
                "peak_price": price, "amount_usd": amt-fees["total"], "fees_paid": fees["total"],
                "token_amount": (amt-fees["total"])/price,
                "tp_hit": False, "features": info["features"], "ml_score": info["ml_score"],
                "auto": True, "entry_time": time.time(), "peak_vol5m": 0.0, "pt_early_done": False}
        state["demo_positions"][mint] = pos
        await db_save_position(mint, pos, True)
        await _safe_notify(app, notif + f"\n📝 *DEMO Auto-bought @ ${price:.6f}*")
    elif state["settings"]["auto_snipe"]:
        price = await get_token_price(mint, pair_data=pair)
        if price <= 0: return
        await _safe_notify(app, notif + "\n🤖 *Auto-sniping...*")
        result = await execute_buy(mint, state["settings"]["trade_amount"])
        if result:
            amt  = state["settings"]["trade_amount"]; fees = calc_fees(amt)
            pos  = {"symbol": info["symbol"], "entry_price": price, "current_price": price,
                    "peak_price": price, "amount_usd": amt-fees["total"],
                    "token_amount": result["out_amount"], "fees_paid": fees["total"],
                    "tp_hit": False, "features": info["features"], "auto": True,
                    "entry_time": time.time(), "peak_vol5m": 0.0, "pt_early_done": False}
            state["positions"][mint] = pos
            await db_save_position(mint, pos, False)
            await _safe_notify(app, f"✅ *Sniped {info['symbol']}!*\nEntry: ${price:.6f}\n"
                f"🔗 [Solscan](https://solscan.io/tx/{result['signature']})")
        else:
            await _safe_notify(app, f"❌ Snipe failed for {info['symbol']}")

def main():
    validate_config()
    global keypair, solana_client
    keypair       = Keypair.from_bytes(base58.b58decode(PRIVATE_KEY_B58))
    solana_client = AsyncClient(RPC_URL, commitment=Confirmed)
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(button_handler)],
        states={
            WAITING_BUY_MINT:           [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_buy_mint)],
            WAITING_BUY_SYMBOL:         [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_buy_symbol)],
            WAITING_CONFIRM_BUY:        [CallbackQueryHandler(handle_confirm_buy)],
            WAITING_SET_TP:             [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_TRAIL:          [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_STOP:           [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_AMOUNT:         [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_SLIP:           [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_ENTRY_SLIP:     [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_EXIT_SLIP:      [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_SCORE:          [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_LIQ:            [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_RUGCHECK:       [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_TRAIL_5X:       [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_TRAIL_10X:      [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_TRAIL_20X:      [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_TRAIL_50X:      [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_MIN_AGE:        [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_VOL5M:          [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_MAX_DEMO:       [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_MAX_REAL:       [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_PT_5X:          [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_PT_10X:         [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_PT_20X:         [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_DAILY_LOSS:     [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_BE_MULT:        [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_MOMENTUM_PCT:   [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_SELL_RATIO:     [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_MAX_HOLD:       [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_PT_EARLY:       [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_PT_EARLY_MULT:  [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_STAGNATION_PCT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_STAGNATION_SECS:[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_MULTI_SIGNAL_CNT:[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_PRE_TP_TRAIL:    [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_PRE_TP_TRAIL_ACT:[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
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