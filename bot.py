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
RAYDIUM_PROGRAM   = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

# Conversation states
WAITING_BUY_MINT      = 0
WAITING_BUY_SYMBOL    = 1
WAITING_SELL_MINT     = 2
WAITING_CONFIRM_BUY   = 3
WAITING_SET_TP        = 10
WAITING_SET_TRAIL     = 11
WAITING_SET_STOP      = 12
WAITING_SET_AMOUNT    = 13
WAITING_SET_SLIP      = 14
WAITING_SET_SCORE     = 15
WAITING_SET_LIQ       = 16
WAITING_SET_RUGCHECK  = 17
WAITING_SET_TRAIL_5X  = 20
WAITING_SET_TRAIL_10X = 21
WAITING_SET_TRAIL_20X = 22
WAITING_SET_TRAIL_50X = 23
WAITING_SET_MIN_AGE   = 24
WAITING_SET_VOL5M     = 25
WAITING_SET_MAX_DEMO  = 26
WAITING_SET_MAX_REAL  = 27
WAITING_SET_PT_5X     = 30
WAITING_SET_PT_10X    = 31
WAITING_SET_PT_20X    = 32
WAITING_SET_DAILY_LOSS = 33
WAITING_SET_CONV_SIZE  = 34
WAITING_SET_BE_MULT    = 35

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
    "daily_pnl": 0.0,           # resets at midnight UTC
    "daily_pnl_date": "",       # YYYY-MM-DD of last reset
    "sniper_paused_until": 0.0, # epoch — auto-pause after daily loss limit
    "recent_losses": {},        # mint -> epoch; blocks revenge re-entry
    "api_stats": {
        "price_ok": 0, "price_fail": 0,
        "quote_ok": 0, "quote_fail": 0,
        "swap_ok": 0,  "swap_fail": 0,
        "confirm_ok": 0, "confirm_timeout": 0,
    },
    "settings": {
        # Core trading
        "take_profit": 3.0,
        "trailing_stop": 15,
        "stop_loss": 0.5,
        "trade_amount": 10.0,
        "demo_trade_amount": 100.0,
        "slippage_bps": 100,
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
        "max_wallet_concentration": 40.0,  # skip if top-10 wallets hold >40% supply
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
        # Risk management
        "daily_loss_limit_pct": 20.0,   # pause sniper if daily loss > X% of capital
        "daily_loss_pause_hrs": 4.0,    # hours to pause after daily loss limit hit
        "breakeven_mult": 2.0,          # move stop to entry once price hits this mult
        "conviction_sizing": True,       # scale trade size by ML score
        # Early dump detection
        "sell_ratio_flip_threshold": 1.5,  # sells > buys * this => early exit
        "vol_exhaustion_pct": 30.0,        # exit if 5m vol < X% of peak vol (post-TP)
    },
    "ml_features": [], "ml_labels": [],
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
                features JSONB, hold_seconds FLOAT DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW());
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
            net_pnl,fees_paid,reason,is_demo,tx_sig,features,hold_seconds)
            VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
        """, trade.get("symbol"), trade.get("mint"), trade.get("entry"),
            trade.get("exit"), trade.get("mult"), trade.get("net_pnl"),
            trade.get("fees_paid", 0), trade.get("reason"),
            trade.get("is_demo", False), trade.get("tx_sig"),
            json.dumps(trade.get("features", [])),
            trade.get("hold_seconds", 0.0))

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
                    "hold_seconds": t.get("hold_seconds", 0),
                    "projected_real": t["net_pnl"] * (
                        state["settings"]["trade_amount"] / state["settings"]["demo_trade_amount"])
                })
            else:
                state["trades_history"].append({
                    "symbol": t["symbol"], "mult": t["multiplier"],
                    "net_pnl": t["net_pnl"], "reason": t["reason"],
                    "hold_seconds": t.get("hold_seconds", 0)})
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
        for r in rows:
            feats = json.loads(r["features"])
            if len(feats) < 18: feats += [0.0] * (18 - len(feats))
            state["ml_features"].append(feats)
            state["ml_labels"].append(r["label"])
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
    """16 market features + hour-of-day + buy-size anomaly = 18 features."""
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
        # avg buy size vs avg sell size (wash-trade signal)
        avg_buy_usd  = (vol5m / b5m) if b5m > 0 else 0
        avg_sell_usd = (vol5m / s5m) if s5m > 0 else 0
        buy_size_ratio = avg_buy_usd / (avg_sell_usd + 1)
        # hour of day (UTC) — pump timing feature
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
        if len(X) < 10: return None
        ml_scaler = StandardScaler(); Xs = ml_scaler.fit_transform(X)
        ml_model  = RandomForestClassifier(n_estimators=100, random_state=42,
                        class_weight="balanced")
        ml_model.fit(Xs, y); ml_ready = True
        preds = ml_model.predict(Xs)
        prec  = precision_score(y, preds, zero_division=0)
        rec   = recall_score(y, preds, zero_division=0)
        log.info(f"ML trained: {len(X)} samples | precision={prec:.2f} recall={rec:.2f}")
        return prec
    except Exception as e:
        log_error("train_model", e); return None

def predict_score(features):
    if not ml_ready or ml_model is None: return 0.5
    try:
        f = features[:]
        if len(f) < 18: f += [0.0]*(18-len(f))
        Xs = ml_scaler.transform([f])
        return float(ml_model.predict_proba(Xs)[0][1])
    except Exception: return 0.5

async def record_trade_outcome(features, is_win, is_demo=False):
    s = state["settings"]
    if s.get("ml_real_only") and is_demo: return None
    label = 1 if is_win else 0
    f = features[:]
    if len(f) < 18: f += [0.0]*(18-len(f))
    state["ml_features"].append(f); state["ml_labels"].append(label)
    await db_save_ml_sample(f, label)
    if len(state["ml_features"]) >= 10: return train_model()
    return None

# ============================================================
# FEES
# ============================================================
def calc_fees(amount_usd):
    dex   = amount_usd * 0.003
    gas   = 0.002
    total = dex + gas
    return {"dex_fee": dex, "gas_fee": gas, "total": total}

# ============================================================
# DAILY P&L TRACKER
# ============================================================
def _reset_daily_pnl_if_needed():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state["daily_pnl_date"] != today:
        state["daily_pnl"]      = 0.0
        state["daily_pnl_date"] = today
        log.info(f"Daily P&L reset for {today}")

def _check_daily_loss_limit(app_ref=None):
    """Returns True if sniper should be paused."""
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

# ============================================================
# PRICE / DATA APIs
# ============================================================
_price_cache: dict = {}
_PRICE_CACHE_TTL   = 3
_rugcheck_cache: dict = {}
_RUGCHECK_CACHE_TTL   = 300

async def get_token_price(mint, pair_data=None):
    now = time.time()
    if mint in _price_cache:
        ts, p = _price_cache[mint]
        if now - ts < _PRICE_CACHE_TTL and p > 0: return p

    # stale-price guard: detect if last 3 cached prices are identical
    if mint in _price_cache:
        cached_val = _price_cache[mint][1]
        stale_hist = getattr(get_token_price, "_stale_hist", {})
        hist = stale_hist.get(mint, [])
        hist.append(cached_val)
        if len(hist) > 3: hist.pop(0)
        stale_hist[mint] = hist
        get_token_price._stale_hist = stale_hist
        if len(hist) == 3 and len(set(hist)) == 1:
            log.warning(f"Stale price feed detected for {mint[:8]} — skipping tick")
            return 0.0  # signal to skip this monitoring tick

    if pair_data:
        try:
            p = float(pair_data.get("priceUsd") or 0)
            if p > 0:
                state["api_stats"]["price_ok"] += 1
                _price_cache[mint] = (now, p)
                return p
        except Exception: pass

    # DexScreener
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6)) as s:
            async with s.get(DEXSCREENER_API + mint) as r:
                if r.status == 200:
                    pairs = [p for p in (await r.json()).get("pairs",[]) if p.get("chainId")=="solana"]
                    if pairs:
                        best = max(pairs, key=lambda p: float(p.get("liquidity",{}).get("usd",0) or 0))
                        price = float(best.get("priceUsd") or 0)
                        if price > 0:
                            state["api_stats"]["price_ok"] += 1
                            _price_cache[mint] = (now, price)
                            return price
    except Exception as e: log.warning(f"DexScreener price: {e}")

    # Jupiter Price API v2
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6)) as s:
            async with s.get(f"{JUPITER_PRICE_API}?ids={mint}&showExtraInfo=false") as r:
                if r.status == 200:
                    raw   = (await r.json()).get("data", {}).get(mint, {})
                    price = float(raw.get("price") or 0)
                    if price > 0:
                        state["api_stats"]["price_ok"] += 1
                        _price_cache[mint] = (now, price)
                        return price
    except Exception as e: log.warning(f"Jupiter price API: {e}")

    # Jupiter Quote fallback
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
            async with s.get(JUPITER_QUOTE_API,
                params={"inputMint": USDC_MINT, "outputMint": mint,
                        "amount": 1_000_000, "slippageBps": 500}) as r:
                if r.status == 200:
                    q = await r.json()
                    out = int(q.get("outAmount", 0))
                    decimals = int(q.get("outputDecimals", 6))
                    if out > 0:
                        price = 1.0 / (out / (10 ** decimals))
                        state["api_stats"]["price_ok"] += 1
                        _price_cache[mint] = (now, price)
                        return price
    except Exception as e: log.warning(f"Jupiter quote fallback price: {e}")

    # GeckoTerminal fallback
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6)) as s:
            async with s.get(f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{mint}",
                    headers={"Accept": "application/json"}) as r:
                if r.status == 200:
                    price = float(
                        (await r.json()).get("data",{}).get("attributes",{}).get("price_usd") or 0)
                    if price > 0:
                        state["api_stats"]["price_ok"] += 1
                        _price_cache[mint] = (now, price)
                        return price
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
    except asyncio.TimeoutError:
        log.warning(f"RugCheck timeout {mint[:8]}")
    except Exception as e:
        log.warning(f"RugCheck error {mint[:8]}: {e}")
    safe = {"score": 0, "risks": [], "rugged": False}
    _rugcheck_cache[mint] = (now, safe); return safe

# ============================================================
# WALLET CONCENTRATION CHECK  (Helius)
# ============================================================
async def check_wallet_concentration(mint) -> float:
    """Returns top-10-holder % of supply. Returns 0.0 if unavailable."""
    if not HELIUS_RPC:
        return 0.0
    try:
        url = HELIUS_RPC.rstrip("/") + f"/v0/token-accounts?mint={mint}&limit=10"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6)) as s:
            async with s.get(url) as r:
                if r.status != 200: return 0.0
                data = await r.json()
                accounts = data.get("token_accounts", [])
                if not accounts: return 0.0
                balances = [float(a.get("amount", 0)) for a in accounts]
                total    = sum(balances)
                # Fetch total supply from Solana RPC
                supply_resp = await solana_client.get_token_supply(mint)
                supply = float(supply_resp.value.ui_amount or 0) if supply_resp.value else 0
                if supply <= 0: return 0.0
                top10_pct = (total / supply) * 100
                return top10_pct
    except Exception as e:
        log.warning(f"Wallet concentration check failed: {e}")
        return 0.0

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

async def sign_and_send(tx_b64):
    try:
        tx = VersionedTransaction.from_bytes(base64.b64decode(tx_b64))
        tx.sign([keypair])
        r = await solana_client.send_raw_transaction(
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
        except Exception as e: log.warning(f"Confirm poll {i+1}: {e}")
        await asyncio.sleep(1)
    state["api_stats"]["confirm_timeout"] += 1; return False

def _conviction_amount(ml_score: float) -> float:
    """Scale trade amount by ML confidence if conviction_sizing is ON."""
    s   = state["settings"]
    amt = s["trade_amount"]
    if not s.get("conviction_sizing", True): return amt
    if ml_score >= 0.80: return amt          # full size
    if ml_score >= 0.65: return amt * 0.75  # 75%
    if ml_score >= 0.50: return amt * 0.50  # 50%
    return amt * 0.25                        # minimal

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
    paused = time.time() < state.get("sniper_paused_until", 0)
    sniper_lbl = ("⏸ Sniper PAUSED" if paused else
                  ("🟢 Sniper ON" if s["auto_snipe"] else "🔴 Sniper OFF"))
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💰 Positions",   callback_data="positions"),
         InlineKeyboardButton("📊 P&L",         callback_data="pnl")],
        [InlineKeyboardButton("🛒 Buy Token",   callback_data="buy_prompt"),
         InlineKeyboardButton("💸 Sell Token",  callback_data="sell_prompt")],
        [InlineKeyboardButton(sniper_lbl,        callback_data="toggle_sniper"),
         InlineKeyboardButton("📝 Demo ON"   if s["demo_mode"] else "📝 Demo OFF",
                              callback_data="toggle_demo")],
        [InlineKeyboardButton("📝 Demo Trades", callback_data="demo_menu"),
         InlineKeyboardButton("🧠 ML Stats",    callback_data="mlstats")],
        [InlineKeyboardButton("⚙️ Settings",    callback_data="settings_menu"),
         InlineKeyboardButton("🏥 Health",      callback_data="health")],
        [InlineKeyboardButton("📜 History",     callback_data="history"),
         InlineKeyboardButton("📈 Analytics",   callback_data="analytics")],
    ])

def kb_settings():
    s   = state["settings"]
    hm  = "🏠 House Money: ON"  if s.get("house_money_mode") else "🏠 House Money: OFF"
    mlo = "🧠 ML Real Only: ON" if s.get("ml_real_only")     else "🧠 ML Real Only: OFF"
    cvs = "📐 Conv.Sizing: ON"  if s.get("conviction_sizing") else "📐 Conv.Sizing: OFF"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🎯 Take Profit: {s['take_profit']}x",          callback_data="set_tp"),
         InlineKeyboardButton(f"📉 Trailing: {s['trailing_stop']}%",           callback_data="set_trail")],
        [InlineKeyboardButton(f"🛑 Stop Loss: {s['stop_loss']}x",              callback_data="set_stop"),
         InlineKeyboardButton(f"💵 Amount: ${s['trade_amount']}",              callback_data="set_amount")],
        [InlineKeyboardButton(f"⚡ Slippage: {s['slippage_bps']}bps",          callback_data="set_slip"),
         InlineKeyboardButton(f"🧠 Min Score: {s['min_score']:.0%}",           callback_data="set_score")],
        [InlineKeyboardButton(f"💧 Min Liq: ${s['min_liquidity']:,.0f}",       callback_data="set_liq"),
         InlineKeyboardButton(f"🛡️ Max Rug: {s['min_rugcheck']}",             callback_data="set_rugcheck")],
        [InlineKeyboardButton(f"⏱ Min Age: {s.get('min_token_age_sec',120)}s", callback_data="set_min_age"),
         InlineKeyboardButton(f"📊 Vol5m≥: {s.get('min_vol5m_pct',10)}% liq", callback_data="set_vol5m")],
        [InlineKeyboardButton(f"📂 Max Demo: {s.get('max_demo_positions',5)}",  callback_data="set_max_demo"),
         InlineKeyboardButton(f"📂 Max Real: {s.get('max_real_positions',3)}",  callback_data="set_max_real")],
        [InlineKeyboardButton(f"🚨 Daily Loss Limit: {s.get('daily_loss_limit_pct',20)}%",
                              callback_data="set_daily_loss"),
         InlineKeyboardButton(f"⚖️ Breakeven @ {s.get('breakeven_mult',2.0)}x",
                              callback_data="set_be_mult")],
        [InlineKeyboardButton(hm,  callback_data="toggle_house_money"),
         InlineKeyboardButton(mlo, callback_data="toggle_ml_real_only")],
        [InlineKeyboardButton(cvs, callback_data="toggle_conviction_sizing")],
        [InlineKeyboardButton("📐 Tiered Trail & Profit-Take", callback_data="tiered_trail_menu")],
        [InlineKeyboardButton("⬅️ Back to Menu",               callback_data="main_menu")],
    ])

def kb_tiered_trail():
    s = state["settings"]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🟢 ≥5x  → {s.get('trail_5x', 4.0)}% trail",  callback_data="set_trail_5x")],
        [InlineKeyboardButton(f"🟡 ≥10x → {s.get('trail_10x', 3.0)}% trail", callback_data="set_trail_10x")],
        [InlineKeyboardButton(f"🟠 ≥20x → {s.get('trail_20x', 2.0)}% trail", callback_data="set_trail_20x")],
        [InlineKeyboardButton(f"🔴 ≥50x → {s.get('trail_50x', 1.5)}% trail", callback_data="set_trail_50x")],
        [InlineKeyboardButton("── Profit-Take Milestones ──",                  callback_data="noop")],
        [InlineKeyboardButton(f"💰 @5x  sell {s.get('pt_5x_pct', 25.0):.0f}%",  callback_data="set_pt_5x")],
        [InlineKeyboardButton(f"💰 @10x sell {s.get('pt_10x_pct', 25.0):.0f}%", callback_data="set_pt_10x")],
        [InlineKeyboardButton(f"💰 @20x sell {s.get('pt_20x_pct', 25.0):.0f}%", callback_data="set_pt_20x")],
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
    pause_txt = ""
    if paused:
        rem = (state["sniper_paused_until"] - time.time()) / 3600
        pause_txt = f"\n⏸ *Sniper paused {rem:.1f}h* (daily loss limit)\n"
    msg = (
        f"🤖 *Solana Meme Coin Bot*\n{'─'*28}\n\n"
        f"*💼 Portfolio*\n"
        f"├ Real P&L:       {'📈' if state['total_pnl']>=0 else '📉'} *${state['total_pnl']:+.2f}*\n"
        f"├ Today P&L:      {'📈' if state['daily_pnl']>=0 else '📉'} ${state['daily_pnl']:+.2f}\n"
        f"├ Demo P&L:       📝 ${state['demo_total_pnl']:+.2f}\n"
        f"├ Trades:         {total} ({wins}W / {total-wins}L)\n\n"
        f"*🤖 Bot Status*\n"
        f"├ Auto-Sniper: {'🟢 ON' if s['auto_snipe'] else '🔴 OFF'}\n"
        f"├ Demo Mode:   {'🟢 ON' if s['demo_mode']  else '🔴 OFF'}\n"
        f"├ ML Model:    {'✅ Ready' if ml_ready else '⏳ Training'} ({n} samples)\n"
        f"{pause_txt}\n"
        f"*⚙️ Active Settings*\n"
        f"├ Take Profit:  {s['take_profit']}x  |  Stop: {s['stop_loss']}x\n"
        f"├ Breakeven @:  {s.get('breakeven_mult',2.0)}x\n"
        f"├ House Money:  {'🏠 ON' if s.get('house_money_mode') else 'OFF'}\n"
        f"├ Daily Limit:  -{s.get('daily_loss_limit_pct',20)}%\n"
        f"├ Min Liq:      ${s['min_liquidity']:,.0f}\n"
        f"├ Min Score:    {s['min_score']:.0%}\n"
        f"├ Max Pos:      {s.get('max_demo_positions',5)}D / {s.get('max_real_positions',3)}R\n"
        f"└ Trade Amount: ${s['trade_amount']}\n"
    )
    if state["positions"]:
        msg += f"\n*📂 Open Positions ({len(state['positions'])})*\n"
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
        usdc_back = tokens_to_sell * current_price
        await _safe_notify(app,
            f"🏠 {pfx}*House Money — Capital Recovered!*\n\n*{pos['symbol']}*\n{'─'*20}\n"
            f"├ Sold:          {sell_pct:.0%} of position\n"
            f"├ USDC recovered: ${usdc_back:.2f}\n"
            f"├ Tokens riding: {int(tokens_remaining):,}\n"
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
            f"├ Tokens riding:  {int(tokens_remaining):,}\n"
            f"└ 🚀 Pure profit from here — trail active!\n\n"
            f"🔗 [Solscan](https://solscan.io/tx/{result['signature']})")

    pos["token_amount"] = int(tokens_remaining)
    pos["amount_usd"]   = 0.0
    pos["fees_paid"]    = 0.0
    pos["capital_recovered"] = True
    return True

# ============================================================
# TIERED PROFIT-TAKE
# ============================================================
async def _tiered_profit_take(app, mint, pos, current_price, milestone: str, is_demo: bool):
    s       = state["settings"]
    pct     = s.get(f"pt_{milestone}_pct", 0.0) / 100.0
    if pct <= 0: return

    total_tokens     = pos.get("token_amount", 0)
    if total_tokens <= 0 or current_price <= 0: return

    # Accelerate sell size if sell pressure is already rising
    td = await get_token_data(mint)
    sell_pressure_boost = 1.0
    if td:
        b5m = float(td.get("txns",{}).get("m5",{}).get("buys",0) or 0)
        s5m = float(td.get("txns",{}).get("m5",{}).get("sells",0) or 0)
        if b5m > 0 and s5m > b5m * state["settings"].get("sell_ratio_flip_threshold", 1.5):
            sell_pressure_boost = 1.5  # sell 50% more than planned
            log.info(f"Sell pressure detected at {milestone} for {pos['symbol']} — boosting PT size")

    tokens_to_sell   = int(total_tokens * min(pct * sell_pressure_boost, 0.95))
    tokens_remaining = total_tokens - tokens_to_sell
    usdc_estimate    = tokens_to_sell * current_price
    pfx = "📝 DEMO " if is_demo else ""
    boost_note = " _(sell pressure detected — size boosted)_" if sell_pressure_boost > 1 else ""

    if is_demo:
        sell_fee      = calc_fees(usdc_estimate)["dex_fee"]
        usdc_received = usdc_estimate - sell_fee
        sig_link      = ""
    else:
        result = await execute_sell(mint, tokens_to_sell)
        if not result:
            await _safe_notify(app, f"⚠️ *{pfx}Profit-take FAILED at {milestone} for {pos['symbol']}*")
            return
        usdc_received = result["usdc_received"]
        sig_link      = f"\n🔗 [Solscan](https://solscan.io/tx/{result['signature']})"

    pos["token_amount"]          = tokens_remaining
    pos[f"pt_{milestone}_done"]  = True
    await db_save_position(mint, pos, is_demo)

    await _safe_notify(app,
        f"💰 {pfx}*Profit-Take @ {milestone} — {pos['symbol']}*\n{'─'*22}\n"
        f"├ Sold:      {tokens_to_sell:,} tokens ({pct*sell_pressure_boost:.0%}){boost_note}\n"
        f"├ USDC in:   ${usdc_received:.2f}\n"
        f"├ Remaining: {tokens_remaining:,} tokens\n"
        f"└ Riding with tiered trail 🚀{sig_link}")

# ============================================================
# CLOSE POSITION
# ============================================================
async def _close_position(app, mint, pos, price, reason, is_demo=False):
    entry = pos["entry_price"]
    mult  = price / entry if entry > 0 else 1
    hm    = pos.get("capital_recovered", False)
    hold_secs = time.time() - pos.get("entry_time", time.time())

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
        # Label as win only if ≥2x (trains model toward moonshots, not marginal wins)
        is_real_win = mult >= 2.0
        acc    = await record_trade_outcome(pos["features"], is_real_win, is_demo=is_demo)
        ml_msg = f"\n🧠 ML updated ({len(state['ml_features'])} samples)"
        if acc: ml_msg += f" | Precision: {acc:.0%}"

    await db_save_trade({"symbol": pos["symbol"], "mint": mint, "entry": entry, "exit": price,
        "mult": mult, "net_pnl": net_pnl, "fees_paid": pos["fees_paid"]+sell_fee,
        "reason": reason, "is_demo": is_demo, "tx_sig": tx_sig, "features": pos.get("features",[]),
        "hold_seconds": hold_secs})
    await db_delete_position(mint)

    if is_demo:
        state["demo_total_pnl"] += net_pnl
        state["demo_trades"].append({"symbol": pos["symbol"], "mult": mult, "net_pnl": net_pnl,
            "reason": reason, "hold_seconds": hold_secs,
            "projected_real": net_pnl*(state["settings"]["trade_amount"]/state["settings"]["demo_trade_amount"])})
        del state["demo_positions"][mint]
    else:
        state["total_pnl"]  += net_pnl
        state["daily_pnl"]  += net_pnl
        if net_pnl < 0:
            state["recent_losses"][mint] = time.time()  # block revenge re-entry 24h
        state["trades_history"].append({"symbol": pos["symbol"], "mult": mult,
            "net_pnl": net_pnl, "reason": reason, "hold_seconds": hold_secs})
        del state["positions"][mint]
        _check_daily_loss_limit()

    pfx       = "📝 DEMO " if is_demo else ""
    pnl_total = state["demo_total_pnl"] if is_demo else state["total_pnl"]
    lbl       = "Demo" if is_demo else "Real"
    hm_tag    = "\n🏠 _(Capital already recovered — pure profit)_" if hm else ""
    hold_txt  = f"{int(hold_secs//60)}m {int(hold_secs%60)}s"
    await _safe_notify(app,
        f"{'✅' if net_pnl>0 else '❌'} {pfx}{reason}\n\n"
        f"*{pos['symbol']}* | {mult:.2f}x | held {hold_txt}\n{'─'*20}\n"
        f"├ Entry:    ${entry:.6f}\n├ Exit:     ${price:.6f}\n"
        f"├ Invested: ${pos['amount_usd']:.2f}\n├ Back:     ${usdc_bk:.2f}\n"
        f"├ Fees:     -${pos['fees_paid']+sell_fee:.4f}\n"
        f"└ *Net P&L: ${net_pnl:+.2f}*\n\n"
        f"{proj_txt}💰 {lbl} P&L: ${pnl_total:+.2f}{sig_link}{ml_msg}{hm_tag}")

# ============================================================
# SAFE NOTIFY
# ============================================================
async def _safe_notify(app, text):
    try:
        await app.bot.send_message(chat_id=AUTHORIZED_USER, text=text,
            parse_mode="Markdown", disable_web_page_preview=True)
    except TelegramError as e:
        log.error(f"Notify failed: {e}")

# ============================================================
# TOKEN EVALUATION
# ============================================================
async def evaluate_new_token(td):
    s      = state["settings"]
    symbol = (td.get("baseToken") or td.get("token") or {}).get("symbol","???")
    liq    = float(td.get("liquidity",{}).get("usd",0) or 0)
    vol24  = float(td.get("volume",{}).get("h24",0) or 0)
    vol5m  = float(td.get("volume",{}).get("m5",0) or 0)
    pc1    = float(td.get("priceChange",{}).get("h1",0) or 0)
    mcap   = float(td.get("marketCap",0) or 0)
    age_ms = td.get("pairCreatedAt") or (time.time()*1000)
    age_s  = max(0, (time.time()*1000 - age_ms) / 1000)
    b5m    = float(td.get("txns",{}).get("m5",{}).get("buys",0) or 0)
    s5m    = float(td.get("txns",{}).get("m5",{}).get("sells",0) or 0)
    mint   = (td.get("baseToken") or td.get("token") or {}).get("address","")

    safety   = await check_token_safety(mint) if mint else {"score":0,"risks":[],"rugged":False}
    features = extract_features(td)
    ml_score = predict_score(features)
    vol5m_pct = (vol5m / liq * 100) if liq > 0 else 0

    # Wallet concentration check
    top10_pct = await check_wallet_concentration(mint) if mint else 0.0
    max_conc  = s.get("max_wallet_concentration", 40.0)

    # Wash-trade detection: avg buy size vs sell size
    avg_buy_usd  = (vol5m / b5m) if b5m > 0 else 0
    avg_sell_usd = (vol5m / s5m) if s5m > 0 else 0
    wash_suspect = (avg_buy_usd > 0 and avg_sell_usd > 0 and
                    avg_buy_usd / avg_sell_usd > 10)  # buys 10x larger = few wallets

    passes = (
        liq    >= s["min_liquidity"] and
        age_s  >= s["min_token_age_sec"] and
        safety["score"] <= s["min_rugcheck"] and
        not safety["rugged"] and
        vol5m_pct >= s.get("min_vol5m_pct", 10.0) and
        (top10_pct == 0 or top10_pct <= max_conc) and
        not wash_suspect
    )

    return {
        "symbol": symbol, "liquidity": liq, "volume": vol24,
        "vol5m": vol5m, "vol5m_pct": vol5m_pct,
        "price_change": pc1, "market_cap": mcap,
        "age_sec": age_s, "safety": safety,
        "features": features, "ml_score": ml_score,
        "rc_score": safety["score"],
        "top10_pct": top10_pct,
        "wash_suspect": wash_suspect,
        "passes_rules": passes,
    }

# ============================================================
# AUTO-SNIPER HELPERS
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
    for m in [m for m, t in list(state["seen_pairs"].items()) if now - t > expiry]:
        del state["seen_pairs"][m]

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
        all_mints = list({m for sub in discovered for m in sub})

        tasks   = [_fetch_full_pair(session, m) for m in all_mints]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if r and not isinstance(r, Exception)]

async def _handle_snipe(app, mint, pair_data, info):
    s = state["settings"]
    if mint in state["recent_losses"]:
        if time.time() - state["recent_losses"][mint] < 86400:
            log.info(f"Skipping {info['symbol']} — recent loss on this token (revenge guard)")
            return
    price = await get_token_price(mint, pair_data=pair_data)
    if price <= 0: return
    amt   = _conviction_amount(info["ml_score"])
    result = await execute_buy(mint, amt)
    if result:
        fees = calc_fees(amt)
        pos  = {"symbol": info["symbol"], "entry_price": price, "current_price": price,
                "peak_price": price, "amount_usd": amt-fees["total"],
                "token_amount": result["out_amount"], "fees_paid": fees["total"],
                "tp_hit": False, "features": info["features"], "auto": True,
                "entry_time": time.time(),
                "peak_vol5m": float(pair_data.get("volume",{}).get("m5",0) or 0)}
        state["positions"][mint] = pos
        await db_save_position(mint, pos, False)
        await _safe_notify(app,
            f"✅ *Sniped {info['symbol']}!*\n"
            f"├ Entry:   ${price:.6f}\n"
            f"├ Amount:  ${amt:.2f} ({info['ml_score']:.0%} confidence)\n"
            f"└ 🔗 [Solscan](https://solscan.io/tx/{result['signature']})")
    else:
        await _safe_notify(app, f"❌ Snipe failed for {info['symbol']}")

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

async def auto_sniper_loop(app):
    log.info("Auto-sniper started.")
    consecutive_errors = 0; loop_count = 0
    while True:
        try:
            _reset_daily_pnl_if_needed()
            sniper_paused = _check_daily_loss_limit()

            if not state["settings"]["auto_snipe"] and not state["settings"]["demo_mode"]:
                await asyncio.sleep(2); continue
            if sniper_paused and not state["settings"]["demo_mode"]:
                await asyncio.sleep(30); continue

            loop_count += 1
            new_pairs = await fetch_new_pairs(); consecutive_errors = 0
            if loop_count % 12 == 0:
                log.info(f"Sniper #{loop_count}: {len(new_pairs)} pairs | demo={len(state['demo_positions'])} real={len(state['positions'])}")

            for pair in new_pairs:
                base = pair.get("baseToken") or pair.get("token") or {}
                mint = base.get("address") or pair.get("tokenAddress","")
                if not mint or mint in state["seen_pairs"]: continue
                state["seen_pairs"][mint] = time.time()

                info = await evaluate_new_token(pair)
                if not info["passes_rules"]: continue
                if mint in state["positions"] or mint in state["demo_positions"]: continue
                if mint in state["recent_losses"]:
                    if time.time() - state["recent_losses"][mint] < 86400: continue

                s = state["settings"]
                if s["demo_mode"] and len(state["demo_positions"]) >= s.get("max_demo_positions",5): continue
                if s["auto_snipe"] and not s["demo_mode"] and len(state["positions"]) >= s.get("max_real_positions",3): continue

                age_min = info.get("age_sec",0)/60
                risks   = ", ".join([r.get("name","") for r in info["safety"].get("risks",[])]) or "None"
                conc_txt = f"{info['top10_pct']:.1f}%" if info['top10_pct'] > 0 else "N/A"
                notif = (
                    f"🔍 *New Token Detected*\n{'─'*24}\n🪙 *{info['symbol']}*\n"
                    f"├ Liquidity:    ${info['liquidity']:,.0f}\n"
                    f"├ Volume 24h:   ${info['volume']:,.0f}\n"
                    f"├ Vol5m:        ${info.get('vol5m',0):,.0f} ({info.get('vol5m_pct',0):.1f}% of liq)\n"
                    f"├ Age:          {age_min:.1f} min\n"
                    f"├ Market Cap:   ${info['market_cap']:,.0f}\n"
                    f"├ Price 1h:     {info['price_change']:+.1f}%\n"
                    f"├ ML Score:     {info['ml_score']:.0%} confidence\n"
                    f"├ Top-10 Hold:  {conc_txt}\n"
                    f"├ RugCheck:     {info['rc_score']} (lower=safer)\n"
                    f"└ Risks:        {risks}\n"
                )
                if s["demo_mode"]:
                    price = await get_token_price(mint, pair_data=pair)
                    if price <= 0: continue
                    amt  = s["demo_trade_amount"]; fees = calc_fees(amt)
                    pos  = {"symbol": info["symbol"], "entry_price": price, "current_price": price,
                            "peak_price": price, "amount_usd": amt-fees["total"], "fees_paid": fees["total"],
                            "token_amount": (amt-fees["total"])/price,
                            "tp_hit": False, "features": info["features"], "ml_score": info["ml_score"],
                            "auto": True, "entry_time": time.time(),
                            "peak_vol5m": float(pair.get("volume",{}).get("m5",0) or 0)}
                    state["demo_positions"][mint] = pos
                    await db_save_position(mint, pos, True)
                    await _safe_notify(app, notif + f"\n📝 *DEMO Auto-bought @ ${price:.6f}*")
                elif s["auto_snipe"]:
                    if info["ml_score"] < s["min_score"]:
                        await _safe_notify(app, notif + f"\n⚠️ *Skipped — ML {info['ml_score']:.0%} < {s['min_score']:.0%}*")
                        continue
                    await _safe_notify(app, notif + "\n🤖 *Auto-sniping...*")
                    await _handle_snipe(app, mint, pair, info)

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
    price_history: dict = {}  # mint -> [(ts, price), ...]
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
                    if price <= 0: continue  # stale feed — skip tick
                    pos["current_price"] = price
                    entry = pos["entry_price"]; mult = price / entry
                    tp = state["settings"]["take_profit"]; sl = state["settings"]["stop_loss"]

                    # Price history (momentum)
                    if mint not in price_history: price_history[mint] = []
                    price_history[mint].append((time.time(), price))
                    if len(price_history[mint]) > 20: price_history[mint].pop(0)

                    # Peak tracking
                    if price > pos.get("peak_price", entry): pos["peak_price"] = price
                    peak_mult        = pos["peak_price"] / entry if entry > 0 else 1.0
                    post_tp_trail    = _tiered_trail(peak_mult)
                    trailing_trigger = pos["peak_price"] * (1 - post_tp_trail)

                    # ── Breakeven stop ──────────────────────────────────────────
                    be_mult = state["settings"].get("breakeven_mult", 2.0)
                    if mult >= be_mult and not pos.get("breakeven_active"):
                        pos["breakeven_active"] = True
                        pos["stop_loss_floor"]  = entry  # dynamic floor at entry
                        await db_save_position(mint, pos, is_demo)
                        log.info(f"Breakeven activated for {pos['symbol']} @ {mult:.2f}x")

                    # ── Tiered profit-take milestones ───────────────────────────
                    for milestone, threshold in [("5x", 5.0), ("10x", 10.0), ("20x", 20.0)]:
                        if mult >= threshold and not pos.get(f"pt_{milestone}_done"):
                            await _tiered_profit_take(app, mint, pos, price, milestone, is_demo)

                    # ── Live 5m volume tracking (dump exhaustion) ───────────────
                    td = await get_token_data(mint)
                    b5m_live = 0.0; s5m_live = 0.0; vol5m_live = 0.0
                    if td:
                        vol5m_live = float(td.get("volume",{}).get("m5",0) or 0)
                        b5m_live   = float(td.get("txns",{}).get("m5",{}).get("buys",0) or 0)
                        s5m_live   = float(td.get("txns",{}).get("m5",{}).get("sells",0) or 0)
                        # Update peak 5m volume
                        if vol5m_live > pos.get("peak_vol5m", 0):
                            pos["peak_vol5m"] = vol5m_live
                        if mint not in vol5m_history: vol5m_history[mint] = []
                        vol5m_history[mint].append(vol5m_live)
                        if len(vol5m_history[mint]) > 10: vol5m_history[mint].pop(0)

                    reason = None

                    # ── Stop loss ───────────────────────────────────────────────
                    if mult <= sl:
                        reason = f"🔴 Stop Loss at {mult:.2f}x"
                    # ── Breakeven floor (never lose once 2x hit) ────────────────
                    elif pos.get("breakeven_active") and price <= entry:
                        reason = f"🔒 Breakeven Stop at {mult:.2f}x"
                    elif pos.get("tp_hit"):
                        # House money recovery guard
                        if (state["settings"].get("house_money_mode")
                                and not pos.get("capital_recovered")):
                            ok = await _partial_sell_for_capital_recovery(app, mint, pos, price, is_demo)
                            await db_save_position(mint, pos, is_demo)
                            if ok: continue

                        # ── Early dump detection ────────────────────────────────
                        early_exit_reason = None

                        # 1. Buy/sell ratio flip
                        flip_thresh = state["settings"].get("sell_ratio_flip_threshold", 1.5)
                        if b5m_live > 0 and s5m_live > b5m_live * flip_thresh:
                            early_exit_reason = f"🚨 Sell Pressure Exit at {mult:.2f}x (sells {s5m_live:.0f} vs buys {b5m_live:.0f})"

                        # 2. Volume exhaustion
                        if not early_exit_reason:
                            peak_v = pos.get("peak_vol5m", 0)
                            exhaust_pct = state["settings"].get("vol_exhaustion_pct", 30.0) / 100
                            if peak_v > 0 and vol5m_live > 0 and vol5m_live < peak_v * exhaust_pct:
                                early_exit_reason = f"📉 Volume Exhaustion Exit at {mult:.2f}x (vol {vol5m_live:.0f} < {exhaust_pct:.0%} of peak {peak_v:.0f})"

                        # 3. Momentum drop (price action)
                        if not early_exit_reason:
                            hist = price_history.get(mint, [])
                            if len(hist) >= 5:
                                old_p = hist[-5][1]
                                drop  = (old_p - price) / old_p if old_p > 0 else 0
                                if drop > 0.03:
                                    early_exit_reason = f"⚡ Momentum Exit at {mult:.2f}x (-{drop:.1%} in 5 ticks)"

                        if early_exit_reason:
                            reason = early_exit_reason
                        elif price <= trailing_trigger:
                            reason = f"🟡 Trailing Stop at {mult:.2f}x ({int(post_tp_trail*100)}% trail from {peak_mult:.1f}x peak)"

                    elif mult >= tp and not pos.get("tp_hit"):
                        pos["tp_hit"] = True
                        trail_pct = int(_tiered_trail(mult) * 100)
                        pfx = "📝 DEMO " if is_demo else ""
                        if state["settings"].get("house_money_mode") and not pos.get("capital_recovered"):
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

                    if reason:
                        price_history.pop(mint, None)
                        vol5m_history.pop(mint, None)
                        await _close_position(app, mint, pos, price, reason, is_demo)
                except Exception as e:
                    log_error(f"monitor/{mint[:8]}", e)

# ============================================================
# CALLBACK HANDLER
# ============================================================
def auth(fn):
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id if update.effective_user else None
        if uid != AUTHORIZED_USER:
            await update.effective_message.reply_text("⛔ Unauthorized.")
            return ConversationHandler.END
        return await fn(update, ctx)
    wrapper.__name__ = fn.__name__
    return wrapper

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
                be   = " 🔒" if pos.get("breakeven_active") else ""
                lines.append(f"{'🟢' if pnl>=0 else '🔴'} *{pos['symbol']}*{hm}{be} | {mult:.2f}x | ${pnl:+.2f}")
            await q.edit_message_text("*📂 Open Positions*\n\n" + "\n".join(lines),
                parse_mode="Markdown", reply_markup=kb_positions(state["positions"]))

    elif data == "pnl":
        t = state["trades_history"]; wins = sum(1 for x in t if x["net_pnl"] > 0)
        await q.edit_message_text(
            f"{'📈' if state['total_pnl']>=0 else '📉'} *P&L Summary*\n\n"
            f"├ Real P&L:     *${state['total_pnl']:+.2f}*\n"
            f"├ Today P&L:    ${state['daily_pnl']:+.2f}\n"
            f"├ Demo P&L:     ${state['demo_total_pnl']:+.2f}\n"
            f"├ Total Trades: {len(t)}\n├ Wins: {wins}\n└ Losses: {len(t)-wins}",
            parse_mode="Markdown", reply_markup=kb_main())

    elif data == "analytics":
        t = state["trades_history"]
        if not t:
            await q.edit_message_text("📭 No trades yet for analytics.", reply_markup=kb_main()); return
        wins   = [x for x in t if x["net_pnl"] > 0]
        losses = [x for x in t if x["net_pnl"] <= 0]
        avg_hold_w = sum(x.get("hold_seconds",0) for x in wins) / max(len(wins),1)
        avg_hold_l = sum(x.get("hold_seconds",0) for x in losses) / max(len(losses),1)
        avg_mult_w = sum(x["mult"] for x in wins) / max(len(wins),1)
        avg_mult_l = sum(x["mult"] for x in losses) / max(len(losses),1)
        # Win rate by reason
        reason_counts: dict = {}
        for x in t:
            r = x.get("reason","?")[:30]
            reason_counts[r] = reason_counts.get(r,{"w":0,"l":0})
            if x["net_pnl"] > 0: reason_counts[r]["w"] += 1
            else: reason_counts[r]["l"] += 1
        reason_lines = "\n".join(
            f"  {r}: {d['w']}W/{d['l']}L" for r,d in list(reason_counts.items())[:6])
        # ML score buckets (if features exist — use stored trades)
        paused = time.time() < state.get("sniper_paused_until", 0)
        pause_txt = ""
        if paused:
            rem = (state["sniper_paused_until"] - time.time()) / 3600
            pause_txt = f"\n⏸ Sniper paused {rem:.1f}h remaining"
        await q.edit_message_text(
            f"📈 *Trade Analytics*\n{'─'*26}\n\n"
            f"*Win/Loss*\n"
            f"├ Wins:         {len(wins)} ({len(wins)/max(len(t),1):.0%})\n"
            f"├ Avg win:      {avg_mult_w:.2f}x / ${sum(x['net_pnl'] for x in wins)/max(len(wins),1):+.2f}\n"
            f"├ Avg loss:     {avg_mult_l:.2f}x / ${sum(x['net_pnl'] for x in losses)/max(len(losses),1):+.2f}\n\n"
            f"*Hold Time*\n"
            f"├ Avg win hold:  {avg_hold_w/60:.1f} min\n"
            f"└ Avg loss hold: {avg_hold_l/60:.1f} min\n\n"
            f"*Exit Reasons*\n{reason_lines}"
            f"{pause_txt}",
            parse_mode="Markdown", reply_markup=kb_main())

    elif data == "toggle_sniper":
        state["settings"]["auto_snipe"] = not state["settings"]["auto_snipe"]
        await db_save_settings()
        await q.edit_message_text(await build_dashboard() + f"\n\n*Auto-Sniper {'🟢 ON' if state['settings']['auto_snipe'] else '🔴 OFF'}*",
            parse_mode="Markdown", reply_markup=kb_main())

    elif data == "toggle_demo":
        state["settings"]["demo_mode"] = not state["settings"]["demo_mode"]
        await db_save_settings()
        await q.edit_message_text(await build_dashboard() + f"\n\n*Demo Mode {'🟢 ON' if state['settings']['demo_mode'] else '🔴 OFF'}*",
            parse_mode="Markdown", reply_markup=kb_main())

    elif data == "toggle_house_money":
        state["settings"]["house_money_mode"] = not state["settings"].get("house_money_mode", True)
        await db_save_settings()
        status = "🟢 ON" if state["settings"]["house_money_mode"] else "🔴 OFF"
        await q.edit_message_text(
            f"⚙️ *Settings*\n\n🏠 House Money Mode turned *{status}*\n\n"
            f"_At TP hit, sells enough tokens to recover your capital.\nThe rest rides free with the trail._",
            parse_mode="Markdown", reply_markup=kb_settings())

    elif data == "toggle_ml_real_only":
        state["settings"]["ml_real_only"] = not state["settings"].get("ml_real_only", False)
        await db_save_settings()
        status = "🟢 ON" if state["settings"]["ml_real_only"] else "🔴 OFF"
        await q.edit_message_text(
            f"⚙️ *Settings*\n\n🧠 ML Real Only turned *{status}*\n\n"
            f"_When ON, ML trains only on real trades, not demo.\nRecommended once you have real trade history._",
            parse_mode="Markdown", reply_markup=kb_settings())

    elif data == "toggle_conviction_sizing":
        state["settings"]["conviction_sizing"] = not state["settings"].get("conviction_sizing", True)
        await db_save_settings()
        status = "🟢 ON" if state["settings"]["conviction_sizing"] else "🔴 OFF"
        await q.edit_message_text(
            f"⚙️ *Settings*\n\n📐 Conviction Sizing turned *{status}*\n\n"
            f"_When ON, trade size scales with ML confidence:\n"
            f"≥80% → full size | 65–79% → 75% | 50–64% → 50%_",
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
                tags = ("🎯" if pos.get("tp_hit") else "") + (" 🏠" if pos.get("capital_recovered") else "") + (" 🔒" if pos.get("breakeven_active") else "")
                lines.append(f"{'🟢' if pnl>=0 else '🔴'} *{pos['symbol']}*{tags} | {mult:.2f}x | ${pnl:+.2f} (real≈${proj:+.2f})")
            await q.edit_message_text("*📝 Demo Positions*\n\n" + "\n".join(lines),
                parse_mode="Markdown", reply_markup=kb_positions(state["demo_positions"], is_demo=True))

    elif data == "demohistory":
        if not state["demo_trades"]:
            await q.edit_message_text("📭 No demo history yet.", reply_markup=kb_main())
        else:
            lines = [f"{'✅' if t['net_pnl']>0 else '❌'} {t['symbol']} | {t['mult']:.2f}x | ${t['net_pnl']:+.2f} (real≈${t.get('projected_real',0):+.2f})"
                     for t in state["demo_trades"][-10:]]
            await q.edit_message_text("*📜 Demo History (last 10)*\n\n" + "\n".join(lines),
                parse_mode="Markdown", reply_markup=kb_main())

    elif data == "mlstats":
        n = len(state["ml_features"])
        wins = sum(state["ml_labels"])
        if n < 10:
            await q.edit_message_text(f"🧠 *ML Stats*\n\n⏳ Need {10-n} more samples to train.\nCurrently: {n} samples.",
                parse_mode="Markdown", reply_markup=kb_main()); return
        feat_names = ["liquidity","vol24","pc1h","pc6h","pc24h","b1h","s1h","buy_sell_1h",
                      "age_min","mcap","vol5m","b5m","s5m","buy_sell_5m","liq_mcap","vol_liq",
                      "buy_size_ratio","hour_utc"]
        top = []
        if ml_ready and ml_model:
            imp = ml_model.feature_importances_
            top = sorted(zip(feat_names[:len(imp)], imp), key=lambda x: -x[1])[:5]
        await q.edit_message_text(
            f"🧠 *ML Model Stats*\n\n"
            f"├ Samples:    {n} ({wins}W / {n-wins}L)\n"
            f"├ Status:     {'✅ Ready' if ml_ready else '⏳ Training'}\n"
            f"├ Win label:  ≥2x multiplier\n\n"
            f"*Top Features:*\n" + "\n".join([f"  {nm}: {v:.1%}" for nm,v in top]),
            parse_mode="Markdown", reply_markup=kb_main())

    elif data == "settings_menu":
        await q.edit_message_text("⚙️ *Settings*\n\nTap any setting to change it:",
            parse_mode="Markdown", reply_markup=kb_settings())

    elif data == "set_tp":
        ctx.user_data["setting"] = "take_profit"
        await q.edit_message_text(f"🎯 *Take Profit*\nCurrent: {state['settings']['take_profit']}x\n\nSend a new multiplier (e.g. `3.0`):",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_TP

    elif data == "set_trail":
        ctx.user_data["setting"] = "trailing_stop"
        await q.edit_message_text(f"📉 *Trailing Stop*\nCurrent: {state['settings']['trailing_stop']}%\n\nSend a new percentage:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_TRAIL

    elif data == "set_stop":
        ctx.user_data["setting"] = "stop_loss"
        await q.edit_message_text(f"🛑 *Stop Loss*\nCurrent: {state['settings']['stop_loss']}x\n\nSend a new multiplier (e.g. `0.5`):\n_Recommended: 0.4–0.6_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_STOP

    elif data == "set_amount":
        ctx.user_data["setting"] = "trade_amount"
        await q.edit_message_text(f"💵 *Trade Amount*\nCurrent: ${state['settings']['trade_amount']}\n\nSend a new USD amount:",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_AMOUNT

    elif data == "set_slip":
        ctx.user_data["setting"] = "slippage_bps"
        await q.edit_message_text(f"⚡ *Slippage*\nCurrent: {state['settings']['slippage_bps']}bps\n\nSend new bps (100 = 1%):",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_SLIP

    elif data == "set_score":
        ctx.user_data["setting"] = "min_score"
        await q.edit_message_text(f"🧠 *Min ML Score*\nCurrent: {state['settings']['min_score']:.0%}\n\nSend a value 0–1 (e.g. `0.5`):\n_Recommended: 0.5–0.7_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_SCORE

    elif data == "set_liq":
        ctx.user_data["setting"] = "min_liquidity"
        await q.edit_message_text(f"💧 *Min Liquidity*\nCurrent: ${state['settings']['min_liquidity']:,}\n\nSend a new USD value:\n_Recommended: $50,000+_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_LIQ

    elif data == "set_rugcheck":
        ctx.user_data["setting"] = "min_rugcheck"
        await q.edit_message_text(f"🛡️ *Max RugCheck Score*\nCurrent: {state['settings']['min_rugcheck']}\n\nLower = safer. Send a new value:\n_Recommended: 500–1000_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_RUGCHECK

    elif data == "set_min_age":
        ctx.user_data["setting"] = "min_token_age_sec"
        await q.edit_message_text(f"⏱ *Min Token Age*\nCurrent: {state['settings'].get('min_token_age_sec',120)}s\n\nSkip tokens younger than this (seconds).\n_Recommended: 120–300_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_MIN_AGE

    elif data == "set_vol5m":
        ctx.user_data["setting"] = "min_vol5m_pct"
        await q.edit_message_text(f"📊 *Min 5m Volume % of Liquidity*\nCurrent: {state['settings'].get('min_vol5m_pct',10)}%\n\nFilters tokens with liquidity but no trading.\n_Recommended: 5–15_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_VOL5M

    elif data == "set_max_demo":
        ctx.user_data["setting"] = "max_demo_positions"
        await q.edit_message_text(f"📂 *Max Demo Positions*\nCurrent: {state['settings'].get('max_demo_positions',5)}\n\nSniper stops at this limit.\n_Recommended: 3–8_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_MAX_DEMO

    elif data == "set_max_real":
        ctx.user_data["setting"] = "max_real_positions"
        await q.edit_message_text(f"📂 *Max Real Positions*\nCurrent: {state['settings'].get('max_real_positions',3)}\n\nSniper stops at this limit.\n_Recommended: 2–5_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_MAX_REAL

    elif data == "set_daily_loss":
        ctx.user_data["setting"] = "daily_loss_limit_pct"
        await q.edit_message_text(
            f"🚨 *Daily Loss Limit*\nCurrent: {state['settings'].get('daily_loss_limit_pct',20)}%\n\n"
            f"If real P&L drops by this % of your capital in one day,\n"
            f"the sniper auto-pauses for {state['settings'].get('daily_loss_pause_hrs',4)}h.\n"
            f"Send a percentage (e.g. `20`):\n_Recommended: 15–25%_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_DAILY_LOSS

    elif data == "set_be_mult":
        ctx.user_data["setting"] = "breakeven_mult"
        await q.edit_message_text(
            f"⚖️ *Breakeven Stop Multiplier*\nCurrent: {state['settings'].get('breakeven_mult',2.0)}x\n\n"
            f"Once price hits this multiple, stop-loss moves up to entry.\nYou cannot lose on that trade.\n"
            f"Send a multiplier (e.g. `2.0`):\n_Recommended: 1.5–3.0_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_BE_MULT

    elif data == "tiered_trail_menu":
        s = state["settings"]
        await q.edit_message_text(
            f"📐 *Tiered Trailing Stop & Profit-Take*\n\n"
            f"*Trailing Stop* — tightens as price pumps:\n"
            f"├ Peak ≥ 5x  → {s.get('trail_5x',4.0)}% trail\n"
            f"├ Peak ≥ 10x → {s.get('trail_10x',3.0)}% trail\n"
            f"├ Peak ≥ 20x → {s.get('trail_20x',2.0)}% trail\n"
            f"└ Peak ≥ 50x → {s.get('trail_50x',1.5)}% trail\n"
            f"_(Below 5x: default 5%)_\n\n"
            f"*Profit-Take* — auto-sell a slice at each milestone:\n"
            f"├ @ 5x  → sell {s.get('pt_5x_pct',25.0):.0f}% of position\n"
            f"├ @ 10x → sell {s.get('pt_10x_pct',25.0):.0f}% of position\n"
            f"└ @ 20x → sell {s.get('pt_20x_pct',25.0):.0f}% of position\n"
            f"_(Set any to 0 to disable. Size auto-boosts if sell pressure detected.)_",
            parse_mode="Markdown", reply_markup=kb_tiered_trail())

    elif data == "set_trail_5x":
        ctx.user_data["setting"] = "trail_5x"
        await q.edit_message_text(f"🟢 *Trail % at ≥5x*\nCurrent: {state['settings'].get('trail_5x',4.0)}%\n\nSend a percentage:\n_Recommended: 3–6%_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_TRAIL_5X

    elif data == "set_trail_10x":
        ctx.user_data["setting"] = "trail_10x"
        await q.edit_message_text(f"🟡 *Trail % at ≥10x*\nCurrent: {state['settings'].get('trail_10x',3.0)}%\n\nSend a percentage:\n_Recommended: 2–5%_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_TRAIL_10X

    elif data == "set_trail_20x":
        ctx.user_data["setting"] = "trail_20x"
        await q.edit_message_text(f"🟠 *Trail % at ≥20x*\nCurrent: {state['settings'].get('trail_20x',2.0)}%\n\nSend a percentage:\n_Recommended: 1.5–3%_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_TRAIL_20X

    elif data == "set_trail_50x":
        ctx.user_data["setting"] = "trail_50x"
        await q.edit_message_text(f"🔴 *Trail % at ≥50x*\nCurrent: {state['settings'].get('trail_50x',1.5)}%\n\nSend a percentage:\n_Recommended: 1–2%_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_TRAIL_50X

    elif data == "set_pt_5x":
        ctx.user_data["setting"] = "pt_5x_pct"
        await q.edit_message_text(
            f"💰 *Profit-Take @ 5x*\nCurrent: {state['settings'].get('pt_5x_pct',25.0):.0f}%\n\n"
            f"Sell this % of your position when price hits 5x.\nSet to 0 to disable.\n_Recommended: 20–33%_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_PT_5X

    elif data == "set_pt_10x":
        ctx.user_data["setting"] = "pt_10x_pct"
        await q.edit_message_text(
            f"💰 *Profit-Take @ 10x*\nCurrent: {state['settings'].get('pt_10x_pct',25.0):.0f}%\n\n"
            f"Sell this % of your position when price hits 10x.\nSet to 0 to disable.\n_Recommended: 20–33%_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_PT_10X

    elif data == "set_pt_20x":
        ctx.user_data["setting"] = "pt_20x_pct"
        await q.edit_message_text(
            f"💰 *Profit-Take @ 20x*\nCurrent: {state['settings'].get('pt_20x_pct',25.0):.0f}%\n\n"
            f"Sell this % of your position when price hits 20x.\nSet to 0 to disable.\n_Recommended: 15–25%_",
            parse_mode="Markdown", reply_markup=kb_back()); return WAITING_SET_PT_20X

    elif data == "noop":
        pass

    elif data == "health":
        err_txt = "\n".join([f"  [{e['time']}] {e['context']}: {e['error']}" for e in state["errors"][-5:]]) or "  None"
        st = state["api_stats"]
        await q.edit_message_text(
            f"🏥 *Bot Health*\n\n"
            f"*API Stats:*\n"
            f"├ Price:   {st['price_ok']}✅ {st['price_fail']}❌\n"
            f"├ Quote:   {st['quote_ok']}✅ {st['quote_fail']}❌\n"
            f"├ Swap:    {st['swap_ok']}✅ {st['swap_fail']}❌\n"
            f"└ Confirm: {st['confirm_ok']}✅ timeout={st['confirm_timeout']}\n\n"
            f"*State:*\n"
            f"├ Positions: {len(state['positions'])} real / {len(state['demo_positions'])} demo\n"
            f"├ ML: {len(state['ml_features'])} samples | {'Ready' if ml_ready else 'Training'}\n"
            f"├ Daily P&L: ${state['daily_pnl']:+.2f}\n"
            f"└ DB: {'✅ Connected' if db_pool else '❌ Disconnected'}\n\n"
            f"*Recent Errors:*\n{err_txt}",
            parse_mode="Markdown", reply_markup=kb_main())

    elif data == "history":
        if not state["trades_history"]:
            await q.edit_message_text("📭 No history yet.", reply_markup=kb_main())
        else:
            lines = [f"{'✅' if t['net_pnl']>0 else '❌'} {t['symbol']} | {t['mult']:.2f}x | ${t['net_pnl']:+.2f}"
                     for t in state["trades_history"][-10:]]
            await q.edit_message_text("*📜 Last 10 Trades*\n\n" + "\n".join(lines),
                parse_mode="Markdown", reply_markup=kb_main())

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
            f"💸 *Confirm Sell*\n\nToken: *{pos['symbol']}*\n"
            f"Entry: ${pos['entry_price']:.6f}\nAmount: ${pos['amount_usd']:.2f}\n\nAre you sure?",
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
            await q.edit_message_text("❌ Price unavailable, try again.", reply_markup=kb_main()); return
        mult = price / pos["entry_price"]
        await q.edit_message_text(
            f"💸 *Close Demo Position*\n\nToken: *{pos['symbol']}*\n"
            f"Entry: ${pos['entry_price']:.6f}\nCurrent: ${price:.6f} ({mult:.2f}x)\n\nConfirm?",
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
        price = await get_token_price(mint)
        if price <= 0:
            await q.edit_message_text("❌ Price unavailable, try again.", reply_markup=kb_main()); return
        await _close_position(app, mint, pos, price, "✂️ Manual TP Close", True)
        await q.edit_message_text(await build_dashboard(), parse_mode="Markdown", reply_markup=kb_main())

# ============================================================
# MESSAGE HANDLERS
# ============================================================
async def handle_setting_input(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    key = ctx.user_data.get("setting"); txt = update.message.text.strip()
    try:
        val = float(txt)
        state["settings"][key] = int(val) if key in (
            "slippage_bps","max_demo_positions","max_real_positions","min_token_age_sec"
        ) else val
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
        await q.edit_message_text("❌ Price unavailable for demo trade. Try again.", reply_markup=kb_main())
        return ConversationHandler.END
    if is_demo:
        td  = await get_token_data(mint)
        pos = {"symbol": symbol, "entry_price": price, "current_price": price, "peak_price": price,
               "amount_usd": amt-fees["total"], "token_amount": (amt-fees["total"])/price,
               "fees_paid": fees["total"], "tp_hit": False,
               "features": extract_features(td) if td else [0.0]*18, "auto": False,
               "entry_time": time.time(), "peak_vol5m": 0.0}
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
               "entry_time": time.time(), "peak_vol5m": 0.0}
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

def main():
    validate_config()
    global keypair, solana_client
    keypair       = Keypair.from_bytes(base58.b58decode(PRIVATE_KEY_B58))
    solana_client = AsyncClient(RPC_URL, commitment=Confirmed)
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
            WAITING_SET_MIN_AGE:   [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_VOL5M:     [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_MAX_DEMO:  [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_MAX_REAL:  [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_PT_5X:     [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_PT_10X:    [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_PT_20X:    [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_DAILY_LOSS:[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_CONV_SIZE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
            WAITING_SET_BE_MULT:   [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_input)],
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