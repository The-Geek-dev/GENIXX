"""
agent_link.py — GENIXX monitoring agent support.

Three jobs, kept free of bot.py globals so they can be tested on their own:

1. Watchdog rules  — pure functions that look at recent trades / counters and
   return alert texts. bot.py runs them every minute and sends deduped alerts.
2. GitHub exchange — upload trade snapshots to a PRIVATE repo and read the
   review agent's suggestions back (GitHub REST "contents" API).
3. Suggestion safety — every setting the agent may touch has hard bounds and a
   maximum step size. Anything outside them is rejected, whatever the agent says.
"""
import base64
import json
import time
from statistics import median

import aiohttp

# ════════════════════════════════════════════════════════════════════════════
# 3. SUGGESTION SAFETY — the ONLY settings the agent may change
# ════════════════════════════════════════════════════════════════════════════
# key: (min, max, max_step_per_change)
AGENT_BOUNDS = {
    "min_score":             (0.30, 0.75, 0.10),
    "stop_loss":             (0.80, 0.95, 0.05),
    "take_profit":           (1.30, 5.00, 0.50),
    "price5m_max_pct":       (10.0, 100.0, 20.0),
    "price1h_max_pct":       (100.0, 2000.0, 300.0),
    "min_token_age_sec":     (60, 1800, 600),
    "max_token_age_sec":     (3600, 28800, 7200),
    "min_liquidity":         (20000, 150000, 20000),
    "min_vol5m_pct":         (2.0, 50.0, 10.0),
    "min_rugcheck":          (100, 800, 200),
    "min_entry_bs_ratio":    (0.0, 2.0, 0.3),
    "pre_tp_trail_pct":      (1.0, 10.0, 2.0),
    "pre_tp_trail_act_mult": (1.05, 1.50, 0.10),
}
INT_KEYS = {"min_token_age_sec", "max_token_age_sec", "min_liquidity", "min_rugcheck"}
MAX_CHANGES_PER_SUGGESTION = 3
AUTO_APPLY_COOLDOWN_SEC = 11 * 3600   # at most one auto-applied batch per review cycle


def validate_changes(changes, current: dict):
    """
    Split a list of {"key","to","why"} into (accepted, rejected).
    accepted items gain "from"; rejected items gain "error".
    """
    accepted, rejected = [], []
    if not isinstance(changes, list):
        return [], [{"change": changes, "error": "changes must be a list"}]
    for ch in changes:
        if not isinstance(ch, dict) or "key" not in ch or "to" not in ch:
            rejected.append({"change": ch, "error": "needs key and to"}); continue
        key = ch["key"]
        if key not in AGENT_BOUNDS:
            rejected.append({**ch, "error": "setting not allowed for the agent"}); continue
        lo, hi, step = AGENT_BOUNDS[key]
        try:
            to = float(ch["to"])
        except (TypeError, ValueError):
            rejected.append({**ch, "error": "not a number"}); continue
        if not lo <= to <= hi:
            rejected.append({**ch, "error": f"outside bounds {lo}–{hi}"}); continue
        cur = current.get(key)
        if cur is not None:
            try:
                cur_f = float(cur)
            except (TypeError, ValueError):
                cur_f = None
            if cur_f is not None:
                if abs(to - cur_f) < 1e-9:
                    rejected.append({**ch, "error": "already at that value"}); continue
                if abs(to - cur_f) > step + 1e-9:
                    rejected.append({**ch, "error": f"step {abs(to-cur_f):g} > max {step:g}"}); continue
        if any(a["key"] == key for a in accepted):
            rejected.append({**ch, "error": "duplicate key"}); continue
        if len(accepted) >= MAX_CHANGES_PER_SUGGESTION:
            rejected.append({**ch, "error": f"max {MAX_CHANGES_PER_SUGGESTION} changes per suggestion"}); continue
        accepted.append({"key": key, "from": cur, "to": int(round(to)) if key in INT_KEYS else to,
                         "why": str(ch.get("why", ""))[:300]})
    return accepted, rejected


def fmt_value(key, v):
    if v is None: return "—"
    if key == "min_score": return f"{float(v):.0%}"
    if key in ("stop_loss", "take_profit", "pre_tp_trail_act_mult"): return f"{float(v):g}x"
    if key.endswith("_pct"): return f"{float(v):g}%"
    if key.endswith("_sec"): return f"{int(v)//60} min" if int(v) >= 120 else f"{int(v)}s"
    if key == "min_liquidity": return f"${int(v):,}"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


# ════════════════════════════════════════════════════════════════════════════
# 1. WATCHDOG RULES
# ════════════════════════════════════════════════════════════════════════════
def is_stop_exit(reason: str) -> bool:
    return isinstance(reason, str) and "Stop Loss" in reason and "Breakeven" not in reason


def check_stop_slippage(trades, stop_level: float, n=15, tolerance=0.08):
    """Alert when recent stop-loss exits fill well below the configured stop."""
    stops = [t for t in trades if is_stop_exit(t.get("reason"))][-n:]
    if len(stops) < n: return None
    mults = [float(t.get("mult") or 0) for t in stops]
    avg = sum(mults) / len(mults)
    if avg >= stop_level - tolerance: return None
    ages = [t["price_age"] for t in stops if isinstance(t.get("price_age"), (int, float))]
    age_txt = f"\nMedian price age at stop: {median(ages):.1f}s" if ages else ""
    rugs = sum(1 for m in mults if m < 0.5)
    return (f"📉 *Stops filling badly*\nLast {n} stop-loss exits averaged *{avg:.2f}x* "
            f"against a {stop_level}x stop ({rugs} were rugs below 0.5x).{age_txt}")


def check_win_rate(trades, n=30, floor=0.35):
    recent = trades[-n:]
    if len(recent) < n: return None
    wins = sum(1 for t in recent if (t.get("net_pnl") or 0) > 0)
    net = sum((t.get("net_pnl") or 0) for t in recent)
    wr = wins / n
    if wr >= floor: return None
    return (f"📊 *Win rate dropped*\nLast {n} trades: *{wr:.0%}* win rate "
            f"({wins}W/{n-wins}L), net ${net:+.2f}.")


def check_losing_streak(trades, n=8):
    recent = trades[-n:]
    if len(recent) < n: return None
    if all((t.get("net_pnl") or 0) <= 0 for t in recent):
        net = sum((t.get("net_pnl") or 0) for t in recent)
        return f"🧊 *{n} losses in a row* (net ${net:+.2f})."
    return None


def check_no_trades(last_buy_ts: float, active: bool, hours=3.0, now=None):
    now = now or time.time()
    if not active or not last_buy_ts: return None
    idle = (now - last_buy_ts) / 3600
    if idle < hours: return None
    return (f"😴 *No new buys for {idle:.1f}h*\nSniper is on but nothing passed the filters. "
            f"Check API health or whether filters are too strict.")


def check_spike_rejects(reject_times, now=None, window=3600, limit=5):
    now = now or time.time()
    recent = [t for t in reject_times if now - t <= window]
    if len(recent) <= limit: return None
    return (f"⚠️ *{len(recent)} price spikes ignored in the last hour*\n"
            f"A price source may be returning bad data.")


def check_restart_loop(restart_times, now=None, window=900, limit=3):
    now = now or time.time()
    recent = [t for t in restart_times if now - t <= window]
    if len(recent) < limit: return None
    return f"🔁 *Restart loop*\nBot restarted {len(recent)} times in {window//60} minutes. Check Railway logs."


def check_monitor_stalled(heartbeat_ts: float, open_positions: int, now=None, limit=60):
    now = now or time.time()
    if open_positions <= 0 or not heartbeat_ts: return None
    gap = now - heartbeat_ts
    if gap < limit: return None
    return (f"🚨 *Position monitor stalled*\nNo price check for {gap:.0f}s with "
            f"{open_positions} open position(s). Stops are NOT being checked.")


# ════════════════════════════════════════════════════════════════════════════
# 2. GITHUB EXCHANGE (private data repo)
# ════════════════════════════════════════════════════════════════════════════
class GitHubData:
    API = "https://api.github.com"

    def __init__(self, token: str, repo: str):
        self.token, self.repo = token, repo

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.repo and "/" in self.repo)

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "genixx-bot"}

    async def get_file(self, session: aiohttp.ClientSession, path: str):
        """Return (decoded_text, sha) or (None, None) if missing."""
        url = f"{self.API}/repos/{self.repo}/contents/{path}"
        async with session.get(url, headers=self._headers(),
                               timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status == 404: return None, None
            if r.status != 200:
                raise RuntimeError(f"GitHub GET {path}: HTTP {r.status} {(await r.text())[:200]}")
            data = await r.json()
            return base64.b64decode(data.get("content", "")).decode("utf-8"), data.get("sha")

    async def put_file(self, session: aiohttp.ClientSession, path: str, text: str, message: str,
                       create_only: bool = False):
        """
        Create or replace a file. With create_only=True an existing file is never
        overwritten (no sha is sent, so GitHub rejects the write with HTTP 422).
        """
        url = f"{self.API}/repos/{self.repo}/contents/{path}"
        sha = None
        if not create_only:
            _, sha = await self.get_file(session, path)
        body = {"message": message, "content": base64.b64encode(text.encode("utf-8")).decode("ascii")}
        if sha: body["sha"] = sha
        async with session.put(url, headers=self._headers(), json=body,
                               timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status not in (200, 201):
                raise RuntimeError(f"GitHub PUT {path}: HTTP {r.status} {(await r.text())[:200]}")
            return True


def snapshot_path(ts: float, first_id=None, last_id=None) -> str:
    """
    Unique per upload: second-level timestamp + trade id range.
    (v2.635 used HH MM only, so two uploads in the same minute overwrote each other.)
    """
    t = time.gmtime(ts)
    rng = f"-{first_id}-{last_id}" if first_id is not None else "-none"
    return time.strftime("snapshots/%Y/%m/%d/%H%M%S", t) + rng + ".json"


def build_snapshot(*, trades, settings, open_positions, counters, demo_total_pnl,
                   ml_samples, version, settings_changes, now=None) -> dict:
    now = now or time.time()
    return {
        "generated_at": now,
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "bot_version": version,
        "settings": settings,
        "trades": trades,
        "open_positions": open_positions,
        "counters": counters,
        "demo_total_pnl": demo_total_pnl,
        "ml_samples": ml_samples,
        "settings_changes": settings_changes,
    }


def parse_suggestion(text: str):
    """Parse control/suggestions.json. Returns dict or None."""
    try:
        d = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(d, dict) or not d.get("id") or "changes" not in d:
        return None
    return d
