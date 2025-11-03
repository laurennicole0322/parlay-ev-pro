import os, json, requests, pandas as pd, csv
from datetime import datetime

# ==========================================================
# CONFIG
# ==========================================================
DISCORD_BEST_PARLAY = os.getenv("DISCORD_WEBHOOK_BEST_PARLAY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")  # export ODDS_API_KEY="..."
SPORT = "basketball_nba"
BOOK_KEY = "hardrockbet"  # change if needed
MODEL_FILE = "data/model_trueprobs.csv"
HISTORY_FILE = "data/parlay_history.csv"
BANKROLL = 100  # USD

# Map your human Market labels -> Odds API "markets" keys + outcome field names
MARKET_KEY_MAP = {
    # CSV "Market" : (api_market_key, outcome_stat_name_substring_for_over)
    "3PT Made Over": ("player_threes", "over"),
    "Points Over":   ("player_points", "over"),
    "Assists Over":  ("player_assists", "over"),
    "Rebounds Over": ("player_rebounds", "over"),
}

# ==========================================================
# UTILITIES
# ==========================================================
def implied_prob(american):
    if american is None: return None
    if american > 0:  return 100 / (american + 100)
    else:             return (-american) / ((-american) + 100)

def decimal_odds(american):
    if american is None: return None
    if american > 0:  return (american / 100) + 1
    else:             return (100 / abs(american)) + 1

def kelly_fraction(true_prob, dec_odds):
    b = dec_odds - 1
    p = true_prob
    q = 1 - p
    f_star = ((b * p) - q) / b
    return max(f_star / 2, 0)  # Half-Kelly, floor at 0

def send_best_parlay_to_discord(title, message):
    if not DISCORD_BEST_PARLAY:
        print("⚠️ Discord webhook missing (DISCORD_WEBHOOK_BEST_PARLAY).")
        return None
    payload = {"embeds":[{"title":title,"description":message,"color":0x00FF7F,
                          "footer":{"text":"Parlay Pro Engine | " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}]}
    r = requests.post(DISCORD_BEST_PARLAY, data=json.dumps(payload), headers={"Content-Type":"application/json"})
    if r.status_code not in (200, 204):
        print(f"⚠️ Discord post failed ({r.status_code}): {r.text}")
        return None
    print("✅ Posted parlay to #best-parlay Discord channel.")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_parlay(history_file, log_entry):
    file_exists = os.path.exists(history_file)
    with open(history_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(log_entry)
    print(f"🧾 Logged parlay to {history_file}")

# ==========================================================
# LIVE ODDS FETCH
# ==========================================================
def fetch_live_odds_for_market(api_market_key):
    """
    Returns a list of games with bookmakers/markets/outcomes for the requested market key.
    """
    if not ODDS_API_KEY:
        raise RuntimeError("Missing ODDS_API_KEY env var.")
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": api_market_key,      # e.g., player_threes
        "oddsFormat": "american"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def find_book_price_for_player_line(live_json, player_name, api_market_key, book_key, wanted_line):
    """
    Walk the live odds json to find a matching price for player + Over + line at the specified book.
    Returns an integer American odds (e.g., -145 or +130) or None.
    """
    # Basic normalization for name matching
    norm = lambda s: s.lower().replace(".", "").replace("'", "").strip()
    player_norm = norm(player_name)

    for game in live_json:
        for bk in game.get("bookmakers", []):
            if bk.get("key") != book_key:
                continue
            for market in bk.get("markets", []):
                if market.get("key") != api_market_key:
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name","")
                    # Expect patterns like "Player Name Over 1.5", some APIs split into fields
                    # We try to detect "Over" and match player + line
                    if norm(player_name) in norm(name) and "over" in norm(name):
                        # Try to read the line
                        line = outcome.get("point", None)
                        try:
                            line = float(line) if line is not None else None
                        except:
                            line = None
                        if line is not None and abs(line - float(wanted_line)) < 1e-6:
                            return outcome.get("price", None)
    return None

# ==========================================================
# MAIN
# ==========================================================
if not os.path.exists(MODEL_FILE):
    print(f"❌ Model file not found at {MODEL_FILE}")
    raise SystemExit(1)

df = pd.read_csv(MODEL_FILE)
required_cols = {"Player","Market","Line","TrueProb"}
missing = required_cols - set(df.columns)
if missing:
    raise SystemExit(f"❌ model_trueprobs.csv missing columns: {missing}")
print("📊 Loaded model_trueprobs.csv")

# Fetch per-market live odds once and cache in a dict
market_cache = {}
for market_label, (api_market_key, _tag) in MARKET_KEY_MAP.items():
    market_cache[api_market_key] = fetch_live_odds_for_market(api_market_key)

# Resolve live odds per row
live_rows = []
for _, row in df.iterrows():
    player = str(row["Player"]).strip()
    market_label = str(row["Market"]).strip()
    line = float(row["Line"])
    true_prob = float(row["TrueProb"])

    if market_label not in MARKET_KEY_MAP:
        # Unsupported market; skip
        continue
    api_market_key, _ = MARKET_KEY_MAP[market_label]
    live_json = market_cache.get(api_market_key, [])

    price = find_book_price_for_player_line(live_json, player, api_market_key, BOOK_KEY, line)
    if price is None:
        # no price found — skip
        continue

    ip = implied_prob(price)
    edge_pct = (true_prob - ip) * 100

    live_rows.append({
        "Player": player,
        "Market": market_label,
        "Line": line,
        "Live_Odds": price,
        "ImpliedProb": ip,
        "TrueProb": true_prob,
        "Edge%": round(edge_pct, 2)
    })

live_df = pd.DataFrame(live_rows)
if live_df.empty:
    raise SystemExit("❌ No live prices found that match your CSV (player/market/line).")

live_df = live_df.sort_values("Edge%", ascending=False)

# pick top 3 for parlay
top3 = live_df.head(3)
true_prob = round(top3["TrueProb"].prod(), 4)
implied_prob_total = round(top3["ImpliedProb"].prod(), 4)

# parlay payout multiple = product of individual decimal odds
payout_multiple = 1.0
for o in top3["Live_Odds"]:
    payout_multiple *= decimal_odds(o)
payout_multiple = round(payout_multiple, 2)

expected_value = round((true_prob - implied_prob_total) * 100, 2)

# Kelly stake sizing on parlay
kelly = kelly_fraction(true_prob, payout_multiple)
stake = round(BANKROLL * kelly, 2)
potential_win = round(stake * (payout_multiple - 1), 2)
expected_profit = round((true_prob * potential_win) - ((1 - true_prob) * stake), 2)
new_bankroll = round(BANKROLL - stake + expected_profit, 2)


# ==========================================================
# 🧠 Build Leg Lines — with full SGO market name compatibility
# ==========================================================

def detect_market_column(df):
    """Find the best market name column, case-insensitive."""
    possible_cols = [
        "MarketName", "MarketType", "MarketLabel",
        "Market", "MarketDesc",
        "market_name", "market_type", "market_label",
        "market", "market_desc"
    ]
    for col in possible_cols:
        if col in df.columns:
            return col
    return None

market_col = detect_market_column(top3)

def infer_market_from_game_or_name(row):
    """Fallback if market name is still missing."""
    text_fields = [
        str(row.get("MarketName", "")),
        str(row.get("MarketType", "")),
        str(row.get("MarketLabel", "")),
        str(row.get("Market", "")),
        str(row.get("MarketDesc", "")),
        str(row.get("market_name", "")),
        str(row.get("market_type", "")),
        str(row.get("market_label", "")),
        str(row.get("market_desc", "")),
        str(row.get("Game", "")),
        str(row.get("Player", "")),
    ]
    combined = " ".join([t for t in text_fields if t and t != "nan"]).lower()
    if "points" in combined:
        return "Points"
    if "rebounds" in combined:
        return "Rebounds"
    if "assists" in combined:
        return "Assists"
    if "threes" in combined or "3pt" in combined:
        return "3PT Made"
    if "yards" in combined:
        return "Yards"
    if "touchdowns" in combined or "td" in combined:
        return "Touchdowns"
    if "hits" in combined:
        return "Hits"
    if "strikeouts" in combined:
        return "Strikeouts"
    return "Unknown Market"

def format_leg(row):
    """Format a readable leg line for Discord posts."""
    player = row.get("Player", "")
    side = row.get("Side", "")
    line = row.get("Line", "")
    odds = row.get("BookOdds", "")
    
    # Get market name directly from SGO
    market = None
    if market_col and row.get(market_col):
        market = row.get(market_col)
    elif "market_name" in row and row["market_name"]:
        market = row["market_name"]
    else:
        market = infer_market_from_game_or_name(row)

    # Clean text
    market = str(market).replace("_", " ").title().strip()
    replacements = {
        "Player Points": "Points",
        "Player Rebounds": "Rebounds",
        "Player Assists": "Assists",
        "Three Pointers Made": "3PT Made",
        "Shots On Goal": "Shots on Goal",
        "Passing Yards": "Passing Yards",
        "Receiving Yards": "Receiving Yards",
        "Rushing Yards": "Rushing Yards",
        "Home Runs": "Home Runs",
        "Strikeouts": "Strikeouts",
    }
    for k, v in replacements.items():
        if k.lower() in market.lower():
            market = v
            break

    line_str = f"{line}" if pd.notna(line) and line != "" else ""
    odds_str = f"({int(odds):+d})" if pd.notna(odds) and str(odds).replace('-', '').isdigit() else ""
    side_str = f"{side} " if side else ""

    formatted = f"**{player}** — {side_str}{market} {line_str} {odds_str}".strip()
    return formatted

# Combine formatted legs
leg_lines = [format_leg(r) for _, r in top3.iterrows()]

# Extract headers
sport = top3.iloc[0].get("League", "Unknown League")
game = top3.iloc[0].get("Game", "Unknown Game")

# ==========================================================
# 🧾 Compose Discord Message (clean and clear)
# ==========================================================
message = (
    f"**{sport} | {game}**\n\n"
    "**Legs:**\n" + "\n".join(f"• {x}" for x in leg_lines) + "\n\n"
    f"**True Prob:** {true_prob:.1%}\n"
    f"**Implied Prob:** {implied_prob_total:.1%}\n"
    f"**EV Gain:** {expected_value:.2f}%\n"
    f"**Payout Multiple:** ×{payout_multiple:.2f}\n\n"
    f"💰 **Bet Size:** ${stake:.2f}\n"
    f"🎯 **Potential Win:** ${potential_win:.2f}\n"
    f"📈 **Expected Profit:** ${expected_profit:.2f}\n"
    f"🏦 **Updated Bankroll (simulated):** ${new_bankroll:.2f}"
)
