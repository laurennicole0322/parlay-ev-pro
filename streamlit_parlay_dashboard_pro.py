# streamlit_parlay_dashboard_pro.py
# Parlay +EV Pro (clean fixed build with SGO v2/events, The Odds API, SportsData.io inputs)
# Run: streamlit run streamlit_parlay_dashboard_pro.py

import os
import math
import json
import time
import pandas as pd
import numpy as np
import requests
import streamlit as st
import pytz
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union
from dotenv import load_dotenv  # 👈 load environment variables
from props_engine_plus import load_user_bet_history



# --- Import AI Betting Engine ---
from props_engine_plus import (
    scan_recommended,
    push_recommended,
    push_placed_bet,
    todays_matchups_mlb_probables,
    format_recommended_msg,
    append_bet_feedback,
    logistic_calibrate,
    update_bet_outcomes,   # 🏁 auto-updates Outcome column
)

# --- Unified Odds + Extractors (Streamlit Cached) ---
from unified_odds_engine_v3_1 import (
    fetch_sgo_events,
    fetch_oddsapi_games,
    extract_sgo_df,
    extract_oddsapi_df,
    extract_sportsdataio_df,
    quality_filter
)

# =====================================
# 🎨 UNIVERSAL STYLE HELPERS (EV/Kelly)
# =====================================
import pandas as pd

def color_row(row):
    """Full-row color highlight based on EV% value."""
    ev = row.get("EV_Pct", 0)
    if pd.isna(ev):
        return [""] * len(row)
    if ev >= 5:
        color = "#166534"   # strong green
    elif ev >= 2:
        color = "#22c55e"   # moderate green
    elif ev > 0:
        color = "#eab308"   # yellow
    else:
        color = "#b91c1c"   # red
    return [f"background-color: {color}; color: white;"] * len(row)

def style_table(df: pd.DataFrame):
    """Apply row highlighting and percent formatting safely."""
    try:
        return (
            df.style
            .apply(color_row, axis=1)
            .format({
                "EV_Pct": "{:.2f}%",
                "Kelly_Pct": "{:.2f}%",
                "HalfKelly_Pct": "{:.2f}%",
                "ImpliedProb": "{:.2%}",
                "TrueProb": "{:.2%}",
            })
        )
    except Exception as e:
        print(f"[WARN] style_table failed: {e}")
        return df

# --- Load environment variables (API keys, Discord webhooks, etc.) ---
load_dotenv()  # 👈 loads .env file automatically

# --- Define global environment variables for all tabs ---
SPORTSDATA_KEY = os.getenv("SPORTSDATA_KEY", "")
ODDSAPI_KEY = os.getenv("ODDSAPI_KEY", "")
SGO_KEY = os.getenv("SGO_KEY", "")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")

# --- Optional: log confirmation (masked) ---
if DISCORD_WEBHOOK:
    print("✅ Discord webhook loaded from .env")
else:
    print("⚠️ Discord webhook not found in .env")

# --- Streamlit page config ---
st.set_page_config(page_title="Parlay +EV Pro", layout="wide")

# ============== Helpers ==============
def us_to_prob(american_odds: Optional[str]) -> Optional[float]:
    """Convert American odds to implied probability (0..1)."""
    if american_odds is None:
        return None
    try:
        s = str(american_odds).strip()
        if not s:
            return None
        if s[0] == '+':
            s = s[1:]
        o = int(s)
        if o > 0:
            return 100 / (o + 100)
        else:
            return abs(o) / (abs(o) + 100)
    except Exception:
        return None


def fmt_pct(x: Optional[float]) -> str:
    return f"{x*100:0.1f}%" if isinstance(x, (int, float)) and not pd.isna(x) else ""


def today_str(tz_fix_hours: int = 0) -> str:
    return (datetime.utcnow() + timedelta(hours=tz_fix_hours)).strftime("%Y-%m-%d")


def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


# ============================================================
# 🔍 Check SportsGameOdds API Usage
# ============================================================
def check_sgo_usage(api_key: str):
    """
    Fetches and displays your current SportsGameOdds usage limits.
    Warns if you're near or past your monthly or per-minute caps.
    """
    try:
        url = f"https://api.sportsgameodds.com/v2/account/usage?apiKey={api_key}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", {})
        tier = data.get("tier", "Unknown")
        rate = data.get("ratelimits", {})

        per_min = rate.get("per-minute", {})
        per_day = rate.get("per-day", {})
        per_month = rate.get("per-month", {})

        min_used = per_min.get("current-requests", 0)
        min_limit = per_min.get("max-requests", 0)

        day_used = per_day.get("current-requests", 0)
        day_limit = per_day.get("max-requests", 0)

        month_used = per_month.get("current-entities", 0)
        month_limit = per_month.get("max-entities", 0)

        st.markdown(f"### ⚙️ SportsGameOdds API Usage — *{tier.title()}* tier")
        st.write(f"**Per Minute:** {min_used} / {min_limit}")
        st.write(f"**Per Day:** {day_used} / {day_limit}")
        st.write(f"**Per Month (Entities):** {month_used:,} / {month_limit:,}")

        # Highlight warnings visually
        if month_limit and month_used >= month_limit:
            st.error("🚫 You’ve reached your **monthly entity limit** — SGO requests will fail until reset.")
        elif month_limit and month_used >= 0.9 * month_limit:
            st.warning("⚠️ You’re using over **90%** of your monthly SGO limit. Consider upgrading or throttling refreshes.")
        elif min_limit != "unlimited" and min_used >= min_limit:
            st.warning("⏱ You’ve hit your **per-minute rate limit** — wait 60 seconds before retrying.")
        else:
            st.success("✅ SGO usage within limits.")
        return data
    except Exception as e:
        st.warning(f"Could not check SGO usage: {e}")
        return None
    
# ============================================================
# 🎯 PERSONALIZATION HELPER (Recalibrate Odds Using User Bets)
# ============================================================
def apply_personalization(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Adjust TrueProb, EV%, and Edge% based on the user's historical betting preferences
    (from my_bet_history.xlsx). Adds personalized columns to the dataframe.
    """

    try:
        from props_engine_plus import load_user_bet_history
        user_bets = load_user_bet_history(verbose=False)

        if user_bets.empty:
            if verbose:
                print("ℹ️ No personal bet history found — skipping personalization.")
            return df

        # --- Calculate user preferences ---
        league_pref = user_bets["League"].value_counts(normalize=True).to_dict()
        market_pref = user_bets["Market"].value_counts(normalize=True).to_dict()

        # --- Apply bias weighting ---
        def bias_row(row):
            league_weight = league_pref.get(row.get("League"), 0.05)
            market_weight = market_pref.get(row.get("MarketType", row.get("Market")), 0.05)
            bias_factor = 1 + ((league_weight + market_weight) / 3)
            return min(row.get("TrueProb", 0) * bias_factor, 1.0)

        df["PersonalizedProb"] = df.apply(bias_row, axis=1)
        df["PersonalizedEdge%"] = (df["PersonalizedProb"] - df["ImpliedProb"]) * 100
        df["PersonalizedEV%"] = ((df["PersonalizedProb"] * df["BookOdds"]) - 1) * 100

        if verbose:
            print(f"✅ Personalized recalibration applied to {len(df)} rows.")
        return df

    except Exception as e:
        print(f"⚠️ Personalization failed: {e}")
        return df
    
# ============== Finalize DataFrame Helper ==============
def finalize_odds_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Helper to finalize odds DataFrame with consistent numeric formatting,
    probability percentages, and preferred column ordering.
    Used by all extract_* functions.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Convert numeric columns safely
    for c in ["Line", "ImpliedProb", "TrueProb", "EdgePct",
              "EV_Pct", "Kelly_Pct", "HalfKelly_Pct", "HalfKellyCapped_Pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Add formatted % columns for display
    if "ImpliedProb" in df.columns:
        df["ImpliedProb%"] = df["ImpliedProb"].apply(fmt_pct)
    if "TrueProb" in df.columns:
        df["TrueProb%"] = df["TrueProb"].apply(fmt_pct)
    if "EdgePct" in df.columns:
        df["Edge%"] = df["EdgePct"].map(lambda x: f"{x:0.2f}%" if pd.notna(x) else "")

    # Sort logically for readability
    sort_cols = [c for c in ["Game", "Player", "MarketType", "Side"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ignore_index=True)

    # Preferred column order (keeps flexibility across APIs)
    ordered_cols = [
        "League", "EventID", "Game", "Player", "Team", "MarketType",
        "MarketName", "Side", "Line", "BookOdds", "TrueOdds",
        "EV_Pct", "Kelly_Pct", "HalfKelly_Pct", "HalfKellyCapped_Pct",
        "BooksAvailable", "ImpliedProb%", "TrueProb%", "Edge%"
    ]
    extras = [c for c in df.columns if c not in ordered_cols]
    df = df[[c for c in ordered_cols if c in df.columns] + extras]

    return df

# ==========================================
# 🧠 ODDS + PROBABILITY CALCULATORS
# ==========================================

def american_to_decimal(american_odds: Optional[str]) -> Optional[float]:
    """Convert American odds to decimal odds."""
    if american_odds is None:
        return None
    try:
        s = str(american_odds).strip()
        if not s:
            return None
        if s[0] == '+':
            s = s[1:]
        o = int(s)
        if o > 0:
            return (o / 100) + 1
        else:
            return (100 / abs(o)) + 1
    except Exception:
        return None


def implied_probability(american_odds: Optional[str]) -> Optional[float]:
    """Convert American odds to implied probability (0..1)."""
    if american_odds is None:
        return None
    try:
        s = str(american_odds).strip()
        if not s:
            return None
        if s[0] == '+':
            s = s[1:]
        o = int(s)
        if o > 0:
            return 100 / (o + 100)
        else:
            return abs(o) / (abs(o) + 100)
    except Exception:
        return None


def kelly_fraction(p_true: float, odds_decimal: float) -> float:
    """Return the Kelly fraction (decimal odds)."""
    if p_true is None or odds_decimal is None or odds_decimal <= 1:
        return 0.0
    b = odds_decimal - 1
    q = 1 - p_true
    f = ((b * p_true) - q) / b
    return max(f, 0)



# --- Optional Auto-Refresh every 5 minutes ---
import time, json
from datetime import datetime

st.divider()

AUTO_REFRESH = st.sidebar.checkbox("⏱ Auto-refresh engine every 5 minutes", value=False)
HISTORY_PATH = "data/parlay_history.csv"

def get_last_parlay_signature():
    """Get a unique signature from the latest parlay in history (legs only)."""
    if not os.path.exists(HISTORY_PATH):
        return None
    df = pd.read_csv(HISTORY_PATH)
    if df.empty or "Legs" not in df.columns:
        return None
    return str(df.iloc[-1]["Legs"]).strip()

if AUTO_REFRESH:
    st.info("Auto-refresh is active. The engine will run every 5 minutes and post only if a *new* parlay is detected.")
    prev_signature = get_last_parlay_signature()
    last_run = st.empty()

    while True:
        res = subprocess.run(
            ["python3", "refresh_live_odds.py"],
            capture_output=True,
            text=True,
            timeout=120
        )

        if res.stdout:
            st.code(res.stdout)
            new_signature = get_last_parlay_signature()
            if new_signature and new_signature != prev_signature:
                st.success("🚀 New parlay detected and posted to Discord!")
                prev_signature = new_signature
            else:
                st.info("No new parlay found this cycle — skipping Discord post.")
        else:
            st.warning("(no output from refresh script)")

        last_run.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(300)  # 5 minutes
        
# ============== Sidebar ==============

st.sidebar.title("🔑 API Keys & Settings")

# Keys (all manual input, nothing hardcoded)
sdata_key = st.sidebar.text_input("SportsData.io API Key", type="password", key="sdata_key")
odds_key  = st.sidebar.text_input("The Odds API Key", type="password", key="odds_key")
sgo_key   = st.sidebar.text_input("SportsGameOdds API Key", type="password", key="sgo_key")

st.sidebar.markdown("---")

sport = st.sidebar.selectbox(
    "Sport / League",
    [
        "NBA", "NFL", "NHL", "MLB", "NCAAF",
        "Soccer - EPL",
        "Soccer - La Liga",
        "Soccer - Serie A",
        "Soccer - Bundesliga",
        "Soccer - Ligue 1",
        "Soccer - MLS",
        "Soccer - UEFA Champions League",
        "Soccer - UEFA Europa League",
        "Soccer - World Cup",
    ],
    index=0,
    key="sport_select_main"
)

book_filter = st.sidebar.multiselect(
    "Bookmakers (SGO filter)",
    options=[
        "draftkings","fanduel","betmgm","caesars","betrivers","pointsbet","espnbet",
        "bovada","betonline","unibet","williamhill","betway","prophetexchange","betfairexchange",
        "hardrockbet","tipico","mybookie","everygame","lowvig","betus","thescorebet","gtbets",
    ],
    default=[],
    key="sgo_book_filter"
)

st.sidebar.markdown("---")

edge_floor = st.sidebar.number_input("Min EV Edge % (display)", value=0.0, step=0.5, key="edge_floor")
auto_refresh = st.sidebar.checkbox("Auto-refresh SGO (NBA/NFL) every 60s", value=True, key="sgo_auto_refresh")

st.sidebar.markdown("---")

run_btn = st.sidebar.button("🚀 Run Dashboard", type="primary", key="run_button")

# ==========================================================
# Unified Sports Map (for OddsAPI, SportsData.io, and SGO)
# ==========================================================
sport_map_odds = {
    # 🏀 Basketball
    "NBA": "basketball_nba",
    "NCAAB": "basketball_ncaab",

    # 🏈 Football
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",

    # ⚾ Baseball
    "MLB": "baseball_mlb",

    # 🏒 Hockey
    "NHL": "icehockey_nhl",  # ✅ Added NHL support

    # ⚽ Soccer
    "Soccer - EPL": "soccer_epl",
    "Soccer - La Liga": "soccer_spain_la_liga",
    "Soccer - Serie A": "soccer_italy_serie_a",
    "Soccer - Bundesliga": "soccer_germany_bundesliga",
    "Soccer - Ligue 1": "soccer_france_ligue_one",
    "Soccer - MLS": "soccer_usa_mls",
    "Soccer - UEFA Champions League": "soccer_uefa_champions_league",
    "Soccer - UEFA Europa League": "soccer_uefa_europa_league",
    "Soccer - World Cup": "soccer_fifa_world_cup",
}

# Get the selected sport key dynamically
sport_key = sport_map_odds.get(sport, "icehockey_nhl")



# ===================================================
# 🧠 NORMALIZATION HELPERS (used by extractors)
# ===================================================
def normalize_market_name(market_key: str, market_name: str) -> str:
    """
    Convert raw market names/keys into readable labels (e.g., 'Player Points Over').
    """
    key = (market_key or "").lower()
    name = (market_name or "").lower()

    def contains(*words):
        return all(w in key or w in name for w in words)

    if "over" in key or "over" in name:
        if "points" in key or "points" in name:
            return "Player Points Over"
        if "rebounds" in key or "rebounds" in name:
            return "Player Rebounds Over"
        if "assists" in key or "assists" in name:
            return "Player Assists Over"
        if "three" in key or "3pt" in key:
            return "Player 3PT Made Over"
        if "touchdowns" in key or "rushing_touchdowns" in key:
            return "Player Touchdowns Over"
        if "passing yards" in name or contains("passing", "yards"):
            return "QB Passing Yards Over"
        if "rushing yards" in name or contains("rushing", "yards"):
            return "Rushing Yards Over"
        if "receiving yards" in name or contains("receiving", "yards"):
            return "Receiving Yards Over"

    if "under" in key or "under" in name:
        if "points" in key or "points" in name:
            return "Player Points Under"
        if "rebounds" in key or "rebounds" in name:
            return "Player Rebounds Under"
        if "assists" in key or "assists" in name:
            return "Player Assists Under"
        if "three" in key or "3pt" in key:
            return "Player 3PT Made Under"
        if "touchdowns" in key or "rushing_touchdowns" in key:
            return "Player Touchdowns Under"
        if "passing yards" in name or contains("passing", "yards"):
            return "QB Passing Yards Under"
        if "rushing yards" in name or contains("rushing", "yards"):
            return "Rushing Yards Under"
        if "receiving yards" in name or contains("receiving", "yards"):
            return "Receiving Yards Under"

    return market_name or market_key


def choose_line(odd_obj: Dict[str, Any]) -> Optional[float]:
    """Select the most relevant numeric line (O/U or spread)."""
    for k in ("bookOverUnder", "bookSpread", "fairOverUnder", "fairSpread"):
        v = odd_obj.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None


def choose_odds(odd_obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return (bookOdds, fairOdds) as normalized strings like '+120' or '-130'."""
    bo = odd_obj.get("bookOdds")
    fo = odd_obj.get("fairOdds")

    def norm(x):
        if x is None:
            return None
        s = str(x).strip()
        if s and s[0] not in "+-":
            try:
                n = int(s)
                return f"+{n}" if n > 0 else str(n)
            except Exception:
                return s
        return s

    return norm(bo), norm(fo)
    
# ==========================================
# 🏥 INJURIES + WEATHER + MATCHUPS FETCHERS
# ==========================================

@st.cache_data(ttl=300)
def fetch_injuries(sport: str = "nba") -> pd.DataFrame:
    """Pull current injury data from SportsData.io"""
    url = f"https://api.sportsdata.io/v4/{sport}/scores/json/Injuries"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_KEY}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            return pd.DataFrame(r.json())
        else:
            st.warning(f"⚠️ Injury fetch failed: {r.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Injury API error: {e}")
        return pd.DataFrame()



# ============================================
# 🌤️ WEATHER FETCHER — OPEN-METEO (WITH CACHE)
# ============================================
import requests
from datetime import datetime, timedelta

# Simple cache to prevent rate-limit issues
_weather_cache = {}  # {city: (timestamp, weather_dict)}

def fetch_weather(city: str) -> dict:
    """
    Fetch weather for a given NFL city using Open-Meteo (no key required).
    Caches results for 60 minutes to avoid rate limits.
    """
    if not city:
        return {}

    city = city.lower().strip()

    # ✅ Use cached data if less than 1 hour old
    now = datetime.utcnow()
    if city in _weather_cache:
        ts, cached_data = _weather_cache[city]
        if (now - ts) < timedelta(hours=1):
            return cached_data

    # NFL city → coordinates
    city_to_coords = {
        # AFC East
        "buffalo": (42.7738, -78.7868),
        "miami": (25.958, -80.2389),
        "new england": (42.0909, -71.2643),
        "new york": (40.8136, -74.0744),
        # AFC North
        "baltimore": (39.278, -76.6227),
        "cincinnati": (39.0954, -84.5161),
        "cleveland": (41.5061, -81.6995),
        "pittsburgh": (40.4468, -80.0158),
        # AFC South
        "houston": (29.6847, -95.4107),
        "indianapolis": (39.7601, -86.1639),
        "jacksonville": (30.3239, -81.6373),
        "nashville": (36.1664, -86.7713),
        # AFC West
        "denver": (39.7439, -105.0201),
        "kansas city": (39.0489, -94.4841),
        "las vegas": (36.0908, -115.183),
        "los angeles": (34.0139, -118.2851),
        # NFC East
        "washington": (38.9072, -77.0369),
        "philadelphia": (39.9008, -75.1675),
        "dallas": (32.7473, -97.0928),
        "new york giants": (40.8136, -74.0744),
        # NFC North
        "chicago": (41.8623, -87.6167),
        "detroit": (42.34, -83.0456),
        "green bay": (44.5013, -88.0622),
        "minneapolis": (44.9735, -93.2577),
        # NFC South
        "atlanta": (33.7554, -84.4008),
        "carolina": (35.2251, -80.8528),
        "new orleans": (29.9511, -90.0812),
        "tampa": (27.9759, -82.5033),
        # NFC West
        "arizona": (33.5277, -112.2626),
        "san francisco": (37.403, -121.97),
        "seattle": (47.5952, -122.3316),
        "los angeles rams": (34.0139, -118.2851),
        # International
        "london": (51.5074, -0.1278),
        "frankfurt": (50.1109, 8.6821),
        "munich": (48.1351, 11.5820),
    }

    coords = city_to_coords.get(city)
    if not coords:
        return {}

    lat, lon = coords
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current_weather=true&hourly=temperature_2m,precipitation,wind_speed_10m"
        f"&timezone=auto"
    )

    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 429:
            print(f"⚠️ Open-Meteo rate limit hit for {city} — using cached or default weather.")
            return _weather_cache.get(city, ({},))[1] if city in _weather_cache else {}

        if r.status_code != 200:
            print(f"⚠️ Weather API error {r.status_code}: {r.text[:120]}")
            return {}

        data = r.json()
        current = data.get("current_weather", {})
        hourly = data.get("hourly", {})

        weather = {
            "temperature": current.get("temperature") or hourly.get("temperature_2m", [None])[-1],
            "wind_speed": current.get("windspeed") or hourly.get("wind_speed_10m", [None])[-1],
            "precipitation": hourly.get("precipitation", [0])[-1],
        }

        # ✅ Cache result for 1 hour
        _weather_cache[city] = (now, weather)
        return weather

    except Exception as e:
        print(f"⚠️ Weather fetch failed for {city}: {e}")
        return _weather_cache.get(city, ({},))[1] if city in _weather_cache else {}
    
# ============================================
# 🏟️ AUTO-DETECT GAME CITY FROM MATCHUP STRING
# ============================================
def detect_city_from_game(game_name: str) -> Optional[str]:
    """
    Detect the likely NFL city from a matchup string, such as:
      'Green Bay Packers @ Dallas Cowboys' -> 'Dallas'
    Returns a city string matching the keys in city_to_coords.
    """
    if not game_name:
        return None

    game_lower = game_name.lower()

    # Mapping team keywords → city names
    team_to_city = {
        # AFC East
        "bills": "buffalo",
        "dolphins": "miami",
        "patriots": "new england",
        "jets": "new york",

        # AFC North
        "ravens": "baltimore",
        "bengals": "cincinnati",
        "browns": "cleveland",
        "steelers": "pittsburgh",

        # AFC South
        "texans": "houston",
        "colts": "indianapolis",
        "jaguars": "jacksonville",
        "titans": "nashville",

        # AFC West
        "broncos": "denver",
        "chiefs": "kansas city",
        "raiders": "las vegas",
        "chargers": "los angeles",

        # NFC East
        "commanders": "washington",
        "eagles": "philadelphia",
        "cowboys": "dallas",
        "giants": "new york",

        # NFC North
        "bears": "chicago",
        "lions": "detroit",
        "packers": "green bay",
        "vikings": "minneapolis",

        # NFC South
        "falcons": "atlanta",
        "panthers": "carolina",
        "saints": "new orleans",
        "buccaneers": "tampa",

        # NFC West
        "cardinals": "arizona",
        "49ers": "san francisco",
        "seahawks": "seattle",
        "rams": "los angeles",
    }

    # Detect home team (after '@' symbol)
    if "@" in game_lower:
        parts = game_lower.split("@")
        home_side = parts[-1].strip()
    else:
        home_side = game_lower

    for team, city in team_to_city.items():
        if team in home_side:
            return city

    return None

# ============================================================
# 🧮 TRUE PROBABILITY ADJUSTMENT ENGINE (Injury + Weather)
# ============================================================
def adjust_true_probability(row, injuries_df):
    """
    Dynamically adjust TrueProb based on:
      - Player injury status
      - Real-time weather (via Open-Meteo)
      - Stadium type (dome vs outdoor)
    """
    prob = float(row.get("TrueProb", 0) or 0)
    player = str(row.get("Player", "")).strip()
    game = str(row.get("Game", "")).strip()
    market = str(row.get("MarketType", "")).lower()

    # ================================
    # 🚑 Injury Adjustment
    # ================================
    if isinstance(injuries_df, pd.DataFrame) and not injuries_df.empty:
        injury = injuries_df[injuries_df["Name"].str.lower() == player.lower()]
        if not injury.empty:
            status = injury.iloc[0].get("Status", "")
            if "out" in status.lower():
                prob *= 0.80
            elif "doubtful" in status.lower():
                prob *= 0.85
            elif "questionable" in status.lower():
                prob *= 0.90

    # ================================
    # 🏟️ Stadium Type Logic
    # ================================
    dome_stadiums = {
        "las vegas", "new orleans", "indianapolis", "detroit",
        "minneapolis", "arizona", "atlanta", "houston", "los angeles",
        "dallas"
    }

    city = detect_city_from_game(game)
    is_dome = city in dome_stadiums

    # ================================
    # 🌦 Weather Adjustment (if outdoor)
    # ================================
    if city and not is_dome:
        weather_data = fetch_weather(city)
        if weather_data:
            temp = weather_data.get("temperature", 70)
            wind = weather_data.get("wind_speed", 0)
            rain = weather_data.get("precipitation", 0)

            # Passing/Receiving/Kicking sensitivity
            if any(x in market for x in ["pass", "receiv", "field goal", "kicking"]):
                if wind > 15:
                    prob *= 0.93
                if rain > 0.2:
                    prob *= 0.90

            # Rushing props benefit from weather
            if "rush" in market:
                if wind > 15 or rain > 0.2:
                    prob *= 1.05

            # Extreme temperature adjustments
            if temp < 40:
                prob *= 0.96
            elif temp > 90:
                prob *= 0.98

    # ================================
    # 🧩 Final Cap
    # ================================
    prob = max(0, min(prob, 1.0))
    return prob


# ============== Layout / Tabs ==============
st.title("📊 Parlay +EV Pro")

st.caption(
    "Enter your API keys in the left sidebar, pick a sport, then click **Run Dashboard**. "
    "SGO tab auto-computes implied vs **true** probabilities and supports bookmaker filtering."
)

# Create tab layout
tabs = st.tabs([
    "🏀/🏈 Player Props (SportsData.io)",
    "📈 Game Lines (The Odds API)",
    "🎯 SportsgameOdds (Player Props + Markets)",
    "🎯 Recommended Bets",
    "AI-Driven Bets",
    "⚙️ Engine & Discord"
])


## -------- TAB 1: Player Props (SportsData.io) --------
with tabs[0]:
    st.subheader("🎯 Player Props — SportsData.io")
    st.caption("Pulls official NBA/NFL player props from SportsData.io with live odds and implied probabilities.")

    # --- User Inputs ---
    props_date = st.date_input(
        "Select Props Date",
        value=date.today(),
        key="sportsdata_date_input"
    )
    props_date_str = props_date.strftime("%Y-%m-%d")

    sport_choice = st.selectbox(
        "Select Sport",
        ["NBA", "NFL"],
        index=0,
        key="sportsdata_sport_select"
    )

    # --- Check for API key ---
    if not SPORTSDATA_KEY:
        st.info("⚠️ Please enter your SportsData.io API key in the sidebar or .env to enable this feed.")
    else:
        # --- Construct API URL ---
        sport_lower = sport_choice.lower()
        url = f"https://api.sportsdata.io/v3/{sport_lower}/odds/json/PlayerPropsByDate/{props_date_str}"

        # --- Fetch raw JSON ---
        try:
            st.write("⏳ Fetching live player props from SportsData.io...")
            headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_KEY}
            response = requests.get(url, headers=headers, timeout=25)

            if response.status_code != 200:
                st.warning(f"⚠️ SportsData.io returned {response.status_code}: {response.text[:200]}")
                sdata_df = pd.DataFrame()
            else:
                data = response.json()
                sdata_df = extract_sportsdataio_df(data)

        except Exception as e:
            st.error(f"❌ Error fetching props: {e}")
            sdata_df = pd.DataFrame()

        # --- Display results ---
        if sdata_df.empty:
            st.warning(f"No props found for {sport_choice} on {props_date_str}. Try another date or verify your key.")
        else:
            st.success(f"✅ Loaded {len(sdata_df)} player props for {sport_choice} ({props_date_str})")

            # --- Summary Metrics ---
            avg_edge = sdata_df["EV_Pct"].mean() if "EV_Pct" in sdata_df.columns else None
            if avg_edge is not None:
                st.metric("Average EV%", f"{avg_edge:,.2f}%")

            # --- Data Table ---
            with st.expander("📋 View Player Props Data", expanded=True):
                st.dataframe(
                    sdata_df.head(150),
                    use_container_width=True,
                    hide_index=True
                )

            # --- CSV Export ---
            csv = sdata_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download Props CSV",
                data=csv,
                file_name=f"SportsDataIO_Props_{sport_choice}_{props_date_str}.csv",
                mime="text/csv",
                key="dl_sportsdata_props"
            )


# -------- TAB 2: Game Lines — The Odds API --------
with tabs[1]:
    st.subheader("🏈 Game Lines — The Odds API")
    st.caption("Pulls live Moneyline, Spread, and Total odds for multiple sports.")

    # --- Sport selection mapping ---
    sport_map_odds = {
        "NBA": "basketball_nba",
        "NFL": "americanfootball_nfl",
        "MLB": "baseball_mlb",
        "NCAAF": "americanfootball_ncaaf",
        "NCAAM": "basketball_ncaab",  # ✅ correct key for college basketball
        "Soccer - EPL": "soccer_epl",
        "Soccer - La Liga": "soccer_spain_la_liga",
        "Soccer - Serie A": "soccer_italy_serie_a",
        "Soccer - Bundesliga": "soccer_germany_bundesliga",
        "Soccer - Ligue 1": "soccer_france_ligue_one",
        "Soccer - MLS": "soccer_usa_mls",
        "Soccer - UEFA Champions League": "soccer_uefa_champions_league",
        "Soccer - UEFA Europa League": "soccer_uefa_europa_league",
        "Soccer - World Cup": "soccer_fifa_world_cup",
    }

    # --- Sport selection dropdown ---
    selected_sport = st.selectbox(
        "Select Sport / League",
        list(sport_map_odds.keys()),
        index=list(sport_map_odds.keys()).index("NBA"),
        key="oddsapi_sport_select"
    )
    sport_key = sport_map_odds[selected_sport]

    # --- Bookmaker Filter ---
    available_books = ["hardrockbet", "draftkings", "fanduel", "caesars", "espnbet"]
    selected_books = st.multiselect(
        "Select Bookmakers",
        options=available_books,
        default=["hardrockbet"],
        key="oddsapi_book_filter"
    )
    books_param = ",".join(selected_books)

    # --- Min edge filter ---
    min_edge = st.slider(
        "Minimum EV% (Edge Filter)",
        min_value=0.0, max_value=10.0, value=0.0, step=0.5,
        key="oddsapi_min_edge"
    )

    # --- Check for API key ---
    if not ODDSAPI_KEY:
        st.info("⚠️ Please enter your The Odds API key in the sidebar or .env file.")
    else:
        st.write(f"🎯 Querying **{selected_sport}** | Bookmakers: `{books_param}`")

        try:
            from props_engine_plus import fetch_the_odds_api_games, extract_odds_api_df
            raw_odds = fetch_the_odds_api_games(
                api_key=ODDSAPI_KEY,
                sport_key=sport_key,
                bookmakers=books_param
            )
            odds_df = extract_odds_api_df(raw_odds)
        except Exception as e:
            st.error(f"❌ Failed to retrieve or process Odds API data: {e}")
            odds_df = pd.DataFrame()

        # --- Validate and Display ---
        if odds_df.empty:
            st.warning("No game lines found for this sport or bookmaker.")
        else:
            st.success(f"✅ Retrieved {len(odds_df)} lines from The Odds API")

            # --- Apply filters ---
            if "EV_Pct" in odds_df.columns:
                odds_df = odds_df[odds_df["EV_Pct"].fillna(0) >= min_edge]

            if odds_df.empty:
                st.warning(f"No lines meet EV ≥ {min_edge}%")
            else:
                # --- Sort by EV% and TrueProb, then limit to top 50 ---
                sort_cols = [c for c in ["EV_Pct", "TrueProb"] if c in odds_df.columns]
                if sort_cols:
                    odds_df = odds_df.sort_values(sort_cols, ascending=[False, False])

                top50 = odds_df.head(50).reset_index(drop=True)

                # --- Display table ---
                st.markdown("### 📊 Top 50 Value-Ranked Odds (Sorted by EV% & TrueProb)")
                st.caption(f"Displaying top {len(top50)} of {len(odds_df)} total results.")
                styled_df = style_table(top50)
                st.dataframe(styled_df, use_container_width=True, height=520, hide_index=True)

                # --- CSV Download (Top 50 Only) ---
                csv = top50.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download CSV (Top 50 Odds API Game Lines)",
                    data=csv,
                    file_name=f"odds_api_{sport_key}_{today_str()}.csv",
                    mime="text/csv",
                    key="dl_oddsapi_lines"
                )

                # --- Discord Push (Top 5) ---
                if st.button("📢 Send Top 5 Value Lines to Discord", key="send_discord_oddsapi"):
                    try:
                        from props_engine_plus import format_recommended_msg, push_recommended
                        top5 = top50.head(5).copy()
                        msg = format_recommended_msg(
                            f"💸 Top 5 Game Lines — {selected_sport.upper()} ({today_str()})",
                            top5, top_n=5
                        )
                        push_recommended(top5, title=f"Top 5 Game Lines — {selected_sport.upper()}")
                        st.success("✅ Sent Top 5 Game Lines to Discord!")
                    except Exception as e:
                        st.error(f"❌ Failed to send to Discord: {e}")                       

# -------- TAB 3: SportsGameOdds — Player Props + Markets --------
with tabs[2]:
    st.subheader("🎯 SportsGameOdds — Player Props + Market Value Analysis")
    st.caption("Real-time player props, edges, and probabilities from the SportsGameOdds API.")

    if auto_refresh and sport.upper() in ("NBA", "NFL"):
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=300 * 1000, key=f"auto_refresh_{sport.lower()}")
        st.caption("🔄 Auto-refreshing every 300 seconds (NBA/NFL).")

    sgo_league_map = {
        "NBA": "NBA",
        "NFL": "NFL",
        "MLB": "MLB",
        "NHL": "NHL",
        "NCAAM":"NCAAM",
        "NCAAF": "NCAAF",
        "Soccer": "SOC",
    }
    league_id = sgo_league_map.get(sport, "NBA")

    if not SGO_KEY:
        st.info("⚠️ Please enter your SportsGameOdds API key in the sidebar or .env file.")
    else:
        try:
            check_sgo_usage(SGO_KEY)
        except Exception:
            pass

        st.write(f"📡 Fetching live {league_id} props & markets from SportsGameOdds...")
        try:
            payload = fetch_sgo_events(
                sgo_api_key=SGO_KEY,
                league_id=league_id,
                limit=100
            )
            sgo_df = extract_sgo_df(
                payload,
                wanted_books=["hardrockbet", "draftkings", "fanduel", "caesars", "espnbet"]
            )
        except Exception as e:
            st.error(f"❌ Error fetching SGO data: {e}")
            sgo_df = pd.DataFrame()

        if sgo_df.empty:
            st.warning("No props or markets returned. Try adjusting filters or check API tier limits.")
        else:
            st.success(f"✅ Loaded {len(sgo_df)} props from SportsGameOdds ({league_id})")

            st.markdown("### 🔍 Filters")
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

            with col1:
                unique_markets = sorted(sgo_df["MarketType"].dropna().unique().tolist())
                chosen_market = st.selectbox("Market Type", ["All"] + unique_markets, index=0)

            with col2:
                player_q = st.text_input("Filter by Player Name", "")

            with col3:
                unique_games = sorted(sgo_df["Game"].dropna().unique().tolist())
                chosen_game = st.selectbox("Game Matchup", ["All"] + unique_games, index=0)

            with col4:
                edge_min = st.number_input("Min Edge %", value=3.0, step=0.5)

            filtered = sgo_df.copy()
            if chosen_market != "All":
                filtered = filtered[filtered["MarketType"] == chosen_market]
            if chosen_game != "All":
                filtered = filtered[filtered["Game"] == chosen_game]
            if player_q.strip():
                filtered = filtered[
                    filtered["Player"].fillna("").str.contains(player_q.strip(), case=False, na=False)
                ]
            if edge_min is not None:
                filtered = filtered[(filtered["EdgePct"].fillna(-9999) >= float(edge_min))]

            if filtered.empty:
                st.info("ℹ️ No props matched your filters.")
            else:
                st.markdown(f"### 📊 Filtered Results ({len(filtered)} props shown)")
                styled_df = style_table(filtered)
                st.dataframe(styled_df, use_container_width=True, height=550, hide_index=True)

                csv = filtered.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Download CSV (SGO Props + Markets)",
                    data=csv,
                    file_name=f"sgo_{league_id}_{today_str()}.csv",
                    mime="text/csv"
                )

                if st.button("📢 Send Top 5 Recommended Bets to Discord", key="send_discord_sgo"):
                    try:
                        from props_engine_plus import format_recommended_msg, send_discord, DISCORD_WEBHOOK
                        top5 = filtered.copy()
                        if "TrueProb" in top5.columns:
                            top5 = top5[top5["TrueProb"] >= 0.53]
                        top5 = top5.sort_values("EV_Pct", ascending=False).head(5)
                        if top5.empty:
                            st.warning("⚠️ No bets meet the True Probability ≥ 53% threshold.")
                        else:
                            top5["DisplayName"] = top5.apply(
                                lambda x: x["Player"] if pd.notna(x["Player"]) and x["Player"] != "None"
                                else x.get("Game", "Unknown Game"),
                                axis=1,
                            )
                            msg_title = f"🎯 Top 5 Recommended Bets — {sport.upper()} ({today_str()})"
                            msg_lines = [
                                f"**{row['DisplayName']}** — {row['MarketName']} | {row['Side']} {row['Line']} | ({row['BookOdds']})"
                                for _, row in top5.iterrows()
                            ]
                            msg = msg_title + "\n" + "\n".join(msg_lines)
                            if DISCORD_WEBHOOK:
                                send_discord(msg, DISCORD_WEBHOOK)
                                st.success("✅ Sent Top 5 Recommended Bets to Discord!")
                            else:
                                st.warning("⚠️ Discord webhook not found in .env.")
                    except Exception as e:
                        st.error(f"❌ Failed to send to Discord: {e}")
                        
 # -------- TAB 4: Recommended Bets (Unified + Discord) --------
with tabs[3]:
    st.subheader("💡 Recommended Bets — Unified (SGO + OddsAPI + SportsData)")
    st.caption("Combines player props and team odds across sources. Filters for high-value edges and win probability thresholds.")

    # ==============================
    # 🧠 STEP 1 – Combine Available Data Sources
    # ==============================
    try:
        available_dfs = []
        if "odds_df" in locals() and isinstance(odds_df, pd.DataFrame) and not odds_df.empty:
            available_dfs.append(odds_df)
        if "sgo_df" in locals() and isinstance(sgo_df, pd.DataFrame) and not sgo_df.empty:
            available_dfs.append(sgo_df)
        if "sdata_df" in locals() and isinstance(sdata_df, pd.DataFrame) and not sdata_df.empty:
            available_dfs.append(sdata_df)

        combined_df = (
            pd.concat(available_dfs, ignore_index=True, sort=False)
            if available_dfs else pd.DataFrame()
        )
    except Exception as e:
        st.warning(f"⚠️ Error combining datasets: {e}")
        combined_df = pd.DataFrame()

    # Ensure EV/Kelly present if any source missed them
    if not combined_df.empty:
        try:
            if not set(["EV_Pct", "Kelly_Pct", "HalfKelly_Pct", "HalfKellyCapped_Pct"]).issubset(combined_df.columns):
                from props_engine_plus import compute_value_metrics
                combined_df = compute_value_metrics(combined_df, odds_col="BookOdds")
        except Exception:
            pass  # non-blocking; most sources already include these

    # ============================================================
    # ⚙️ STEP 2 — Apply Real-Time Adjustments (Injury + Weather)
    # ============================================================
    if not combined_df.empty:
        try:
            # Pick a sport for injury endpoint (simple heuristic)
            try:
                sport_choice = "nba" if "NBA" in combined_df["League"].astype(str).unique().tolist() else "nfl"
            except Exception:
                sport_choice = "nfl"

            # Pull injuries (ok if empty)
            try:
                injuries_df = fetch_injuries(sport_choice)
            except Exception as e:
                st.warning(f"⚠️ Injury fetch failed: {e}")
                injuries_df = pd.DataFrame()

            # Adjust each row; adjust_true_probability() handles weather internally
            combined_df["AdjTrueProb"] = combined_df.apply(
                lambda r: adjust_true_probability(r, injuries_df), axis=1
            )

            # If AdjTrueProb created, optionally recalc EV/Kelly off adjusted prob (non-destructive)
            if "AdjTrueProb" in combined_df.columns:
                from props_engine_plus import american_to_decimal

                def _recalc(row):
                    p = row.get("AdjTrueProb", None)
                    o = row.get("BookOdds", None)
                    try:
                        if p is None or pd.isna(p) or o is None or str(o).strip() == "":
                            return pd.Series([
                                row.get("EV_Pct"), row.get("Kelly_Pct"),
                                row.get("HalfKelly_Pct"), row.get("HalfKellyCapped_Pct")
                            ])
                        dec = american_to_decimal(o)
                        ev = ((p * dec) - 1) * 100.0 if (dec not in (None, 0)) else row.get("EV_Pct")
                        # Kelly with "true" p
                        b = dec - 1 if dec else None
                        kelly = ((p * b - (1 - p)) / b * 100.0) if (b and b != 0) else row.get("Kelly_Pct")
                        half = (kelly / 2.0) if kelly is not None else None
                        half_cap = min(half, 10.0) if half is not None else None
                        return pd.Series([ev, kelly, half, half_cap])
                    except Exception:
                        return pd.Series([
                            row.get("EV_Pct"), row.get("Kelly_Pct"),
                            row.get("HalfKelly_Pct"), row.get("HalfKellyCapped_Pct")
                        ])

                combined_df[["EV_Pct", "Kelly_Pct", "HalfKelly_Pct", "HalfKellyCapped_Pct"]] = \
                    combined_df.apply(_recalc, axis=1)
        except Exception as e:
            st.warning(f"⚠️ Adjustment step skipped due to error: {e}")
            combined_df["AdjTrueProb"] = combined_df.get("TrueProb", 0)

    # ============================================================
    # ♻️ STEP 3 — Recalibrate Model Using Bet History (optional)
    # ============================================================
    st.subheader("♻️ Model Recalibration (Learning from My Bets)")
    st.caption("Calibrates TrueProb via logistic regression when ≥ 50 bets are logged (model_feedback_log.csv).")

    auto_calibrated = False
    try:
        if os.path.exists("model_feedback_log.csv"):
            feedback = pd.read_csv("model_feedback_log.csv")
            if len(feedback) >= 50:
                feedback, calib_model = logistic_calibrate(
                    feedback,
                    prob_col="TrueProb",
                    outcome_col="Outcome"
                )
                feedback.to_csv("model_feedback_log_calibrated.csv", index=False)
                st.info(f"🤖 Auto-calibrated using {len(feedback)} historical bets.")
                auto_calibrated = True
    except Exception as e:
        st.warning(f"⚠️ Auto-calibration skipped due to error: {e}")

    if st.button("🚀 Recalibrate Model from Bet History", key="retrain_model"):
        try:
            feedback = pd.read_csv("model_feedback_log.csv")
            feedback, calib_model = logistic_calibrate(
                feedback,
                prob_col="TrueProb",
                outcome_col="Outcome"
            )
            feedback.to_csv("model_feedback_log_calibrated.csv", index=False)
            st.success("✅ Model recalibrated manually using logistic regression.")
        except Exception as e:
            st.warning(f"⚠️ Manual calibration failed: {e}")

    if not auto_calibrated:
        st.caption("ℹ️ Auto-calibration triggers once 50+ bets are logged.")

    # ============================================================
    # 🎯 STEP 4 — Filter for Recommended Bets (tunable controls)
    # ============================================================
    rec_df = pd.DataFrame()  # guard for later use

    if not combined_df.empty:
        st.markdown("### 🎛️ Recommendation Filters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            min_ev = st.number_input("Min EV %", value=5.0, step=0.5, key="rec_min_ev")
        with c2:
            max_ev = st.number_input("Max EV %", value=20.0, step=0.5, key="rec_max_ev")
        with c3:
            min_half_kelly = st.number_input("Min Half-Kelly %", value=2.0, step=0.5, key="rec_min_half_kelly")
        with c4:
            min_true = st.number_input("Min True Prob", value=0.50, step=0.01, key="rec_min_true")

        # Prefer adjusted prob if present; else fallback to TrueProb
        prob_col = "AdjTrueProb" if "AdjTrueProb" in combined_df.columns else "TrueProb"

        try:
            rec_df = combined_df[
                (combined_df["EV_Pct"].fillna(0).between(min_ev, max_ev)) &
                (combined_df["HalfKelly_Pct"].fillna(0) > float(min_half_kelly)) &
                (combined_df[prob_col].fillna(0) >= float(min_true))
            ].copy()
        except Exception:
            rec_df = pd.DataFrame()

        if rec_df.empty:
            st.warning(
                f"⚠️ No bets found with EV {min_ev:.1f}%–{max_ev:.1f}%, "
                f"Half Kelly > {min_half_kelly:.1f}%, and {prob_col} ≥ {min_true:.2f}."
            )
        else:
            rec_df = rec_df.sort_values(["EV_Pct", "HalfKelly_Pct"], ascending=False).reset_index(drop=True)
            st.dataframe(rec_df.head(50), use_container_width=True, height=520)

            # CSV download
            csv = rec_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Recommended Bets (CSV)",
                data=csv,
                file_name=f"recommended_bets_{today_str()}.csv",
                mime="text/csv",
                key="dl_recommended_csv",
            )
    else:
        st.warning("⚠️ No combined data available for recommendations.")

    # ============================================================
    # 🔔 STEP 5 — Send to Discord
    # ============================================================
    if st.button("📤 Send Top Bets to Discord", key="send_discord_tab4"):
        try:
            if rec_df.empty:
                st.warning("⚠️ No recommended bets to send. Adjust filters first.")
            else:
                top10 = rec_df.head(10).copy()
                msg = format_recommended_msg(f"🔥 Top Value Picks — {today_str()}", top10, top_n=10)

                from props_engine_plus import send_discord, DISCORD_WEBHOOK
                if DISCORD_WEBHOOK:
                    send_discord(msg, DISCORD_WEBHOOK)
                    st.success("✅ Sent Top 10 Picks to Discord!")
                else:
                    st.warning("⚠️ Discord webhook not found in .env file.")
        except Exception as e:
            st.error(f"❌ Failed to send to Discord: {e}")
            
# -------- TAB 5: AI-Driven Recommended Bets --------
with tabs[4]:
    st.subheader("🤖 AI-Driven Recommended Bets (Props Engine + EV + Kelly)")
    st.caption("Auto-scans props & game lines, filters for True Prob ≥ 60 % and strong +EV signals.")

    bankroll = st.number_input("💰 Bankroll ($)", 10, 10000, 500, 50)
    st.markdown("### 🚀 Running AI Auto-Scan Across All Sources…")

    # --- Normalize league naming for consistency (so NCAAM / NCAAB work) ---
    for df_name in ["odds_df", "sgo_df", "sdata_df"]:
        if df_name in locals():
            df = locals()[df_name]
            if isinstance(df, pd.DataFrame) and not df.empty:
                if "League" in df.columns:
                    df["League"] = (
                        df["League"]
                        .astype(str)
                        .str.replace("basketball_ncaam", "NCAAB", regex=False)
                        .str.replace("basketball_ncaab", "NCAAB", regex=False)
                        .str.replace("college_basketball", "NCAAB", regex=False)
                    )
                locals()[df_name] = df

    # --- Combine available dataframes ---
    sources = []
    for df_name in ["odds_df", "sgo_df", "sdata_df"]:
        if df_name in locals():
            df = locals()[df_name]
            if isinstance(df, pd.DataFrame) and not df.empty:
                sources.append(df)
    all_data = pd.concat(sources, ignore_index=True, sort=False) if sources else pd.DataFrame()

    # --- Ensure required columns exist ---
    base_cols = [
        "Player", "Game", "MarketType", "Side", "Line",
        "Bookmaker", "BookOdds", "TrueProb", "EV_Pct", "HalfKelly_Pct"
    ]
    for c in base_cols:
        if c not in all_data.columns:
            all_data[c] = np.nan

    # --- Display name logic ---
    all_data["DisplayName"] = all_data.apply(
        lambda x: x["Player"]
        if pd.notna(x["Player"]) and str(x["Player"]).strip()
        else (x["Game"] if pd.notna(x["Game"]) else "Unknown"),
        axis=1,
    )

    # --- Market detection for default slider presets ---
    market_hint = (
        "props"
        if all_data["Player"].notna().any()
        else "moneyline"
        if all_data["MarketType"].astype(str).str.contains("h2h", case=False, na=False).any()
        else "spread"
        if all_data["MarketType"].astype(str).str.contains("spread", case=False, na=False).any()
        else "default"
    )
    st.markdown(f"Detected Market Type: **{market_hint.title()}**")

    presets = {
        "moneyline": {"ev": 3.0, "kelly": 1.0, "true": 0.58},
        "spread": {"ev": 5.0, "kelly": 2.0, "true": 0.60},
        "props": {"ev": 6.0, "kelly": 2.5, "true": 0.62},
        "default": {"ev": 5.0, "kelly": 2.0, "true": 0.60},
    }
    p = presets.get(market_hint, presets["default"])

    min_ev = st.slider("Min EV %", 0.0, 15.0, p["ev"], 0.5)
    min_kelly = st.slider("Min Half-Kelly %", 0.0, 10.0, p["kelly"], 0.5)
    min_true = st.slider("Min True Prob (Adj)", 0.0, 1.0, p["true"], 0.01)

    # --- Stop early if no data ---
    if all_data.empty:
        st.warning("⚠️ No betting data available to scan.")
        st.stop()

    try:
        ai_df = all_data[
            (all_data["EV_Pct"].fillna(0) >= min_ev)
            & (all_data["HalfKelly_Pct"].fillna(0) >= min_kelly)
            & (all_data.get("AdjTrueProb", all_data["TrueProb"]).fillna(0) >= min_true)
        ].copy()

        if ai_df.empty:
            st.info(
                f"No AI bets met thresholds (EV ≥ {min_ev:.1f} %, Half Kelly ≥ {min_kelly:.1f} %, True Prob ≥ {min_true:.2f})."
            )
        else:
            ai_df = ai_df.sort_values(["EV_Pct", "HalfKelly_Pct"], ascending=False).reset_index(drop=True)
            top10 = ai_df.head(10)
            st.success(f"✅ Found {len(ai_df)} AI-qualified bets — showing top 10")

            # --- Format line column with color emojis ---
            def color_line(row):
                line = str(row.get("Line", "")).strip()
                if not line:
                    return ""
                mt = str(row.get("MarketType", "")).lower()
                if "spread" in mt:
                    return f"🟩 {line}" if "+" in line else f"🟥 {line}"
                if "total" in mt or "totals" in mt:
                    return f"🟧 {line}"
                return line

            top10["Line"] = top10.apply(color_line, axis=1)

            # --- Columns to display ---
            display_cols = [
                "DisplayName", "MarketType", "Side", "Line",
                "Bookmaker", "BookOdds", "TrueProb", "EV_Pct", "HalfKelly_Pct"
            ]

            # --- Style helper ---
            def style_ai(df):
                return (
                    df.style
                    .bar(subset=["EV_Pct"], color=["#e63946", "#2a9d8f"], vmin=0, vmax=df["EV_Pct"].max())
                    .bar(subset=["HalfKelly_Pct"], color=["#f4a261", "#2a9d8f"], vmin=0, vmax=df["HalfKelly_Pct"].max())
                    .format({
                        "BookOdds": "{:.2f}",
                        "TrueProb": "{:.2f}",
                        "EV_Pct": "{:.2f} %",
                        "HalfKelly_Pct": "{:.2f} %",
                    }, na_rep="")
                    .hide(axis="index")
                )

            # --- Display main table ---
            st.markdown("### 📊 AI-Ranked Opportunities (Top 10)")

            # --- Legend for line colors and EV% bars ---
            st.markdown("""
            <div style='padding: 10px; background-color: #1a1a1a; border-radius: 8px; border: 1px solid #333;'>
            <b>🎨 Line & EV% Color Legend</b><br>
            🟩 <b>Positive Line / Underdog Value</b> — indicates a favorable or +spread line.<br>
            🟥 <b>Negative Line / Favorite</b> — indicates a tighter or high-confidence side.<br>
            🟧 <b>Total / Over–Under</b> — represents game total lines (not spread or moneyline).<br><br>
            <b>📊 EV% Color Bar:</b><br>
            <span style='color:#e63946;'>🔴 Low Edge</span> → <span style='color:#f4a261;'>🟠 Moderate Edge</span> → <span style='color:#2a9d8f;'>🟢 High Edge</span><br>
            Bars represent relative expected value and Kelly strength.
            </div>
            """, unsafe_allow_html=True)

            styled_df = style_ai(top10[display_cols])
            st.dataframe(styled_df, use_container_width=True, height=550)

            # --- CSV Export (Top 50 only) ---
            csv = ai_df.head(50).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download AI Recommendations (CSV – Top 50)",
                data=csv,
                file_name=f"ai_recommended_bets_{today_str()}.csv",
                mime="text/csv",
            )

            # --- Discord Push (Top 5) ---
            st.markdown("### 🔔 Send AI Picks to Discord")
            if st.button("📢 Push AI Top 5 to Discord", key="send_discord_ai_top5"):
                try:
                    from props_engine_plus import send_discord, DISCORD_WEBHOOK

                    # Pick sport emoji
                    def sport_emoji(name):
                        name = str(name).lower()
                        if "nba" in name or "basketball" in name:
                            return "🏀"
                        if "nfl" in name or "football" in name:
                            return "🏈"
                        if "mlb" in name or "baseball" in name:
                            return "⚾"
                        if "soccer" in name or "fifa" in name:
                            return "⚽"
                        if "nhl" in name or "hockey" in name:
                            return "🏒"
                        if "ncaab" in name:
                            return "🎓🏀"
                        return "🎯"

                    league_sample = str(top10.get("League", "")).lower()
                    emoji = sport_emoji(league_sample)

                    msg_lines = [f"**{emoji} AI Top Value Picks — {today_str()}**\n"]
                    for _, row in top10.head(5).iterrows():
                        name = row.get("DisplayName", "Unknown Game")
                        market = row.get("MarketType", "Unknown")
                        side = row.get("Side", "")
                        line = row.get("Line", "")
                        odds = row.get("BookOdds", "")
                        book = row.get("Bookmaker", "")
                        ev = f"{row.get('EV_Pct', 0):.2f}%"
                        tp = f"{row.get('TrueProb', 0):.2f}"
                        msg_lines.append(f"🎯 **{name}** — {market.title()} | {side} {line} ({odds}) @ *{book}* ↳ TrueProb: {tp}, EV: {ev}")

                    msg = "\n".join(msg_lines)
                    if DISCORD_WEBHOOK:
                        send_discord(msg, DISCORD_WEBHOOK)
                        st.success("✅ AI Top 5 Picks sent to Discord!")
                    else:
                        st.warning("⚠️ Discord webhook not found in .env")
                except Exception as e:
                    st.error(f"❌ Discord push failed: {e}")

    except Exception as e:
        st.error(f"❌ Error during AI auto-scan: {e}")
        
# ==========================================================
# TAB 6: ⚙️ Engine & Discord Control Panel (Manual Run + History)
# ==========================================================
with tabs[5]:
    import pandas as pd, os, altair as alt, subprocess

    st.header("⚙️ Engine & Discord Control Panel")
    st.caption("Manually trigger the AI engine, monitor bankroll trends, and review Discord output logs.")

    st.divider()

    # ===================================================
    # 🚀 ENGINE RUN SECTION
    # ===================================================
    st.markdown("### 🚀 Run Parlay Engine")
    st.write("Click below to refresh live odds, rebuild parlays, and push results to Discord.")

    engine_script = "refresh_live_odds.py"
    if not os.path.exists(engine_script):
        st.warning(f"⚠️ Engine script not found: {engine_script}")
    else:
        if st.button("🎯 Run Engine & Post to #best-parlay", key="run_engine"):
            try:
                with st.spinner("Running engine... this may take up to 2 minutes"):
                    res = subprocess.run(
                        ["python3", engine_script],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                if res.returncode == 0:
                    st.success("✅ Engine executed successfully!")
                    st.code(res.stdout or "(no output)")
                else:
                    st.warning(f"⚠️ Engine completed with warnings (code {res.returncode})")
                    st.code(res.stderr or "(no stderr)")
            except subprocess.TimeoutExpired:
                st.error("❌ Engine timed out (120 s limit). Try again or check your script.")
            except Exception as e:
                st.error(f"❌ Error while running engine: {e}")

    st.divider()

    # ===================================================
    # 📊 HISTORY + BANKROLL TREND
    # ===================================================
    st.markdown("### 📊 Parlay History & Bankroll Trend")

    hist_path = "data/parlay_history.csv"
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)

    if os.path.exists(hist_path):
        try:
            hist = pd.read_csv(hist_path)
            if "Timestamp" in hist.columns:
                hist["Timestamp"] = pd.to_datetime(hist["Timestamp"], errors="coerce")

            # --- Summary metrics
            total_profit = round(hist.get("ExpectedProfit", pd.Series(dtype=float)).sum(), 2)
            avg_ev = round(hist.get("EV%", pd.Series(dtype=float)).mean(), 2)
            current_bankroll = round(hist.get("BankrollAfter", pd.Series([0])).iloc[-1], 2)
            win_rate = (
                (hist["Outcome"].str.lower().eq("win").mean() * 100)
                if "Outcome" in hist.columns
                else 0
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("💰 Total Profit", f"${total_profit}")
            c2.metric("📈 Avg EV %", f"{avg_ev:.2f}%")
            c3.metric("🏦 Current Bankroll", f"${current_bankroll}")
            c4.metric("🎯 Win Rate", f"{win_rate:.1f}%")

            # --- Recent history table
            st.markdown("#### Recent History")
            st.dataframe(hist.tail(10), use_container_width=True)

            # --- Bankroll trend chart
            if {"BankrollAfter", "Timestamp"}.issubset(hist.columns):
                chart = (
                    alt.Chart(hist.dropna(subset=["BankrollAfter"]))
                    .mark_line(point=True, color="#00b4d8")
                    .encode(
                        x=alt.X("Timestamp:T", title="Date/Time"),
                        y=alt.Y("BankrollAfter:Q", title="Bankroll ($)"),
                        tooltip=["Timestamp", "BankrollAfter", "ExpectedProfit"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Failed to load history: {e}")
    else:
        st.warning("No history found yet. Run the engine to create data/parlay_history.csv.")

    st.divider()

    # ===================================================
    # 🧠 DISCORD OUTPUT LOG
    # ===================================================
    st.markdown("### 🧠 Discord Output Log")
    st.info("Displays the latest results posted to your Discord webhook via the engine or AI tabs.")

    if os.path.exists(hist_path):
        try:
            hist = pd.read_csv(hist_path)
            cols = [c for c in ["Timestamp", "Legs", "ExpectedProfit", "BankrollAfter"] if c in hist.columns]
            st.dataframe(hist.tail(5)[cols], use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Failed to read Discord log: {e}")
    else:
        st.warning("No parlay log available yet — run the engine once to generate logs.")

# ============== Footer ==============
st.markdown("---")
st.caption("© Parlay +EV Pro — all odds and props are for informational purposes only.")
