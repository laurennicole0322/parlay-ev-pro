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

# --- Import AI Betting Engine ---
from props_engine_plus import (
    scan_recommended,
    push_recommended,
    push_placed_bet,
    todays_matchups_mlb_probables,
    format_recommended_msg,  # 👈 required for Discord message formatting
)

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
        "NBA", "NFL", "MLB", "NCAAF",
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


# ============== Data Fetchers ==============
# Optional: simple SportsData.io props pull (only for NBA/NFL, by date)
@st.cache_data(ttl=120)
def fetch_sportsdataio_props(key: str, sport: str, the_date: str) -> pd.DataFrame:
    """
    Minimal props feed:
      - NBA: /v3/nba/odds/json/PlayerPropsByDate/{date}
      - NFL: /v3/nfl/odds/json/PlayerPropsByDate/{date}
    Returns empty if key missing or unsupported sport.
    """
    if not key:
        return pd.DataFrame()

    sport_lower = sport.lower()
    if sport_lower not in ("nba", "nfl"):
        return pd.DataFrame()

    base = f"https://api.sportsdata.io/v3/{sport_lower}/odds/json/PlayerPropsByDate/{the_date}"
    params = {"key": key}
    try:
        r = requests.get(base, params=params, timeout=25)
        if r.status_code != 200:
            st.warning(f"SportsData.io {r.status_code}: {r.text[:200]}")
            return pd.DataFrame()
        data = r.json()
    except Exception as e:
        st.error(f"SportsData.io error: {e}")
        return pd.DataFrame()

    rows = []
    for p in data or []:
        rows.append(dict(
            Date=the_date,
            Team=p.get("Team"),
            Player=p.get("PlayerName"),
            Market=p.get("BetType"),
            StatLine=p.get("PlayerPropTotal"),
            Book=p.get("Sportsbook"),
            OddsAmerican=p.get("PayoutAmerican"),
            ImpliedProb=us_to_prob(p.get("PayoutAmerican")) if p.get("PayoutAmerican") else None,
            Game=p.get("GameDisplay") or f"{p.get('AwayTeam')} @ {p.get('HomeTeam')}"
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ImpliedProb"] = pd.to_numeric(df["ImpliedProb"], errors="coerce")
    return df

# ==========================================
# 🧠 ODDS API FETCHER + EXTRACTOR (FINAL)
# ==========================================
# --- Clean Odds API Fetcher (v4, no debug) ---
@st.cache_data(ttl=60)
def fetch_the_odds_api_games(
    odds_api_key: str,
    sport_key: str,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
) -> list:
    """
    Fetch game lines (Moneyline, Spreads, Totals) from The Odds API v4.
    Returns the raw JSON (list of events). No Streamlit prints.
    """
    if not odds_api_key:
        return []

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": odds_api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            st.warning(f"The Odds API {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        st.error(f"The Odds API error: {e}")
        return []


# --- Odds API Extractor (Moneyline, Spreads, Totals) ---
def extract_odds_api_df(raw_data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize The Odds API v4 output into a unified DataFrame.
    Handles Moneyline (h2h), Spreads, and Totals (Over/Under).
    Adds ImpliedProb, EV%, Kelly%, and Half-Kelly calculations.
    """
    if not raw_data:
        return pd.DataFrame()

    events = raw_data if isinstance(raw_data, list) else [raw_data]
    rows: List[Dict[str, Any]] = []

    for ev in events:
        sport_title = ev.get("sport_title", "")
        event_id = ev.get("id", "")
        commence = ev.get("commence_time")
        home_team = ev.get("home_team", "")
        teams = ev.get("teams") or []
        game_name = " @ ".join(teams) if teams else home_team

        for bk in ev.get("bookmakers", []) or []:
            book = bk.get("title", "")

            for mk in bk.get("markets", []) or []:
                mkey = mk.get("key", "")
                outcomes = mk.get("outcomes", []) or []

                # --- MONEYLINE ---
                if mkey == "h2h":
                    for o in outcomes:
                        side = o.get("name", "")
                        odds_str = str(o.get("price")) if o.get("price") is not None else None
                        implied = us_to_prob(odds_str)
                        dec = american_to_decimal(odds_str)
                        ev_pct = (((implied * dec) - 1) * 100.0) if (implied and dec) else None
                        kelly = (kelly_fraction(implied, dec) * 100.0) if (implied and dec) else None

                        rows.append(dict(
                            League=sport_title,
                            EventID=event_id,
                            Commence=commence,
                            Game=game_name,
                            MarketType="moneyline",
                            MarketName="moneyline",
                            Side=side,
                            Line=None,
                            BookOdds=odds_str,
                            TrueOdds=None,
                            ImpliedProb=implied,
                            TrueProb=implied,
                            EdgePct=0.0,
                            EV_Pct=ev_pct,
                            Kelly_Pct=kelly,
                            HalfKelly_Pct=(kelly / 2.0) if kelly is not None else None,
                            HalfKellyCapped_Pct=min(kelly / 2.0, 10.0) if kelly is not None else None,
                            BooksAvailable=book,
                        ))

                # --- SPREADS ---
                elif mkey == "spreads":
                    for o in outcomes:
                        side = o.get("name", "")
                        point = o.get("point")
                        odds_str = str(o.get("price")) if o.get("price") is not None else None
                        implied = us_to_prob(odds_str)
                        dec = american_to_decimal(odds_str)
                        ev_pct = (((implied * dec) - 1) * 100.0) if (implied and dec) else None
                        kelly = (kelly_fraction(implied, dec) * 100.0) if (implied and dec) else None

                        try:
                            side_label = f"{side} {float(point):+g}" if point is not None else side
                        except Exception:
                            side_label = side

                        rows.append(dict(
                            League=sport_title,
                            EventID=event_id,
                            Commence=commence,
                            Game=game_name,
                            MarketType="spread",
                            MarketName="spread",
                            Side=side_label,
                            Line=point,
                            BookOdds=odds_str,
                            TrueOdds=None,
                            ImpliedProb=implied,
                            TrueProb=implied,
                            EdgePct=0.0,
                            EV_Pct=ev_pct,
                            Kelly_Pct=kelly,
                            HalfKelly_Pct=(kelly / 2.0) if kelly is not None else None,
                            HalfKellyCapped_Pct=min(kelly / 2.0, 10.0) if kelly is not None else None,
                            BooksAvailable=book,
                        ))

                # --- TOTALS (OVER/UNDER) ---
                elif mkey == "totals":
                    for o in outcomes:
                        side = o.get("name", "")
                        point = o.get("point")
                        odds_str = str(o.get("price")) if o.get("price") is not None else None
                        implied = us_to_prob(odds_str)
                        dec = american_to_decimal(odds_str)
                        ev_pct = (((implied * dec) - 1) * 100.0) if (implied and dec) else None
                        kelly = (kelly_fraction(implied, dec) * 100.0) if (implied and dec) else None

                        try:
                            side_label = f"{side} {float(point):g}" if point is not None else side
                        except Exception:
                            side_label = side

                        rows.append(dict(
                            League=sport_title,
                            EventID=event_id,
                            Commence=commence,
                            Game=game_name,
                            MarketType="total_points",
                            MarketName="total_points",
                            Side=side_label,
                            Line=point,
                            BookOdds=odds_str,
                            TrueOdds=None,
                            ImpliedProb=implied,
                            TrueProb=implied,
                            EdgePct=0.0,
                            EV_Pct=ev_pct,
                            Kelly_Pct=kelly,
                            HalfKelly_Pct=(kelly / 2.0) if kelly is not None else None,
                            HalfKellyCapped_Pct=min(kelly / 2.0, 10.0) if kelly is not None else None,
                            BooksAvailable=book,
                        ))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Cleanup / Sort
    df["Commence"] = pd.to_datetime(df["Commence"], errors="coerce")
    df = df.sort_values(["Commence", "Game", "MarketType", "Side"], ignore_index=True)

    # Add formatted %
    df["ImpliedProb%"] = df["ImpliedProb"].apply(fmt_pct)
    df["TrueProb%"] = df["TrueProb"].apply(fmt_pct)
    df["Edge%"] = df["EdgePct"].map(lambda x: f"{x:0.2f}%" if pd.notna(x) else "")

    # Order columns for dashboard
    ordered_cols = [
        "League", "EventID", "Commence", "Game",
        "MarketType", "MarketName", "Side", "Line",
        "BookOdds", "TrueOdds", "ImpliedProb%", "TrueProb%",
        "EV_Pct", "Kelly_Pct", "HalfKelly_Pct", "HalfKellyCapped_Pct",
        "BooksAvailable"
    ]
    extras = [c for c in df.columns if c not in ordered_cols]
    return df[[c for c in ordered_cols if c in df.columns] + extras]

# ==========================================
# 🎯 SPORTS GAME ODDS (SGO) FETCHER (FINAL)
# ==========================================
@st.cache_data(ttl=60)
def fetch_sgo_events(sgo_api_key: str, league_id: str, limit: int = 50) -> Dict[str, Any]:
    """
    Pull full market data from SportsGameOdds v2/events endpoint.
    Returns JSON payload for use with extract_sgo_df().
    """
    if not sgo_api_key:
        return {}

    params = {
        "apiKey": sgo_api_key,
        "oddsAvailable": "true",
        "leagueID": league_id,
        "limit": limit,
        "includeAltLines": "true",
        "includeOpposingOdds": "true",
        "include": "players,teams,markets,odds",  # ✅ full context for parsing
    }

    url = "https://api.sportsgameodds.com/v2/events?"
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            try:
                body = r.json()
                st.warning(f"SGO {r.status_code}: {body.get('error') or str(body)[:200]}")
            except Exception:
                st.warning(f"SGO {r.status_code}: {r.text[:200]}")
            return {}
        return r.json()
    except Exception as e:
        st.error(f"SGO error: {e}")
        return {}

def normalize_market_name(market_key: str, market_name: str) -> str:
    """
    Convert raw market names/keys into friendly categories (e.g., 'Player Points O/U').
    This helps with filtering and grouping in the Streamlit dashboard.
    """
    key = (market_key or "").lower()
    name = (market_name or "").lower()

    def contains(*words):
        return all(w in key or w in name for w in words)


    # ----- Explicit Over / Under detection -----
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

    # Default: return readable fallback
    return market_name or market_key
def choose_line(odd_obj: Dict[str, Any]) -> Optional[float]:
    """
    Pick the best available numeric line (O/U or spread) for display.
    Priority order: bookOverUnder → bookSpread → fairOverUnder → fairSpread
    """
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
    """
    Return (bookOdds, fairOdds) as normalized strings like '+120' or '-130'.
    """
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

# ============================================
# 🧠 UNIFIED ENHANCED EXTRACTORS (v3.1) — Over / Under / Yes / No support
# ============================================
def extract_sgo_df(payload: Dict[str, Any], wanted_books: List[str]) -> pd.DataFrame:
    """
    Flatten SportsGameOdds (SGO) events JSON into a DataFrame.
    Ensures Over, Under, Yes, and No sides are all captured and normalized.
    Adds implied vs true probability columns and betting metrics.
    """
    if not payload or not payload.get("success"):
        return pd.DataFrame()

    events = payload.get("data") or []
    rows: List[Dict[str, Any]] = []

    for ev in events:
        event_id = ev.get("eventID")
        league_id = ev.get("leagueID")

        teams = safe_get(ev, "teams") or {}
        away = safe_get(teams, "away", "names", "long", default="Away")
        home = safe_get(teams, "home", "names", "long", default="Home")
        game_name = f"{away} @ {home}"

        players_map = safe_get(ev, "players") or {}
        odds = ev.get("odds") or {}

        # --- Merge similar markets (Over/Under, Yes/No) ---
        merged_odds = {}
        for k, v in odds.items():
            base_key = (
                k.replace("_ou_over", "over")
                 .replace("_ou_under", "under")
                 .replace("-ou-over", "-ou")
                 .replace("-ou-under", "-ou")
                 .replace("_yes", "_yn")
                 .replace("_no", "_yn")
                 .replace("-yes", "-yn")
                 .replace("-no", "-yn")
            )
            merged_odds.setdefault(base_key, []).append((k, v))

        for base_key, variants in merged_odds.items():
            for oddID, odd_obj in variants:
                market_name = odd_obj.get("marketName", oddID)
                market_type = normalize_market_name(oddID, market_name)

                # --- Detect Side (Over/Under/Yes/No) ---
                id_lower = oddID.lower()
                name_lower = market_name.lower()
                if "over" in id_lower or "o" in name_lower:
                    side_label = "Over"
                elif "under" in id_lower or "u" in name_lower:
                    side_label = "Under"
                elif "yes" in id_lower or "yes" in name_lower:
                    side_label = "Yes"
                elif "no" in id_lower or "no" in name_lower:
                    side_label = "No"
                else:
                    # fallback to helper
                    side_label = determine_side_label(
                        oddID=oddID,
                        market_name=market_name,
                        book_line=odd_obj.get("point"),
                        odd_obj=odd_obj,
                        home=home,
                        away=away
                    )

                # --- Player / entity info ---
                stat_entity = odd_obj.get("statEntityID")
                player_id = odd_obj.get("playerID") or (
                    stat_entity if stat_entity not in ("home", "away", "all") else None
                )
                player_name = (
                    safe_get(players_map, player_id, "name", default=player_id)
                    if player_id in players_map else None
                )

                # --- Filter for wanted bookmakers ---
                bybk = odd_obj.get("byBookmaker") or {}
                chosen = {
                    bk: val for bk, val in bybk.items()
                    if (not wanted_books or bk in wanted_books) and isinstance(val, dict)
                }
                if not chosen:
                    continue

                # --- Extract price & line ---
                book_line = choose_line(odd_obj)
                book_odds_str, fair_odds_str = choose_odds(odd_obj)

                # --- Calculate metrics ---
                implied_prob = implied_probability(book_odds_str)
                true_prob = implied_probability(fair_odds_str)
                edge_pct = ((true_prob - implied_prob) * 100.0) if (true_prob and implied_prob) else None
                dec = american_to_decimal(book_odds_str)
                ev_pct = (((true_prob * dec) - 1) * 100.0) if (true_prob and dec) else None
                kelly = (kelly_fraction(true_prob, dec) * 100.0) if (true_prob and dec) else None
                half_kelly = (kelly / 2.0) if kelly is not None else None
                half_kelly_capped = min(half_kelly, 10.0) if half_kelly is not None else None

                rows.append(dict(
                    League=league_id,
                    EventID=event_id,
                    Game=game_name,
                    Player=player_name,
                    PlayerID=player_id,
                    MarketType=market_type,
                    MarketName=market_name,
                    Side=side_label,  # ✅ Over / Under / Yes / No
                    Line=book_line,
                    BookOdds=book_odds_str,
                    TrueOdds=fair_odds_str,
                    ImpliedProb=implied_prob,
                    TrueProb=true_prob,
                    EdgePct=edge_pct,
                    EV_Pct=ev_pct,
                    Kelly_Pct=kelly,
                    HalfKelly_Pct=half_kelly,
                    HalfKellyCapped_Pct=half_kelly_capped,
                    BooksAvailable=",".join(sorted(chosen.keys())) if chosen else ""
                ))

    return finalize_odds_df(rows)

# ============== Enhanced extract_odds_api_df (clean + O/U grouping) ==============
from typing import Union

def extract_odds_api_df(raw_data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize The Odds API v4 output into a unified DataFrame.
    Handles Moneyline (h2h), Spreads, and Totals (Over/Under) explicitly.
    Adds ImpliedProb, EV%, Kelly%, plus formatted % columns.
    """
    if not raw_data:
        return pd.DataFrame()

    events = raw_data if isinstance(raw_data, list) else [raw_data]
    rows: List[Dict[str, Any]] = []

    for ev in events:
        sport_title = ev.get("sport_title", "")
        event_id = ev.get("id", "")
        commence = ev.get("commence_time")
        home_team = ev.get("home_team", "")
        teams = ev.get("teams") or []
        game_name = " @ ".join(teams) if teams else home_team

        for bk in ev.get("bookmakers", []) or []:
            book = bk.get("title", "")

            for mk in bk.get("markets", []) or []:
                mkey = mk.get("key", "")  # 'h2h', 'spreads', 'totals'
                outcomes = mk.get("outcomes", []) or []

                if mkey == "h2h":
                    # Moneyline: two teams, name==team, no point
                    for o in outcomes:
                        side = o.get("name", "")
                        odds_str = str(o.get("price")) if o.get("price") is not None else None
                        implied = us_to_prob(odds_str)
                        dec = american_to_decimal(odds_str)
                        ev_pct = (((implied * dec) - 1) * 100.0) if (implied and dec) else None
                        kelly = (kelly_fraction(implied, dec) * 100.0) if (implied and dec) else None

                        rows.append(dict(
                            League=sport_title,
                            EventID=event_id,
                            Game=game_name,
                            Player=None,
                            PlayerID=None,
                            MarketType="moneyline",
                            MarketName="moneyline",
                            Side=side,
                            Line=None,
                            BookOdds=odds_str,
                            TrueOdds=None,
                            ImpliedProb=implied,
                            TrueProb=implied,
                            EdgePct=0.0,
                            EV_Pct=ev_pct,
                            Kelly_Pct=kelly,
                            HalfKelly_Pct=(kelly/2.0) if kelly is not None else None,
                            HalfKellyCapped_Pct=min(kelly/2.0, 10.0) if kelly is not None else None,
                            BooksAvailable=book,
                            Commence=commence,
                        ))

                elif mkey == "spreads":
                    for o in outcomes:
                        side = o.get("name", "")
                        point = o.get("point")
                        odds_str = str(o.get("price")) if o.get("price") is not None else None
                        implied = us_to_prob(odds_str)
                        dec = american_to_decimal(odds_str)
                        ev_pct = (((implied * dec) - 1) * 100.0) if (implied and dec) else None
                        kelly = (kelly_fraction(implied, dec) * 100.0) if (implied and dec) else None

                        # Label as "<Team> -X.X" or "<Team> +X.X"
                        try:
                            spread_lbl = f"{side} {float(point):+g}" if point is not None else side
                        except Exception:
                            spread_lbl = side

                        rows.append(dict(
                            League=sport_title,
                            EventID=event_id,
                            Game=game_name,
                            Player=None,
                            PlayerID=None,
                            MarketType="spread",
                            MarketName="spread",
                            Side=spread_lbl,
                            Line=point,
                            BookOdds=odds_str,
                            TrueOdds=None,
                            ImpliedProb=implied,
                            TrueProb=implied,
                            EdgePct=0.0,
                            EV_Pct=ev_pct,
                            Kelly_Pct=kelly,
                            HalfKelly_Pct=(kelly/2.0) if kelly is not None else None,
                            HalfKellyCapped_Pct=min(kelly/2.0, 10.0) if kelly is not None else None,
                            BooksAvailable=book,
                            Commence=commence,
                        ))

                elif mkey == "totals":
                    # Over/Under
                    for o in outcomes:
                        side_raw = o.get("name", "")  # "Over" / "Under"
                        point = o.get("point")
                        odds_str = str(o.get("price")) if o.get("price") is not None else None
                        implied = us_to_prob(odds_str)
                        dec = american_to_decimal(odds_str)
                        ev_pct = (((implied * dec) - 1) * 100.0) if (implied and dec) else None
                        kelly = (kelly_fraction(implied, dec) * 100.0) if (implied and dec) else None

                        # Side label: "Over 223.5" / "Under 223.5"
                        try:
                            side_lbl = f"{side_raw} {float(point):g}" if point is not None else side_raw
                        except Exception:
                            side_lbl = side_raw

                        rows.append(dict(
                            League=sport_title,
                            EventID=event_id,
                            Game=game_name,
                            Player=None,
                            PlayerID=None,
                            MarketType="total_points",
                            MarketName="total_points",
                            Side=side_lbl,
                            Line=point,
                            BookOdds=odds_str,
                            TrueOdds=None,
                            ImpliedProb=implied,
                            TrueProb=implied,
                            EdgePct=0.0,
                            EV_Pct=ev_pct,
                            Kelly_Pct=kelly,
                            HalfKelly_Pct=(kelly/2.0) if kelly is not None else None,
                            HalfKellyCapped_Pct=min(kelly/2.0, 10.0) if kelly is not None else None,
                            BooksAvailable=book,
                            Commence=commence,
                        ))
                else:
                    # Ignore other markets (e.g., player-level props) since your call doesn't request them
                    continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Dtypes + pretty %
    for c in ["Line", "ImpliedProb", "TrueProb", "EdgePct", "EV_Pct", "Kelly_Pct", "HalfKelly_Pct", "HalfKellyCapped_Pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ImpliedProb%"] = df["ImpliedProb"].apply(fmt_pct)
    df["TrueProb%"]    = df["TrueProb"].apply(fmt_pct)
    df["Edge%"]        = df["EdgePct"].map(lambda x: f"{x:0.2f}%" if pd.notna(x) else "")

    df["Commence"] = pd.to_datetime(df["Commence"], errors="coerce")
    df = df.sort_values(["Commence", "Game", "MarketType", "Side"], ignore_index=True)

    ordered_cols = [
        "League", "EventID", "Commence", "Game",
        "MarketType", "MarketName", "Side", "Line",
        "BookOdds", "TrueOdds",
        "EV_Pct", "Kelly_Pct", "HalfKelly_Pct", "HalfKellyCapped_Pct",
        "BooksAvailable", "ImpliedProb%", "TrueProb%", "Edge%"
    ]
    extras = [c for c in df.columns if c not in ordered_cols]
    return df[[c for c in ordered_cols if c in df.columns] + extras]

def extract_sportsdataio_df(raw_data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize SportsData.io props into the unified format.
    Handles Over/Under inference, implied probabilities, and Kelly metrics.
    """
    if not raw_data:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    # Always treat as list
    for p in (raw_data if isinstance(raw_data, list) else [raw_data]):
        game = p.get("GameDisplay") or f"{p.get('AwayTeam')} @ {p.get('HomeTeam')}"
        player = p.get("PlayerName")
        team = p.get("Team")
        market = p.get("BetType") or ""
        stat_line = p.get("PlayerPropTotal")
        odds = p.get("PayoutAmerican")
        book = p.get("Sportsbook")

        # --- Infer Side ---
        # Try to read explicit side, or derive from text
        market_lower = market.lower()
        if "over" in market_lower:
            side_label = f"Over {stat_line}" if stat_line else "Over"
        elif "under" in market_lower:
            side_label = f"Under {stat_line}" if stat_line else "Under"
        else:
            # default to “Both” (some props come as combined market)
            side_label = determine_side_label(market, market, stat_line, p, p.get("HomeTeam"), p.get("AwayTeam"))

        # --- Compute metrics ---
        implied_prob = us_to_prob(odds)
        true_prob = implied_prob
        dec = american_to_decimal(odds)
        ev_pct = (((true_prob * dec) - 1) * 100.0) if (true_prob and dec) else None
        kelly = (kelly_fraction(true_prob, dec) * 100.0) if (true_prob and dec) else None
        half_kelly = (kelly / 2.0) if kelly is not None else None
        half_kelly_capped = min(half_kelly, 10.0) if half_kelly is not None else None

        rows.append(dict(
            League=p.get("League") or "",
            EventID=p.get("EventID") or "",
            Game=game,
            Player=player,
            PlayerID=None,
            MarketType=market,
            MarketName=market,
            Side=side_label,
            Line=stat_line,
            BookOdds=odds,
            TrueOdds=None,
            ImpliedProb=implied_prob,
            TrueProb=true_prob,
            EV_Pct=ev_pct,
            Kelly_Pct=kelly,
            HalfKelly_Pct=half_kelly,
            HalfKellyCapped_Pct=half_kelly_capped,
            BooksAvailable=book
        ))

    return finalize_odds_df(rows)

# ============================================
# 🔍 Helper functions shared by all extractors
# ============================================
def determine_side_label(oddID, market_name, book_line, odd_obj, home, away) -> str:
    """
    Universal side-label detector for SportsGameOdds + Odds API.
    Combines explicit pattern detection with intelligent fallbacks.
    Covers:
      - Over/Under (Game, Team, Player)
      - Spread & Moneyline
      - Player props (Points, Assists, Rebounds, Steals, 3PT, Touchdowns, etc.)
      - Generic player stats (e.g., Hits, Strikeouts, Tackles)
      - Yes/No and Team Totals
    Works across NBA, NCAAB, WNBA, NFL, NCAAF, MLB, etc.
    """
    id_lower = str(oddID or "").lower()
    name_lower = str(market_name or "").lower()
    combined = f"{id_lower} {name_lower}"

    # ------------------------------------------------
    # 🟢 OVER / UNDER MARKETS (explicit)
    # ------------------------------------------------
    if any(x in combined for x in [
        # Game totals
        "points-all-game-ou-over", "points_all_game_ou_over",
        # Team totals
        "points-home-game-ou-over", "points-away-game-ou-over",
        "points_home_game_ou_over", "points_away_game_ou_over",
        # Player props (NBA/NFL)
        "points-any_player_id-game-ou-over",
        "assists-any_player_id-game-ou-over",
        "rebounds-any_player_id-game-ou-over",
        "steals-any_player_id-game-ou-over",
        "three_pointers-made-any_player_id-game-ou-over",
        "threepointers-made-any_player_id-game-ou-over",
        "touchdowns-any_player_id-game-ou-over",
        "rushing_touchdowns-any_player_id-game-ou-over",
        "points_any_player_id_game_ou_over",
        "assists_any_player_id_game_ou_over",
        "rebounds_any_player_id_game_ou_over",
        "steals_any_player_id_game_ou_over",
        "three_pointers_made_any_player_id_game_ou_over",
        "touchdowns_any_player_id_game_ou_over",
        "rushing_touchdowns_any_player_id_game_ou_over",
        # Generic
        "ou-over", " o/u over", " over", "over:"
    ]):
        return f"Over {book_line}" if book_line not in (None, "", "None") else "Over"

    if any(x in combined for x in [
        # Game totals
        "points-all-game-ou-under", "points_all_game_ou_under",
        # Team totals
        "points-home-game-ou-under", "points-away-game-ou-under",
        "points_home_game_ou_under", "points_away_game_ou_under",
        # Player props (NBA/NFL)
        "points-any_player_id-game-ou-under",
        "assists-any_player_id-game-ou-under",
        "rebounds-any_player_id-game-ou-under",
        "steals-any_player_id-game-ou-under",
        "three_pointers-made-any_player_id-game-ou-under",
        "threepointers-made-any_player_id-game-ou-under",
        "touchdowns-any_player_id-game-ou-under",
        "rushing_touchdowns-any_player_id-game-ou-under",
        "points_any_player_id_game_ou_under",
        "assists_any_player_id_game_ou_under",
        "rebounds_any_player_id_game_ou_under",
        "steals_any_player_id_game_ou_under",
        "three_pointers_made_any_player_id_game_ou_under",
        "touchdowns_any_player_id_game_ou_under",
        "rushing_touchdowns_any_player_id_game_ou_under",
        # Generic
        "ou-under", " o/u under", " under", "under:"
    ]):
        return f"Under {book_line}" if book_line not in (None, "", "None") else "Under"

    # ------------------------------------------------
    # 🟠 FALLBACK FOR STRUCTURED "O/U" BUT NO SIDE WORD
    # ------------------------------------------------
    if ("over/under" in combined or " o/u " in combined) and not any(x in combined for x in ["over", "under"]):
        return f"Over {book_line or ''}".strip()

    # ------------------------------------------------
    # 🟣 SPREAD MARKETS
    # ------------------------------------------------
    if any(x in combined for x in ["points-home-game-sp-home", "points_home_game_sp_home"]):
        return f"{home} Spread {book_line}" if book_line else f"{home} Spread"

    if any(x in combined for x in ["points-away-game-sp-away", "points_away_game_sp_away"]):
        return f"{away} Spread {book_line}" if book_line else f"{away} Spread"

    # ------------------------------------------------
    # 🔵 MONEYLINE MARKETS
    # ------------------------------------------------
    if any(x in combined for x in ["points-home-game-ml-home", "points_home_game_ml_home"]):
        return f"{home} ML"
    if any(x in combined for x in ["points-away-game-ml-away", "points_away_game_ml_away"]):
        return f"{away} ML"

    if "moneyline" in combined or combined.strip().endswith(" ml"):
        side_id = (odd_obj or {}).get("sideID", "").lower() if odd_obj else ""
        if "home" in combined or side_id == "home":
            return home
        if "away" in combined or side_id == "away":
            return away
        return "Moneyline"

    # ------------------------------------------------
    # 🟡 YES / NO MARKETS
    # ------------------------------------------------
    if any(x in combined for x in [" yes", "will score", "first touchdown", "to score", "made", "converted", "achieve"]):
        return "Yes"
    if any(x in combined for x in [" no", "will not", "not score", "missed", "failed"]):
        return "No"

    # ------------------------------------------------
    # 🟤 TEAM TOTALS
    # ------------------------------------------------
    if "team total" in combined or "team points" in combined:
        if "home" in combined:
            return f"{home} Team Total"
        if "away" in combined:
            return f"{away} Team Total"
        return "Team Total"

    # ------------------------------------------------
    # ⚫ GENERAL PLAYER PROP DETECTION (fallbacks)
    # ------------------------------------------------
    for label, key in [
        ("Strikeouts", "strikeouts"),
        ("Hits", "hits"),
        ("RBIs", "rbi"),
        ("Home Run", "home run"),
        ("Passing Yards", "passing yards"),
        ("Receiving Yards", "receiving yards"),
        ("Rushing Yards", "rushing yards"),
        ("Assists", "assists"),
        ("Points", "points"),
        ("Rebounds", "rebounds"),
        ("3PT", "3pt"),
        ("Three Pointers", "three pointers"),
        ("Tackles", "tackles"),
        ("Interceptions", "interceptions"),
    ]:
        if key in combined:
            return f"{label} {book_line}" if book_line not in (None, "", "None") else label

    # ------------------------------------------------
    # 🧩 DEFAULT CATCH-ALL
    # ------------------------------------------------
    return "Unknown"

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
    "AI-Driven Bets"
])

## -------- TAB 1: Player Props (SportsData.io) --------
with tabs[0]:
    st.subheader("Player Props — SportsData.io")
    st.caption("Works best for NBA/NFL. Enter your SportsData.io key to enable.")

    props_date = st.date_input(
        "Props Date",
        value=date.today(),
        key="sdata_date_picker"
    )
    props_date_str = props_date.strftime("%Y-%m-%d")

    # --- Check for API key ---
    if not sdata_key:
        st.info("Enter your SportsData.io API key in the sidebar to fetch props.")
    else:
        # --- Fetch raw data ---
        raw_sdata = fetch_sportsdataio_props(sdata_key, sport, props_date_str)

        # --- Normalize + extract ---
        sdata_df = extract_sportsdataio_df(
            raw_sdata.to_dict(orient="records")
            if isinstance(raw_sdata, pd.DataFrame)
            else raw_sdata
        )

        if sdata_df.empty:
            st.warning("No props returned right now for this sport/date.")
        else:
            with st.expander("Sample (head)", expanded=True):
                st.dataframe(sdata_df.head(100), use_container_width=True)

            # --- CSV Export ---
            csv = sdata_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV (SportsData.io Props)",
                data=csv,
                file_name=f"sportsdataio_props_{sport}_{props_date_str}.csv",
                mime="text/csv",
                key="dl_sdata_props"
            )

# -------- TAB 2: Game Lines (The Odds API) --------
with tabs[1]:
    st.subheader("Game Lines — The Odds API")

    sport_map_odds = {
    "NBA": "basketball_nba",
    "NFL": "americanfootball_nfl",
    "MLB": "baseball_mlb",
    "NCAAF": "americanfootball_ncaaf",

    # ⚽ All major soccer leagues supported by The Odds API
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
    sport_key = sport_map_odds.get(sport, "basketball_nba")

    if not odds_key:
        st.info("Enter your The Odds API key in the sidebar to fetch game lines.")
    else:
        raw_odds = fetch_the_odds_api_games(odds_key, sport_key)
        odds_df = extract_odds_api_df(raw_odds)

        if odds_df.empty:
            st.warning("No odds returned.")
        else:
            with st.expander("Data (head)", expanded=True):
                st.dataframe(odds_df.head(100), width="stretch")

            csv = odds_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV (The Odds API Game Lines)",
                data=csv,
                file_name=f"odds_api_{sport_key}_{today_str()}.csv",
                mime="text/csv",
                key="dl_odds_lines",
            )
            
# -------- TAB 3: SportsgameOdds (Player Props + Markets) --------
with tabs[2]:
    st.subheader("SportsgameOdds — Player Props + Markets (Implied vs True)")

    # Optional auto-refresh for NBA/NFL
    if auto_refresh and sport.upper() in ("NBA", "NFL"):
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60 * 1000, key=f"auto_refresh_{sport.lower()}")
        st.caption("🔄 Auto-refreshing every 60 seconds (NBA/NFL).")

    sgo_league_map = {
        "NBA": "NBA",
        "NFL": "NFL",
        "MLB": "MLB",
        "NCAAF": "NCAAF",
        "Soccer": "SOC"
    }
    league_id = sgo_league_map.get(sport, "NBA")

    if not sgo_key:
        st.info("Enter your SportsGameOdds API key in the sidebar to fetch live props & markets.")
    else:
        # --- Fetch & process ---
        payload = fetch_sgo_events(sgo_key, league_id, limit=50)
        sgo_df = extract_sgo_df(payload, wanted_books=book_filter)

        if sgo_df.empty:
            st.warning("No markets/props returned. (If your tier is limited, try a smaller league or remove filters.)")
        else:
            # --- Filters ---
            colf1, colf2, colf3 = st.columns([1, 1, 1])

            with colf1:
                unique_markets = sorted(sgo_df["MarketType"].dropna().unique().tolist())
                chosen_market = st.selectbox(
                    "Market Type",
                    ["All"] + unique_markets,
                    index=0,
                    key="sgo_market_select"
                )

            with colf2:
                player_q = st.text_input("Player name contains (optional)", "", key="sgo_player_query")

            with colf3:
                edge_min = st.number_input("Min Edge %", value=edge_floor, step=0.5, key="sgo_edge_min")

            # --- Apply filters ---
            filtered = sgo_df.copy()
            if chosen_market != "All":
                filtered = filtered[filtered["MarketType"] == chosen_market]
            if player_q.strip():
                filtered = filtered[filtered["Player"].fillna("").str.contains(player_q.strip(), case=False, na=False)]
            if edge_min is not None:
                filtered = filtered[(filtered["EdgePct"].fillna(-9999) >= float(edge_min))]

            # --- Display ---
            if not filtered.empty:
                show = filtered.copy()
                show["ImpliedProb%"] = show["ImpliedProb"].apply(fmt_pct)
                show["TrueProb%"] = show["TrueProb"].apply(fmt_pct)
                show["Edge%"] = show["EdgePct"].map(lambda x: f"{x:0.2f}%" if pd.notna(x) else "")

                show = show.drop(columns=["ImpliedProb", "TrueProb", "EdgePct"], errors="ignore")
                st.dataframe(show, use_container_width=True, height=520)

                # --- CSV Export ---
                csv = filtered.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV (SGO Props + Markets)",
                    data=csv,
                    file_name=f"sgo_{league_id}_{today_str()}.csv",
                    mime="text/csv",
                    key="dl_sgo_csv"
                )
            else:
                st.info("No rows match your filters.")
                
# -------- TAB 4: Recommended Bets (Legacy) --------
# -------- TAB 4: Recommended Bets (Unified + Discord) --------
with tabs[3]:
    st.subheader("🎯 Recommended Bets — Unified (SGO + OddsAPI + SportsData)")
    st.caption("Combines player props and team odds across sources. Filters for high-value edges and win probability ≥ 60%.")

    # ==========================================================
    # 🧩 STEP 1 — Combine Available Data Sources
    # ==========================================================
    try:
        available_dfs = []

        if "odds_df" in locals() and isinstance(odds_df, pd.DataFrame) and not odds_df.empty:
            available_dfs.append(odds_df)
        if "sgo_df" in locals() and isinstance(sgo_df, pd.DataFrame) and not sgo_df.empty:
            available_dfs.append(sgo_df)
        if "sdata_df" in locals() and isinstance(sdata_df, pd.DataFrame) and not sdata_df.empty:
            available_dfs.append(sdata_df)

        combined_df = pd.concat(available_dfs, ignore_index=True, sort=False) if available_dfs else pd.DataFrame()
    except Exception as e:
        st.warning(f"Error combining datasets: {e}")
        combined_df = pd.DataFrame()

    # ==========================================================
    # ⚙️ STEP 2 — Apply Real-Time Adjustments (Injury + Weather)
    # ==========================================================
    if not combined_df.empty:
        try:
            sport_choice = "nba" if "NBA" in combined_df["League"].astype(str).unique().tolist() else "nfl"
        except Exception:
            sport_choice = "nfl"

        # detect home city from first game
        first_game = str(combined_df["Game"].iloc[0]) if "Game" in combined_df.columns else ""
        city_guess = detect_city_from_game(first_game) or "miami"

        try:
            injuries_df = fetch_injuries(sport_choice)
        except Exception as e:
            st.warning(f"⚠️ Injury fetch failed: {e}")
            injuries_df = pd.DataFrame()

        try:
            weather_data = fetch_weather(city_guess)
        except Exception as e:
            st.warning(f"⚠️ Weather fetch failed: {e}")
            weather_data = {}

        # ✅ Apply adjustments only if data is available
        try:
            if (not injuries_df.empty) or (weather_data and len(weather_data) > 0):
                combined_df["AdjTrueProb"] = combined_df.apply(
                    lambda r: adjust_true_probability(r, injuries_df), axis=1
                )
            else:
                st.info("No injury or weather data available — skipping adjustments.")
                combined_df["AdjTrueProb"] = combined_df.get("TrueProb", 0)
        except Exception as e:
            st.warning(f"⚠️ Adjustment step skipped due to error: {e}")
            combined_df["AdjTrueProb"] = combined_df.get("TrueProb", 0)

    # ==========================================================
    # 🎯 STEP 3 — Filter for Recommended Bets
    # ==========================================================
    if not combined_df.empty:
        rec_df = combined_df[
            (combined_df["EV_Pct"].fillna(0) > 5) &
            (combined_df["HalfKelly_Pct"].fillna(0) > 2) &
            (combined_df["AdjTrueProb"].fillna(0) >= 0.60)
        ].copy()

        if rec_df.empty:
            st.warning("No high-edge bets found (EV > 5%, Half Kelly > 2%, True Prob ≥ 60%).")
        else:
            rec_df = rec_df.sort_values(["EV_Pct", "HalfKelly_Pct"], ascending=False)
            st.dataframe(rec_df.head(50), use_container_width=True, height=520)

            # --- CSV export ---
            csv = rec_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Recommended Bets (CSV)",
                data=csv,
                file_name=f"recommended_bets_{today_str()}.csv",
                mime="text/csv",
                key="dl_recbets"
            )

            # ======================================================
            # 🔔 STEP 4 — Send to Discord
            # ======================================================
            if st.button("📢 Send Top Bets to Discord", key="send_discord_tab4"):
                top10 = rec_df.head(10)
                msg = format_recommended_msg(f"Top Value Picks — {today_str()}", top10, top_n=10)
                push_recommended(top10, title="Top Value Picks")
                st.success("✅ Sent Top 10 Picks to Discord!")
                
# -------- TAB 5: AI-Driven Recommended Bets (Props Engine) --------
with tabs[4]:
    st.subheader("🤖 AI-Driven Recommended Bets (Props Engine + EV + Kelly)")
    st.caption("Uses player prop history, opponent stats, live odds, weather, and bankroll sizing logic.")

    # --- Choose Sport ---
    sport_choice = st.selectbox(
        "Choose Sport", 
        ["NBA", "MLB", "NFL", "NCAAF", "NCAAB"]
    )
    bankroll = st.number_input("Bankroll ($)", min_value=10, max_value=10000, value=100)

    # --- Default Inputs by Sport ---
    if sport_choice == "NBA":
        default_players = ["Cade Cunningham", "Jalen Brunson", "Trae Young"]
        default_stat = "assists"
        default_opp = "BOS"
    elif sport_choice == "MLB":
        default_players = ["Shohei Ohtani", "Mookie Betts", "Juan Soto"]
        default_stat = "hits"
        default_opp = "TOR"
    elif sport_choice == "NFL":
        default_players = ["Justin Jefferson", "CeeDee Lamb"]
        default_stat = "receiving_yards"
        default_opp = "GB"
    elif sport_choice == "NCAAF":
        default_players = ["Bo Nix", "Jalen Milroe"]
        default_stat = "passing_yards"
        default_opp = "UGA"
    else:  # NCAAB
        default_players = ["Zach Edey", "Armando Bacot"]
        default_stat = "points"
        default_opp = "MSU"

    # --- Input Fields ---
    st.markdown("### Enter Players to Scan")
    players = st.text_area("Player Names (one per line):", "\n".join(default_players))
    opponent = st.text_input("Opponent (abbr):", default_opp)
    stat = st.text_input("Stat Type:", default_stat)

    # --- Run Scan ---
    if st.button("🔍 Scan AI-Driven Recommendations"):
        tickets = [
            {"player": p.strip(), "opponent": opponent, "stat": stat}
            for p in players.splitlines() if p.strip()
        ]

        with st.spinner("Fetching odds and calculating EV / Kelly values..."):
            results = scan_recommended(
                sport=sport_choice,
                tickets=tickets,
                seasons=[2023, 2024],
                bankroll=bankroll,
                preferred_book_key="hardrockbet",
                odds_sport_key={
                    "NBA": "basketball_nba",
                    "MLB": "baseball_mlb",
                    "NFL": "americanfootball_nfl",
                    "NCAAF": "americanfootball_ncaaf",
                    "NCAAB": "basketball_ncaab",  # 👈 added for college basketball
                }[sport_choice],
            )

        # --- Display Results ---
        if results is not None and not results.empty:
            st.success(f"Found {len(results)} opportunities ✅")
            st.dataframe(results, use_container_width=True, height=520)

            # ✅ Store results so the Legacy Recommended Bets tab (Tab 4) can access them
            st.session_state["ai_results"] = results

            st.markdown("#### 🔔 Send to Discord (#recommended-bets)")
            if st.button("📢 Push Top Results to Discord"):
                push_recommended(results, title=f"{sport_choice} Recommended Bets ({stat.capitalize()})")
                st.info("✅ Sent to Discord channel.")
        else:
            st.warning("No opportunities found for these parameters.")
            
# ============== Footer ==============
st.markdown("---")
st.caption("© Parlay +EV Pro — all odds and props are for informational purposes only.")
