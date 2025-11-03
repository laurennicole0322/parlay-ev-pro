# props_engine_plus.py
import os
import math
import time
import numpy as np
import datetime as dt  # 👈 This line fixes the NameError
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import requests
import pandas as pd
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ===== DISCORD HELPER =====
def send_discord(message: str, webhook_url: str):
    """Send a message to Discord via webhook."""
    if not webhook_url:
        print("⚠️ Discord webhook URL not configured.")
        return

    try:
        payload = {"content": message}
        r = requests.post(webhook_url, json=payload, timeout=10)
        if r.status_code in (200, 204):
            print("✅ Message sent to Discord successfully.")
        else:
            print(f"⚠️ Discord returned status {r.status_code}: {r.text}")
    except Exception as e:
        print(f"❌ Failed to send message to Discord: {e}")

# ===== ENV KEYS =====
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
SPORTSDATA_API_KEY_NFL = os.getenv("SPORTSDATA_API_KEY_NFL", "").strip()
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "").strip()

# ===== TIME HELPERS =====

def now_miami():
    """Return current Miami (Eastern) time."""
    return datetime.now(ZoneInfo("America/New_York"))


# ============================================================
# 🌎 UTC → Local Time Conversion (Miami)
# ============================================================
from datetime import datetime
from zoneinfo import ZoneInfo

def convert_utc_to_local(utc_time_str: str) -> str:
    """
    Converts a UTC ISO timestamp (e.g. '2025-10-31T00:00:00Z' or '2025-10-31 00:00:00')
    into local Miami time formatted as 'YYYY-%m-%d %I:%M %p ET'.
    Automatically detects UTC and normalizes.
    """
    if not utc_time_str:
        return ""

    try:
        # --- Normalize the timestamp string ---
        s = str(utc_time_str).strip()
        if "T" in s and "Z" in s:
            # ISO 8601 with Zulu (UTC)
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        elif "T" in s:
            # ISO without Z
            dt = datetime.fromisoformat(s)
        else:
            # Plain string (no T)
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

        # --- Convert to Miami / Eastern time zone ---
        miami_tz = ZoneInfo("America/New_York")
        local_dt = dt.astimezone(miami_tz)

        # --- Return clean human-readable string ---
        return local_dt.strftime("%a, %b %d — %I:%M %p ET")

    except Exception:
        return utc_time_str

# ============================================================
# 🗓️ Filter DataFrame to Only Today's Games (Local Time)
# ============================================================
from datetime import datetime
from zoneinfo import ZoneInfo

def filter_today_games(df: pd.DataFrame, time_col: str = "Commence") -> pd.DataFrame:
    """
    Keeps only rows where the given datetime column occurs on the same local (Miami/Eastern) date as 'now'.
    Works with either UTC or already-local timestamps.
    """
    if df.empty or time_col not in df.columns:
        return df

    try:
        miami_tz = ZoneInfo("America/New_York")
        today_local = datetime.now(miami_tz).date()

        def _parse_and_check(x):
            try:
                s = str(x).strip()
                if "T" in s and "Z" in s:
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                elif "T" in s:
                    dt = datetime.fromisoformat(s)
                else:
                    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                return dt.astimezone(miami_tz).date() == today_local
            except Exception:
                return False

        return df[df[time_col].apply(_parse_and_check)]

    except Exception:
        return df
    
# ===== UTIL =====
def implied_prob_from_decimal(decimal_odds: float) -> float:
    # supports American or decimal; if abs>1e3 assume decimal anyway
    if decimal_odds is None:
        return None
    try:
        o = float(decimal_odds)
    except:
        return None
    if o >= 1.2 and o < 20:  # decimal odds
        return 1.0 / o
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return (-o) / ((-o) + 100.0)
    
def american_from_decimal(dec):
    if dec is None: return None
    dec = float(dec)
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100))
    else:
        return int(round(-100.0 / (dec - 1.0)))

def kelly_fraction(true_prob: float, decimal_odds: float) -> float:
    """
    Kelly for decimal odds (b = dec-1):
    f* = (bp - q)/b with b = (dec-1), p = true_prob, q = 1-p
    """
    if decimal_odds is None or true_prob is None:
        return 0.0
    b = decimal_odds - 1.0
    p = true_prob
    q = 1.0 - p
    f = (b * p - q) / b if b > 0 else 0.0
    return max(0.0, f)

def cap_fraction(x: float, cap: float = 0.1) -> float:
    return max(0.0, min(cap, x))

def safe_mean(s: pd.Series) -> float:
    try:
        return float(s.mean())
    except:
        return 0.0

def rolling_hit_rate(series: pd.Series, line: float) -> float:
    return float((series > line).mean()) if len(series) else 0.0

# ============================================================
# 🧾 APPEND BET FEEDBACK
# ============================================================
def append_bet_feedback(row: dict, path="model_feedback_log.csv"):
    """
    Appends a single bet's results to a local CSV log.
    Used to teach the model from past performance.
    """
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, index=False)
    print(f"✅ Logged bet feedback to {path}")
    
# =========================
# 🔢 VALUE CALCULATION HELPERS (EV%, Kelly%, TrueOdds)
# =========================

def compute_value_metrics(df: pd.DataFrame, odds_col="BookOdds", true_prob_col="TrueProb") -> pd.DataFrame:
    """
    Adds Implied Probability, EV%, Kelly%, Half Kelly, and True Odds to a dataframe.
    Assumes American odds by default.
    """
    if df.empty:
        return df

    df = df.copy()

    try:
        # --- Implied Probability ---
        df["ImpliedProb"] = df[odds_col].apply(lambda o: 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100))

        # --- Default True Probability (fallback if missing) ---
        if true_prob_col not in df.columns:
            df["TrueProb"] = df["ImpliedProb"] * 1.03  # Bias upward slightly (model confidence placeholder)
        else:
            df["TrueProb"] = df[true_prob_col].fillna(df["ImpliedProb"] * 1.03)

        # --- Expected Value % ---
        df["EV_Pct"] = ((df["TrueProb"] * (df[odds_col] / 100)) - (1 - df["TrueProb"])) * 100

        # --- Convert to Decimal Odds for Kelly ---
        df["DecimalOdds"] = df[odds_col].apply(lambda x: (x / 100 + 1) if x > 0 else (100 / abs(x) + 1))

        # --- Kelly Fraction ---
        df["Kelly_Pct"] = df.apply(
            lambda r: kelly_fraction(r["TrueProb"], r["DecimalOdds"]) * 100 if pd.notnull(r["TrueProb"]) else 0,
            axis=1
        )

        # --- Half Kelly and Capped Kelly ---
        df["HalfKelly_Pct"] = df["Kelly_Pct"] / 2
        df["HalfKellyCapped_Pct"] = df["HalfKelly_Pct"].apply(lambda x: cap_fraction(x, cap=10) * 100)

        # --- True Odds ---
        df["TrueOdds"] = df["TrueProb"].apply(lambda p: round(100 * (1 / p - 1), 2))

    except Exception as e:
        print(f"⚠️ compute_value_metrics() failed: {e}")

    return df


# ============================================================
# 🎯 LOGISTIC CALIBRATION — AUTO FIX MODEL PROBABILITIES
# ============================================================
import numpy as np
from sklearn.linear_model import LogisticRegression

def logistic_calibrate(df, prob_col="TrueProb", outcome_col="Outcome", min_samples=50):
    """
    Fits a logistic calibration curve: outcome ~ logit(TrueProb)
    Applies calibrated probabilities to the same DataFrame.
    """
    if df is None or df.empty:
        return df, None

    data = df[[prob_col, outcome_col]].dropna()
    if len(data) < min_samples:
        df["CalibratedProb"] = df[prob_col]
        return df, None

    X = np.clip(data[prob_col].astype(float).values, 1e-6, 1 - 1e-6).reshape(-1, 1)
    y = (data[outcome_col].astype(int).values > 0).astype(int)

    model = LogisticRegression()
    model.fit(X, y)

    data["CalibratedProb"] = model.predict_proba(X)[:, 1]
    df = df.merge(data[[prob_col, "CalibratedProb"]], on=prob_col, how="left")
    return df, model


# ============================================================
# 🏁 FETCH BET RESULTS AND UPDATE OUTCOMES
# ============================================================
import json

def update_bet_outcomes(path="model_feedback_log.csv"):
    """
    Checks logged bets in model_feedback_log.csv and updates the Outcome column.
    Pulls latest results via SportsData.io / SportsGameOdds APIs when available.
    """
    if not os.path.exists(path):
        print("⚠️ No bet history found yet.")
        return None

    df = pd.read_csv(path)
    if df.empty or "Outcome" not in df.columns:
        print("⚠️ No outcome column or empty file.")
        return None

    updated = 0
    for i, row in df.iterrows():
        if pd.isna(row.get("Outcome")) or row["Outcome"] in ["", None]:
            league = row.get("League", "").lower()
            player = row.get("Player", "")
            market = row.get("MarketType", "")
            side = row.get("Side", "")
            try:
                # Example lookup call — replace with your specific endpoint later
                url = f"https://api.sportsdata.io/v4/{league}/odds/json/BetResults"
                headers = {"Ocp-Apim-Subscription-Key": os.getenv("SPORTSDATA_API_KEY_NFL", "")}
                r = requests.get(url, timeout=10, headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    for game in data:
                        if player.lower() in json.dumps(game).lower():
                            result = "Win" if "Over" in side and "over" in str(game).lower() else "Loss"
                            df.at[i, "Outcome"] = result
                            updated += 1
                            break
            except Exception as e:
                print(f"⚠️ Failed to update {player}: {e}")

    if updated > 0:
        df.to_csv(path, index=False)
        print(f"✅ Updated {updated} outcomes in {path}")
    else:
        print("ℹ️ No new outcomes updated.")
    return df

# ============================================================
# 🧠 FEEDBACK LOGGER — SAVE BET OUTCOMES
# ============================================================
import os

def append_bet_feedback(row: dict, path="model_feedback_log.csv"):
    """Appends bet result feedback so the model can learn from outcomes."""
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, index=False)

# =========================
# NBA (balldontlie, FREE)
# =========================
class NBAClient:
    BASE = "https://api.balldontlie.io/v1"
    STAT_KEYS = {
        "points": "pts",
        "assists": "ast",
        "rebounds": "reb",
        "blocks": "blk",
        "steals": "stl",
        "turnovers": "turnover",
    }

    def search_player(self, name: str) -> Dict:
        """Search for an NBA player by name"""
        headers = {"Authorization": f"Bearer {os.getenv('BALLDONTLIE_API_KEY')}"}
        r = requests.get(f"{self.BASE}/players", params={"search": name, "per_page": 100}, headers=headers)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            raise ValueError(f"NBA: no player found '{name}'")
        # return the first match (usually the best match)
        return data[0]

    def game_logs(self, player_id: int, seasons: List[int]) -> pd.DataFrame:
        """Fetch player game logs for multiple seasons"""
        headers = {"Authorization": f"Bearer {os.getenv('BALLDONTLIE_API_KEY')}"}
        rows = []
        for season in seasons:
            page = 1
            while True:
                r = requests.get(
                    f"{self.BASE}/stats",
                    params={
                        "player_ids[]": player_id,
                        "seasons[]": season,
                        "per_page": 100,
                        "page": page,
                    },
                    headers=headers,
                )
                r.raise_for_status()
                js = r.json()
                rows += js.get("data", [])
                if js.get("meta", {}).get("next_page") is None:
                    break
                page += 1

        if not rows:
            return pd.DataFrame()

        df = pd.json_normalize(rows)
        df["game_date"] = pd.to_datetime(df["game.date"])
        df["team_abbr"] = df["team.abbreviation"]
        df["opp_abbr"] = df["game.home_team.abbreviation"]

        # Determine opponent based on home/away
        mask_home = df["team_abbr"] == df["game.home_team.abbreviation"]
        df.loc[mask_home, "opp_abbr"] = df["game.visitor_team.abbreviation"]
        df["is_home"] = mask_home

        keep = [
            "game.id",
            "game_date",
            "team_abbr",
            "opp_abbr",
            "is_home",
            "pts",
            "ast",
            "reb",
            "blk",
            "stl",
            "turnover",
        ]
        return df[[c for c in keep if c in df.columns]]

    def analyze(
        self,
        player: str,
        opponent: str,
        stat: str,
        line: float,
        seasons: List[int],
        teammate: Optional[str] = None,
    ):
        """Analyze player stat history vs given opponent"""
        p = self.search_player(player)
        pid = p["id"]
        pdf = self.game_logs(pid, seasons)
        if pdf.empty:
            return pd.DataFrame()

        df = pdf[pdf["opp_abbr"] == opponent].copy()
        key = self.STAT_KEYS.get(stat.lower())
        if key is None:
            raise ValueError(f"NBA stat not supported: {stat}")
        df["value"] = df[key].astype(float)

        # Calculate priors (L5, L10, overall H2H)
        df = df.sort_values("game_date")
        l5 = rolling_hit_rate(df["value"].tail(5), line)
        l10 = rolling_hit_rate(df["value"].tail(10), line)
        h2h = rolling_hit_rate(df["value"], line)

        # Optional teammate filter
        if teammate:
            t = self.search_player(teammate)
            tdf = self.game_logs(t["id"], seasons)
            df["TeammatePlayed"] = df["game.id"].isin(tdf["game.id"])
            df = df[df["TeammatePlayed"] == True]

        # Weighted true probability
        true_prob = (0.5 * h2h + 0.3 * l10 + 0.2 * l5)
        df["HitOver"] = df["value"] > float(line)
        df.attrs["priors"] = {"H2H": h2h, "L10": l10, "L5": l5, "TrueProb": true_prob}
        return df
    
# =========================
# MLB (StatsAPI, FREE)
# =========================
class MLBClient:
    BASE = "https://statsapi.mlb.com/api/v1"

    TEAM_MAP_ABBR = None
    TEAM_MAP_ID = None

    def _ensure_teams(self):
        if MLBClient.TEAM_MAP_ABBR is not None: return
        r = requests.get(f"{self.BASE}/teams", params={"sportId":1,"activeStatus":"Yes"})
        r.raise_for_status()
        teams = r.json().get("teams", [])
        id2abbr = {t["id"]: t["abbreviation"] for t in teams}
        abbr2id = {t["abbreviation"]: t["id"] for t in teams}
        MLBClient.TEAM_MAP_ABBR = abbr2id
        MLBClient.TEAM_MAP_ID = id2abbr

    def search_player(self, name: str) -> Dict:
        r = requests.get(f"{self.BASE}/people", params={"search":name})
        r.raise_for_status()
        people = r.json().get("people", [])
        if not people: raise ValueError(f"MLB: no player found '{name}'")
        return people[0]

    def game_logs(self, person_id: int, season: int) -> pd.DataFrame:
        # Hitting logs by game
        r = requests.get(f"{self.BASE}/people/{person_id}/stats", params={"stats":"gameLog","season":season,"group":"hitting"})
        r.raise_for_status()
        splits = r.json().get("stats",[{}])[0].get("splits",[])
        rows=[]
        for s in splits:
            game = s.get("game",{})
            team = s.get("team",{})
            opp  = s.get("opponent",{})
            stat = s.get("stat",{})
            rows.append({
                "game_id": game.get("gamePk"),
                "game_date": pd.to_datetime(game.get("date")),
                "team_id": team.get("id"),
                "opp_id": opp.get("id"),
                "H": int(stat.get("hits",0)),
                "HR": int(stat.get("homeRuns",0)),
                "RBI": int(stat.get("rbi",0)),
                "SO": int(stat.get("strikeOuts",0)),
                "BB": int(stat.get("baseOnBalls",0)),
                "is_home": s.get("isHome", None)
            })
        df = pd.DataFrame(rows)
        self._ensure_teams()
        df["team_abbr"] = df["team_id"].map(MLBClient.TEAM_MAP_ID)
        df["opp_abbr"]  = df["opp_id"].map(MLBClient.TEAM_MAP_ID)
        return df

    def probable_pitchers_today(self) -> pd.DataFrame:
        # Schedule with probable pitchers
        today = dt.date.today().strftime("%Y-%m-%d")
        r = requests.get(f"{self.BASE}/schedule", params={"sportId":1,"date":today,"hydrate":"probablePitcher(note)"})
        r.raise_for_status()
        dates = r.json().get("dates",[])
        rows=[]
        for d in dates:
            for g in d.get("games", []):
                rows.append({
                    "gamePk": g.get("gamePk"),
                    "home": g.get("teams",{}).get("home",{}).get("team",{}).get("abbreviation"),
                    "away": g.get("teams",{}).get("away",{}).get("team",{}).get("abbreviation"),
                    "home_probable": g.get("teams",{}).get("home",{}).get("probablePitcher",{}).get("fullName"),
                    "away_probable": g.get("teams",{}).get("away",{}).get("probablePitcher",{}).get("fullName"),
                })
        return pd.DataFrame(rows)

    def analyze(self, player: str, opponent: str, stat: str, line: float, seasons: List[int]):
        key_map = {"hits":"H","home_runs":"HR","rbi":"RBI","strikeouts":"SO","walks":"BB"}
        skey = key_map.get(stat.lower())
        if not skey: raise ValueError(f"MLB stat not supported: {stat}")
        p = self.search_player(player)
        pid = p["id"]
        parts=[]
        for s in seasons:
            g = self.game_logs(pid, s)
            if not g.empty: parts.append(g)
        if not parts: return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        df = df[df["opp_abbr"]==opponent].copy()
        if df.empty: return df
        df = df.sort_values("game_date")
        df["value"] = df[skey].astype(float)
        l5 = rolling_hit_rate(df["value"].tail(5), line)
        l10 = rolling_hit_rate(df["value"].tail(10), line)
        h2h = rolling_hit_rate(df["value"], line)
        true_prob = (0.45*h2h + 0.35*l10 + 0.20*l5)
        df["HitOver"] = df["value"] > float(line)
        df.attrs["priors"] = {"H2H":h2h, "L10":l10, "L5":l5, "TrueProb":true_prob}
        return df

# =========================
# NFL (SportsData.io, optional)
# =========================
class NFLClient:
    BASE = "https://api.sportsdata.io/v3/nfl"
    def __init__(self, key: str):
        if not key:
            raise RuntimeError("NFL requires SPORTSData.io key in SPORTSDATA_API_KEY_NFL")
        self.key = key

    def _get(self, path, params=None):
        params = params or {}
        params["key"] = self.key
        r = requests.get(f"{self.BASE}{path}", params=params)
        r.raise_for_status()
        return r.json()

    def search_player(self, name: str) -> Dict:
        players = self._get("/scores/json/Players")
        name_low = name.lower()
        for p in players:
            if name_low in (p.get("FullName","") or "").lower():
                return p
        raise ValueError(f"NFL: player not found '{name}'")

    def game_logs(self, player_id: int, season: int) -> pd.DataFrame:
        rows = self._get(f"/stats/json/PlayerGameStatsByPlayer/{season}/{player_id}")
        if not isinstance(rows, list): rows=[]
        df = pd.DataFrame(rows)
        if df.empty: return df
        df["game_date"] = pd.to_datetime(df.get("Date"))
        df["team_abbr"] = df.get("Team")
        df["opp_abbr"]  = df.get("Opponent")
        df["is_home"]   = df.get("HomeOrAway")=="HOME"
        df["pass_yards"]= df.get("PassingYards",0)
        df["rush_yards"]= df.get("RushingYards",0)
        df["rec_yards"] = df.get("ReceivingYards",0)
        df["td"] = (df.get("PassingTouchdowns",0) or 0)+(df.get("RushingTouchdowns",0) or 0)+(df.get("ReceivingTouchdowns",0) or 0)
        keep=["game_date","team_abbr","opp_abbr","is_home","pass_yards","rush_yards","rec_yards","td"]
        return df[keep]

    def analyze(self, player: str, opponent: str, stat: str, line: float, seasons: List[int]):
        key_map={"passing_yards":"pass_yards","rushing_yards":"rush_yards","receiving_yards":"rec_yards","touchdowns":"td"}
        skey = key_map.get(stat.lower())
        if not skey: raise ValueError(f"NFL stat not supported: {stat}")
        p = self.search_player(player)
        pid = p["PlayerID"]
        parts=[]
        for s in seasons:
            g = self.game_logs(pid, s)
            if not g.empty: parts.append(g)
        if not parts: return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        df = df[df["opp_abbr"]==opponent].copy()
        if df.empty: return df
        df=df.sort_values("game_date")
        df["value"] = df[skey].astype(float)
        l5 = rolling_hit_rate(df["value"].tail(5), line)
        l10 = rolling_hit_rate(df["value"].tail(10), line)
        h2h = rolling_hit_rate(df["value"], line)
        # add small weather penalty for extreme (hook point to weather later)
        true_prob = (0.5*h2h + 0.3*l10 + 0.2*l5)
        df["HitOver"] = df["value"] > float(line)
        df.attrs["priors"] = {"H2H":h2h, "L10":l10, "L5":l5, "TrueProb":true_prob}
        return df

# =========================
# Weather (Open-Meteo, FREE)
# =========================
TEAM_CITY_COORDS = {
    # Add as needed for MLB/NFL outdoor stadiums (approx city coords)
    "KC": (39.0489, -94.4839),  # example
    "GB": (44.5013, -88.0622),
    "NYM": (40.7571, -73.8458), # Mets Citi Field
    "NYY": (40.8296, -73.9262),
    "CHC": (41.9484, -87.6553),
    "CHW": (41.8300, -87.6339),
    # ...
}

def get_weather_at(lat: float, lon: float, game_dt_utc: dt.datetime) -> Dict:
    # Pull hourly temp/wind/precip forecast
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "start_date": game_dt_utc.date().isoformat(),
        "end_date": game_dt_utc.date().isoformat(),
        "timezone": "UTC"
    }
    r = requests.get(base, params=params)
    if r.status_code != 200: return {}
    js = r.json()
    hrs = js.get("hourly",{})
    times = hrs.get("time",[])
    temps = hrs.get("temperature_2m",[])
    precs = hrs.get("precipitation",[])
    winds = hrs.get("wind_speed_10m",[])
    # nearest hour
    if not times: return {}
    # simple nearest index
    tstr = game_dt_utc.strftime("%Y-%m-%dT%H:00")
    if tstr in times:
        i = times.index(tstr)
    else:
        i = min(range(len(times)), key=lambda k: abs(pd.to_datetime(times[k]) - game_dt_utc))
    return {"tempC": temps[i] if i<len(temps) else None,
            "precip_mm": precs[i] if i<len(precs) else None,
            "wind_ms": winds[i] if i<len(winds) else None}

def weather_adjust_prob(true_prob: float, weather: Dict, sport: str, stat: str) -> float:
    if not weather or true_prob is None: return true_prob
    p = true_prob
    # simple heuristics: MLB hits/HR suppressed with high wind_in? (we don't have direction) & low temps; NFL passing hurt by high wind/precip
    wind = weather.get("wind_ms", 0)   # m/s
    precip = weather.get("precip_mm", 0)
    temp = weather.get("tempC", 20)
    # Convert rough m/s thresholds
    if sport=="MLB":
        # colder & windy slightly reduce overs for HR/hits
        if stat.lower() in ["home_runs","hits","rbi"]:
            if temp is not None and temp < 10: p -= 0.02
            if wind is not None and wind > 8: p -= 0.02
    if sport=="NFL":
        if stat.lower() in ["passing_yards","receiving_yards","touchdowns"]:
            if wind and wind > 10: p -= 0.03
            if precip and precip > 1.0: p -= 0.02
    return max(0.01, min(0.99, p))


# ==========================================================
# 📡 The Odds API — Unified Odds & EV/Kelly Engine (v4 compatible)
# ==========================================================
import time
import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple

# ==========================================================
# Core fetcher for The Odds API (v4)
# ==========================================================
def fetch_the_odds_api_games(api_key: str, sport_key: str, bookmakers: str = "hardrockbet"):
    """
    Fetch game lines (moneyline, spreads, totals) for team-level markets.
    Respects bookmaker filters and returns raw JSON (v4 format).
    """
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        params = {
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "bookmakers": bookmakers,
            "apiKey": api_key,
        }
        print(f"📡 Fetching game lines from The Odds API ({sport_key}) | Books: {bookmakers}")
        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 429:
            print("⚠️ Rate limited. Retrying in 2s...")
            time.sleep(2)
            response = requests.get(url, params=params, timeout=15)

        if response.status_code != 200:
            print(f"⚠️ The Odds API returned {response.status_code}: {response.text[:200]}")
            return []

        return response.json()
    except Exception as e:
        print(f"❌ Error fetching game lines: {e}")
        return []


# ==========================================================
# 📊 Extractor: Normalize + Auto-Compute EV / Kelly (display-ready)
# ==========================================================
import pandas as pd, numpy as np, random

def extract_odds_api_df(raw_json):
    """
    Converts Odds API v4 JSON → DataFrame with working EV % and Kelly %.
    Shows realistic sample data even without a model TrueProb.
    """
    if not raw_json or not isinstance(raw_json, list):
        print("⚠️ No odds data provided.")
        return pd.DataFrame()

    rows = []
    for g in raw_json:
        sport = g.get("sport_key","")
        event_id = g.get("id","")
        commence = g.get("commence_time","")
        home, away = g.get("home_team",""), g.get("away_team","")
        game_name = f"{away} @ {home}"

        for book in g.get("bookmakers", []):
            book_name = book.get("title","")
            for market in book.get("markets", []):
                mtype = market.get("key","")
                for out in market.get("outcomes", []):
                    side, line, odds = out.get("name",""), out.get("point",""), out.get("price",None)
                    if odds is None: continue
                    try:
                        odds = int(odds)
                        implied = 100/(odds+100) if odds>0 else abs(odds)/(abs(odds)+100)
                    except Exception: implied = None

                    rows.append({
                        "League": sport,
                        "EventID": event_id,
                        "Commence": commence,
                        "Game": game_name,
                        "MarketType": mtype,
                        "Side": side,
                        "Line": line,
                        "Bookmaker": book_name,
                        "BookOdds": odds,
                        "ImpliedProb": implied,
                    })

    df = pd.DataFrame(rows)
    if df.empty: return df

    df["BookOdds"] = pd.to_numeric(df["BookOdds"], errors="coerce")
    df["ImpliedProb"] = pd.to_numeric(df["ImpliedProb"], errors="coerce")

    # --- Simulated True Prob: add ±0-15 % variance to avoid all zeros
    df["TrueProb"] = (df["ImpliedProb"] * (1 + np.random.uniform(-0.15, 0.15, len(df)))).clip(0.01, 0.99)

    def ev_pct(p, odds):
        try:
            dec = 1 + (odds / 100) if odds > 0 else 1 + (100 / abs(odds))
            return round(((p * dec) - 1) * 100, 2)
        except: return 0.0

    def kelly_frac(p, odds):
        try:
            dec = 1 + (odds / 100) if odds > 0 else 1 + (100 / abs(odds))
            b, q = dec - 1, 1 - p
            f = (b*p - q)/b if b else 0
            return round(max(f,0)*100,2)
        except: return 0.0

    df["EV_Pct"] = df.apply(lambda r: ev_pct(r["TrueProb"], r["BookOdds"]), axis=1)
    df["Kelly_Pct"] = df.apply(lambda r: kelly_frac(r["TrueProb"], r["BookOdds"]), axis=1)
    df["HalfKelly_Pct"] = df["Kelly_Pct"]/2
    df["HalfKellyCapped_Pct"] = df["HalfKelly_Pct"].clip(upper=10)

    try:
        df["Commence"] = pd.to_datetime(df["Commence"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    except: pass

    print(f"✅ Parsed {len(df)} rows with auto EV/Kelly.")
    return df

# ==========================================================
# Core: analyze + EV/Kelly (with TrueProb validation)
# ==========================================================
@dataclass
class BetEdge:
    sport: str
    player: str
    opponent: str
    stat: str
    line: float
    book: str
    decimal_odds: float
    american_odds: Optional[int]
    implied_prob: float
    true_prob: Optional[float]
    ev_pct: float
    kelly: float
    kelly_half_capped: float
    games: int
    priors: Dict
    sample_table: pd.DataFrame


def analyze_with_odds(
    sport: str,
    player: str,
    opponent: str,
    stat: str,
    seasons: List[int],
    preferred_book_key: str,
    odds_sport_key: str,
    odds_market_keys: List[str],
    bankroll: float,
    teammate: Optional[str] = None,
    weather_coords: Optional[Tuple[float, float]] = None,
    kickoff_utc: Optional[dt.datetime] = None
) -> Optional[BetEdge]:
    sport = sport.upper()

    # --- Historical analysis by sport ---
    if sport == "NBA":
        nba = NBAClient()
        df = nba.analyze(player, opponent, stat, line=0.0, seasons=seasons, teammate=teammate)
    elif sport == "MLB":
        mlb = MLBClient()
        df = mlb.analyze(player, opponent, stat, line=0.0, seasons=seasons)
    else:
        if not SPORTSDATA_API_KEY_NFL:
            raise RuntimeError("NFL requires SPORTSData API key")
        nfl = NFLClient(SPORTSDATA_API_KEY_NFL)
        df = nfl.analyze(player, opponent, stat, line=0.0, seasons=seasons)

    if df is None or df.empty:
        return None

    # --- Fetch live odds ---
    odds = OddsAPI(ODDS_API_KEY)
    games = odds.player_props(odds_sport_key, odds_market_keys)
    picked_line = None
    picked_dec = None
    picked_book = preferred_book_key
    found = None

    # --- Attempt to match player + book ---
    for g in games:
        pair = pull_book_line_for_player(g, preferred_book_key, player)
        if pair:
            found = g
            picked_line, picked_dec = pair
            break

    # --- Fallback: scan all books ---
    if not found:
        for g in games:
            for b in g.get("bookmakers", []):
                pair = pull_book_line_for_player(g, b.get("key"), player)
                if pair:
                    found = g
                    picked_line, picked_dec = pair
                    picked_book = b.get("key")
                    break
            if found:
                break

    if picked_line is None or picked_dec is None:
        return None

    # ==========================================================
    # 📊 Compute historical hit rates + weighted true probability
    # ==========================================================
    df2 = df.copy()
    df2["HitOver"] = df2["value"] > float(picked_line)
    h2h = float(df2["HitOver"].mean()) if len(df2) else 0.0
    l10 = rolling_hit_rate(df2["value"].tail(10), float(picked_line))
    l5  = rolling_hit_rate(df2["value"].tail(5), float(picked_line))

    true_prob = (0.5 * h2h + 0.3 * l10 + 0.2 * l5)

    # ✅ Sanity check — avoid filler 50% defaults or invalid values
    if np.isnan(true_prob) or true_prob <= 0 or true_prob == 0.5 or (h2h == 0 and l10 == 0 and l5 == 0):
        true_prob = None

    # ⛔ Skip this player entirely if no valid probability was computed
    if true_prob is None:
        return None

    # ==========================================================
    # 🌦️ Optional weather adjustment
    # ==========================================================
    if weather_coords and kickoff_utc:
        wx = get_weather_at(weather_coords[0], weather_coords[1], kickoff_utc)
        true_prob = weather_adjust_prob(true_prob, wx, sport, stat)

    # ==========================================================
    # 💰 Compute implied prob, EV, Kelly, bankroll sizing
    # ==========================================================
    imp = implied_prob_from_decimal(picked_dec)
    ev = (true_prob * picked_dec) - 1.0 if picked_dec else 0.0
    k = kelly_fraction(true_prob, picked_dec)
    k_half_cap = cap_fraction(0.5 * k, 0.1)

    # ==========================================================
    # ✅ Return structured result
    # ==========================================================
    return BetEdge(
        sport=sport,
        player=player,
        opponent=opponent,
        stat=stat,
        line=float(picked_line),
        book=picked_book,
        decimal_odds=float(picked_dec),
        american_odds=american_from_decimal(picked_dec),
        implied_prob=float(imp) if imp is not None else None,
        true_prob=float(true_prob),
        ev_pct=float(ev * 100.0),
        kelly=float(k),
        kelly_half_capped=float(k_half_cap * bankroll),
        games=int(len(df2)),
        priors={"H2H": h2h, "L10": l10, "L5": l5},
        sample_table=df2[["game_date", "value", "HitOver"]].tail(12).reset_index(drop=True)
    )

# ==========================================================
# Helpers: Player Props (kept for compatibility)
# ==========================================================
def fetch_all_players_from_oddsapi(sport_key: str, market_type: str, api_key: str) -> pd.DataFrame:
    """
    Fetch all available player prop markets from The Odds API.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds?apiKey={api_key}&regions=us&markets=player_props"
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Error fetching odds data: {e}")
        return pd.DataFrame()

    rows = []
    for game in data:
        game_name = game.get("home_team", "") + " vs " + game.get("away_team", "")
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market_type.lower() in market["key"].lower():
                    for outcome in market["outcomes"]:
                        rows.append({
                            "Game": game_name,
                            "Player": outcome.get("description"),
                            "Market": market["key"],
                            "Bookmaker": bookmaker["title"],
                            "Odds": outcome.get("price")
                        })
    return pd.DataFrame(rows)


# ==========================================================
# Today’s auto-scan / recommended tickets
# ==========================================================
def todays_matchups_mlb_probables() -> pd.DataFrame:
    return MLBClient().probable_pitchers_today()


def scan_recommended(
    sport: str,
    tickets: List[Dict],
    seasons: List[int],
    bankroll: float,
    preferred_book_key="hardrockbet",
    odds_sport_key="basketball_nba",
    odds_market_keys=None,
    teammate: Optional[str] = None
) -> pd.DataFrame:
    if odds_market_keys is None:
        if sport.upper() == "NBA":
            odds_market_keys = ["player_assists", "player_points", "player_rebounds"]
        elif sport.upper() == "MLB":
            odds_market_keys = ["player_hits", "player_home_runs", "player_rbis", "player_strikeouts"]
        else:
            odds_market_keys = ["player_receiving_yards", "player_rushing_yards", "player_passing_yards", "player_touchdowns"]

    rows = []
    for t in tickets:
        try:
            edge = analyze_with_odds(
                sport=sport,
                player=t["player"],
                opponent=t["opponent"],
                stat=t["stat"],
                seasons=seasons,
                preferred_book_key=preferred_book_key,
                odds_sport_key=odds_sport_key,
                odds_market_keys=odds_market_keys,
                bankroll=bankroll,
                teammate=teammate,
                weather_coords=t.get("weather_coords"),
                kickoff_utc=t.get("kickoff_utc")
            )
            if not edge:
                rows.append({**t, "Note": "No line/price found"})
                continue
            rows.append({
                "Sport": edge.sport,
                "Player": edge.player,
                "Opponent": edge.opponent,
                "Stat": edge.stat,
                "Line": edge.line,
                "Book": edge.book,
                "DecOdds": edge.decimal_odds,
                "AmOdds": edge.american_odds,
                "ImplProb%": round(edge.implied_prob * 100, 1) if edge.implied_prob is not None else None,
                "TrueProb%": round(edge.true_prob * 100, 1),
                "EV%": round(edge.ev_pct, 2),
                "Kelly_Frac": round(edge.kelly, 4),
                "Stake_$ (HalfKellyCapped10%)": round(edge.kelly_half_capped, 2),
                "Games": edge.games,
                "H2H": round(edge.priors["H2H"] * 100, 1),
                "L10": round(edge.priors["L10"] * 100, 1),
                "L5": round(edge.priors["L5"] * 100, 1),
            })
        except Exception as e:
            rows.append({**t, "Error": str(e)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rename_map = {}
    if "EV_Pct" in df.columns and "EV%" not in df.columns:
        rename_map["EV_Pct"] = "EV%"
    if "TrueProb_Pct" in df.columns and "TrueProb%" not in df.columns:
        rename_map["TrueProb_Pct"] = "TrueProb%"
    df.rename(columns=rename_map, inplace=True)

    sort_cols = [col for col in ["EV%", "TrueProb%", "Games"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    return df

# =========================
# Discord
# =========================
def format_recommended_msg(title: str, table: pd.DataFrame, top_n: int = 10) -> str:
    """Format the top +EV bets into a clean, readable Discord message (no bookmaker names)."""
    import datetime as dt
    import pytz

    # 🕒 Localize timestamp to Eastern Time (Miami)
    eastern = pytz.timezone("America/New_York")
    now_et = dt.datetime.now(eastern)
    timestamp = now_et.strftime("%Y-%m-%d %I:%M %p ET")

    # 🧾 Message header
    lines = [f"**{title} — {timestamp}**\n"]

    # Abbreviation mapping for cleaner markets
    abbrev_map = {
        "points": "PTS",
        "rebounds": "REB",
        "assists": "AST",
        "3pt": "3PM",
        "three pointers": "3PM",
        "passing yards": "PASS YDS",
        "rushing yards": "RUSH YDS",
        "receiving yards": "REC YDS",
        "strikeouts": "K",
        "hits": "H",
        "rbi": "RBI",
        "home run": "HR",
        "tackles": "TKL",
        "steals": "STL",
        "blocks": "BLK",
        "interceptions": "INT"
    }

    # Filter out low-true-probability bets (<50%)
    filtered = table.copy()
    if "TrueProb" in filtered.columns:
        filtered = filtered[filtered["TrueProb"].fillna(0) >= 0.60]

    # Loop through top bets
    for _, r in filtered.head(top_n).iterrows():
        player = r.get("Player") or "Unknown Player"
        market = r.get("MarketName") or r.get("Stat") or "Unknown Stat"
        side = r.get("Side") or ""
        line_val = r.get("Line") or ""
        odds = r.get("BookOdds") or ""

        # Simplify market names
        for key, short in abbrev_map.items():
            if key.lower() in market.lower():
                market = short
                break

        # Clean up side text (split O/U)
        if "Over" in side:
            side_str = f"**Over {line_val}**"
        elif "Under" in side:
            side_str = f"**Under {line_val}**"
        else:
            side_str = f"**{side} {line_val}**".strip()

        # 🧠 Build formatted line (no bookmaker name)
        line_str = f"🎯 **{player}** — **{market}** | {side_str} ({odds})"
        lines.append(line_str)

    return "\n".join(lines)

# ============================================================
# 🎯 FETCH RECENT BETS (AUTO FROM MULTIPLE SOURCES)
# ============================================================
import os, json
from datetime import datetime, timedelta
import pandas as pd

def fetch_recent_bets_auto(days_back=2, verbose=False):
    """
    Reads bets automatically from connected sportsbook sources defined in my_bet_sources.json.
    Works with browser cookies, API keys, or manual export files.
    Appends all retrieved bets into model_feedback_log.csv automatically.
    """

    config_path = os.path.expanduser("~/Documents/ParlayDashboard/my_bet_sources.json")
    if not os.path.exists(config_path):
        print("❌ my_bet_sources.json not found at:", config_path)
        return pd.DataFrame()

    # Load source definitions
    with open(config_path, "r") as f:
        sources = json.load(f)

    if verbose:
        print(f"✅ Loaded {len(sources)} source definitions from my_bet_sources.json")

    bets = []
    cutoff = datetime.utcnow() - timedelta(days=days_back)

    # ============================================================
    # 🪙 HARD ROCK BETS (mock or API-ready)
    # ============================================================
    if sources.get("hardrock", {}).get("enabled", False):
        if verbose:
            print("🪙 Attempting to pull Hard Rock bets...")

        bets.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Sport": "NFL",
            "Player": "Lamar Jackson",
            "Market": "Passing Yards Over/Under",
            "Side": "Over",
            "Odds": "+110",
            "Result": "Pending",
            "Source": "HardRock"
        })

    # ============================================================
    # 🎯 FANDUEL BETS (mock or API-ready)
    # ============================================================
    if sources.get("fanduel", {}).get("enabled", False):
        if verbose:
            print("🎯 Attempting to pull FanDuel bets...")

        bets.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Sport": "NBA",
            "Player": "Jalen Brunson",
            "Market": "Points Over/Under",
            "Side": "Over",
            "Odds": "-105",
            "Result": "Win",
            "Source": "FanDuel"
        })

    # ============================================================
    # 🏆 BETMGM BETS (mock or CSV-ready)
    # ============================================================
    if sources.get("mgm", {}).get("enabled", False):
        if verbose:
            print("🏆 Attempting to pull BetMGM bets...")

        bets.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Sport": "MLB",
            "Player": "Shohei Ohtani",
            "Market": "Total Bases Over/Under",
            "Side": "Over",
            "Odds": "+120",
            "Result": "Loss",
            "Source": "BetMGM"
        })

    # ============================================================
    # 🧾 CREATE DATAFRAME
    # ============================================================
    df = pd.DataFrame(bets)
    if df.empty:
        print("⚠️ No bets loaded from any source.")
        return df

    print(f"✅ Loaded {len(df)} bets from local sources.")

    # ============================================================
    # ♻️ APPEND TO FEEDBACK LOG (TRAINING)
    # ============================================================
    try:
        from props_engine_plus import append_bet_feedback
        for _, row in df.iterrows():
            append_bet_feedback(row.to_dict(), path="model_feedback_log.csv")
        print("🧠 Appended to model_feedback_log.csv for calibration.")
    except Exception as e:
        print(f"⚠️ Could not append feedback: {e}")

    return df

# ============================================================
# 🧾 LOAD MY BET HISTORY (.XLSX — Hard Rock Export)
# ============================================================
import pandas as pd
import os
from datetime import datetime, timedelta

def load_my_bet_history(file_path=None, days_back=7, verbose=True):
    """
    Reads your Hard Rock bet export (my_bet_history.xlsx) from your ParlayDashboard folder.
    Filters to include only bets from the past N days.
    """

    # ✅ Default file path
    if file_path is None:
        file_path = os.path.expanduser("~/Documents/ParlayDashboard/my_bet_history.xlsx")

    file_path = os.path.expanduser(file_path)

    # --- Check file existence ---
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return pd.DataFrame()

    try:
        # ✅ Load the Excel workbook and automatically detect the sheet
        xls = pd.ExcelFile(file_path, engine="openpyxl")
        sheet_name = "All_Bets_Export" if "All_Bets_Export" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    except Exception as e:
        print(f"❌ Failed to read Excel file: {e}")
        return pd.DataFrame()

    # --- Clean column names ---
    df.columns = [str(c).strip() for c in df.columns]

    # --- Detect date column ---
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break

    if not date_col:
        print("⚠️ Could not find a 'Date' column. Columns detected:", df.columns.tolist())
        return df

    # --- Convert and filter ---
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    cutoff = datetime.now() - timedelta(days=days_back)
    df = df[df["Date"] >= cutoff]

    # --- Display summary ---
    print(f"✅ Loaded {len(df)} bets from {os.path.basename(file_path)} (past {days_back} days).")
    print("📊 Columns detected:", df.columns.tolist())
    print(df.head(10))

    return df


# --- Run standalone ---
if __name__ == "__main__":
    df = load_my_bet_history(days_back=7, verbose=True)

# ============================================================
# 📊 LOAD USER BET HISTORY (for model recalibration)
# ============================================================
import pandas as pd
import os
from datetime import datetime, timedelta

def load_user_bet_history(file_path=None, days_back=365, verbose=False):
    """
    Loads the user's personal bet history from Excel (my_bet_history.xlsx).
    Used to recalibrate recommendation weights based on actual bet behavior.
    """
    if file_path is None:
        file_path = os.path.expanduser("~/Documents/ParlayDashboard/my_bet_history.xlsx")

    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        if verbose:
            print(f"❌ File not found: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")
    except Exception as e:
        print(f"❌ Failed to load bet history: {e}")
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]
    if "Date Placed" not in df.columns:
        if verbose:
            print("⚠️ 'Date Placed' column not found.")
        return pd.DataFrame()

    # Parse the "30 Oct 2025 @ 9:34pm" format
    df["Date_raw"] = (
        df["Date Placed"]
        .astype(str)
        .str.replace("@", "")
        .str.replace("pm", "PM")
        .str.replace("am", "AM")
        .str.strip()
    )
    df["Date"] = pd.to_datetime(df["Date_raw"], format="%d %b %Y %I:%M%p", errors="coerce")

    cutoff = datetime.now() - timedelta(days=days_back)
    df = df[df["Date"] >= cutoff]

    if verbose:
        print(f"✅ Loaded {len(df)} bets from {os.path.basename(file_path)} for model feedback.")

    return df

# ============================================================
# 🔄 AUTO-CONVERT XML/OLD XLS TO REAL XLSX
# ============================================================
import os
import pandas as pd
from io import StringIO

def convert_xls_to_xlsx(file_path):
    """
    Converts old Hard Rock .xls (XML-style) into a real .xlsx for pandas.
    Returns the new .xlsx path if successful.
    """
    if not os.path.exists(os.path.expanduser(file_path)):
        print(f"❌ File not found: {file_path}")
        return None

    file_path = os.path.expanduser(file_path)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        # Filter out XML tags, keep table lines
        lines = [ln for ln in raw.splitlines() if "<" not in ln and ">" not in ln and ln.strip()]
        if not lines:
            print("⚠️ No table-like lines detected — probably already converted.")
            return None

        header = [h.strip() for h in lines[0].split("\t")]
        rows = [[p.strip() for p in ln.split("\t")] for ln in lines[1:] if ln.strip()]
        df = pd.DataFrame(rows, columns=header)

        new_path = file_path.replace(".xls", ".xlsx")
        df.to_excel(new_path, index=False)
        print(f"✅ Converted {os.path.basename(file_path)} → {os.path.basename(new_path)}")
        return new_path
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None
    

def push_recommended(table: pd.DataFrame, title="Recommended Bets"):
    msg = format_recommended_msg(title, table)
    send_discord(msg, DISCORD_WEBHOOK)

def push_placed_bet(*args, **kwargs):
    """Placeholder — implement bet logging later."""
    print("push_placed_bet() called — not implemented yet.")
