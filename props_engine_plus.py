# props_engine_plus.py
import os
import math
import time
import datetime as dt  # 👈 This line fixes the NameError
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import requests
import pandas as pd
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ===== ENV KEYS =====
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
SPORTSDATA_API_KEY_NFL = os.getenv("SPORTSDATA_API_KEY_NFL", "").strip()
DISCORD_WEBHOOK_ = os.getenv("DISCORD_WEBHOOK", "").strip()

# ===== TIME HELPERS =====
def now_miami():
    """Return current Miami (Eastern) time."""
    return datetime.now(ZoneInfo("America/New_York"))

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

# =========================
# The Odds API (live odds)
# =========================
class OddsAPI:
    BASE = "https://api.the-odds-api.com/v4"

    def __init__(self, key: str):
        if not key:
            raise RuntimeError("Set ODDS_API_KEY in .env")
        self.key = key

    def player_props(self, sport_key: str, markets: List[str], regions="us", odds_format="decimal") -> List[Dict]:
        # sport_key examples: basketball_nba, baseball_mlb, americanfootball_nfl
        # markets examples: ["player_points", "player_assists", "player_hits"]
        out=[]
        for m in markets:
            url = f"{self.BASE}/sports/{sport_key}/odds"
            params = {"apiKey": self.key, "markets": m, "regions": regions, "oddsFormat": odds_format, "dateFormat":"iso"}
            r = requests.get(url, params=params)
            if r.status_code == 429:
                time.sleep(1); r = requests.get(url, params=params)
            if r.status_code != 200: 
                continue
            js = r.json()
            # each item is a game, with bookmakers -> markets -> outcomes
            for g in js:
                gcopy = dict(g)
                gcopy["_market"] = m
                out.append(gcopy)
        return out

def pull_book_line_for_player(game_obj: Dict, bookmaker_key: str, player_name: str) -> Optional[Tuple[float, float]]:
    """
    Return (line, decimal_odds) for a specific player's outcome if present.
    Scans markets/outcomes for matching 'name' containing player_name and "Over" side (for simplicity).
    """
    bks = game_obj.get("bookmakers", [])
    for b in bks:
        if b.get("key") != bookmaker_key:
            continue
        for mk in b.get("markets", []):
            for out in mk.get("outcomes", []):
                name = (out.get("description") or out.get("name") or "")
                if player_name.lower() in name.lower() and ("over" in name.lower() or "Over" in name):
                    line = out.get("point")
                    price = out.get("price") or out.get("odds") or out.get("price_decimal")
                    return (line, float(price) if price is not None else None)
    return None

# =========================
# Core: analyze + EV/Kelly
# =========================
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
    true_prob: float
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
    teammate: Optional[str]=None,
    weather_coords: Optional[Tuple[float,float]]=None,
    kickoff_utc: Optional[dt.datetime]=None
) -> Optional[BetEdge]:
    sport = sport.upper()
    # 1) historical analysis
    if sport=="NBA":
        nba = NBAClient()
        df = nba.analyze(player, opponent, stat, line=0.0, seasons=seasons, teammate=teammate)  # line set later
    elif sport=="MLB":
        mlb = MLBClient()
        df = mlb.analyze(player, opponent, stat, line=0.0, seasons=seasons)
    else:
        if not SPORTSDATA_API_KEY_NFL:
            raise RuntimeError("NFL requires SPORTSData API key")
        nfl = NFLClient(SPORTSDATA_API_KEY_NFL)
        df = nfl.analyze(player, opponent, stat, line=0.0, seasons=seasons)

    if df is None or df.empty:
        return None

    # 2) Live odds/line from The Odds API
    odds = OddsAPI(ODDS_API_KEY)
    games = odds.player_props(odds_sport_key, odds_market_keys)  # list of games
    picked_line = None
    picked_dec = None
    picked_book = preferred_book_key

    # naive match: find a game with the opponent and our player's name
    player_lower = player.lower()
    found = None
    for g in games:
        # try to locate the preferred_book data first
        pair = pull_book_line_for_player(g, preferred_book_key, player)
        if pair:
            found = g
            picked_line, picked_dec = pair
            break
    if not found:
        # fallback: scan any bookmaker
        for g in games:
            bks = g.get("bookmakers", [])
            for b in bks:
                pair = pull_book_line_for_player(g, b.get("key"), player)
                if pair:
                    found = g
                    picked_line, picked_dec = pair
                    picked_book = b.get("key")
                    break
            if found: break

    if picked_line is None or picked_dec is None:
        # can't compute EV/Kelly without a line/price
        return None

    # 3) compute priors at this specific line
    df2 = df.copy()
    df2["HitOver"] = df2["value"] > float(picked_line)
    h2h = float(df2["HitOver"].mean()) if len(df2) else 0.0
    l10 = rolling_hit_rate(df2["value"].tail(10), float(picked_line))
    l5  = rolling_hit_rate(df2["value"].tail(5), float(picked_line))
    true_prob = (0.5*h2h + 0.3*l10 + 0.2*l5)

    # 4) weather adjustment (if outdoor + coords provided)
    if weather_coords and kickoff_utc:
        wx = get_weather_at(weather_coords[0], weather_coords[1], kickoff_utc)
        true_prob = weather_adjust_prob(true_prob, wx, sport, stat)

    imp = implied_prob_from_decimal(picked_dec)
    ev = (true_prob * picked_dec) - 1.0 if picked_dec else 0.0
    k = kelly_fraction(true_prob, picked_dec)
    k_half_cap = cap_fraction(0.5*k, 0.1)  # half-kelly, hard cap at 10% bankroll

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
        ev_pct=float(ev*100.0),
        kelly=float(k),
        kelly_half_capped=float(k_half_cap*bankroll),
        games=int(len(df2)),
        priors={"H2H":h2h,"L10":l10,"L5":l5},
        sample_table=df2[["game_date","value","HitOver"]].tail(12).reset_index(drop=True)
    )

# =========================
# Helpers: Odds API Player Fetch
# =========================
def fetch_all_players_from_oddsapi(sport_key: str, market_type: str, api_key: str) -> pd.DataFrame:
    """
    Fetch all available player markets for a given sport and market type
    (e.g. all NFL receiving_yards props) from The Odds API.
    
    Parameters:
        sport_key (str): The Odds API sport key (e.g. 'basketball_nba', 'americanfootball_nfl')
        market_type (str): The player market type (e.g. 'assists', 'receiving_yards')
        api_key (str): The Odds API key from .env
    
    Returns:
        pd.DataFrame: DataFrame containing all available players, games, markets, and odds
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

# =========================
# Today’s auto-scan
# =========================
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
    teammate: Optional[str]=None
) -> pd.DataFrame:
    """
    tickets: list of dicts like:
      {"player":"Cade Cunningham","opponent":"BOS","stat":"assists"}
    Returns: sorted DF with EV/Kelly
    """
    if odds_market_keys is None:
        if sport.upper()=="NBA":
            odds_market_keys = ["player_assists","player_points","player_rebounds"]
        elif sport.upper()=="MLB":
            odds_market_keys = ["player_hits","player_home_runs","player_rbis","player_strikeouts"]
        else:
            odds_market_keys = ["player_receiving_yards","player_rushing_yards","player_passing_yards","player_touchdowns"]

    rows=[]
    for t in tickets:
        try:
            edge = analyze_with_odds(
                sport=sport, player=t["player"], opponent=t["opponent"], stat=t["stat"],
                seasons=seasons, preferred_book_key=preferred_book_key,
                odds_sport_key=odds_sport_key, odds_market_keys=odds_market_keys,
                bankroll=bankroll, teammate=teammate,
                weather_coords=t.get("weather_coords"), kickoff_utc=t.get("kickoff_utc")
            )
            if not edge: 
                rows.append({**t,"Note":"No line/price found"})
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
                "ImplProb%": round(edge.implied_prob*100,1) if edge.implied_prob is not None else None,
                "TrueProb%": round(edge.true_prob*100,1),
                "EV%": round(edge.ev_pct,2),
                "Kelly_Frac": round(edge.kelly,4),
                "Stake_$ (HalfKellyCapped10%)": round(edge.kelly_half_capped,2),
                "Games": edge.games,
                "H2H": round(edge.priors["H2H"]*100,1),
                "L10": round(edge.priors["L10"]*100,1),
                "L5": round(edge.priors["L5"]*100,1),
            })
        except Exception as e:
            rows.append({**t,"Error":str(e)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # normalize column names if needed
    rename_map = {}
    if "EV_Pct" in df.columns and "EV%" not in df.columns:
        rename_map["EV_Pct"] = "EV%"
    if "TrueProb_Pct" in df.columns and "TrueProb%" not in df.columns:
        rename_map["TrueProb_Pct"] = "TrueProb%"
    df.rename(columns=rename_map, inplace=True)

    # sort by whatever columns exist
    sort_cols = [col for col in ["EV%", "TrueProb%", "Games"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False]*len(sort_cols)).reset_index(drop=True)
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


def push_recommended(table: pd.DataFrame, title="Recommended Bets"):
    msg = format_recommended_msg(title, table)
    send_discord(msg, DISCORD_WEBHOOK)

def push_placed_bet(*args, **kwargs):
    """Placeholder — implement bet logging later."""
    print("push_placed_bet() called — not implemented yet.")
