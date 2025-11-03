# ==========================================================
# unified_odds_engine_v3_1.py
# Streamlit-ready module (cached fetchers + robust extractors)
# Supports: SportsGameOdds v2/events, The Odds API v4, SportsData.io
# Includes Over / Under / Yes / No / Spread / Moneyline parsing
# ==========================================================

from __future__ import annotations

import os
import time
import math
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import requests
import streamlit as st


# ==========================================================
# 🔧 General Helpers
# ==========================================================
def fmt_pct(x: Optional[float]) -> str:
    return f"{x * 100:0.1f}%" if isinstance(x, (int, float)) and not pd.isna(x) else ""


def safe_get(d: Any, *keys, default=None):
    """Safely descend dict-like objects (tolerates missing keys / non-dicts)."""
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            # Try attribute access for SDK objects
            cur = getattr(cur, k, None)
    return default if cur is None else cur


# ---------- Odds / Probability math ----------
def american_to_decimal(odds: Optional[str]) -> Optional[float]:
    """Convert American odds string ('+120', '-130') to decimal (1.0+)."""
    if odds is None:
        return None
    try:
        s = str(odds).strip()
        if not s:
            return None
        if s[0] == '+':
            s = s[1:]
        o = int(s)
        if o > 0:
            return 1 + (o / 100.0)
        else:
            return 1 + (100.0 / abs(o))
    except Exception:
        return None


def implied_probability(american_odds: Optional[str]) -> Optional[float]:
    """Implied probability from American odds (0..1)."""
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
            return 100.0 / (o + 100.0)
        else:
            return abs(o) / (abs(o) + 100.0)
    except Exception:
        return None


def kelly_fraction(p: Optional[float], dec: Optional[float]) -> float:
    """
    Kelly fraction for decimal odds. p is win prob (0..1), dec is decimal odds.
    Returns fraction of bankroll (0..1); negative if –EV.
    """
    if p is None or dec is None or p <= 0 or p >= 1 or dec <= 1.0:
        return 0.0
    b = dec - 1.0  # net odds in decimal
    q = 1 - p
    num = (b * p) - q
    den = b
    return num / den if den != 0 else 0.0


def normalize_market_name(market_key: str, market_name: str) -> str:
    """
    Convert raw market names/keys into friendly categories, keeping O/U polarity
    when words are present in either the key or market_name.
    """
    key = (market_key or "").lower()
    name = (market_name or "").lower()

    def contains(*words):
        return all((w in key) or (w in name) for w in words)

    # Over
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
        if "passing" in key or contains("passing", "yards"):
            return "QB Passing Yards Over"
        if "rushing" in key or contains("rushing", "yards"):
            return "Rushing Yards Over"
        if "receiving" in key or contains("receiving", "yards"):
            return "Receiving Yards Over"

    # Under
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
        if "passing" in key or contains("passing", "yards"):
            return "QB Passing Yards Under"
        if "rushing" in key or contains("rushing", "yards"):
            return "Rushing Yards Under"
        if "receiving" in key or contains("receiving", "yards"):
            return "Receiving Yards Under"

    return market_name or market_key


def choose_line(odd_obj: Dict[str, Any]) -> Optional[float]:
    """Pick the best available numeric line (O/U or spread) for display."""
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
        if not s:
            return None
        if s[0] not in "+-":
            try:
                n = int(s)
                return f"+{n}" if n > 0 else str(n)
            except Exception:
                return s
        return s

    return norm(bo), norm(fo)


# ==========================================================
# 🧠 SIDE LABEL DETECTOR (Universal Fallback)
# ==========================================================
def determine_side_label(oddID, market_name, book_line, odd_obj, home, away) -> str:
    """Universal fallback detector for Over/Under, Spread, ML, Yes/No."""
    id_lower = str(oddID or "").lower()
    name_lower = str(market_name or "").lower()
    combined = f"{id_lower} {name_lower}"

    if "over" in combined:
        return f"Over {book_line}" if book_line else "Over"
    if "under" in combined:
        return f"Under {book_line}" if book_line else "Under"
    if any(x in combined for x in [" yes", "will score", "to score", "first touchdown", "made", "converted"]):
        return "Yes"
    if any(x in combined for x in [" no", "will not", "not score", "missed", "failed"]):
        return "No"

    # Moneyline
    if "moneyline" in combined or combined.strip().endswith(" ml"):
        side_id = (odd_obj or {}).get("sideID", "").lower() if odd_obj else ""
        if "home" in combined or side_id == "home":
            return f"{home} ML" if home else "Moneyline"
        if "away" in combined or side_id == "away":
            return f"{away} ML" if away else "Moneyline"
        return "Moneyline"

    # Spread
    if "spread" in combined:
        if "home" in combined:
            return f"{home} Spread {book_line}" if (home and book_line) else f"{home} Spread" if home else "Spread"
        if "away" in combined:
            return f"{away} Spread {book_line}" if (away and book_line) else f"{away} Spread" if away else "Spread"
        return "Spread"

    # Team totals
    if "team total" in combined or "team points" in combined:
        if "home" in combined:
            return f"{home} Team Total" if home else "Team Total"
        if "away" in combined:
            return f"{away} Team Total" if away else "Team Total"
        return "Team Total"

    # Generic player stat fallback
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

    return "Unknown"


# ==========================================================
# 🧩 FINALIZE & CLEAN ODDS DATAFRAME (shared post-step)
# ==========================================================
def finalize_odds_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Cleans and standardizes final odds DataFrame across all extractors.
    Includes formatting, sorting, NaN handling, and deduplication.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Enforce numeric types
    for c in ["Line", "ImpliedProb", "TrueProb", "EdgePct", "EV_Pct", "Kelly_Pct",
              "HalfKelly_Pct", "HalfKellyCapped_Pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Pretty %
    if "ImpliedProb" in df.columns:
        df["ImpliedProb%"] = df["ImpliedProb"].apply(fmt_pct)
    if "TrueProb" in df.columns:
        df["TrueProb%"] = df["TrueProb"].apply(fmt_pct)
    if "EdgePct" in df.columns:
        df["Edge%"] = df["EdgePct"].map(lambda x: f"{x:0.2f}%" if pd.notna(x) else "")

    # Keep rows with real odds
    if "BookOdds" in df.columns:
        df = df[df["BookOdds"].notna() & (df["BookOdds"].astype(str) != "")]

    # Sort & de-dupe
    sort_cols = [c for c in ["League", "Commence", "Game", "Player", "MarketType", "Side"] if c in df.columns]
    df = df.drop_duplicates(ignore_index=True)
    if sort_cols:
        if "Commence" in df.columns:
            df["Commence"] = pd.to_datetime(df["Commence"], errors="coerce")
        df = df.sort_values(sort_cols, ignore_index=True)

    # Column order
    ordered_cols = [
        "League", "EventID", "Commence", "Game", "Player", "PlayerID",
        "MarketType", "MarketName", "Side", "Line",
        "BookOdds", "TrueOdds", "ImpliedProb", "TrueProb",
        "EV_Pct", "Kelly_Pct", "HalfKelly_Pct", "HalfKellyCapped_Pct",
        "BooksAvailable", "ImpliedProb%", "TrueProb%", "Edge%"
    ]
    extras = [c for c in df.columns if c not in ordered_cols]
    return df[[c for c in ordered_cols if c in df.columns] + extras]


# ==========================================================
# ⚙️ FETCHERS (Streamlit-cached)
# ==========================================================
@st.cache_data(ttl=300)
def fetch_sgo_events(
    sgo_api_key: str,
    league_id: str,
    limit: int = 100,
    include_markets: str = "player_points,player_rebounds,player_assists,player_touchdowns",
    include_books: str = "hardrockbet,draftkings,fanduel,caesars,espnbet",
    include_alt_lines: bool = False,
    include_opposing: bool = False
) -> Dict[str, Any]:
    """
    Pull limited, high-value market data from SportsGameOdds v2/events endpoint.
    Rate-limit safe: trims to ~limit and caches for 5 minutes.
    """
    if not sgo_api_key:
        return {}

    url = "https://api.sportsgameodds.com/v2/events"
    params = {
        "apiKey": sgo_api_key.strip(),
        "oddsAvailable": "true",
        "leagueID": league_id,
        "limit": int(limit),
        "include": "players,teams,markets,odds",
        "includeAltLines": "true" if include_alt_lines else "false",
        "includeOpposingOdds": "true" if include_opposing else "false",
        "includeMarkets": include_markets,
        "includeBooks": include_books,
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            # Gentle backoff then one retry
            time.sleep(1.0)
            r = requests.get(url, params=params, timeout=30)

        if r.status_code != 200:
            try:
                body = r.json()
                st.warning(f"SGO {r.status_code}: {body.get('error') or str(body)[:200]}")
            except Exception:
                st.warning(f"SGO {r.status_code}: {r.text[:200]}")
            return {}

        data = r.json()
        if isinstance(data, dict) and "data" in data:
            valid_events = [ev for ev in data["data"] if ev.get("odds")]
            data["data"] = valid_events[: int(limit)]
        return data

    except Exception as e:
        st.error(f"SGO fetch error: {e}")
        return {}


@st.cache_data(ttl=300)
def fetch_oddsapi_games(
    odds_api_key: str,
    sport_key: str = "basketball_nba",
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    bookmakers: str = "hardrockbet,draftkings,fanduel,caesars,espnbet",
    limit: int = 100
) -> list:
    """
    Fetch limited game-line odds from The Odds API (v4).
    Returns up to ~limit events.
    """
    if not odds_api_key:
        return []

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": odds_api_key.strip(),
        "regions": regions,
        "markets": markets,
        "bookmakers": bookmakers,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(1.0)
            r = requests.get(url, params=params, timeout=30)

        if r.status_code != 200:
            try:
                body = r.json()
                st.warning(f"OddsAPI {r.status_code}: {body.get('message') or str(body)[:200]}")
            except Exception:
                st.warning(f"OddsAPI {r.status_code}: {r.text[:200]}")
            return []

        data = r.json()
        if isinstance(data, list):
            data = data[: int(limit)]
        return data

    except Exception as e:
        st.error(f"OddsAPI fetch error: {e}")
        return []


# Optional generic SportsData.io fetch (you can pass your prepared URL/params)
@st.cache_data(ttl=300)
def fetch_sportsdataio(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None):
    try:
        r = requests.get(url, headers=headers or {}, params=params or {}, timeout=30)
        if r.status_code == 429:
            time.sleep(1.0)
            r = requests.get(url, headers=headers or {}, params=params or {}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"SportsData.io fetch error: {e}")
        return []


# ==========================================================
# 🟢 SPORTS GAME ODDS EXTRACTOR
# ==========================================================
def extract_sgo_df(payload: Any, wanted_books: List[str]) -> pd.DataFrame:
    """
    Unified SportsGameOdds (SGO) extractor.
    Handles Over/Under/Yes/No markets and Anytime Touchdowns.
    Preserves EV, Kelly, and probability calculations.
    """
    # Accept SDK-like (obj) or dict payload
    if hasattr(payload, "data"):
        events = payload.data or []
    elif isinstance(payload, dict) and payload.get("data"):
        events = payload["data"]
    else:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for ev in events:
        event_id = getattr(ev, "eventID", None) or ev.get("eventID")
        league_id = getattr(ev, "leagueID", None) or ev.get("leagueID")

        teams = safe_get(ev, "teams") or {}
        away = safe_get(teams, "away", "names", "long", default="Away")
        home = safe_get(teams, "home", "names", "long", default="Home")
        game_name = f"{away} @ {home}"

        players_map = safe_get(ev, "players") or {}
        odds = getattr(ev, "odds", None) or ev.get("odds") or {}

        # Merge O/U and Yes/No style keys so we see both sides
        merged_odds: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
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
                market_name = odd_obj.get("marketName") or odd_obj.get("betTypeID") or oddID
                market_type = normalize_market_name(oddID, market_name)

                # Identify TD/YesNo groupings
                stat_id = (odd_obj.get("statID") or "").lower()
                stat_entity = (odd_obj.get("statEntityID") or "").lower()
                bet_type = (odd_obj.get("betTypeID") or "").lower()
                if stat_id == "touchdowns" and bet_type == "yn":
                    if stat_entity == "any_player_id":
                        market_name = "Player Anytime Touchdowns Yes/No"
                    elif stat_entity in ["home", "away"]:
                        market_name = "Team Anytime Touchdowns Yes/No"
                    elif stat_entity == "all":
                        market_name = "Any Touchdowns Yes/No"
                    market_type = market_name

                # Side detection (robust)
                id_lower = str(oddID).lower()
                name_lower = str(market_name).lower()
                desc_lower = str(odd_obj.get("name") or "").lower()
                desc_extra = str(odd_obj.get("description") or "").lower()

                if "over" in (id_lower + name_lower + desc_lower + desc_extra):
                    side_label = "Over"
                elif "under" in (id_lower + name_lower + desc_lower + desc_extra):
                    side_label = "Under"
                elif "yes" in (id_lower + name_lower + desc_lower + desc_extra):
                    side_label = "Yes"
                elif "no" in (id_lower + name_lower + desc_lower + desc_extra):
                    side_label = "No"
                else:
                    side_label = determine_side_label(
                        oddID=oddID, market_name=market_name, book_line=odd_obj.get("point"),
                        odd_obj=odd_obj, home=home, away=away
                    )

                # Player / entity info
                stat_entity = odd_obj.get("statEntityID")
                player_id = odd_obj.get("playerID") or (
                    stat_entity if stat_entity not in ("home", "away", "all") else None
                )
                player_name = (
                    safe_get(players_map, player_id, "name", default=player_id)
                    if player_id in players_map else None
                )

                # Filter by wanted books
                bybk = odd_obj.get("byBookmaker") or {}
                chosen = {
                    bk: val for bk, val in bybk.items()
                    if (not wanted_books or bk in wanted_books) and isinstance(val, dict)
                }
                if not chosen:
                    continue

                # Extract price / line
                book_line = choose_line(odd_obj) or odd_obj.get("line")
                book_odds_str, fair_odds_str = choose_odds(odd_obj)
                if not book_odds_str and "price" in odd_obj:
                    book_odds_str = odd_obj.get("price")

                if not book_line or str(book_line).strip().lower() in ["", "none"]:
                    if any(x in market_name.lower() for x in ["touchdown", "score", "td"]):
                        book_line = "Yes/No"
                    else:
                        book_line = "-"

                # Metrics
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
                    Side=side_label,
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


# ==========================================================
# ⚙️ ODDS API EXTRACTOR (Updated for NCAAM & Multi-League Support)
# ==========================================================
def extract_oddsapi_df(raw_data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize The Odds API v4 output into a unified DataFrame.
    Handles Moneyline (h2h), Spreads, and Totals (Over/Under).
    Adds ImpliedProb, EV%, Kelly%, and formatted % columns.
    Compatible with all basketball/football/baseball/soccer leagues (NBA, NFL, NCAAM, etc.).
    """
    if not raw_data:
        return pd.DataFrame()

    events = raw_data if isinstance(raw_data, list) else [raw_data]
    rows: List[Dict[str, Any]] = []

    for ev in events:
        sport_title = ev.get("sport_title", "")
        sport_key = ev.get("sport_key", "")
        event_id = ev.get("id", "")
        commence = ev.get("commence_time")
        home_team = ev.get("home_team", "")
        away_team = ev.get("away_team", "")
        teams = ev.get("teams") or []

        # Prefer away/home format when available
        if away_team and home_team:
            game_name = f"{away_team} @ {home_team}"
        elif teams:
            game_name = " @ ".join(teams)
        else:
            game_name = home_team or "Unknown Matchup"

        for bk in ev.get("bookmakers", []) or []:
            book = bk.get("title", "") or bk.get("key", "")

            for mk in bk.get("markets", []) or []:
                mkey = mk.get("key", "").lower()  # 'h2h', 'spreads', 'totals'
                outcomes = mk.get("outcomes", []) or []

                for o in outcomes:
                    side = o.get("name", "")
                    point = o.get("point")
                    odds_str = str(o.get("price")) if o.get("price") is not None else None
                    implied = implied_probability(odds_str)
                    dec = american_to_decimal(odds_str)
                    ev_pct = (((implied * dec) - 1) * 100.0) if (implied and dec) else None
                    kelly = (kelly_fraction(implied, dec) * 100.0) if (implied and dec) else None

                    # Map markets to readable labels
                    if mkey == "h2h":
                        market_type = "moneyline"
                        side_label = side
                    elif mkey == "spreads":
                        market_type = "spread"
                        side_label = f"{side} {float(point):+g}" if point is not None else side
                    elif mkey == "totals":
                        market_type = "total_points"
                        side_label = f"{side} {float(point):g}" if point is not None else side
                    else:
                        continue

                    rows.append(dict(
                        League=sport_title or sport_key,
                        EventID=event_id,
                        Commence=commence,
                        Game=game_name,
                        MarketType=market_type,
                        MarketName=market_type,
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

    # Return formatted DataFrame
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Add convenience columns
    df["League"] = df["League"].replace(
        {
            "basketball_nba": "NBA",
            "basketball_ncaab": "NCAAM",
            "americanfootball_nfl": "NFL",
            "americanfootball_ncaaf": "NCAAF",
        }
    )

    return finalize_odds_df(rows)


# ==========================================================
# 🏈 SPORTSDATA.IO EXTRACTOR
# ==========================================================
def extract_sportsdataio_df(raw_data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize SportsData.io props into the unified format.
    Handles Over/Under inference, implied probabilities, and Kelly metrics.
    """
    if not raw_data:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for p in (raw_data if isinstance(raw_data, list) else [raw_data]):
        game = p.get("GameDisplay") or f"{p.get('AwayTeam')} @ {p.get('HomeTeam')}"
        player = p.get("PlayerName")
        market = p.get("BetType") or ""
        stat_line = p.get("PlayerPropTotal")
        odds = p.get("PayoutAmerican")
        book = p.get("Sportsbook")

        market_lower = market.lower()
        if "over" in market_lower:
            side_label = f"Over {stat_line}" if stat_line else "Over"
        elif "under" in market_lower:
            side_label = f"Under {stat_line}" if stat_line else "Under"
        else:
            side_label = determine_side_label(
                market, market, stat_line, p, p.get("HomeTeam"), p.get("AwayTeam")
            )

        implied_prob = implied_probability(odds)
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


# ==========================================================
# 🧹 QUALITY FILTER (optional helper for dashboards)
# ==========================================================
def quality_filter(
    df: pd.DataFrame,
    min_trueprob: Optional[float] = None,    # e.g., 0.53
    min_edge_pct: Optional[float] = None,    # e.g., 5.0
    books_whitelist: Optional[List[str]] = None,
    max_rows: Optional[int] = 100
) -> pd.DataFrame:
    """
    Keep the best rows based on TrueProb & Edge, whitelisted books, and row cap.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    if books_whitelist and "BooksAvailable" in out.columns:
        out = out[out["BooksAvailable"].astype(str).str.lower().apply(
            lambda s: any(b.lower() in s for b in books_whitelist)
        )]

    if (min_trueprob is not None) and ("TrueProb" in out.columns):
        out = out[out["TrueProb"].fillna(0) >= float(min_trueprob)]

    if (min_edge_pct is not None) and ("EdgePct" in out.columns):
        out = out[out["EdgePct"].fillna(-9999) >= float(min_edge_pct)]

    # Prefer highest EV, then TrueProb, then recent commence if available
    sort_cols = [c for c in ["EV_Pct", "TrueProb", "Commence"] if c in out.columns]
    if sort_cols:
        ascending = [False if c in ["EV_Pct", "TrueProb"] else True for c in sort_cols]
        out = out.sort_values(sort_cols, ascending=ascending, ignore_index=True)

    if max_rows is not None:
        out = out.head(int(max_rows))

    return out
