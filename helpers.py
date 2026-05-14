"""
Reusable functions for the international football BI pipeline.

Each function does one thing. Each function has a docstring explaining what it
does, what it expects as input, and what it returns. Read these before reading
main.py.

Authors: <add your group names here>
Course: Business Intelligence 1, University of Vienna, Summer 2026
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tournament categorization
# -----------------------------------------------------------------------------

# Bucketing rules for the ~193 distinct tournament names. Order matters:
# checks run top-to-bottom, first match wins. We keep this as a config list so
# future maintainers can extend without touching code logic.
TOURNAMENT_CATEGORY_RULES = [
    # (substring matched against tournament name (case-insensitive), category)
    # Most specific first. "qualification" must be checked BEFORE "FIFA World Cup"
    # otherwise WC qualifiers would be misclassified as the World Cup itself.
    ("qualification", "Qualifier"),
    ("qualifying", "Qualifier"),
    ("FIFA World Cup", "FIFA World Cup"),
    ("UEFA Euro", "Continental"),
    ("Copa América", "Continental"),
    ("African Cup of Nations", "Continental"),
    ("AFC Asian Cup", "Continental"),
    ("Gold Cup", "Continental"),
    ("Oceania Nations Cup", "Continental"),
    ("UEFA Nations League", "Nations League"),
    ("CONCACAF Nations League", "Nations League"),
    ("Confederations Cup", "Confederations Cup"),
    ("Friendly", "Friendly"),
]


def categorize_tournament(name: str) -> str:
    """Map a raw tournament name to a coarse category.

    Args:
        name: The raw tournament string from the matches dataset.

    Returns:
        One of: 'FIFA World Cup', 'Qualifier', 'Continental', 'Nations League',
        'Confederations Cup', 'Friendly', 'Other'.
    """
    if not isinstance(name, str):
        return "Other"
    name_lower = name.lower()
    for substring, category in TOURNAMENT_CATEGORY_RULES:
        if substring.lower() in name_lower:
            return category
    return "Other"


def is_knockout_tournament(name: str) -> bool:
    """Heuristic: is this match part of a knockout tournament stage?

    Friendlies, qualifiers, and league-format tournaments (Nations League) are
    NOT knockouts. Major tournaments (World Cup, continental cups, Confeds Cup)
    include both group and knockout stages — without per-match stage labels we
    treat the whole tournament as 'knockout-eligible'. Document this limitation
    in the slides — it's a real caveat for Q1 interpretation.
    """
    category = categorize_tournament(name)
    return category in {"FIFA World Cup", "Continental", "Confederations Cup"}


def is_competitive_tournament(category: str) -> bool:
    """Friendlies are non-competitive; everything else is competitive."""
    return category != "Friendly"


# -----------------------------------------------------------------------------
# Country name reconciliation
# -----------------------------------------------------------------------------

def load_country_mapping(path: str | Path) -> dict[str, str]:
    """Load the manual mapping from match-side names to ranking-side names.

    The mapping CSV has columns: match_name, ranking_name, note.
    Returns a dict suitable for pandas Series.replace().

    Names not in the mapping (the vast majority) pass through unchanged.
    """
    df = pd.read_csv(path)
    expected_cols = {"match_name", "ranking_name", "note"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"country_mapping.csv missing columns: {missing}")
    mapping = dict(zip(df["match_name"], df["ranking_name"]))
    logger.info("Loaded %d country name mappings", len(mapping))
    return mapping


def apply_country_mapping(
    df: pd.DataFrame, columns: list[str], mapping: dict[str, str]
) -> pd.DataFrame:
    """Apply the name mapping to one or more columns in a DataFrame.

    Returns a copy. Columns not in the mapping pass through unchanged.
    """
    df = df.copy()
    for col in columns:
        df[col] = df[col].replace(mapping)
    return df


# -----------------------------------------------------------------------------
# Date parsing
# -----------------------------------------------------------------------------

def parse_ranking_dates(rankings: pd.DataFrame) -> pd.DataFrame:
    """Convert the rank_date column from DD-MM-YY string to proper datetime.

    The Kaggle dataset has shipped in two formats over time:
      - Older snapshots: '31-12-92'   (DD-MM-YY, no century)
      - Newer snapshots: '1992-12-31' (ISO YYYY-MM-DD)
    We detect which format is in use by inspecting the first non-null value
    and parse accordingly. This makes the pipeline robust to re-downloads
    even if Kaggle changes the export format again.
    """
    rankings = rankings.copy()
    sample = rankings["rank_date"].dropna().iloc[0]
    if len(str(sample)) == 8 and str(sample)[2] == "-":
        # DD-MM-YY format (e.g. '31-12-92'). pandas' 2-digit year cutoff
        # treats >= 69 as 19xx, < 69 as 20xx — correct for our range.
        date_format = "%d-%m-%y"
    else:
        # ISO YYYY-MM-DD format (e.g. '1992-12-31').
        date_format = "%Y-%m-%d"
    rankings["rank_date"] = pd.to_datetime(
        rankings["rank_date"], format=date_format, errors="raise"
    )
    return rankings


def parse_match_dates(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert match date column to datetime. Format is YYYY-MM-DD."""
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], format="%Y-%m-%d", errors="raise")
    return matches


# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------

def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Drop matches with missing scores and deduplicate.

    A match with NaN score is an abandoned/awarded result that pollutes win-rate
    calculations. We drop them and log the count for the audit trail.
    Duplicates on (date, home_team, away_team) are also dropped.
    """
    matches = matches.copy()
    n0 = len(matches)
    matches = matches.dropna(subset=["home_score", "away_score"])
    n1 = len(matches)
    matches = matches.drop_duplicates(subset=["date", "home_team", "away_team"])
    n2 = len(matches)
    logger.info(
        "clean_matches: dropped %d null-score rows, %d duplicates (%d -> %d)",
        n0 - n1, n1 - n2, n0, n2,
    )
    matches["home_score"] = matches["home_score"].astype(int)
    matches["away_score"] = matches["away_score"].astype(int)
    return matches


def keep_fifa_ranked_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Keep only matches where both teams have complete FIFA ranking data.

    The results dataset contains FIFA countries as well as regional/non-FIFA
    teams such as Martinique, Guadeloupe, Jersey, and French Guiana. Those
    teams cannot be connected to the FIFA ranking dataset, so keeping them in
    the BI fact table creates null foreign keys and null ranking measures.

    Returns a copy containing only rows with both home and away rank/points.
    """
    required_cols = ["home_rank", "away_rank", "home_points", "away_points"]
    missing_cols = set(required_cols) - set(matches.columns)
    if missing_cols:
        raise ValueError(f"Ranking columns missing before FIFA filter: {missing_cols}")

    n0 = len(matches)
    filtered = matches.dropna(subset=required_cols).copy()
    logger.info(
        "keep_fifa_ranked_matches: dropped %d matches without complete FIFA rankings (%d -> %d)",
        n0 - len(filtered), n0, len(filtered),
    )
    return filtered.reset_index(drop=True)


# -----------------------------------------------------------------------------
# As-of join: attach FIFA rankings to each match
# -----------------------------------------------------------------------------

def attach_ranking(
    matches: pd.DataFrame,
    rankings: pd.DataFrame,
    team_column: str,
    rank_prefix: str,
) -> pd.DataFrame:
    """Attach the most recent FIFA ranking BEFORE the match date for `team_column`.

    This is the centerpiece integration logic. We use pandas.merge_asof with
    direction='backward' so that for each match we pick the ranking that was
    in force on (or before) the match date — never a future ranking.

    Both DataFrames must be sorted by their date column. The `by` parameter
    ensures we look up rankings within the same country.

    Args:
        matches: Match-level DataFrame with 'date' (datetime) and `team_column`.
        rankings: Rankings DataFrame with 'rank_date' (datetime), 'country_full',
                  'rank', 'total_points', 'confederation'.
        team_column: 'home_team' or 'away_team' (canonical names already applied).
        rank_prefix: 'home' or 'away'. New columns become e.g. 'home_rank'.

    Returns:
        matches with new columns: <prefix>_rank, <prefix>_points, <prefix>_confederation.
        Rows where no ranking exists for that country/date stay as NaN — by design.
        That null-ness is the signal we use to exclude non-FIFA entities and
        pre-1993 matches from the analytical questions.
    """
    if not matches["date"].is_monotonic_increasing:
        matches = matches.sort_values("date").reset_index(drop=True)
    rankings_sorted = rankings.sort_values("rank_date").reset_index(drop=True)

    # We only need a few cols from rankings; renaming avoids collisions when
    # we run this twice (once for home, once for away).
    rankings_slim = rankings_sorted[
        ["rank_date", "country_full", "rank", "total_points", "confederation"]
    ].rename(
        columns={
            "rank": f"{rank_prefix}_rank",
            "total_points": f"{rank_prefix}_points",
            "confederation": f"{rank_prefix}_confederation",
            "country_full": team_column,  # so 'by=' aligns
        }
    )

    merged = pd.merge_asof(
        matches,
        rankings_slim,
        left_on="date",
        right_on="rank_date",
        by=team_column,
        direction="backward",
        allow_exact_matches=True,
    )
    # rank_date got merged in; drop it (we already have match date)
    merged = merged.drop(columns=["rank_date"])
    return merged


# -----------------------------------------------------------------------------
# Outcome derivation
# -----------------------------------------------------------------------------

def derive_outcomes(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute derived measures used by all four analytical questions.

    Adds:
        goal_difference: home_score - away_score
        rank_gap: away_rank - home_rank (positive = home is higher-ranked,
                  remembering rank 1 is best)
        points_gap: home_points - away_points
        actual_winner: 'home' / 'away' / 'draw'
        ranking_predicted_winner: 'home' / 'away' / 'draw' (None if either rank missing)
        ranking_was_correct: bool (NA only if either rank is missing)
    """
    df = matches.copy()
    df["goal_difference"] = df["home_score"] - df["away_score"]

    df["rank_gap"] = df["away_rank"] - df["home_rank"]
    df["points_gap"] = df["home_points"] - df["away_points"]

    # Actual winner — always defined when scores are present
    conditions_actual = [
        df["home_score"] > df["away_score"],
        df["home_score"] < df["away_score"],
    ]
    df["actual_winner"] = np.select(
        conditions_actual,
        ["home", "away"],
        default="draw",
    )

    # Ranking-predicted winner — undefined when either rank is missing.
    # NOTE: in FIFA rankings, LOWER rank number = BETTER team. So if home_rank
    # < away_rank, home is the favored team.
    has_both_ranks = df["home_rank"].notna() & df["away_rank"].notna()
    conditions_pred = [
        has_both_ranks & (df["home_rank"] < df["away_rank"]),
        has_both_ranks & (df["home_rank"] > df["away_rank"]),
        has_both_ranks & (df["home_rank"] == df["away_rank"]),
    ]
    df["ranking_predicted_winner"] = np.select(
        conditions_pred,
        ["home", "away", "draw"],
        default=None,
    )

    # Was the ranking correct? Equal ranks are treated as a draw prediction so
    # Tableau receives a complete boolean field after FIFA-ranked filtering.
    can_evaluate = df["ranking_predicted_winner"].isin(["home", "away", "draw"])
    df["ranking_was_correct"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df.loc[can_evaluate, "ranking_was_correct"] = (
        df.loc[can_evaluate, "ranking_predicted_winner"]
        == df.loc[can_evaluate, "actual_winner"]
    )
    return df


# -----------------------------------------------------------------------------
# Dimension builders
# -----------------------------------------------------------------------------

WORLD_CUP_YEARS = {
    1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982,
    1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026,
}


def build_date_dimension(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Generate dim_date covering [start, end] inclusive, daily granularity.

    Surrogate key is YYYYMMDD as int — convenient for Tableau and human-readable
    when debugging.
    """
    dates = pd.date_range(start=start, end=end, freq="D")
    dim = pd.DataFrame({"full_date": dates})
    dim["date_key"] = dim["full_date"].dt.strftime("%Y%m%d").astype(int)
    dim["year"] = dim["full_date"].dt.year
    dim["quarter"] = dim["full_date"].dt.quarter
    dim["month"] = dim["full_date"].dt.month
    dim["month_name"] = dim["full_date"].dt.strftime("%B")
    dim["day_of_week"] = dim["full_date"].dt.day_name()
    dim["decade"] = (dim["year"] // 10) * 10
    dim["is_world_cup_year"] = dim["year"].isin(WORLD_CUP_YEARS)
    return dim[
        ["date_key", "full_date", "year", "quarter", "month", "month_name",
         "day_of_week", "decade", "is_world_cup_year"]
    ]


def build_country_dimension(rankings: pd.DataFrame) -> pd.DataFrame:
    """Build dim_country from the rankings dataset.

    Uses each country's LATEST appearance to capture the current confederation
    assignment. (Edge case: Kazakhstan moved from AFC to UEFA in 2002. By using
    the most recent value, dim_country reflects today's reality.)

    Surrogate keys are 1-based integers.
    """
    latest = (
        rankings.sort_values("rank_date")
        .groupby("country_full", as_index=False)
        .last()[["country_full", "country_abrv", "confederation"]]
    )
    latest = latest.reset_index(drop=True)
    latest.insert(0, "country_key", latest.index + 1)
    latest = latest.rename(
        columns={"country_full": "country_name", "country_abrv": "country_code"}
    )
    return latest[["country_key", "country_name", "country_code", "confederation"]]


def build_tournament_dimension(matches: pd.DataFrame) -> pd.DataFrame:
    """Build dim_tournament from distinct tournament names with derived flags.

    is_knockout and is_competitive are precomputed so Tableau filtering is fast
    and consistent across sheets.
    """
    distinct = (
        matches[["tournament"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .rename(columns={"tournament": "tournament_name"})
    )
    distinct["tournament_category"] = distinct["tournament_name"].apply(categorize_tournament)
    distinct["is_knockout"] = distinct["tournament_name"].apply(is_knockout_tournament)
    distinct["is_competitive"] = distinct["tournament_category"].apply(is_competitive_tournament)
    distinct.insert(0, "tournament_key", distinct.index + 1)
    return distinct


def build_match_context_dimension() -> pd.DataFrame:
    """Static dim_match_context: home / other host / neutral.

    Trivially small but kept as a real dimension to satisfy star-schema purity
    and to give Tableau a clean filter pill.
    """
    return pd.DataFrame(
        {
            "match_context_key": [1, 2, 3],
            "context": ["home", "other_host", "neutral"],
            "description": [
                "Match played at the home team's country",
                "Match not marked neutral, but host country differs from the mapped home team",
                "Match played at a neutral venue (typical for tournaments)",
            ],
        }
    )
