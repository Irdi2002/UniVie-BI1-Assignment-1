"""
Main orchestration for the international football BI pipeline.

Run with:
    python main.py

Inputs (in data/raw/):
    results.csv                    -- 49,287 international matches (1872-2026)
    fifa_ranking-2024-06-20.csv    -- 67,472 monthly FIFA rankings (1992-2024)
    goalscorers.csv                -- optional, not used in this pipeline
    shootouts.csv                  -- optional, not used in this pipeline

Outputs (in data/output/):
    fact_match.csv             -- one row per international match with attached rankings
    dim_date.csv               -- daily date dimension covering the fact table date range
    dim_country.csv            -- one row per country with confederation
    dim_tournament.csv         -- one row per tournament with category flags
    dim_match_context.csv      -- static home/away/neutral lookup

"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import helpers as H

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PIPELINE_DIR = Path(__file__).parent
RAW_DIR = PIPELINE_DIR / "data" / "raw"
OUT_DIR = PIPELINE_DIR / "data" / "output"
MAPPING_PATH = PIPELINE_DIR / "country_mapping.csv"

# Earliest FIFA ranking (rankings start August 1992; we filter to 1993+ to ensure
# both teams in any match have at least a chance of having a baseline ranking).
EARLIEST_FACT_DATE = pd.Timestamp("1993-01-01")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. EXTRACT — read raw CSVs
    # -------------------------------------------------------------------------
    logger.info("=== STAGE 1: EXTRACT ===")
    matches = pd.read_csv(RAW_DIR / "results.csv")
    rankings = pd.read_csv(RAW_DIR / "fifa_ranking-2024-06-20.csv")
    logger.info("Loaded %d matches, %d ranking snapshots", len(matches), len(rankings))

    # -------------------------------------------------------------------------
    # 2. TRANSFORM — types, cleaning, country reconciliation
    # -------------------------------------------------------------------------
    logger.info("=== STAGE 2: TRANSFORM ===")

    matches = H.parse_match_dates(matches)
    rankings = H.parse_ranking_dates(rankings)
    logger.info(
        "Date ranges -- matches: %s to %s, rankings: %s to %s",
        matches["date"].min().date(),
        matches["date"].max().date(),
        rankings["rank_date"].min().date(),
        rankings["rank_date"].max().date(),
    )

    matches = H.clean_matches(matches)

    mapping = H.load_country_mapping(MAPPING_PATH)
    matches = H.apply_country_mapping(matches, ["home_team", "away_team", "country"], mapping)

    # -------------------------------------------------------------------------
    # 3. INTEGRATE — as-of join rankings onto matches
    # -------------------------------------------------------------------------
    logger.info("=== STAGE 3: INTEGRATE ===")

    matches = H.attach_ranking(matches, rankings, team_column="home_team", rank_prefix="home")
    matches = H.attach_ranking(matches, rankings, team_column="away_team", rank_prefix="away")
    logger.info(
        "After as-of join: %d matches; %d have both ranks; %d have neither",
        len(matches),
        matches[["home_rank", "away_rank"]].notna().all(axis=1).sum(),
        matches[["home_rank", "away_rank"]].isna().all(axis=1).sum(),
    )

    # Filter the fact table to the analytically usable window (rankings era).
    # Matches before 1993 are kept out of the FACT TABLE — but we'd note in
    # slides that they exist as historical context. We also stop at the latest
    # ranking snapshot available, otherwise later matches would be evaluated
    # with stale FIFA rankings.
    n_before = len(matches)
    latest_fact_date = rankings["rank_date"].max()
    matches = matches[
        (matches["date"] >= EARLIEST_FACT_DATE)
        & (matches["date"] <= latest_fact_date)
    ].reset_index(drop=True)
    logger.info(
        "Filtered to %s through %s: %d -> %d matches",
        EARLIEST_FACT_DATE.date(),
        latest_fact_date.date(),
        n_before,
        len(matches),
    )

    # The Tableau model focuses on matches between FIFA-ranked teams. Regional
    # and non-FIFA teams from the results dataset cannot be connected to the
    # FIFA ranking dataset and would otherwise produce null keys/measures.
    matches = H.keep_fifa_ranked_matches(matches)
    matches = H.derive_outcomes(matches)

    # -------------------------------------------------------------------------
    # 4. BUILD DIMENSIONS
    # -------------------------------------------------------------------------
    logger.info("=== STAGE 4: BUILD DIMENSIONS ===")

    dim_date = H.build_date_dimension(
        start=matches["date"].min(),
        end=matches["date"].max(),
    )
    dim_country = H.build_country_dimension(rankings)
    dim_tournament = H.build_tournament_dimension(matches)
    dim_match_context = H.build_match_context_dimension()

    logger.info(
        "Dimensions built -- date: %d rows, country: %d rows, tournament: %d rows",
        len(dim_date), len(dim_country), len(dim_tournament),
    )

    # -------------------------------------------------------------------------
    # 5. BUILD FACT TABLE — attach surrogate keys
    # -------------------------------------------------------------------------
    logger.info("=== STAGE 5: BUILD FACT TABLE ===")

    # date_key = YYYYMMDD (matches dim_date)
    fact = matches.copy()
    fact["date_key"] = fact["date"].dt.strftime("%Y%m%d").astype(int)

    # Map team names -> country_key. The FIFA-ranked filter above ensures these
    # foreign keys are complete in the final fact table.
    # We use pandas' nullable 'Int64' dtype so the column stays integer in the
    # CSV output (writes as '77', not '77.0'), which avoids a float-vs-int type
    # mismatch when Tableau builds the dim_country relationship.
    country_lookup = dict(zip(dim_country["country_name"], dim_country["country_key"]))
    fact["home_team_key"] = fact["home_team"].map(country_lookup).astype("Int64")
    fact["away_team_key"] = fact["away_team"].map(country_lookup).astype("Int64")

    # Tournament key
    tournament_lookup = dict(zip(dim_tournament["tournament_name"], dim_tournament["tournament_key"]))
    fact["tournament_key"] = fact["tournament"].map(tournament_lookup)

    # Match context key — derived from the neutral flag and host country.
    # 'home' if the home team is in its country, 'neutral' if explicitly flagged,
    # otherwise 'other_host'. The latter mostly captures historical country-name
    # mismatches and non-FIFA regional teams rather than true away-home fixtures.
    def context_key(row: pd.Series) -> int:
        if row["neutral"]:
            return 3
        if row["country"] == row["home_team"]:
            return 1
        return 2

    fact["match_context_key"] = fact.apply(context_key, axis=1)

    # Surrogate key for the fact itself
    fact = fact.reset_index(drop=True)
    fact.insert(0, "match_key", fact.index + 1)

    # Final column selection -- this is the contract Tableau will read
    fact_final = fact[
        [
            "match_key",
            "date_key",
            "home_team_key",
            "away_team_key",
            "tournament_key",
            "match_context_key",
            # measures
            "home_score",
            "away_score",
            "goal_difference",
            "home_rank",
            "away_rank",
            "home_points",
            "away_points",
            "rank_gap",
            "points_gap",
            # categorical outcome attributes
            "ranking_predicted_winner",
            "actual_winner",
            "ranking_was_correct",
        ]
    ]

    # Diagnostic: how many matches are usable for Q1 (ranking accuracy) analysis?
    usable_q1 = fact_final["ranking_was_correct"].notna().sum()
    logger.info(
        "Fact table built: %d rows total, %d usable for Q1 (complete FIFA ranking data)",
        len(fact_final), usable_q1,
    )

    # -------------------------------------------------------------------------
    # 6. LOAD — write out CSVs
    # -------------------------------------------------------------------------
    logger.info("=== STAGE 6: LOAD ===")

    outputs = {
        "fact_match.csv": fact_final,
        "dim_date.csv": dim_date,
        "dim_country.csv": dim_country,
        "dim_tournament.csv": dim_tournament,
        "dim_match_context.csv": dim_match_context,
    }
    for name, df in outputs.items():
        path = OUT_DIR / name
        df.to_csv(path, index=False)
        logger.info("Wrote %s (%d rows, %d cols)", name, len(df), len(df.columns))

    logger.info("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
