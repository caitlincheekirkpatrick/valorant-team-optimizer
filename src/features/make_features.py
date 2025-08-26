import argparse, os
from typing import List, Dict
import pandas as pd
import numpy as np
from datetime import datetime
from utils.agents import parse_agent_list, role_counts, ROLES
from features.elo import compute_pre_match_elos

def map_one_hot(series: pd.Series) -> pd.DataFrame:
    maps = sorted(series.dropna().unique().tolist())
    oh = pd.get_dummies(series, prefix="map")
    # ensure stable column order
    return oh.reindex(columns=sorted(oh.columns), fill_value=0)

def rolling_map_winrate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team pre-match map winrate (all past matches on that map).
    Returns two new columns: teamA_map_wr_pre, teamB_map_wr_pre in [0,1].
    """
    df = df.copy()
    df["teamA_map_wr_pre"] = np.nan
    df["teamB_map_wr_pre"] = np.nan

    # per team-map historical record
    hist = {}  # (team, map) -> [wins, total]
    for i, row in df.iterrows():
        keyA = (row["teamA"], row["map"])
        keyB = (row["teamB"], row["map"])
        wA, tA = hist.get(keyA, [0, 0])
        wB, tB = hist.get(keyB, [0, 0])
        df.at[i, "teamA_map_wr_pre"] = 0.0 if tA == 0 else wA / tA
        df.at[i, "teamB_map_wr_pre"] = 0.0 if tB == 0 else wB / tB
        # update with current match outcome
        hist[keyA] = [wA + int(row["teamA_win"] == 1), tA + 1]
        hist[keyB] = [wB + int(row["teamA_win"] == 0), tB + 1]
    return df

def build_role_features(df: pd.DataFrame) -> pd.DataFrame:
    """Count roles per team and create diffs."""
    for side in ["teamA", "teamB"]:
        counts = df[side + "_agents"].apply(lambda s: role_counts(parse_agent_list(s)))
        for r in ROLES:
            df[f"{side}_cnt_{r}"] = counts.apply(lambda d: d.get(r, 0))
    for r in ROLES:
        df[f"diff_cnt_{r}"] = df[f"teamA_cnt_{r}"] - df[f"teamB_cnt_{r}"]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    raw = pd.read_csv(args.input)
    # parse and sort by date (chronological)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)

    # Basic validation
    needed = ["match_id","date","event","map","teamA","teamB","teamA_agents","teamB_agents","teamA_win"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    # Pre-match Elo priors (no leakage)
    eloA_pre, eloB_pre = compute_pre_match_elos(
        teamA=raw["teamA"].tolist(),
        teamB=raw["teamB"].tolist(),
        outcomes=raw["teamA_win"].astype(int).tolist(),
        k=32.0, base=1500.0
    )
    raw["eloA_pre"] = eloA_pre
    raw["eloB_pre"] = eloB_pre
    raw["elo_diff"] = raw["eloA_pre"] - raw["eloB_pre"]

    # Map winrate priors (no leakage)
    raw = rolling_map_winrate(raw)
    raw["map_wr_diff"] = raw["teamA_map_wr_pre"] - raw["teamB_map_wr_pre"]

    # Agent role counts & diffs
    raw = build_role_features(raw)

    # Map one-hot
    map_oh = map_one_hot(raw["map"])
    feat = pd.concat([raw, map_oh], axis=1)

    # Final feature selection
    role_diff_cols = [f"diff_cnt_{r}" for r in ROLES]
    map_cols = map_oh.columns.tolist()
    features = ["elo_diff","map_wr_diff"] + role_diff_cols + map_cols

    out = feat[["match_id","date","teamA_win"] + features].copy()
    out.rename(columns={"teamA_win": "y"}, inplace=True)
    out.to_csv(args.out, index=False)
    print(f"Saved features to {args.out}\nColumns: {list(out.columns)}")

if __name__ == "__main__":
    main()
