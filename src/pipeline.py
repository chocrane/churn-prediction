"""ETL pipeline for user-behaviour data."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    log.info("loaded %d rows", len(df))
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["user_id", "event_type", "timestamp"])
    df = df[df["session_duration"].between(1, 86_400)]
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("user_id")
        .agg(sessions=("session_duration","count"),
             avg_dur=("session_duration","mean"),
             events=("event_type","nunique"),
             last_ts=("timestamp","max"))
        .reset_index()
    )
    agg["recency"] = (pd.Timestamp.now() - agg["last_ts"]).dt.days
    return agg.drop(columns=["last_ts"])

def scale(df: pd.DataFrame, cols: list[str]):
    sc = StandardScaler()
    return sc.fit_transform(df[cols].values), sc
