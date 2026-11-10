"""Load the synthetic dataset for modelling.

Only the public tables are loaded. The generator's hidden folder is never
touched here (tests/test_no_leakage.py enforces this across src/).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "synthetic"
EXPERIMENT_CONFIG = REPO_ROOT / "config" / "experiment.yaml"


@dataclass
class Dataset:
    bookings: pd.DataFrame
    items: pd.DataFrame
    services: pd.DataFrame

    @property
    def service_ids(self) -> list[str]:
        return self.services["service_id"].tolist()


def load_dataset(data_dir: str | Path = DEFAULT_DATA_DIR) -> Dataset:
    d = Path(data_dir)
    bookings = pd.read_csv(d / "bookings.csv", parse_dates=["booking_ts", "move_date"])
    items = pd.read_csv(d / "booking_items.csv")
    services = pd.read_csv(d / "services.csv")
    return Dataset(bookings, items, services)


def load_experiment_config(path: str | Path = EXPERIMENT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def basket_matrix(bookings: pd.DataFrame, items: pd.DataFrame, service_ids: list[str]) -> np.ndarray:
    """Boolean (n_bookings, n_services) matrix, rows in the order of `bookings`."""
    row = pd.Index(bookings["booking_id"])
    col = pd.Index(service_ids)
    it = items[items["booking_id"].isin(row)]
    X = np.zeros((len(row), len(col)), dtype=bool)
    X[row.get_indexer(it["booking_id"]), col.get_indexer(it["service_id"])] = True
    return X
