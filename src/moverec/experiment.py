"""Shared experiment setup so every script evaluates on identical data and carts."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .als import ImplicitALS
from .baselines import CoOccurrence, GlobalPopularity, ProfilePopularity
from .data import REPO_ROOT, Dataset, basket_matrix, load_dataset, load_experiment_config
from .evaluate import EvalCarts, by_cart_size, evaluate, make_partial_carts, summarize
from .report import _table
from .split import Split, time_split

PARAMS_DIR = REPO_ROOT / "reports" / "params"


@dataclass
class Context:
    cfg: dict
    ds: Dataset
    split: Split
    X_train: np.ndarray
    carts: dict[str, EvalCarts]   # "val" always; "test" only if unsealed
    free: list[str]

    @property
    def key(self) -> str:
        return f"map@{self.cfg['eval']['map_k']}"

    def run(self, model, part: str = "val", exclude=None) -> pd.DataFrame:
        ev = self.cfg["eval"]
        return evaluate(model, self.carts[part], self.ds.services,
                        ks=ev["ks"], map_k=ev["map_k"], exclude=exclude)

    def score(self, model, part: str = "val") -> float:
        return float(summarize(self.run(model, part))[self.key])


def prepare(include_test: bool = False) -> Context:
    cfg = load_experiment_config()
    ds = load_dataset()
    split = time_split(ds.bookings, **cfg["split"])
    sids = ds.service_ids
    seed = cfg["eval"]["seed"]
    names = ["val", "test"] if include_test else ["val"]
    carts = {}
    for offset, name in enumerate(["val", "test"]):   # fixed seeds per part
        if name in names:
            part = split.parts()[name]
            carts[name] = make_partial_carts(part, basket_matrix(part, ds.items, sids), seed + offset)
    free = ds.services.loc[ds.services["is_free"], "service_id"].tolist()
    return Context(cfg, ds, split, basket_matrix(split.train, ds.items, sids), carts, free)


def save_params(name: str, params: dict) -> None:
    """Persist hyperparameters chosen on validation so later steps reuse them."""
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    (PARAMS_DIR / f"{name}.json").write_text(json.dumps(params, indent=2) + "\n")


def load_params(name: str) -> dict:
    path = PARAMS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/run_{name}.py first")
    return json.loads(path.read_text())


def tuned_baselines(ctx: Context) -> tuple[list, dict]:
    """The three step-2 baselines, with profile smoothing picked on validation."""
    grid = ctx.cfg["baselines"]["profile_smoothing_grid"]
    tuning = {m: ctx.score(ProfilePopularity(m).fit(ctx.split.train, ctx.X_train)) for m in grid}
    best = max(tuning, key=tuning.get)
    save_params("baselines", {"profile_smoothing": best})
    models = [GlobalPopularity(), ProfilePopularity(best), CoOccurrence()]
    for m in models:
        m.fit(ctx.split.train, ctx.X_train)
    return models, tuning


def tuned_als(ctx: Context) -> ImplicitALS:
    """ALS with the configuration chosen in step 3 (reports/params/als.json)."""
    return ImplicitALS(**load_params("als")).fit(ctx.split.train, ctx.X_train)


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    return df.map(lambda v: f"{v:.3f}" if isinstance(v, float) else v)


def comparison_tables(ctx: Context, models: list, part: str) -> list[str]:
    """Markdown: overall metrics (with / without $0 items) and MAP by cart size."""
    out = []
    for label, exclude in (("all services", None),
                           ("excluding $0 items " + ", ".join(ctx.free), ctx.free)):
        rows = {m.name: summarize(ctx.run(m, part, exclude)) for m in models}
        out += [f"## {part.capitalize()}: {label}\n",
                _table(fmt(pd.DataFrame(rows).T.reset_index(names="model"))), ""]
    out.append(f"### {part.capitalize()} by visible cart size ({ctx.key}, all services)\n")
    sliced = {m.name: by_cart_size(ctx.run(m, part)) for m in models}
    tbl = pd.DataFrame({name: s[ctx.key] for name, s in sliced.items()})
    tbl.insert(0, "carts", next(iter(sliced.values()))["carts"])
    out += [_table(fmt(tbl.reset_index(names="cart size"))), ""]
    return out
