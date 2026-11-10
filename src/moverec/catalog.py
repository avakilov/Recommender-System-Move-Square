"""The service catalog: the recommendation targets."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .assumptions import Assumptions

TRUCK_GROUP = "truck"
STORAGE_GROUP = "storage"


@dataclass(frozen=True)
class Service:
    id: str
    group: str
    is_free: bool


class Catalog:
    def __init__(self, services: list[Service]):
        ids = [s.id for s in services]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate service ids in catalog")
        self.services = services
        self.ids = ids
        self.index = {sid: i for i, sid in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.services)

    def in_group(self, group: str) -> list[str]:
        return [s.id for s in self.services if s.group == group]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"service_id": s.id, "group": s.group, "is_free": s.is_free} for s in self.services]
        )


def load_catalog(assumptions: Assumptions) -> Catalog:
    free = set(assumptions["free_services"])
    services = [
        Service(id=sid, group=g["group"], is_free=sid in free)
        for g in assumptions["catalog"]
        for sid in g["services"]
    ]
    catalog = Catalog(services)
    unknown_free = free - set(catalog.ids)
    if unknown_free:
        raise ValueError(f"free_services not in catalog: {sorted(unknown_free)}")
    return catalog
