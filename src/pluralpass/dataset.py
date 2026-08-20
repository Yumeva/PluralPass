from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PassGraphDataset(Dataset):
    def __init__(
        self,
        graph_path: str | Path,
        split_path: str | Path,
        split: str,
        max_nodes: int = 22,
        max_samples: int | None = None,
    ):
        membership = pd.read_csv(split_path)
        allowed = set(membership.loc[membership["split"] == split, "event_id"].astype(str))
        self.rows: list[dict[str, Any]] = []
        with gzip.open(graph_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if str(row["event_id"]) in allowed:
                    self.rows.append(row)
                    if max_samples is not None and len(self.rows) >= max_samples:
                        break
        self.max_nodes = max_nodes

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        n = min(len(row["nodes"]), self.max_nodes)
        feature_dim = len(row["nodes"][0])
        nodes = np.zeros((self.max_nodes, feature_dim), dtype=np.float32)
        nodes[:n] = np.asarray(row["nodes"][:n], dtype=np.float32)
        mask = np.zeros(self.max_nodes, dtype=bool)
        mask[:n] = True
        candidate = np.zeros(self.max_nodes, dtype=bool)
        candidate[:n] = np.asarray(row["candidate_mask"][:n], dtype=bool)
        receiver = int(row["receiver_index"])
        if receiver >= self.max_nodes:
            raise IndexError("Receiver index exceeds configured max_nodes")
        return {
            "nodes": torch.from_numpy(nodes),
            "node_mask": torch.from_numpy(mask),
            "candidate_mask": torch.from_numpy(candidate),
            "receiver_index": torch.tensor(receiver, dtype=torch.long),
            "pass_completed": torch.tensor(row["pass_completed"], dtype=torch.float32),
            "value_delta": torch.tensor(row["value_delta_proxy"], dtype=torch.float32),
            "visible_area_fraction": torch.tensor(
                row["visible_area_fraction"], dtype=torch.float32
            ),
            "event_id": row["event_id"],
            "match_id": str(row["match_id"]),
            "domain": row["domain"],
        }
