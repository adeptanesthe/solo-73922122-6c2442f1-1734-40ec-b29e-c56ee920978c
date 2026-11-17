"""确定性生成 60 条本地样本：10 位贡献者，每人 6 条，四维特征，指令(1)/噪声(0) 标签。

分片布局（样本 id 1..60 连续）：
  片1 = id 1..20，片2 = id 21..40，片3 = id 41..60
  每片 4 个有序切片，每切片 5 条（切片内次序 ord 1..5）
贡献者按 id 连续分块（每 6 条一位），因此多数撤回只波及 1~2 个片，
便于观察“只重训受影响片、未受影响片不训练”。
"""
from __future__ import annotations

import numpy as np

CONTRIBUTORS = [
    "alice", "bob", "carol", "dave", "erin",
    "frank", "grace", "heidi", "ivan", "judy",
]

SEED = 20261116


def generate_samples() -> list[dict]:
    rng = np.random.default_rng(SEED)
    mu_pos = np.array([1.2, -0.8, 0.6, 1.0])    # 指令类中心
    mu_neg = np.array([-1.0, 0.9, -0.5, -0.8])  # 噪声类中心
    rows: list[dict] = []
    for i in range(60):
        label = i % 2  # 30/30 均衡
        mu = mu_pos if label == 1 else mu_neg
        x = rng.normal(mu, 1.1, size=4)
        rows.append(
            {
                "id": i + 1,
                "contributor": CONTRIBUTORS[i // 6],
                "x": [round(float(v), 6) for v in x],
                "label": label,
                "shard": i // 20 + 1,
                "slice": (i % 20) // 5 + 1,
                "ord": i % 5 + 1,
            }
        )
    return rows


def seed_samples(conn) -> int:
    rows = generate_samples()
    conn.executemany(
        "INSERT INTO samples(id, contributor, x1, x2, x3, x4, label, shard, slice, ord, withdrawn)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,0)",
        [
            (
                r["id"], r["contributor"], r["x"][0], r["x"][1], r["x"][2], r["x"][3],
                r["label"], r["shard"], r["slice"], r["ord"],
            )
            for r in rows
        ],
    )
    return len(rows)
