"""机器遗忘核心引擎（SISA 风格：分片 Shards + 有序切片 Slices + 聚合预测）。

- 60 条样本固定分为 3 片，每片 4 个有序切片，每切片 5 条。
- 每片一个逻辑回归（4 维特征 + 偏置），零初始化；按固定次序逐样本训练一遍，
  SGD 学习率 0.1、动量 0.9；动量缓冲与权重一起写入每个切片后的检查点，
  因此“从最早受影响切片的前一切片续算”与“剔除撤回样本后从头训练”逐位一致。
- 预测 = 三片 sigmoid 概率的均值。
- 撤回按贡献者永久生效；只重训受影响片，未受影响片不训练。
- 状态机：READY / STOPPED / RETRAINING / FAILED。
  * 最新撤回全部消除前停止预测；失败或重启仍停供、可重试。
  * 轮次结果（models 表 + 代际 + 已处理撤回指针 + 状态）在单个事务内提交，
    预测端校验所有分片代际一致，杜绝拼用新旧分片恢复预测。
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

import numpy as np

from .seed import seed_samples

# ---------------- 配置 ----------------
_DEFAULT_DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "unlearn.db"))
DB_PATH = os.path.abspath(os.environ.get("DB_PATH", _DEFAULT_DB))

SHARDS = 3               # 片数
SLICES_PER_SHARD = 4     # 每片切片数
SLICE_SIZE = 5           # 每切片样本数
FEATURE_DIM = 4
LR = 0.1                 # SGD 学习率
MOMENTUM = 0.9           # 动量
TOLERANCE = 1e-10        # float64 参数误差阈值

# 每个切片重训后的人为延时（秒），便于在网页上观察进度并在慢重训期间提交新撤回
SLICE_DELAY = float(os.environ.get("RETRAIN_SLICE_DELAY", "0.6"))
# 测试钩子："round_id:shard:slice" 命中时在该切片注入一次失败
INJECT_FAILURE_AT = os.environ.get("INJECT_FAILURE_AT", "")

STATE_READY = "READY"
STATE_STOPPED = "STOPPED"
STATE_RETRAINING = "RETRAINING"
STATE_FAILED = "FAILED"

STATE_LABELS = {
    STATE_READY: "就绪（可预测）",
    STATE_STOPPED: "已停供（存在待消除的撤回）",
    STATE_RETRAINING: "重训中（停供）",
    STATE_FAILED: "重训失败（停供，可重试）",
}

LOCK = threading.RLock()   # 所有 DB 访问与状态迁移的统一锁
WAKE = threading.Event()   # 唤醒后台重训线程
_CONN: sqlite3.Connection | None = None

PROGRESS: dict = {
    "round_id": None, "shard": None, "slice": None,
    "done_units": 0, "total_units": 0, "message": "",
}


class ModelUnavailable(Exception):
    """模型处于停供状态（有待消除的撤回 / 重训中 / 失败 / 代际不一致）。"""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------------- DB 基础 ----------------
def db() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _CONN = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
        _CONN.row_factory = sqlite3.Row
        _CONN.execute("PRAGMA journal_mode=WAL")
        _CONN.execute("PRAGMA busy_timeout=30000")
    return _CONN


def init_schema() -> None:
    db().executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS samples(
            id INTEGER PRIMARY KEY,
            contributor TEXT NOT NULL,
            x1 REAL NOT NULL, x2 REAL NOT NULL, x3 REAL NOT NULL, x4 REAL NOT NULL,
            label INTEGER NOT NULL,
            shard INTEGER NOT NULL,
            slice INTEGER NOT NULL,
            ord INTEGER NOT NULL,
            withdrawn INTEGER NOT NULL DEFAULT 0,
            withdrawal_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS withdrawals(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contributor TEXT NOT NULL,
            sample_count INTEGER NOT NULL,
            requested_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS checkpoints(
            shard INTEGER NOT NULL,
            slice INTEGER NOT NULL,
            generation INTEGER NOT NULL,
            w BLOB NOT NULL,
            b REAL NOT NULL,
            vw BLOB NOT NULL,
            vb REAL NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(shard, slice, generation)
        );
        CREATE TABLE IF NOT EXISTS models(
            shard INTEGER PRIMARY KEY,
            generation INTEGER NOT NULL,
            w BLOB NOT NULL,
            b REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS rounds(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,              -- running / done / failed
            withdrawals_json TEXT NOT NULL,    -- 本轮消除的撤回贡献者
            affected_json TEXT NOT NULL,       -- 实际重训切片 {"1":[3,4]}
            max_error REAL,                    -- 与从头训练参照的最大参数误差
            note TEXT
        );
        """
    )
    db().commit()


def meta_get(key: str, default: str | None = None) -> str | None:
    row = db().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def meta_set(key: str, value) -> None:
    db().execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def get_state() -> str:
    return meta_get("state", STATE_STOPPED)


def get_generation() -> int:
    return int(meta_get("generation", "0"))


def get_processed_withdrawal_id() -> int:
    return int(meta_get("processed_withdrawal_id", "0"))


def unprocessed_withdrawals() -> list[dict]:
    rows = db().execute(
        "SELECT * FROM withdrawals WHERE id > ? ORDER BY id",
        (get_processed_withdrawal_id(),),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------- 训练核心（全 float64，固定次序，逐样本单遍） ----------------
def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def sgd_step(w, b, vw, vb, x: np.ndarray, y: int):
    """一步带动量的 SGD（BCE 损失）。w/vw 为 float64 向量，b/vb 为标量。"""
    z = float(np.dot(w, x)) + b
    d = _sigmoid(z) - float(y)
    vw = MOMENTUM * vw - LR * (d * x)
    vb = MOMENTUM * vb - LR * d
    return w + vw, b + vb, vw, vb


def _zero_state():
    return np.zeros(FEATURE_DIM, dtype=np.float64), 0.0, np.zeros(FEATURE_DIM, dtype=np.float64), 0.0


def _features(s: dict) -> np.ndarray:
    return np.array([s["x1"], s["x2"], s["x3"], s["x4"]], dtype=np.float64)


def _slice_samples(samples: list[dict], shard: int, slice_idx: int) -> list[dict]:
    return sorted(
        (s for s in samples if s["shard"] == shard and s["slice"] == slice_idx),
        key=lambda r: r["ord"],
    )


def save_checkpoint(shard: int, slice_idx: int, generation: int, w, b, vw, vb) -> None:
    with LOCK:
        db().execute(
            "INSERT OR REPLACE INTO checkpoints(shard, slice, generation, w, b, vw, vb, created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                shard, slice_idx, generation,
                np.asarray(w, dtype=np.float64).tobytes(), float(b),
                np.asarray(vw, dtype=np.float64).tobytes(), float(vb),
                now_iso(),
            ),
        )
        db().commit()


def load_latest_checkpoint(shard: int, slice_idx: int):
    """加载某片在指定切片之后的最新检查点（含动量缓冲）；slice 0 表示零初始化。"""
    if slice_idx <= 0:
        return _zero_state()
    with LOCK:
        row = db().execute(
            "SELECT w, b, vw, vb FROM checkpoints WHERE shard=? AND slice=?"
            " ORDER BY generation DESC LIMIT 1",
            (shard, slice_idx),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"缺少检查点: shard={shard} slice={slice_idx}")
    return (
        np.frombuffer(row["w"], dtype=np.float64).copy(),
        float(row["b"]),
        np.frombuffer(row["vw"], dtype=np.float64).copy(),
        float(row["vb"]),
    )


def _maybe_inject_failure(round_id, shard: int, slice_idx: int) -> None:
    if INJECT_FAILURE_AT and INJECT_FAILURE_AT == f"{round_id}:{shard}:{slice_idx}":
        raise RuntimeError(f"注入故障: round={round_id} shard={shard} slice={slice_idx}")


def train_shard_from_slice(samples: list[dict], shard: int, start_slice: int,
                           generation: int, round_id=None,
                           save: bool = True, delay: float = 0.0,
                           progress: bool = False):
    """从 start_slice-1 的检查点续算 start_slice..4 号切片，返回最终 (w, b)。"""
    w, b, vw, vb = load_latest_checkpoint(shard, start_slice - 1)
    for j in range(start_slice, SLICES_PER_SHARD + 1):
        _maybe_inject_failure(round_id, shard, j)
        for s in _slice_samples(samples, shard, j):
            if s["withdrawn"]:
                continue
            w, b, vw, vb = sgd_step(w, b, vw, vb, _features(s), s["label"])
        if save:
            save_checkpoint(shard, j, generation, w, b, vw, vb)
        if progress:
            with LOCK:
                PROGRESS["shard"] = shard
                PROGRESS["slice"] = j
                PROGRESS["done_units"] += 1
                PROGRESS["message"] = f"第 {shard} 片 · 切片 {j}/{SLICES_PER_SHARD} 已完成"
        if delay > 0:
            time.sleep(delay)
    return w, b


def reference_params(samples: list[dict], shard: int):
    """参照：剔除全部已撤回样本后，该片从零初始化从头训练一遍（不落盘）。"""
    w, b, vw, vb = _zero_state()
    for j in range(1, SLICES_PER_SHARD + 1):
        for s in _slice_samples(samples, shard, j):
            if s["withdrawn"]:
                continue
            w, b, vw, vb = sgd_step(w, b, vw, vb, _features(s), s["label"])
    return w, b


def verify_against_reference(final_params: dict, samples: list[dict]):
    """对照误差：实际参数 vs 从头训练参照，取所有片、w 与 b 的最大绝对差。"""
    per_shard = {}
    for shard in range(1, SHARDS + 1):
        rw, rb = reference_params(samples, shard)
        w, b = final_params[shard]
        err = max(float(np.max(np.abs(rw - w))), abs(rb - b))
        per_shard[shard] = err
    return max(per_shard.values()), per_shard


# ---------------- 初始训练 ----------------
def initial_train() -> None:
    samples = [dict(r) for r in db().execute("SELECT * FROM samples ORDER BY id").fetchall()]
    generation = 1
    for shard in range(1, SHARDS + 1):
        w, b = train_shard_from_slice(samples, shard, 1, generation, save=True, delay=0.0)
        db().execute(
            "INSERT INTO models(shard, generation, w, b) VALUES(?,?,?,?) "
            "ON CONFLICT(shard) DO UPDATE SET generation=excluded.generation, w=excluded.w, b=excluded.b",
            (shard, generation, np.asarray(w, dtype=np.float64).tobytes(), float(b)),
        )
    meta_set("state", STATE_READY)
    meta_set("generation", generation)
    meta_set("processed_withdrawal_id", 0)
    meta_set("state_note", "")
    db().commit()


# ---------------- 遗忘：撤回与重训 ----------------
def withdraw_contributor(contributor: str) -> dict | None:
    """永久撤回某贡献者的全部活跃样本。返回撤回信息；已撤完返回 None；不存在抛 KeyError。"""
    with LOCK:
        rows = db().execute(
            "SELECT id FROM samples WHERE contributor=? AND withdrawn=0", (contributor,)
        ).fetchall()
        if not rows:
            exists = db().execute(
                "SELECT 1 FROM samples WHERE contributor=? LIMIT 1", (contributor,)
            ).fetchone()
            if not exists:
                raise KeyError(contributor)
            return None
        cur = db().execute(
            "INSERT INTO withdrawals(contributor, sample_count, requested_at) VALUES(?,?,?)",
            (contributor, len(rows), now_iso()),
        )
        wid = cur.lastrowid
        db().execute(
            "UPDATE samples SET withdrawn=1, withdrawal_id=? WHERE contributor=? AND withdrawn=0",
            (wid, contributor),
        )
        # 与轮次提交同一把锁：要么先停供再提交撤回，要么轮次先完成——不会漏停供
        if get_state() == STATE_READY:
            meta_set("state", STATE_STOPPED)
        meta_set("state_note", f"已受理撤回 #{wid}（{contributor}，{len(rows)} 条），等待重训")
        db().commit()
    WAKE.set()
    return {"withdrawal_id": wid, "samples": len(rows)}


def run_round(pending: list[dict]) -> None:
    """执行一轮遗忘重训。失败时置 FAILED 并抛异常（由调用方捕获）。"""
    names = [w["contributor"] for w in pending]
    with LOCK:
        round_id = db().execute(
            "INSERT INTO rounds(started_at, status, withdrawals_json, affected_json) VALUES(?,?,?,?)",
            (now_iso(), "running", json.dumps(names, ensure_ascii=False), "{}"),
        ).lastrowid
        meta_set("state", STATE_RETRAINING)
        meta_set("state_note", f"第 {round_id} 轮重训中：{', '.join(names)}")
        db().commit()
        # 快照：本轮全程使用同一撤回集合；轮中新撤回留给下一轮
        samples = [dict(r) for r in db().execute("SELECT * FROM samples ORDER BY id").fetchall()]
        generation = get_generation() + 1

    # 本轮新撤回样本（按撤回单号归属）；训练时跳过的是全部已撤回样本
    pending_ids = {w["id"] for w in pending}
    newly_withdrawn = {s["id"] for s in samples if s["withdrawal_id"] in pending_ids}
    final_params: dict[int, tuple] = {}
    affected: dict[int, list[int]] = {}

    # 每片的最早受影响切片 k 按“本轮新撤回”计算：k-1 处最新检查点可证明
    # 对全部已撤回样本干净（任何含后来撤回样本的切片都会被更晚的轮次重写），
    # 因此续算结果与剔除全部撤回样本后从头训练逐位一致；未受影响片完全不训练。
    plan = []  # (shard, start_slice)
    for shard in range(1, SHARDS + 1):
        hit = sorted({s["slice"] for s in samples if s["shard"] == shard and s["id"] in newly_withdrawn})
        if hit:
            plan.append((shard, min(hit)))
    with LOCK:
        PROGRESS.update({
            "round_id": round_id, "shard": None, "slice": None,
            "done_units": 0,
            "total_units": sum(SLICES_PER_SHARD - k + 1 for _, k in plan),
            "message": f"第 {round_id} 轮开始，受影响片：{[s for s, _ in plan] or '无'}",
        })

    try:
        for shard in range(1, SHARDS + 1):
            k = next((k for s, k in plan if s == shard), None)
            if k is None:
                # 未受影响片：不训练，沿用现有参数
                with LOCK:
                    row = db().execute("SELECT w, b FROM models WHERE shard=?", (shard,)).fetchone()
                final_params[shard] = (np.frombuffer(row["w"], dtype=np.float64).copy(), float(row["b"]))
                continue
            w, b = train_shard_from_slice(
                samples, shard, k, generation, round_id=round_id,
                save=True, delay=SLICE_DELAY, progress=True,
            )
            final_params[shard] = (w, b)
            affected[shard] = list(range(k, SLICES_PER_SHARD + 1))

        max_err, _per = verify_against_reference(final_params, samples)

        # 原子提交：模型 + 代际 + 已处理撤回指针 + 轮次记录 + 状态，同事务落库
        with LOCK:
            for shard in range(1, SHARDS + 1):
                w, b = final_params[shard]
                db().execute(
                    "INSERT INTO models(shard, generation, w, b) VALUES(?,?,?,?) "
                    "ON CONFLICT(shard) DO UPDATE SET generation=excluded.generation, w=excluded.w, b=excluded.b",
                    (shard, generation, np.asarray(w, dtype=np.float64).tobytes(), float(b)),
                )
            meta_set("generation", generation)
            meta_set("processed_withdrawal_id", max(w["id"] for w in pending))
            more = unprocessed_withdrawals()
            if not more:
                meta_set("state", STATE_READY)
                meta_set("state_note", "")
            db().execute(
                "UPDATE rounds SET finished_at=?, status='done', affected_json=?, max_error=? WHERE id=?",
                (now_iso(), json.dumps(affected), max_err, round_id),
            )
            db().commit()
            PROGRESS["message"] = (
                f"第 {round_id} 轮完成，对照误差 {max_err:.3e}"
                + ("" if not more else f"；另有 {len(more)} 条撤回待处理，继续下一轮")
            )
    except Exception as exc:
        with LOCK:
            meta_set("state", STATE_FAILED)
            meta_set("state_note", f"第 {round_id} 轮失败：{exc}")
            db().execute(
                "UPDATE rounds SET finished_at=?, status='failed', affected_json=?, note=? WHERE id=?",
                (now_iso(), json.dumps(affected), str(exc), round_id),
            )
            db().commit()
            PROGRESS["message"] = f"第 {round_id} 轮失败：{exc}"
        raise


def worker_loop() -> None:
    """后台重训线程：被撤回/重试唤醒，逐轮处理直到没有未消除的撤回。"""
    while True:
        WAKE.wait()
        WAKE.clear()
        while True:
            with LOCK:
                pending = unprocessed_withdrawals()
            if not pending:
                break
            try:
                run_round(pending)
            except Exception:
                break  # run_round 已置 FAILED，等待人工重试


def retry() -> dict:
    with LOCK:
        state = get_state()
        if state == STATE_READY:
            return {"ok": False, "message": "模型已就绪，无需重试"}
        if state == STATE_RETRAINING:
            return {"ok": False, "message": "重训正在进行中"}
        meta_set("state_note", "已手动触发重试")
        db().commit()
    WAKE.set()
    return {"ok": True, "message": "已触发重训"}


# ---------------- 预测 ----------------
def predict(features: list[float]):
    with LOCK:
        state = get_state()
        if state != STATE_READY:
            note = meta_get("state_note", "")
            raise ModelUnavailable(f"模型停供中（{STATE_LABELS.get(state, state)}）。{note or ''}".strip())
        generation = get_generation()
        rows = db().execute("SELECT shard, generation, w, b FROM models ORDER BY shard").fetchall()
        if len(rows) != SHARDS or any(r["generation"] != generation for r in rows):
            # 兜底防线：任何新旧分片混用都拒绝服务
            raise ModelUnavailable("分片代际不一致，拒绝拼用新旧分片恢复预测")
        x = np.array(features, dtype=np.float64)
        probs = []
        for r in rows:
            w = np.frombuffer(r["w"], dtype=np.float64)
            probs.append(_sigmoid(float(np.dot(w, x)) + float(r["b"])))
    p = sum(probs) / len(probs)
    return p, probs


# ---------------- 状态查询 ----------------
def get_status() -> dict:
    with LOCK:
        state = get_state()
        generation = get_generation()
        samples = [dict(r) for r in db().execute("SELECT * FROM samples ORDER BY id").fetchall()]
        contributors = [
            {
                "name": name,
                "total": sum(1 for s in samples if s["contributor"] == name),
                "withdrawn": sum(1 for s in samples if s["contributor"] == name and s["withdrawn"]),
            }
            for name in sorted({s["contributor"] for s in samples})
        ]
        withdrawals = [dict(r) for r in db().execute("SELECT * FROM withdrawals ORDER BY id").fetchall()]
        rounds = [
            dict(r) for r in db().execute("SELECT * FROM rounds ORDER BY id DESC LIMIT 20").fetchall()
        ]
        for r in rounds:
            r["withdrawals"] = json.loads(r.pop("withdrawals_json") or "[]")
            r["affected"] = json.loads(r.pop("affected_json") or "{}")
        pending = [w for w in withdrawals if w["id"] > get_processed_withdrawal_id()]
        progress = dict(PROGRESS)
        state_note = meta_get("state_note", "")

        verification = None
        if state == STATE_READY:
            final = {}
            for r in db().execute("SELECT shard, w, b FROM models").fetchall():
                final[r["shard"]] = (np.frombuffer(r["w"], dtype=np.float64).copy(), float(r["b"]))
            if len(final) == SHARDS:
                max_err, per_shard = verify_against_reference(final, samples)
                verification = {
                    "max_error": max_err,
                    "per_shard": {str(k): v for k, v in per_shard.items()},
                    "tolerance": TOLERANCE,
                    "ok": max_err <= TOLERANCE,
                }

    return {
        "state": state,
        "state_label": STATE_LABELS.get(state, state),
        "state_note": state_note,
        "can_predict": state == STATE_READY,
        "generation": generation,
        "stats": {
            "total_samples": len(samples),
            "active_samples": sum(1 for s in samples if not s["withdrawn"]),
            "withdrawn_samples": sum(1 for s in samples if s["withdrawn"]),
            "total_withdrawals": len(withdrawals),
            "pending_withdrawals": len(pending),
        },
        "contributors": contributors,
        "rounds": rounds,
        "progress": progress,
        "verification": verification,
        "config": {
            "shards": SHARDS, "slices_per_shard": SLICES_PER_SHARD, "slice_size": SLICE_SIZE,
            "lr": LR, "momentum": MOMENTUM, "tolerance": TOLERANCE, "slice_delay": SLICE_DELAY,
        },
        "samples": [
            {
                "id": s["id"], "contributor": s["contributor"], "shard": s["shard"],
                "slice": s["slice"], "ord": s["ord"], "label": s["label"],
                "withdrawn": bool(s["withdrawn"]),
                "x": [s["x1"], s["x2"], s["x3"], s["x4"]],
            }
            for s in samples
        ],
    }


# ---------------- 启动 ----------------
def startup() -> None:
    with LOCK:
        init_schema()
        if meta_get("state") is None:
            n = seed_samples(db())
            initial_train()
            PROGRESS["message"] = f"初始化完成：{n} 条样本，3 片 × 4 切片"
        else:
            state = get_state()
            if state == STATE_RETRAINING:
                # 重启打断：保持停供，标记失败，等待人工重试
                meta_set("state", STATE_FAILED)
                meta_set("state_note", "服务重启导致重训中断，保持停供，可点击重试")
                db().execute(
                    "UPDATE rounds SET status='failed', finished_at=?,"
                    " note=COALESCE(note,'') || ' [重启中断]' WHERE status='running'",
                    (now_iso(),),
                )
                db().commit()
    worker = threading.Thread(target=worker_loop, name="unlearn-worker", daemon=True)
    worker.start()
