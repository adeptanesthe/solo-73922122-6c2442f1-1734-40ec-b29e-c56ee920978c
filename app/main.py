"""FastAPI 入口：机器遗忘网页与 API。"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import core

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(_: FastAPI):
    core.startup()
    yield


app = FastAPI(title="机器遗忘平台 · 指令/噪声分类器", lifespan=lifespan)


class PredictIn(BaseModel):
    features: list[float] = Field(min_length=4, max_length=4)


class WithdrawIn(BaseModel):
    contributor: str = Field(min_length=1, max_length=64)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/status")
def status():
    return core.get_status()


@app.post("/api/predict")
def predict(inp: PredictIn):
    try:
        p, shard_probs = core.predict(inp.features)
    except core.ModelUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "probability": p,
        "label": int(p >= 0.5),
        "label_name": "指令" if p >= 0.5 else "噪声",
        "shard_probabilities": shard_probs,
    }


@app.post("/api/withdraw")
def withdraw(inp: WithdrawIn):
    contributor = inp.contributor.strip()
    try:
        result = core.withdraw_contributor(contributor)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"贡献者不存在：{contributor}")
    if result is None:
        raise HTTPException(status_code=400, detail=f"贡献者 {contributor} 的样本已全部撤回")
    return {"ok": True, **result}


@app.post("/api/retry")
def retry():
    return core.retry()


# 仅自测用途：设置 UNLEARN_TEST_SHUTDOWN=1 后暴露，立即结束进程（等价崩溃，无清理）
if os.environ.get("UNLEARN_TEST_SHUTDOWN") == "1":
    @app.post("/api/_shutdown")
    def _shutdown(code: int = 0):
        os._exit(code)
