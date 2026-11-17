"""端到端自测：启动真实服务进程，验证机器遗忘关键流程。

场景：
  basic   —— 初始就绪、预测、撤回触发增量重训、对照误差≤1e-10、未受影响片不训练、
              慢重训期间受理新撤回且旧轮不恢复其影响、停供期间预测被拒
  restart —— 重训中途 SIGKILL，重启后保持停供(FAILED)，重试后恢复且误差≤1e-10
  failure —— 注入故障使轮次失败，停供可重试，最终与从头训练一致

用法： python3 tests/selftest.py [basic|restart|failure|all]
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time

import httpx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, port: int, dbdir: str, delay: str = "0.5", inject: str = ""):
        self.port, self.dbdir = port, dbdir
        env = dict(os.environ)
        env.update({
            "DB_PATH": os.path.join(dbdir, "unlearn.db"),
            "RETRAIN_SLICE_DELAY": delay,
            "INJECT_FAILURE_AT": inject,
            "UNLEARN_TEST_SHUTDOWN": "1",
            "PYTHONUNBUFFERED": "1",
        })
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
            cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        self.base = f"http://127.0.0.1:{port}"

    def wait_ready(self, timeout: float = 40.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                r = httpx.get(f"{self.base}/api/status", timeout=2)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.3)
        return False

    def _shutdown(self, code: int) -> None:
        """通过测试端点让进程立即退出（os._exit，等价崩溃/停止）。"""
        try:
            httpx.post(f"{self.base}/api/_shutdown", params={"code": code}, timeout=2)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def kill(self) -> None:
        # 模拟崩溃：进程立即以非零码退出，不做任何清理
        self._shutdown(1)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self._shutdown(0)

    def logs(self) -> str:
        out = ""
        if self.proc.stdout:
            try:
                out = self.proc.stdout.read() or ""
            except Exception:
                pass
        return out


def status(s: Server) -> dict:
    return httpx.get(f"{s.base}/api/status", timeout=5).json()


def wait_state(s: Server, want: str, timeout: float = 90.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = status(s)
        if last["state"] == want:
            return last
        time.sleep(0.2)
    raise AssertionError(f"等待状态 {want} 超时，当前 {last and last['state']}")


def predict(s: Server):
    return httpx.post(f"{s.base}/api/predict", json={"features": [1.0, -0.5, 0.2, 0.8]}, timeout=5)


def withdraw(s: Server, name: str) -> httpx.Response:
    return httpx.post(f"{s.base}/api/withdraw", json={"contributor": name}, timeout=5)


def scenario_basic() -> None:
    print("\n=== 场景 basic：主流程 ===")
    dbdir = tempfile.mkdtemp(prefix="unlearn-basic-")
    srv = Server(free_port(), dbdir, delay="0.5")
    try:
        check("服务启动", srv.wait_ready())
        st = wait_state(srv, "READY")
        check("初始 READY 且代际=1", st["generation"] == 1, f"gen={st['generation']}")
        check("60 条样本全活跃", st["stats"]["active_samples"] == 60)
        check("10 位贡献者", len(st["contributors"]) == 10)
        check("初始校验误差≤1e-10", st["verification"]["ok"], f"err={st['verification']['max_error']}")

        r = predict(srv)
        check("就绪时可预测", r.status_code == 200 and 0.0 <= r.json()["probability"] <= 1.0,
              f"p={r.json()['probability']:.4f} 分片={['%.4f' % x for x in r.json()['shard_probabilities']]}")

        # 撤回 carol（id 13-18 → 仅片1 切片3,4，k=3）
        r = withdraw(srv, "carol")
        check("撤回 carol 受理", r.status_code == 200 and r.json()["samples"] == 6)
        r = predict(srv)
        check("撤回后立即停供（409）", r.status_code == 409)
        st = wait_state(srv, "READY")
        check("重训后恢复 READY", True)
        check("活跃样本 54", st["stats"]["active_samples"] == 54)
        rd = st["rounds"][0]
        check("第1轮只重训片1切片3-4", rd["affected"] == {"1": [3, 4]}, str(rd["affected"]))
        check("第1轮对照误差≤1e-10", rd["max_error"] <= 1e-10, f"err={rd['max_error']}")
        check("轮次校验与实时校验一致", st["verification"]["ok"])

        # 慢重训期间提交新撤回：dave（片1切片4 + 片2切片1..）进行中再撤 erin（片2）
        r = withdraw(srv, "dave")
        check("撤回 dave 受理", r.status_code == 200)
        wait_state(srv, "RETRAINING", timeout=10)
        r = predict(srv)
        check("重训中预测被拒（409）", r.status_code == 409)
        r = withdraw(srv, "erin")
        check("慢重训期间新撤回 erin 仍受理", r.status_code == 200)
        st = wait_state(srv, "READY")
        check("两轮后恢复 READY", True)
        check("活跃样本 42", st["stats"]["active_samples"] == 42)
        rounds = {r["id"]: r for r in st["rounds"]}
        check("共 3 轮", len(rounds) == 3, f"rounds={sorted(rounds)}")
        check("第2轮(dave)：片1切片4、片2切片1-4",
              rounds[2]["affected"] == {"1": [4], "2": [1, 2, 3, 4]}, str(rounds[2]["affected"]))
        check("第3轮(erin)：仅片2切片1-4",
              rounds[3]["affected"] == {"2": [1, 2, 3, 4]}, str(rounds[3]["affected"]))
        check("第3轮未触碰片1/片3", "1" not in rounds[3]["affected"] and "3" not in rounds[3]["affected"])
        check("第2轮误差≤1e-10", rounds[2]["max_error"] <= 1e-10, f"{rounds[2]['max_error']}")
        check("第3轮误差≤1e-10", rounds[3]["max_error"] <= 1e-10, f"{rounds[3]['max_error']}")
        check("最终实时校验≤1e-10", st["verification"]["ok"], f"{st['verification']['max_error']}")
        check("旧轮未抢先生效：第2轮只含 dave", rounds[2]["withdrawals"] == ["dave"])
        check("新撤回由后续轮处理：第3轮含 erin", rounds[3]["withdrawals"] == ["erin"])
        r = predict(srv)
        check("恢复后可预测", r.status_code == 200)

        # 边界：未知贡献者 / 重复撤回
        check("未知贡献者 404", withdraw(srv, "nobody").status_code == 404)
        wait_state(srv, "READY")
        check("重复撤回 400", withdraw(srv, "carol").status_code == 400)
    finally:
        srv.stop()


def scenario_restart() -> None:
    print("\n=== 场景 restart：重训中途强杀，重启后停供可重试 ===")
    dbdir = tempfile.mkdtemp(prefix="unlearn-restart-")
    port = free_port()
    srv = Server(port, dbdir, delay="0.8")
    try:
        check("服务启动", srv.wait_ready())
        wait_state(srv, "READY")
        withdraw(srv, "alice")  # 片1 切片1,2 → k=1，整片重训约 3.2s
        wait_state(srv, "RETRAINING", timeout=10)
        time.sleep(1.0)  # 确保处于重训中段
        srv.kill()
        check("重训中途进程崩溃（os._exit(1)）", srv.proc.returncode == 1)

        srv2 = Server(port, dbdir, delay="0.5")
        try:
            check("重启后服务启动", srv2.wait_ready())
            st = status(srv2)
            check("重启后保持停供（FAILED）", st["state"] == "FAILED", st["state"])
            r = httpx.post(f"{srv2.base}/api/predict",
                           json={"features": [1, 0, 0, 0]}, timeout=5)
            check("重启后预测仍被拒（409）", r.status_code == 409)
            r = httpx.post(f"{srv2.base}/api/retry", timeout=5)
            check("重试受理", r.status_code == 200 and r.json()["ok"])
            st = wait_state(srv2, "READY")
            check("重试后恢复 READY", True)
            check("活跃样本 54", st["stats"]["active_samples"] == 54)
            check("重试后轮次误差≤1e-10", st["rounds"][0]["max_error"] <= 1e-10,
                  f"err={st['rounds'][0]['max_error']}")
            check("重启恢复后实时校验≤1e-10", st["verification"]["ok"])
            check("中断轮次已标记失败", any(r["status"] == "failed" for r in st["rounds"]))
        finally:
            srv2.stop()
    finally:
        srv.stop()


def scenario_failure() -> None:
    print("\n=== 场景 failure：注入故障 → 停供 → 重试成功 ===")
    dbdir = tempfile.mkdtemp(prefix="unlearn-failure-")
    port = free_port()
    # bob: id 7-12 → 片1 切片2,3 → k=2；在 round1 片1 切片2 注入失败
    srv = Server(port, dbdir, delay="0.4", inject="1:1:2")
    try:
        check("服务启动", srv.wait_ready())
        wait_state(srv, "READY")
        withdraw(srv, "bob")
        st = wait_state(srv, "FAILED", timeout=30)
        check("注入故障后进入 FAILED（停供）", True)
        r = predict(srv)
        check("失败后预测被拒（409）", r.status_code == 409)
        check("失败轮次有记录", st["rounds"][0]["status"] == "failed",
              st["rounds"][0].get("note", ""))
    finally:
        srv.stop()

    # 去掉故障注入重启（模拟修复后重试）
    srv2 = Server(port, dbdir, delay="0.4")
    try:
        check("重启后服务启动", srv2.wait_ready())
        st = status(srv2)
        check("重启后仍停供（FAILED 保持）", st["state"] == "FAILED", st["state"])
        httpx.post(f"{srv2.base}/api/retry", timeout=5)
        st = wait_state(srv2, "READY")
        check("重试后恢复 READY", True)
        rd = [r for r in st["rounds"] if r["status"] == "done"][0]
        check("重试轮只重训片1切片2-4", rd["affected"] == {"1": [2, 3, 4]}, str(rd["affected"]))
        check("重试轮误差≤1e-10", rd["max_error"] <= 1e-10, f"{rd['max_error']}")
        check("最终实时校验≤1e-10", st["verification"]["ok"])
    finally:
        srv2.stop()


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    scenarios = {"basic": scenario_basic, "restart": scenario_restart, "failure": scenario_failure}
    for name, fn in scenarios.items():
        if which in ("all", name):
            fn()
    print(f"\n===== 结果：{len(PASS)} 通过，{len(FAIL)} 失败 =====")
    if FAIL:
        print("失败项：", ", ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
