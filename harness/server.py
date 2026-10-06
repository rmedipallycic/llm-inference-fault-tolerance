"""Start, health-check, and hard-kill an inference server process.

The server is any process exposing an OpenAI-compatible HTTP API
(vLLM, SGLang, or llama.cpp's llama-server for laptop smoke tests).
Killing takes down the whole process tree, because vLLM and SGLang
spawn child processes that would otherwise keep holding GPU memory.
"""
import os
import shlex
import signal
import subprocess
import sys
import time

import requests

IS_WINDOWS = sys.platform.startswith("win")


class ServerProcess:
    def __init__(self, command, base_url, log_path):
        self.command = command
        self.base_url = base_url.rstrip("/")
        self.log_path = log_path
        self.proc = None
        self._log = None

    def start(self):
        self._log = open(self.log_path, "a", encoding="utf-8")
        self._log.write(f"\n=== start {time.time():.6f} cmd: {self.command}\n")
        self._log.flush()
        kwargs = {"stdout": self._log, "stderr": subprocess.STDOUT}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            args = self.command
            kwargs["shell"] = True
        else:
            kwargs["start_new_session"] = True  # own process group for killpg
            args = shlex.split(self.command)
        self.proc = subprocess.Popen(args, **kwargs)
        return self.proc.pid

    def wait_ready(self, timeout_s=900.0, poll_s=1.0):
        """Block until GET /v1/models answers 200. Returns seconds waited."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {self.proc.returncode}; see {self.log_path}"
                )
            try:
                r = requests.get(f"{self.base_url}/v1/models", timeout=5)
                if r.status_code == 200:
                    return time.monotonic() - t0
            except requests.RequestException:
                pass
            time.sleep(poll_s)
        raise TimeoutError(f"server not ready after {timeout_s}s; see {self.log_path}")

    def kill(self):
        """Fail-stop: SIGKILL the whole process tree. Returns wall-clock kill time."""
        t_kill = time.time()
        if self.proc is None:
            return t_kill
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        if self._log:
            self._log.write(f"=== killed {t_kill:.6f}\n")
            self._log.flush()
        return t_kill

    def stop(self):
        self.kill()
        if self._log:
            self._log.close()
            self._log = None
