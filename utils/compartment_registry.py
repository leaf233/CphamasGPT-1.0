"""
utils/compartment_registry.py

跨房室（1/2/3 cmt）并行进程的 BIC 数据共享注册表。
使用原子 rename（os.replace）避免多进程读写竞态。
"""
import json
import os
import time


class CompartmentRegistry:
    """跨进程共享的房室 BIC 注册表。"""

    def __init__(self, path: str):
        self.path = path

    # ---- 写 ---- #

    def write(self, n_cmt: int, ofv, n_params: int,
              best_code_path: str = None) -> None:
        registry = self.read()
        registry[str(n_cmt)] = {
            'n_cmt': n_cmt,
            'ofv': ofv,
            'n_params': n_params,
            'best_code_path': best_code_path,
            'ts': time.time(),
        }
        self._atomic_dump(registry)

    # ---- 读 ---- #

    def read(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    # ---- 等待齐备 ---- #

    def wait_for(self, expected: set, timeout_s: int = 300,
                 poll_s: int = 5) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            registry = self.read()
            present = {int(k) for k in registry.keys()}
            if expected.issubset(present):
                return registry
            time.sleep(poll_s)
        return self.read()

    # ---- 内部 ---- #

    def _atomic_dump(self, registry: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2)
        os.replace(tmp, self.path)