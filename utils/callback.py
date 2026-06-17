"""
修改gym的callback函数，实现训练过程中保存模型和日志
"""
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict
import os, csv
import pandas as pd

class RewardComponentsCallback(BaseCallback):
    def __init__(self, csv_path=None):
        super().__init__()
        self.csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None
        self._reset_sums()

    def _reset_sums(self):
        self.count = 0
        self.sums = defaultdict(float)

    def _on_training_start(self) -> None:
        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = None

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    def _on_rollout_start(self) -> None:
        self._reset_sums()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        info = infos[0]

        row = {"timesteps": int(self.num_timesteps)}
        wrote = False
        for key, val in info.items():
            if isinstance(key, str) and key.startswith("r/"):
                try:
                    v = float(val)
                except Exception:
                    continue
                self.sums[key] += v
                row[key] = v
                wrote = True

        # 逐步 CSV（可选）
        if wrote and self._csv_file is not None:
            if self._csv_writer is None:
                headers = ["timesteps"] + sorted(k for k in row.keys() if k != "timesteps")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=headers)
                self._csv_writer.writeheader()
            self._csv_writer.writerow(row)

        if wrote:
            self.count += 1
        return True

    def _on_rollout_end(self) -> None:
        if self.count == 0:
            return
        for k, s in self.sums.items():
            self.logger.record(f"reward/{k}_mean", s / self.count)
        self._reset_sums()
