"""
Simple vs Simple 双端冒烟：unit0 + unit1 各开一个 TrainEnv 线程同步 step。

用法（UE 房间已开、已进入对战画面）:
    python scripts/dual_smoke_test.py --max-steps 200
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.train_env import TrainEnv


def make_env(config_path, port, unit_id):
  with open(config_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
  cfg = dict(cfg)
  cfg["port"] = port
  cfg["unit_id"] = unit_id
  tmp = ROOT / "config" / f"_tmp_unit{unit_id}.yaml"
  tmp.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
  return TrainEnv(config_path=str(tmp)), tmp


def run_unit(env, unit_id, max_steps, log_interval, done_event, stats):
  try:
    obs, _ = env.reset()
    print(f"[unit{unit_id}] reset OK obs={obs.shape} my_hp={env.my_state[12]:.0f}")
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
      action = env.action_space.sample()
      obs, reward, terminated, truncated, _ = env.step(action)
      steps += 1
      if steps % log_interval == 0:
        print(
          f"[unit{unit_id}] step={steps} my=({env.my_state[0]:.0f},{env.my_state[1]:.0f},{env.my_state[2]:.0f}) "
          f"hp={env.my_state[12]:.0f}/{env.enemy_state[12]:.0f}"
        )
      if max_steps and steps >= max_steps:
        break
    stats[unit_id] = steps
    print(f"[unit{unit_id}] done steps={steps} term={terminated} trunc={truncated}")
  except Exception as exc:
    stats[unit_id] = f"error: {exc}"
    print(f"[unit{unit_id}] ERROR: {exc}")
  finally:
    done_event.set()
    env.adaptor.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=str(ROOT / "config" / "envs.yaml"))
  parser.add_argument("--max-steps", type=int, default=200)
  parser.add_argument("--log-interval", type=int, default=50)
  args = parser.parse_args()

  with open(args.config, encoding="utf-8") as f:
    base = yaml.safe_load(f)
  port0 = base["port"]
  port1 = base.get("unit1_port", port0 + 4)

  print(f"dual smoke room={base.get('room_id')} ports={port0}/{port1}")
  env0, tmp0 = make_env(args.config, port0, 0)
  env1, tmp1 = make_env(args.config, port1, 1)

  done0 = threading.Event()
  done1 = threading.Event()
  stats = {}

  t0 = threading.Thread(target=run_unit, args=(env0, 0, args.max_steps, args.log_interval, done0, stats))
  t1 = threading.Thread(target=run_unit, args=(env1, 1, args.max_steps, args.log_interval, done1, stats))

  t0.start()
  time.sleep(0.3)
  t1.start()
  t0.join()
  t1.join()
  tmp0.unlink(missing_ok=True)
  tmp1.unlink(missing_ok=True)
  print("stats:", stats)


if __name__ == "__main__":
  main()
