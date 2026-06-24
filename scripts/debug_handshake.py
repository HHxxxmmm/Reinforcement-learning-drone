"""
探测仿真平台 TCP 握手。房间保持开启时在 Python/ 下运行:
    python scripts/debug_handshake.py
"""
import argparse
import socket
import struct
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.train_env import pack_initial
from utils.initialize import generate_initial_state


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def try_recv(sock, size, label):
    try:
        data = b""
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("connection closed")
            data += chunk
        print(f"  [OK] recv {label}: {len(data)} bytes")
        return data
    except socket.timeout:
        print(f"  [TIMEOUT] recv {label}")
        return None


def try_send(sock, fmt, values, label):
    payload = struct.pack(fmt, *values)
    sock.sendall(payload)
    print(f"  [OK] send {label}: {len(payload)} bytes")


def build_initial(room_id, unit_id, state, body=None):
    if body is None:
        body = np.zeros(24, dtype=np.int32)
    return pack_initial(body, room_id=room_id, unit_id=unit_id, state=state, sync_step=1)


def build_action():
    return [0.5, 0.0, 0.0, 0.0, 0.0]


def run_probe(host, port, room_id, unit_id, state, mode):
    print(f"\n--- mode={mode} host={host} port={port} room={room_id} unit={unit_id} state={state} ---")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(8)
    try:
        sock.connect((host, port))
        print("  [OK] tcp connect")

        initial = build_initial(room_id, unit_id, state)
        action = build_action()

        if mode in ("recv_first", "recv_init_recv", "recv_action_recv"):
            try_recv(sock, 216, "obs_before_send")

        if mode in ("init_recv", "recv_init_recv", "init_action_recv"):
            try_send(sock, "<100i", initial.tolist(), "initial(100i)")

        if mode in ("action_recv", "recv_action_recv", "init_action_recv"):
            try_send(sock, "<5d", action, "action")

        if mode != "connect_only":
            data = try_recv(sock, 216, "obs")
            if data:
                obs = struct.unpack("=27d", data)
                print(
                    f"  sample: my_hp={obs[12]:.0f}, enemy_hp={obs[25]:.0f}, is_done={obs[26]:.1f}"
                )
                return True
        return False
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        return False
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "envs.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    host = cfg["host"]
    port = cfg["port"]
    room_id = cfg.get("room_id", 0)
    unit_id = cfg.get("unit_id", 0)

    modes = [
        "init_recv",
        "init_action_recv",
        "recv_first",
        "action_recv",
        "recv_init_recv",
    ]
    for state in (2, 1):
        for mode in modes:
            if run_probe(host, port, room_id, unit_id, state, mode):
                print("\n>>> 成功模式:", mode, "state=", state)
                return

    unit1_port = cfg.get("unit1_port")
    if unit1_port:
        print(f"\n--- 额外探测 unit1 端口 {unit1_port} (仅 connect) ---")
        run_probe(host, unit1_port, room_id, 1, 2, "connect_only")

    print("\n所有探测均失败。请确认: 1) 已进入对战画面 2) port/room_id 与当前房间一致 3) 问助教握手顺序")


if __name__ == "__main__":
    main()
