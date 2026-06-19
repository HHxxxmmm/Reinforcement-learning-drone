"""快速扫描哪个端口在发 initial 后会回观测包。"""
import socket
import struct
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.train_env import pack_initial


def load_config():
    with open(ROOT / "config" / "envs.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def probe_port(host, port, room_id, unit_id, state, timeout=3):
    body = np.zeros(24, dtype=np.int32)
    initial = pack_initial(body, room_id, unit_id, state, 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        sock.sendall(struct.pack("<100i", *initial.tolist()))
        data = b""
        while len(data) < 216:
            chunk = sock.recv(216 - len(data))
            if not chunk:
                return "closed"
            data += chunk
        obs = struct.unpack("=27d", data)
        my_hp, enemy_hp = obs[12], obs[25]
        my_pos = obs[0:3]
        enemy_pos = obs[13:16]
        # 发一帧动作，看敌机是否变化（真房间通常两边都会有响应）
        sock.sendall(struct.pack("<5d", 0.5, 0.0, 0.0, 0.0, 0.0))
        data2 = b""
        while len(data2) < 216:
            chunk = sock.recv(216 - len(data2))
            if not chunk:
                break
            data2 += chunk
        enemy_moved = False
        if len(data2) == 216:
            obs2 = struct.unpack("=27d", data2)
            enemy_moved = any(abs(obs2[i] - obs[i]) > 1e-6 for i in range(13, 16))
        likely_real = my_hp >= 500 and enemy_hp >= 500
        tag = "LIKELY_UE" if likely_real else "orphan?"
        return (
            f"{tag} hp={my_hp:.0f}/{enemy_hp:.0f} "
            f"my={my_pos[0]:.0f},{my_pos[1]:.0f},{my_pos[2]:.0f} "
            f"enemy={enemy_pos[0]:.0f},{enemy_pos[1]:.0f},{enemy_pos[2]:.0f} "
            f"enemy_moved={enemy_moved}"
        )
    except socket.timeout:
        return "timeout"
    except ConnectionRefusedError:
        return "refused"
    except OSError as exc:
        return f"err:{exc}"
    finally:
        sock.close()


def main():
    cfg = load_config()
    host = cfg["host"]
    room_id = cfg.get("room_id", 0)
    unit_id = cfg.get("unit_id", 0)
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 1020

    print(f"scan {host} ports {start}-{end}, room_id={room_id}")
    for port in range(start, end + 1):
        for state in (2, 1):
            result = probe_port(host, port, room_id, unit_id, state, timeout=2)
            if result != "timeout" and result != "refused":
                print(f"  port {port} state={state} -> {result}")
    print("done")


if __name__ == "__main__":
    main()
