"""Simple vs Simple 双端探测：同时连 unit0/unit1 控制端口。"""
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.train_env import pack_initial


def load_config():
    with open(ROOT / "config" / "envs.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def probe_unit(host, port, room_id, unit_id, state, label, results):
    body = np.zeros(24, dtype=np.int32)
    initial = pack_initial(body, room_id, unit_id, state, 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(15)
    try:
        sock.connect((host, port))
        sock.sendall(struct.pack("<100i", *initial.tolist()))
        data = b""
        while len(data) < 216:
            chunk = sock.recv(216 - len(data))
            if not chunk:
                results[label] = "closed before obs"
                return
            data += chunk
        obs = struct.unpack("=27d", data)
        results[label] = (
            f"OK hp={obs[12]:.0f}/{obs[25]:.0f} "
            f"my=({obs[0]:.0f},{obs[1]:.0f},{obs[2]:.0f}) "
            f"enemy=({obs[13]:.0f},{obs[14]:.0f},{obs[15]:.0f})"
        )
    except Exception as exc:
        results[label] = f"FAIL {exc}"
    finally:
        sock.close()


def main():
    cfg = load_config()
    host = cfg["host"]
    room_id = cfg["room_id"]
    port0 = cfg["port"]
    port1 = cfg.get("unit1_port", port0 + 4)

    print(f"dual probe room={room_id} unit0={port0} unit1={port1} host={host}")
    results = {}
    t0 = threading.Thread(
        target=probe_unit,
        args=(host, port0, room_id, 0, 2, "unit0", results),
    )
    t1 = threading.Thread(
        target=probe_unit,
        args=(host, port1, room_id, 1, 2, "unit1", results),
    )
    t0.start()
    time.sleep(0.2)
    t1.start()
    t0.join()
    t1.join()
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
