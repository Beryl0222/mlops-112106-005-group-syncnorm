"""检查交换资料的形状和索引。"""

import argparse
import json
from pathlib import Path

import numpy as np


def inspect(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate(data)
    return {
        "资料": path.name,
        "字段": sorted(data),
        "字节": path.stat().st_size,
        "数值环境": np.__version__,
    }


def validate(data: dict) -> None:
    assert set(data["expected_workers"]) == data["workers"].keys()
    for worker in data["workers"].values():
        lengths = {len(worker[key]) for key in ("groups", "x", "mask", "dy")}
        assert len(lengths) == 1
        assert all(len(row) == len(data["gamma"]) for row in worker["x"] + worker["dy"])


def main() -> None:
    parser = argparse.ArgumentParser(description="检查本地交换资料")
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.scene), ensure_ascii=False))


if __name__ == "__main__":
    main()
