"""Convert model JSON output into a minimal OpenSCAD scaffold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def shape_to_scad(data: dict) -> str:
    dims = data.get("dimensions", {})
    h = dims.get("H", 10)
    w = dims.get("W", 10)
    d = dims.get("D", 10)
    unit = dims.get("unit", "mm")
    global_shape = data.get("global_shape", "cube").lower()
    comment = "// Generated from fiche JSON\n"
    comments = [comment]
    comments.append(f"// object: {data.get('object', '')}")
    comments.append(f"// function: {data.get('function', '')}")
    comments.append(f"// components: {', '.join(data.get('components', []))}")
    comments.append(f"// interfaces: {', '.join(data.get('interfaces', []))}")

    if "cyl" in global_shape:
        body = f"cylinder(h={h}, r={w/2}); // unit: {unit}\n"
    else:
        body = f"cube([{w}, {d}, {h}], center=false); // unit: {unit}\n"
    return "\n".join(comments) + "\n" + body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path, help="Path to JSON output (model prediction)")
    parser.add_argument("--out", type=Path, default=Path("outputs/sample.scad"))
    args = parser.parse_args()

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    scad = shape_to_scad(data)
    args.out.write_text(scad, encoding="utf-8")
    print(f"Wrote OpenSCAD to {args.out}")


if __name__ == "__main__":
    main()
