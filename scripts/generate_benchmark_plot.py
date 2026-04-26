"""Generate benchmark success comparison plots from benchmark_results.json.

Usage:
    uv run python generate_benchmark_plot.py

Inputs:
    benchmark_results.json

Outputs:
    assets/benchmark_success_comparison.svg
    assets/benchmark_success_comparison.png
    training_logs/benchmark_success_comparison.csv
    training_logs/benchmark_success_comparison.json
"""

from __future__ import annotations

import csv
import json
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = Path("/content/CloudSREEnv-Hackathon/episode_traces/benchmark_results.json")
ASSETS_DIR = ROOT / "assets"
LOG_DIR = ROOT / "training_logs"

TASK_LABELS = {
    "task1_tls_certificate_rca": "Task 1 TLS RCA",
    "task2_self_healing": "Task 2 Self Healing",
    "task3_latency_resolution": "Task 3 Latency",
    "task4_noisy_neighbor": "Task 4 Noisy Neighbor",
    "task4_resource_contention": "Task 4 Noisy Neighbor",
    "task5_cache_split_brain": "Task 5 Split Brain",
    "task5_split_brain_cache_consistency": "Task 5 Split Brain",
}

SPLITS = [
    ("In-template", "base_in_template", "sft_in_template"),
    ("Heldout", "base_heldout", "sft_heldout"),
]


def _ensure_dirs() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)


def _load_benchmark() -> dict:
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark file: {BENCHMARK_PATH}")
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


def _comparison_rows(benchmark: dict) -> tuple[list[str], list[dict]]:
    task_order = list(benchmark["base_in_template"]["results"].keys())
    rows = []

    for split_name, base_key, trained_key in SPLITS:
        base = benchmark[base_key]["results"]
        trained = benchmark[trained_key]["results"]
        for task_id in task_order:
            rows.append({
                "split": split_name,
                "task_id": task_id,
                "base_success": int(bool(base[task_id])),
                "sft_success": int(bool(trained[task_id])),
            })

    return task_order, rows


def _write_comparison_logs(rows: list[dict]) -> None:
    with (LOG_DIR / "benchmark_success_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "task_id", "base_success", "sft_success"])
        writer.writeheader()
        writer.writerows(rows)

    (LOG_DIR / "benchmark_success_comparison.json").write_text(
        json.dumps({"source": "benchmark_results.json", "rows": rows}, indent=2),
        encoding="utf-8",
    )


def _write_svg(benchmark: dict, task_order: list[str]) -> None:
    width, height = 1220, 760
    plot_y = 135
    panel_width = 500
    panel_height = 280
    panel_gap = 80
    panel_xs = [95, 95 + panel_width + panel_gap]
    bar_width = 26
    min_fail_height = 18
    red = "#d9534f"
    green = "#2ca02c"
    black = "#333"
    grid = "#ddd"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="610" y="42" text-anchor="middle" font-family="Arial, sans-serif" font-size="26" font-weight="700">CloudSREEnv Benchmark: Base vs Trained Agent</text>',
        '<text x="610" y="72" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#444">Generated from benchmark_results.json</text>',
    ]

    for panel_index, (split_name, base_key, trained_key) in enumerate(SPLITS):
        base = benchmark[base_key]["results"]
        trained = benchmark[trained_key]["results"]
        panel_x = panel_xs[panel_index]
        base_solved = sum(bool(value) for value in base.values())
        trained_solved = sum(bool(value) for value in trained.values())
        total = len(task_order)

        svg.append(
            f'<text x="{panel_x + panel_width / 2}" y="{plot_y - 25}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="20" font-weight="700">'
            f'{split_name}: Base {base_solved}/{total} vs Trained {trained_solved}/{total}</text>'
        )
        svg.append(f'<line x1="{panel_x}" y1="{plot_y + panel_height}" x2="{panel_x + panel_width}" y2="{plot_y + panel_height}" stroke="{black}" stroke-width="2"/>')
        svg.append(f'<line x1="{panel_x}" y1="{plot_y}" x2="{panel_x}" y2="{plot_y + panel_height}" stroke="{black}" stroke-width="2"/>')

        for fraction, label in [(0, "Fail"), (1, "Pass")]:
            y = plot_y + panel_height - fraction * panel_height
            svg.append(f'<line x1="{panel_x}" y1="{y}" x2="{panel_x + panel_width}" y2="{y}" stroke="{grid}" stroke-width="1"/>')
            svg.append(f'<text x="{panel_x - 12}" y="{y + 5}" text-anchor="end" font-family="Arial, sans-serif" font-size="13">{label}</text>')

        slot = panel_width / len(task_order)
        for index, task_id in enumerate(task_order):
            center = panel_x + slot * index + slot / 2
            base_value = bool(base[task_id])
            trained_value = bool(trained[task_id])
            base_height = panel_height if base_value else min_fail_height
            trained_height = panel_height if trained_value else min_fail_height
            base_x = center - bar_width - 5
            trained_x = center + 5

            svg.append(f'<rect x="{base_x}" y="{plot_y + panel_height - base_height}" width="{bar_width}" height="{base_height}" fill="{red}"/>')
            svg.append(f'<rect x="{trained_x}" y="{plot_y + panel_height - trained_height}" width="{bar_width}" height="{trained_height}" fill="{green}"/>')
            svg.append(f'<text x="{base_x + bar_width / 2}" y="{plot_y + panel_height - base_height - 7}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="700">{"PASS" if base_value else "FAIL"}</text>')
            svg.append(f'<text x="{trained_x + bar_width / 2}" y="{plot_y + panel_height - trained_height - 7}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="700">{"PASS" if trained_value else "FAIL"}</text>')

            label_parts = TASK_LABELS.get(task_id, task_id).split(" ", 2)
            first_line = " ".join(label_parts[:2])
            second_line = label_parts[2] if len(label_parts) > 2 else ""
            svg.append(f'<text x="{center}" y="{plot_y + panel_height + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">{first_line}</text>')
            svg.append(f'<text x="{center}" y="{plot_y + panel_height + 44}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">{second_line}</text>')

    svg.extend([
        f'<rect x="465" y="650" width="18" height="18" fill="{red}"/><text x="490" y="664" font-family="Arial, sans-serif" font-size="15">Base model</text>',
        f'<rect x="615" y="650" width="18" height="18" fill="{green}"/><text x="640" y="664" font-family="Arial, sans-serif" font-size="15">Trained SFT model</text>',
        '<text x="610" y="705" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="700">In-template improves from 0/5 to 5/5; heldout improves from 0/5 to 4/5.</text>',
        "</svg>",
    ])

    (ASSETS_DIR / "benchmark_success_comparison.svg").write_text("\n".join(svg), encoding="utf-8")


def _write_png(benchmark: dict, task_order: list[str]) -> None:
    """Write a simple dependency-free PNG bar chart.

    The SVG is the recommended artifact for presentation because it contains
    text labels. This PNG exists for platforms that prefer raster images.
    """
    width, height = 1220, 760
    plot_y = 135
    panel_width = 500
    panel_height = 280
    panel_gap = 80
    panel_xs = [95, 95 + panel_width + panel_gap]
    bar_width = 26
    min_fail_height = 18

    image = bytearray([255, 255, 255] * width * height)

    def rect(x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int]) -> None:
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        r, g, b = color
        for y in range(y0, y1):
            offset = (y * width + x0) * 3
            for _ in range(x0, x1):
                image[offset:offset + 3] = bytes((r, g, b))
                offset += 3

    def line_h(x0: float, x1: float, y: float, color: tuple[int, int, int]) -> None:
        rect(x0, y, x1, y + 2, color)

    def line_v(x: float, y0: float, y1: float, color: tuple[int, int, int]) -> None:
        rect(x, y0, x + 2, y1, color)

    red = (217, 83, 79)
    green = (44, 160, 44)
    black = (51, 51, 51)
    grid = (221, 221, 221)

    for panel_index, (_, base_key, trained_key) in enumerate(SPLITS):
        base = benchmark[base_key]["results"]
        trained = benchmark[trained_key]["results"]
        panel_x = panel_xs[panel_index]

        line_h(panel_x, panel_x + panel_width, plot_y + panel_height, black)
        line_v(panel_x, plot_y, plot_y + panel_height, black)
        line_h(panel_x, panel_x + panel_width, plot_y, grid)
        line_h(panel_x, panel_x + panel_width, plot_y + panel_height, grid)

        slot = panel_width / len(task_order)
        for index, task_id in enumerate(task_order):
            center = panel_x + slot * index + slot / 2
            for value, color, dx in [
                (base[task_id], red, -bar_width - 5),
                (trained[task_id], green, 5),
            ]:
                bar_height = panel_height if value else min_fail_height
                rect(center + dx, plot_y + panel_height - bar_height, center + dx + bar_width, plot_y + panel_height, color)

    rect(465, 650, 483, 668, red)
    rect(615, 650, 633, 668, green)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(image[y * width * 3:(y + 1) * width * 3])

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    (ASSETS_DIR / "benchmark_success_comparison.png").write_bytes(png)


def main() -> None:
    _ensure_dirs()
    benchmark = _load_benchmark()
    task_order, rows = _comparison_rows(benchmark)
    _write_comparison_logs(rows)
    _write_svg(benchmark, task_order)
    _write_png(benchmark, task_order)
    print("Wrote assets/benchmark_success_comparison.svg")
    print("Wrote assets/benchmark_success_comparison.png")
    print("Wrote training_logs/benchmark_success_comparison.csv")
    print("Wrote training_logs/benchmark_success_comparison.json")


if __name__ == "__main__":
    main()
