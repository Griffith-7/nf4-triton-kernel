"""Generate benchmark comparison graphs for README."""

import torch
import time
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from nf4_kernel import dequant_nf4, quantize_nf4


def benchmark_fn(fn, warmup=200, iterations=2000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations


def run_all():
    import bitsandbytes as bnb

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")

    configs = [
        ("4K", 4096),
        ("16K", 16384),
        ("64K", 65536),
        ("256K", 262144),
    ]

    results = {"device": device_name, "sizes": [], "triton_ms": [], "bnb_ms": [], "speedup": []}

    for label, size in configs:
        tensor = torch.randn(size, dtype=torch.bfloat16, device="cuda")

        # Triton: quantize with our kernel, benchmark dequant
        packed_t, absmax_t, _, _ = quantize_nf4(tensor)
        triton_ms = benchmark_fn(lambda: dequant_nf4(packed_t, absmax_t)) * 1000

        # bitsandbytes: quantize with bnb, benchmark dequant
        packed_bnb, qstate = bnb.functional.quantize_4bit(tensor, quant_type="nf4")
        bnb_ms = benchmark_fn(lambda: bnb.functional.dequantize_4bit(packed_bnb, quant_state=qstate, quant_type="nf4")) * 1000

        speedup = bnb_ms / triton_ms

        results["sizes"].append(label)
        results["triton_ms"].append(round(triton_ms, 4))
        results["bnb_ms"].append(round(bnb_ms, 4))
        results["speedup"].append(round(speedup, 2))

        print(f"  {label}: Triton={triton_ms:.4f}ms  BnB={bnb_ms:.4f}ms  speedup={speedup:.2f}x")

    return results


def plot_speedup_bar(results, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    sizes = results["sizes"]
    speedups = results["speedup"]
    colors = ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]

    bars = ax.bar(sizes, speedups, color=colors, edgecolor="#1e40af", linewidth=0.8, width=0.55)

    for bar, sp in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{sp:.2f}x", ha="center", va="bottom", fontweight="bold", fontsize=11, color="#1e3a5f")

    ax.axhline(y=1.15, color="#dc2626", linestyle="--", linewidth=1.2, label="1.15x threshold")
    ax.set_ylim(0, max(speedups) * 1.25)
    ax.set_xlabel("Tensor Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup vs bitsandbytes", fontsize=12, fontweight="bold")
    ax.set_title("NF4 Triton Kernel Speedup over bitsandbytes C++", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_latency_comparison(results, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    sizes = results["sizes"]
    triton_ms = results["triton_ms"]
    bnb_ms = results["bnb_ms"]

    x = np.arange(len(sizes))
    width = 0.32

    bars1 = ax.bar(x - width / 2, triton_ms, width, label="Triton Kernel", color="#2563eb", edgecolor="#1e40af", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, bnb_ms, width, label="bitsandbytes C++", color="#94a3b8", edgecolor="#475569", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Tensor Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Dequantization Latency: Triton vs bitsandbytes", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_scaling_curve(results, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    sizes = results["sizes"]
    triton_ms = results["triton_ms"]
    bnb_ms = results["bnb_ms"]

    ax.plot(sizes, triton_ms, "o-", color="#2563eb", linewidth=2.5, markersize=8, label="Triton Kernel", zorder=5)
    ax.plot(sizes, bnb_ms, "s--", color="#94a3b8", linewidth=2.5, markersize=8, label="bitsandbytes C++", zorder=5)

    ax.fill_between(sizes, triton_ms, bnb_ms, alpha=0.1, color="#2563eb")

    ax.set_xlabel("Tensor Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Scaling Behavior: Triton vs bitsandbytes", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
    os.makedirs(out_dir, exist_ok=True)

    results = run_all()

    data_path = os.path.join(out_dir, "benchmark_data.json")
    with open(data_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved data: {data_path}")

    plot_speedup_bar(results, os.path.join(out_dir, "speedup_comparison.png"))
    plot_latency_comparison(results, os.path.join(out_dir, "latency_comparison.png"))
    plot_scaling_curve(results, os.path.join(out_dir, "scaling_curve.png"))

    print("\nDone!")
