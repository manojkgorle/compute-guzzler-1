"""
Profile GPT-2 training step to identify actual bottlenecks.

Two profiling approaches:
    1. Wall-clock breakdown: forward vs backward vs optimizer (manual timing)
    2. torch.profiler: per-operation timing with GPU activity tracing

Usage:
    python profile_model.py                  # wall-clock breakdown (10 steps)
    python profile_model.py --torch-profile  # full torch.profiler trace
    python profile_model.py --steps 20       # more steps for stable averages
    python profile_model.py --device cuda    # profile on CUDA
"""

import argparse
import time

import torch
from torch.profiler import profile, ProfilerActivity

from config import GPT2Config, TrainConfig, get_device
from model import GPT2
from data import create_dataloaders


def device_sync(device):
    """Synchronize GPU to get accurate wall-clock times.

    GPU backends queue work asynchronously — without syncing, timing
    would only measure CPU dispatch time, not actual GPU compute time.
    """
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def wall_clock_breakdown(model, train_loader, device, num_steps=10):
    """Manual wall-clock timing of forward, backward, and optimizer phases."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    forward_times = []
    backward_times = []
    optimizer_times = []
    total_times = []

    model.train()
    data_iter = iter(train_loader)

    # Warmup: run 2 steps so GPU compiles/caches kernels
    for _ in range(2):
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    device_sync(device)

    print(f"\nProfiling {num_steps} steps (wall-clock breakdown)...\n")

    for step in range(num_steps):
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        # Forward
        device_sync(device)
        t0 = time.perf_counter()
        _, loss = model(x, y)
        device_sync(device)
        t1 = time.perf_counter()

        # Backward
        loss.backward()
        device_sync(device)
        t2 = time.perf_counter()

        # Optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        device_sync(device)
        t3 = time.perf_counter()

        forward_times.append(t1 - t0)
        backward_times.append(t2 - t1)
        optimizer_times.append(t3 - t2)
        total_times.append(t3 - t0)

    # Print results
    def fmt(times):
        avg = sum(times) / len(times)
        return f"{avg * 1000:7.1f}ms"

    total_avg = sum(total_times) / len(total_times)
    fwd_avg = sum(forward_times) / len(forward_times)
    bwd_avg = sum(backward_times) / len(backward_times)
    opt_avg = sum(optimizer_times) / len(optimizer_times)

    print(f"{'Phase':<12} {'Avg Time':>10} {'% of Total':>12}")
    print("-" * 36)
    print(f"{'Forward':<12} {fmt(forward_times):>10} {fwd_avg/total_avg*100:>11.1f}%")
    print(f"{'Backward':<12} {fmt(backward_times):>10} {bwd_avg/total_avg*100:>11.1f}%")
    print(f"{'Optimizer':<12} {fmt(optimizer_times):>10} {opt_avg/total_avg*100:>11.1f}%")
    print("-" * 36)
    print(f"{'Total':<12} {fmt(total_times):>10} {'100.0%':>12}")
    print(f"\n  Steps/sec: {1.0 / total_avg:.2f}")
    print(f"  Est. epoch time: {total_avg * len(train_loader) / 60:.1f} min")


def torch_profile_trace(model, train_loader, device, num_steps=10):
    """Full torch.profiler trace with per-op GPU timing.

    Produces a table of the top-20 most expensive operations,
    showing both CPU and GPU time.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    data_iter = iter(train_loader)

    # Warmup
    for _ in range(2):
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    device_sync(device)

    # Select profiler activity and sort key based on device
    if device.type == "cuda":
        gpu_activity = ProfilerActivity.CUDA
        gpu_sort_key = "self_cuda_time_total"
        gpu_label = "CUDA"
    else:
        gpu_activity = ProfilerActivity.MPS
        gpu_sort_key = "self_mps_time_total"
        gpu_label = "MPS"

    print(f"\nRunning torch.profiler for {num_steps} steps...\n")

    with profile(
        activities=[ProfilerActivity.CPU, gpu_activity],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(num_steps):
            x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        device_sync(device)

    # Print top-20 ops by GPU time
    print(f"Top 20 operations by {gpu_label} (GPU) time:")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by=gpu_sort_key,
        row_limit=20,
    ))

    # Also print by CPU time for comparison
    print("\nTop 20 operations by CPU time:")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=20,
    ))


def main():
    parser = argparse.ArgumentParser(description="Profile GPT-2 training step")
    parser.add_argument("--torch-profile", action="store_true",
                        help="Use torch.profiler (detailed per-op trace)")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of steps to profile (default: 10)")
    parser.add_argument("--device", type=str, default=get_device(),
                        help=f"Device (default: {get_device()})")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = GPT2Config()
    train_config = TrainConfig(device=args.device)

    print(f"Device: {device}")
    print(f"Context length: {config.context_length}")
    print(f"Batch size: {train_config.batch_size}")

    model = GPT2(config).to(device)

    print("\nPreparing data...")
    train_loader, _ = create_dataloaders(config, train_config)

    if args.torch_profile:
        torch_profile_trace(model, train_loader, device, args.steps)
    else:
        wall_clock_breakdown(model, train_loader, device, args.steps)


if __name__ == "__main__":
    main()
