"""
GPT-2 From Scratch — CLI Entry Point.

Usage:
    Train on WikiText-2:
        python main.py train
        python main.py train --epochs 10 --batch-size 4 --lr 1e-4
        python main.py train --device cpu
        python main.py train --resume checkpoints/best.pt
        python main.py train --viz                          # with live dashboard

    Generate text:
        python main.py generate --prompt "The meaning of life is"
        python main.py generate --prompt "In a shocking finding" --max-tokens 300
        python main.py generate --prompt "Once upon a time" --temperature 1.2 --top-k 100
        python main.py generate --checkpoint checkpoints/best.pt --device cpu

    Visualization server (post-hoc analysis):
        python main.py viz --checkpoint checkpoints/best.pt
        python main.py viz --checkpoint checkpoints/best.pt --port 8080
"""

import argparse
import os

# MPS performance environment variables — set BEFORE importing torch.
# No-ops on non-MPS platforms.
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "1")
os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")

import torch

from config import GPT2Config, TrainConfig, get_device
from model import GPT2
from data import create_dataloaders
from train import train
from generate import generate


def cmd_train(args):
    """Train GPT-2 on WikiText-2."""
    config = GPT2Config()

    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
    )

    print(f"Device: {train_config.device}")
    print(f"Batch size: {train_config.batch_size} "
          f"(effective: {train_config.batch_size * train_config.gradient_accumulation_steps})")
    print()

    # Create model
    model = GPT2(config)

    # Resume from checkpoint if specified
    resume_checkpoint = None
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        print(f"  Loaded model weights from step {resume_checkpoint.get('step', '?')}, "
              f"val_loss={resume_checkpoint.get('val_loss', '?')}")

    # Create data loaders
    print("\nPreparing data...")
    train_loader, val_loader = create_dataloaders(config, train_config)

    # Viz setup (optional)
    metrics_logger = None
    hook_manager = None

    if args.viz:
        import threading
        from viz.metrics import MetricsLogger
        from viz.hooks import HookManager
        from viz.app import init_app, socketio

        metrics_logger = MetricsLogger()
        hook_manager = HookManager(model, mode="training")
        hook_manager.attach()

        init_app(metrics_logger=metrics_logger, device=train_config.device)
        port = args.viz_port

        def run_server():
            socketio.run(
                socketio.server.eio.app if hasattr(socketio.server, 'eio') else init_app(),
                host="0.0.0.0", port=port
            )

        # Use socketio.run with the actual app
        from viz.app import app as viz_app
        server_thread = threading.Thread(
            target=lambda: socketio.run(viz_app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True),
            daemon=True,
        )
        server_thread.start()
        print(f"\n[viz] Dashboard running at http://localhost:{port}\n")

    # Train
    train(model, train_loader, val_loader, train_config, config,
          metrics_logger=metrics_logger, hook_manager=hook_manager,
          resume_checkpoint=resume_checkpoint)

    if hook_manager:
        hook_manager.detach()


def cmd_generate(args):
    """Generate text from a trained checkpoint."""
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Reconstruct model config from checkpoint
    config = GPT2Config(**checkpoint["model_config"])
    model = GPT2(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)

    print(f"Model loaded (step {checkpoint.get('step', '?')}, "
          f"val_loss={checkpoint.get('val_loss', '?'):.4f})")
    print(f"Device: {args.device}")
    print(f"Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)

    output = generate(
        model,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )

    print(output)
    print("-" * 60)


def cmd_viz(args):
    """Launch visualization server for post-hoc model analysis."""
    from viz.app import init_app, socketio

    viz_app = init_app(checkpoint_path=args.checkpoint, device=args.device)
    print(f"\nVisualization server at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.\n")
    socketio.run(viz_app, host="0.0.0.0", port=args.port, allow_unsafe_werkzeug=True)


def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 from scratch: train on WikiText-2 and generate text"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train subcommand ---
    train_parser = subparsers.add_parser("train", help="Train GPT-2 on WikiText-2")
    train_parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs (default: 30)"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Micro-batch size (default: 8)"
    )
    train_parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Peak learning rate (default: 3e-4)"
    )
    default_device = get_device()
    train_parser.add_argument(
        "--device", type=str, default=default_device,
        help=f"Device: cuda, mps, or cpu (default: {default_device})"
    )
    train_parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    train_parser.add_argument(
        "--viz", action="store_true",
        help="Enable visualization dashboard during training (port 5000)"
    )
    train_parser.add_argument(
        "--viz-port", type=int, default=5000,
        help="Port for visualization server (default: 5000)"
    )

    # --- Generate subcommand ---
    gen_parser = subparsers.add_parser("generate", help="Generate text from trained model")
    gen_parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt to condition generation on"
    )
    gen_parser.add_argument(
        "--max-tokens", type=int, default=200,
        help="Maximum number of tokens to generate (default: 200)"
    )
    gen_parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature: <1=focused, >1=creative (default: 0.8)"
    )
    gen_parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k filtering: keep top k tokens (default: 50, 0=disabled)"
    )
    gen_parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="Nucleus sampling threshold (default: 0.95, 1.0=disabled)"
    )
    gen_parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to model checkpoint (default: checkpoints/best.pt)"
    )
    gen_parser.add_argument(
        "--device", type=str, default=default_device,
        help=f"Device: cuda, mps, or cpu (default: {default_device})"
    )

    # --- Viz subcommand ---
    viz_parser = subparsers.add_parser("viz", help="Launch visualization server for model analysis")
    viz_parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to model checkpoint (default: checkpoints/best.pt)"
    )
    viz_parser.add_argument(
        "--port", type=int, default=5000,
        help="Server port (default: 5000)"
    )
    viz_parser.add_argument(
        "--device", type=str, default=default_device,
        help=f"Device: cuda, mps, or cpu (default: {default_device})"
    )

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "viz":
        cmd_viz(args)


if __name__ == "__main__":
    main()
