"""Flask application for mechanistic interpretability visualization.

Serves:
  1. Static files (HTML/JS/CSS)
  2. REST API for post-hoc analysis
  3. WebSocket (flask-socketio) for real-time training updates

Usage:
    Standalone:  python -m viz.app --checkpoint checkpoints/best.pt
    During training: launched as background thread by main.py --viz
"""

import os
import glob as glob_module

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit

from viz.analysis import (
    load_model_from_checkpoint,
    tokenize_prompt,
    get_attention_weights,
    get_activation_analysis,
    get_logit_attribution,
    get_head_ablation,
    get_direct_logit_attribution,
    get_activation_patching,
    get_activation_steering,
    get_activation_swapping,
    get_precomputation_detection,
)

# Module-level state
_model = None
_config = None
_metrics_logger = None
_device = "cpu"

static_dir = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=static_dir, static_url_path="/static")
app.config["SECRET_KEY"] = "gpt2-viz"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


def init_app(checkpoint_path=None, metrics_logger=None, device="cpu"):
    """Initialize the app with a model checkpoint and/or metrics logger."""
    global _model, _config, _metrics_logger, _device
    _device = device
    _metrics_logger = metrics_logger

    if checkpoint_path and os.path.exists(checkpoint_path):
        _model, _config = load_model_from_checkpoint(checkpoint_path, device)
        print(f"[viz] Model loaded from {checkpoint_path}")

    return app


# --- Static file serving ---

@app.route("/")
def index():
    return send_from_directory(static_dir, "index.html")


# --- Model info ---

@app.route("/api/model/info")
def model_info():
    if _config is None:
        return jsonify({"loaded": False})
    return jsonify({
        "loaded": True,
        "n_layer": _config.n_layer,
        "n_head": _config.n_head,
        "n_embd": _config.n_embd,
        "context_length": _config.context_length,
        "vocab_size": _config.vocab_size,
    })


# --- Checkpoint management ---

@app.route("/api/checkpoints")
def list_checkpoints():
    ckpt_dir = "checkpoints"
    if not os.path.isdir(ckpt_dir):
        return jsonify([])
    files = glob_module.glob(os.path.join(ckpt_dir, "*.pt"))
    return jsonify([os.path.basename(f) for f in sorted(files)])


@app.route("/api/load_checkpoint", methods=["POST"])
def load_checkpoint():
    global _model, _config
    data = request.get_json()
    path = data.get("path", "")
    if not os.path.exists(path):
        return jsonify({"error": f"Checkpoint not found: {path}"}), 404

    _model, _config = load_model_from_checkpoint(path, _device)
    return jsonify({"status": "ok", "n_layer": _config.n_layer})


# --- Training metrics ---

@app.route("/api/metrics/all")
def metrics_all():
    if _metrics_logger is None:
        return jsonify({"steps": [], "validations": []})
    return jsonify({
        "steps": _metrics_logger.get_all_steps(),
        "validations": _metrics_logger.get_all_validations(),
    })


@app.route("/api/metrics/since/<int:step>")
def metrics_since(step):
    if _metrics_logger is None:
        return jsonify({"steps": []})
    return jsonify({"steps": _metrics_logger.get_steps_since(step)})


# --- Post-hoc analysis endpoints ---

@app.route("/api/attention", methods=["POST"])
def attention():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids, _ = tokenize_prompt(prompt)
    if len(token_ids) > _config.context_length:
        token_ids = token_ids[:_config.context_length]

    result = get_attention_weights(_model, token_ids, _device)
    return jsonify(result)


@app.route("/api/activations", methods=["POST"])
def activations():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids, _ = tokenize_prompt(prompt)
    if len(token_ids) > _config.context_length:
        token_ids = token_ids[:_config.context_length]

    result = get_activation_analysis(_model, token_ids, _device)
    return jsonify(result)


@app.route("/api/attribution", methods=["POST"])
def attribution():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json()
    prompt = data.get("prompt", "")
    top_k = data.get("top_k", 10)
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids, _ = tokenize_prompt(prompt)
    if len(token_ids) > _config.context_length:
        token_ids = token_ids[:_config.context_length]
    if len(token_ids) < 2:
        return jsonify({"error": "Prompt too short (need at least 2 tokens)"}), 400

    result = get_logit_attribution(_model, token_ids, _device, top_k=top_k)
    return jsonify(result)


@app.route("/api/ablation", methods=["POST"])
def ablation():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids, _ = tokenize_prompt(prompt)
    if len(token_ids) > _config.context_length:
        token_ids = token_ids[:_config.context_length]
    if len(token_ids) < 2:
        return jsonify({"error": "Prompt too short (need at least 2 tokens)"}), 400

    result = get_head_ablation(_model, token_ids, _device)
    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids, token_strings = tokenize_prompt(prompt)
    if len(token_ids) > _config.context_length:
        token_ids = token_ids[:_config.context_length]

    import torch
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=_device)
    with torch.no_grad():
        logits, _ = _model(input_ids)

    probs = torch.softmax(logits[0, -1], dim=-1)
    top_probs, top_ids = probs.topk(10)

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    predictions = [
        {"token": enc.decode([int(tid)]), "prob": float(p)}
        for tid, p in zip(top_ids.tolist(), top_probs.tolist())
    ]

    return jsonify({"tokens": token_strings, "predictions": predictions})


@app.route("/api/dla", methods=["POST"])
def dla():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json()
    prompt = data.get("prompt", "")
    position = data.get("position", -1)
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids, _ = tokenize_prompt(prompt)
    if len(token_ids) > _config.context_length:
        token_ids = token_ids[:_config.context_length]
    if len(token_ids) < 2:
        return jsonify({"error": "Prompt too short (need at least 2 tokens)"}), 400

    result = get_direct_logit_attribution(_model, token_ids, _device, position=position)
    return jsonify(result)


# --- Circuits: Causal intervention endpoints ---

@app.route("/api/circuits/patching", methods=["POST"])
def circuits_patching():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json()
    clean_prompt = data.get("clean_prompt", "")
    corrupted_prompt = data.get("corrupted_prompt", "")
    if not clean_prompt or not corrupted_prompt:
        return jsonify({"error": "Both clean_prompt and corrupted_prompt required"}), 400

    clean_ids, _ = tokenize_prompt(clean_prompt)
    corrupted_ids, _ = tokenize_prompt(corrupted_prompt)
    if len(clean_ids) > _config.context_length:
        clean_ids = clean_ids[:_config.context_length]
    if len(corrupted_ids) > _config.context_length:
        corrupted_ids = corrupted_ids[:_config.context_length]
    if len(clean_ids) < 2 or len(corrupted_ids) < 2:
        return jsonify({"error": "Prompts too short (need at least 2 tokens each)"}), 400

    result = get_activation_patching(_model, clean_ids, corrupted_ids, _device)
    return jsonify(result)


@app.route("/api/circuits/steering", methods=["POST"])
def circuits_steering():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    layer = data.get("layer", 0)
    component = data.get("component", "head")
    head = data.get("head", 0)
    scale = data.get("scale", 0.0)

    if component not in ("head", "mlp"):
        return jsonify({"error": "component must be 'head' or 'mlp'"}), 400
    if not (0 <= layer < _config.n_layer):
        return jsonify({"error": f"layer must be 0-{_config.n_layer - 1}"}), 400
    if component == "head" and not (0 <= head < _config.n_head):
        return jsonify({"error": f"head must be 0-{_config.n_head - 1}"}), 400

    token_ids, _ = tokenize_prompt(prompt)
    if len(token_ids) > _config.context_length:
        token_ids = token_ids[:_config.context_length]

    result = get_activation_steering(
        _model, token_ids, _device,
        layer=layer, component=component, head=head, scale=scale,
    )
    return jsonify(result)


@app.route("/api/circuits/swapping", methods=["POST"])
def circuits_swapping():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json()
    source_prompt = data.get("source_prompt", "")
    target_prompt = data.get("target_prompt", "")
    if not source_prompt or not target_prompt:
        return jsonify({"error": "Both source_prompt and target_prompt required"}), 400

    layer = data.get("layer", 0)
    component = data.get("component", "residual")

    if component not in ("residual", "attn", "mlp"):
        return jsonify({"error": "component must be 'residual', 'attn', or 'mlp'"}), 400
    if not (0 <= layer < _config.n_layer):
        return jsonify({"error": f"layer must be 0-{_config.n_layer - 1}"}), 400

    source_ids, _ = tokenize_prompt(source_prompt)
    target_ids, _ = tokenize_prompt(target_prompt)
    if len(source_ids) > _config.context_length:
        source_ids = source_ids[:_config.context_length]
    if len(target_ids) > _config.context_length:
        target_ids = target_ids[:_config.context_length]

    result = get_activation_swapping(
        _model, source_ids, target_ids, _device,
        layer=layer, component=component,
    )
    return jsonify(result)


@app.route("/api/circuits/precomputation", methods=["POST"])
def circuits_precomputation():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    top_k = data.get("top_k", 5)
    token_ids, _ = tokenize_prompt(prompt)
    if len(token_ids) > _config.context_length:
        token_ids = token_ids[:_config.context_length]
    if len(token_ids) < 3:
        return jsonify({"error": "Prompt too short (need at least 3 tokens)"}), 400

    result = get_precomputation_detection(_model, token_ids, _device, top_k=top_k)
    return jsonify(result)


# --- WebSocket events ---

@socketio.on("connect")
def handle_connect():
    """Send full metric history to newly connected client."""
    if _metrics_logger is not None:
        emit("metrics_history", {
            "steps": _metrics_logger.get_all_steps(),
            "validations": _metrics_logger.get_all_validations(),
        })


def emit_step_update(metrics: dict):
    """Push a training step update to all connected clients."""
    socketio.emit("step_update", metrics)


def emit_val_update(metrics: dict):
    """Push a validation result to all connected clients."""
    socketio.emit("val_update", metrics)


def emit_training_complete(best_val_loss: float):
    """Notify clients that training has finished."""
    socketio.emit("training_complete", {"best_val_loss": best_val_loss})


# --- Standalone entry point ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPT-2 Visualization Server")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    init_app(checkpoint_path=args.checkpoint, device=args.device)
    print(f"Starting visualization server at http://localhost:{args.port}")
    socketio.run(app, host="0.0.0.0", port=args.port, debug=False)
