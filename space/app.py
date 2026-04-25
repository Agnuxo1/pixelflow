"""pixelflow HuggingFace Space demo.

Two tabs:
  1. CA Visualizer  -- step a cellular-automaton reservoir and display intermediate states.
  2. MNIST Classifier Demo -- draw a digit, run a pre-trained reservoir pipeline,
     show predicted class + probability bar chart + reservoir state heatmap.

All computation runs on CPU only (no moderngl/OpenGL required).
Seeds are fixed everywhere for reproducibility.
"""

from __future__ import annotations

import io
import logging
import time
import warnings

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")  # non-interactive backend -- required for server use
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# pixelflow imports
# ---------------------------------------------------------------------------
from pixelflow import Reservoir, ReservoirConfig, LogisticReadout

# ---------------------------------------------------------------------------
# Global: trained MNIST pipeline (built once at startup)
# ---------------------------------------------------------------------------

_MNIST_RESERVOIR: Reservoir | None = None
_MNIST_READOUT: LogisticReadout | None = None
_MNIST_TRAIN_ACC: float = 0.0

MNIST_GRID_W = 28
MNIST_GRID_H = 28
MNIST_CHANNELS = 4
MNIST_STEPS = 6
MNIST_SUBSET = 2000  # samples used for training; keeps startup under 60 s on CPU


def _build_mnist_reservoir() -> Reservoir:
    cfg = ReservoirConfig(
        width=MNIST_GRID_W,
        height=MNIST_GRID_H,
        channels=MNIST_CHANNELS,
        steps=MNIST_STEPS,
        rule="diffusion_reaction",
        input_encoding="tile",
        seed=42,
    )
    return Reservoir(cfg, backend="cpu")


def _make_synthetic_digits(n_per_class: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: create simple synthetic 28x28 digit-like images when MNIST download fails."""
    rng = np.random.default_rng(seed)
    samples, labels = [], []
    for digit in range(10):
        for _ in range(n_per_class):
            img = np.zeros((28, 28), dtype=np.float32)
            # Draw a rough pattern unique to each digit class
            row_off = (digit // 5) * 12 + 4
            col_off = (digit % 5) * 4 + 4
            img[row_off : row_off + 10, col_off : col_off + 6] = 1.0
            img += rng.uniform(0, 0.15, img.shape).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)
            samples.append(img.ravel())
            labels.append(digit)
    X = np.array(samples, dtype=np.float32)
    y = np.array(labels, dtype=np.intp)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def _train_mnist_pipeline() -> None:
    global _MNIST_RESERVOIR, _MNIST_READOUT, _MNIST_TRAIN_ACC

    t0 = time.time()
    reservoir = _build_mnist_reservoir()

    # Try to load real MNIST; fall back to synthetic on failure
    X_train: np.ndarray
    y_train: np.ndarray
    try:
        from pixelflow.tasks.mnist import load as load_mnist

        logger.info("Downloading MNIST subset (%d samples)...", MNIST_SUBSET)
        X_train, y_train, _X_test, _y_test = load_mnist(subset=MNIST_SUBSET, seed=42)
        logger.info("MNIST loaded in %.1f s", time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MNIST download failed (%s); using synthetic fallback.", exc)
        X_train, y_train = _make_synthetic_digits(n_per_class=30, seed=42)

    logger.info("Transforming %d samples through reservoir...", len(X_train))
    H_train = reservoir.transform(X_train)
    logger.info("Transform done in %.1f s", time.time() - t0)

    readout = LogisticReadout(C=0.1, max_iter=300)
    readout.fit(H_train, y_train)

    _MNIST_TRAIN_ACC = float(readout.score(H_train, y_train))
    _MNIST_RESERVOIR = reservoir
    _MNIST_READOUT = readout
    logger.info(
        "Pipeline ready in %.1f s  (train acc %.3f)", time.time() - t0, _MNIST_TRAIN_ACC
    )


# Train at import time (Gradio loads app.py once, then serves requests)
logger.info("Starting MNIST pipeline training at startup...")
_train_mnist_pipeline()

# ---------------------------------------------------------------------------
# Tab 1: CA Visualizer helpers
# ---------------------------------------------------------------------------

RULES = ["diffusion_reaction", "life_like", "wave"]
N_INTERMEDIATE = 8  # number of frames to show in the strip


def _run_ca_steps(rule: str, grid_size: int, total_steps: int, seed: int = 0):
    """Run CA from random initial state, collecting N_INTERMEDIATE + 1 snapshots.

    Returns list of (H, W, C) float32 arrays.
    """
    channels = 4
    cfg = ReservoirConfig(
        width=grid_size,
        height=grid_size,
        channels=channels,
        steps=1,  # we step manually
        rule=rule,
        input_encoding="tile",
        seed=seed,
    )
    res = Reservoir(cfg, backend="cpu")

    rng = np.random.default_rng(seed)
    state = rng.random((grid_size, grid_size, channels)).astype(np.float32)

    # Decide which step indices to snapshot
    if total_steps <= N_INTERMEDIATE:
        snap_at = set(range(total_steps + 1))
    else:
        snap_at = set(
            int(round(i * total_steps / (N_INTERMEDIATE - 1)))
            for i in range(N_INTERMEDIATE)
        )
        snap_at.add(0)
        snap_at.add(total_steps)

    snapshots: list[tuple[int, np.ndarray]] = []
    if 0 in snap_at:
        snapshots.append((0, state.copy()))

    for step_idx in range(1, total_steps + 1):
        state = res.step(state)
        if step_idx in snap_at:
            snapshots.append((step_idx, state.copy()))

    # Deduplicate and sort
    seen: set[int] = set()
    unique: list[tuple[int, np.ndarray]] = []
    for s, arr in snapshots:
        if s not in seen:
            seen.add(s)
            unique.append((s, arr))
    unique.sort(key=lambda x: x[0])
    return unique


def _state_to_rgb(state: np.ndarray) -> np.ndarray:
    """Convert (H, W, C) state to (H, W, 3) RGB uint8 for display."""
    # Use first 3 channels if available; otherwise replicate channel 0
    if state.shape[2] >= 3:
        rgb = state[:, :, :3]
    else:
        rgb = np.stack([state[:, :, 0]] * 3, axis=-1)
    # Normalise to [0, 1]
    lo, hi = rgb.min(), rgb.max()
    if hi > lo:
        rgb = (rgb - lo) / (hi - lo)
    else:
        rgb = np.zeros_like(rgb)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def visualize_ca(rule: str, grid_size: int, steps: int) -> plt.Figure:
    """Gradio callback for Tab 1."""
    grid_size = int(grid_size)
    steps = int(steps)

    snapshots = _run_ca_steps(rule, grid_size, steps, seed=0)

    n = len(snapshots)
    # Layout: 2 rows if many frames, else single row
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(f"Rule: {rule}  |  grid {grid_size}x{grid_size}  |  {steps} steps", fontsize=11)

    axes_flat = np.array(axes).ravel() if n > 1 else [axes]

    for ax_i, (step_idx, state) in enumerate(snapshots):
        ax = axes_flat[ax_i]
        rgb = _state_to_rgb(state)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"step {step_idx}", fontsize=8)
        ax.axis("off")

    # Hide unused axes
    for ax_i in range(len(snapshots), len(axes_flat)):
        axes_flat[ax_i].axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 2: MNIST Classifier helpers
# ---------------------------------------------------------------------------

def _preprocess_sketchpad(sketchpad_input) -> np.ndarray | None:
    """Convert Gradio ImageEditor output to a (784,) float32 array in [0, 1].

    Gradio 5.x ImageEditor returns a dict with keys:
      'background', 'layers' (list of PIL Images), 'composite' (PIL Image)
    or None for an empty canvas.
    """
    if sketchpad_input is None:
        return None

    # Handle dict (Gradio 5 ImageEditor output)
    if isinstance(sketchpad_input, dict):
        # Prefer 'composite' (merged drawing), fall back to 'background'
        img_data = sketchpad_input.get("composite") or sketchpad_input.get("background")
        if img_data is None:
            layers = sketchpad_input.get("layers", [])
            img_data = layers[0] if layers else None
        if img_data is None:
            return None
        sketchpad_input = img_data

    # Now sketchpad_input is PIL Image, np.ndarray, or bytes
    if isinstance(sketchpad_input, bytes):
        sketchpad_input = Image.open(io.BytesIO(sketchpad_input))

    if isinstance(sketchpad_input, np.ndarray):
        pil_img = Image.fromarray(sketchpad_input.astype(np.uint8))
    elif isinstance(sketchpad_input, Image.Image):
        pil_img = sketchpad_input
    else:
        try:
            pil_img = Image.fromarray(np.array(sketchpad_input).astype(np.uint8))
        except Exception:
            return None

    # Convert to grayscale, resize to 28x28
    pil_img = pil_img.convert("L").resize((28, 28), Image.LANCZOS)
    arr = np.array(pil_img, dtype=np.float32) / 255.0

    # If canvas is mostly white (background), invert so digit is bright on dark
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    return arr.ravel()


def classify_digit(sketchpad_input) -> tuple[plt.Figure, plt.Figure, str]:
    """Gradio callback for Tab 2.

    Returns:
        prob_fig  -- bar chart of class probabilities
        heat_fig  -- heatmap of reservoir state (channel 0 after MNIST_STEPS)
        label_str -- predicted class string
    """
    if _MNIST_RESERVOIR is None or _MNIST_READOUT is None:
        err_msg = "Pipeline not ready. Please wait and retry."
        return _empty_fig(err_msg), _empty_fig(err_msg), err_msg

    x = _preprocess_sketchpad(sketchpad_input)
    if x is None or x.sum() < 1e-3:
        msg = "Please draw a digit on the canvas."
        return _empty_fig(msg), _empty_fig(msg), msg

    # Run through reservoir
    H = _MNIST_RESERVOIR.transform(x[np.newaxis, :])  # (1, feature_dim)

    # Predict
    pred_class = int(_MNIST_READOUT.predict(H)[0])
    probs = _MNIST_READOUT.predict_proba(H)[0]  # (10,)

    # --- Probability bar chart ---
    prob_fig, prob_ax = plt.subplots(figsize=(5, 3))
    bar_colors = ["steelblue"] * 10
    bar_colors[pred_class] = "firebrick"
    prob_ax.bar(range(10), probs, color=bar_colors)
    prob_ax.set_xticks(range(10))
    prob_ax.set_xlabel("Digit class")
    prob_ax.set_ylabel("Probability")
    prob_ax.set_title(f"Predicted: {pred_class}  (confidence {probs[pred_class]:.2f})")
    prob_ax.set_ylim(0, 1)
    prob_fig.tight_layout()

    # --- Reservoir state heatmap ---
    # Re-run step-by-step to get intermediate state at final step
    rng = np.random.default_rng([_MNIST_RESERVOIR.config.seed, 0])
    state = _MNIST_RESERVOIR.config  # just reading config
    cfg = _MNIST_RESERVOIR.config
    init_state = _MNIST_RESERVOIR._encoder(
        x, cfg.height, cfg.width, cfg.channels, rng
    )
    cur = init_state.copy()
    for _ in range(cfg.steps):
        cur = _MNIST_RESERVOIR.step(cur)

    heat_fig, heat_ax = plt.subplots(figsize=(4, 4))
    heat_ax.imshow(cur[:, :, 0], cmap="viridis", interpolation="nearest")
    heat_ax.set_title("Reservoir state (channel 0 after evolution)")
    heat_ax.axis("off")
    heat_fig.tight_layout()

    label_str = f"Predicted class: {pred_class}   Train accuracy: {_MNIST_TRAIN_ACC:.1%}"
    return prob_fig, heat_fig, label_str


def _empty_fig(msg: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.axis("off")
    return fig


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="pixelflow — GPU Texture Reservoir Computing") as demo:
    gr.Markdown(
        "## pixelflow: GPU Texture Reservoir Computing\n"
        "Cellular-automaton reservoirs on a 2-D pixel grid. "
        "All computation runs on CPU in this demo. "
        "[GitHub](https://github.com/Agnuxo1/pixelflow)"
    )

    with gr.Tab("CA Visualizer"):
        gr.Markdown(
            "Choose a CA rule, grid size, and number of steps. "
            "The panel shows a strip of intermediate states from step 0 to step N."
        )
        with gr.Row():
            with gr.Column(scale=1):
                rule_radio = gr.Radio(
                    choices=RULES,
                    value="diffusion_reaction",
                    label="CA rule",
                )
                grid_slider = gr.Slider(
                    minimum=16, maximum=128, step=8, value=32,
                    label="Grid size (NxN)",
                )
                steps_slider = gr.Slider(
                    minimum=1, maximum=64, step=1, value=16,
                    label="Number of steps",
                )
                run_btn = gr.Button("Run CA")
            with gr.Column(scale=3):
                ca_plot = gr.Plot(label="CA evolution")

        run_btn.click(
            fn=visualize_ca,
            inputs=[rule_radio, grid_slider, steps_slider],
            outputs=ca_plot,
        )
        # Auto-run on load with defaults
        demo.load(
            fn=visualize_ca,
            inputs=[rule_radio, grid_slider, steps_slider],
            outputs=ca_plot,
        )

    with gr.Tab("MNIST Classifier Demo"):
        gr.Markdown(
            "Draw a digit (0-9) on the canvas. "
            "The reservoir pipeline (trained at startup on 2000 MNIST samples) "
            "classifies it and shows the reservoir's internal state."
        )
        with gr.Row():
            with gr.Column(scale=1):
                sketchpad = gr.ImageEditor(
                    label="Draw a digit here",
                    type="pil",
                    sources=[],           # disable file-upload; drawing only
                    transforms=[],        # disable crop/rotate
                    brush=gr.Brush(
                        default_size=14,
                        colors=["#000000"],
                        color_mode="fixed",
                    ),
                )
                classify_btn = gr.Button("Classify")
                label_out = gr.Textbox(label="Result", interactive=False)
            with gr.Column(scale=2):
                prob_plot = gr.Plot(label="Class probabilities")
                heat_plot = gr.Plot(label="Reservoir state heatmap")

        classify_btn.click(
            fn=classify_digit,
            inputs=sketchpad,
            outputs=[prob_plot, heat_plot, label_out],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
