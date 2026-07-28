"""Generate architecture-diagram assets from one real ACFNet sample.

The script intentionally targets ACFNet and Houston2013 only.  It reuses the
project data pipeline so the PCA-HSI and LiDAR patches match training exactly.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-acfnet")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from matplotlib import cm
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.data import getMyData, set_random_seed
from src.runner import ExperimentRunner


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "model/ACFNet/Houston2013_model_train_experiment_hsi_x_pca=30_window=11_lr=0.0001_epochs=100.pth"
)
HOUSTON_DATA = PROJECT_ROOT.parent / "data/Houston2013"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize one Houston2013 sample and ACFNet internals."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/acfnet_architecture")
    parser.add_argument("--sample-index", type=int, default=1796,
                        help="Index in the test set (default: 1796, a visually distinct sample).")
    parser.add_argument("--search-samples", type=int, default=1024,
                        help="Maximum test samples considered during automatic selection.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def normalize_image(array, lower=2.0, upper=98.0):
    array = np.asarray(array, dtype=np.float32)
    lo, hi = np.percentile(array, [lower, upper])
    if hi <= lo:
        return np.zeros_like(array)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0)


def save_image(array, path, cmap=None, dpi=300):
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.imshow(array, cmap=cmap, interpolation="bicubic")
    ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_heatmap(array, path, cmap="turbo", dpi=300):
    save_image(normalize_image(array, 0.0, 100.0), path, cmap=cmap, dpi=dpi)


def save_matrix(array, path, cmap="magma", dpi=300):
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    image = ax.imshow(array, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def crop_raw_patch(cube, row, col, size):
    pad = size // 2
    mode = "reflect" if size % 2 else "symmetric"
    padded = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode=mode)
    return padded[row:row + size, col:col + size]


def save_hsi_cube(raw_patch, path, dpi=300):
    """Render the raw HSI patch as a textured spectral cube."""
    height, width, bands = raw_patch.shape
    rgb_indices = [int(bands * 0.72), int(bands * 0.50), int(bands * 0.25)]
    rgb = np.stack([normalize_image(raw_patch[..., index]) for index in rgb_indices], axis=-1)

    spectral = normalize_image(raw_patch)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x_front, z_front = np.meshgrid(np.arange(width), np.arange(bands))
    y_side, z_side = np.meshgrid(np.arange(height), np.arange(bands))

    fig = plt.figure(figsize=(4.2, 4.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, np.full_like(x, bands - 1), facecolors=rgb,
                    shade=False, linewidth=0, antialiased=False)
    ax.plot_surface(x_front, np.full_like(x_front, height - 1), z_front,
                    facecolors=cm.viridis(spectral[-1, :, :].T),
                    shade=False, linewidth=0, antialiased=False)
    ax.plot_surface(np.full_like(y_side, width - 1), y_side, z_side,
                    facecolors=cm.viridis(spectral[:, -1, :].T),
                    shade=False, linewidth=0, antialiased=False)
    ax.set_box_aspect((width, height, bands * 0.08))
    ax.view_init(elev=27, azim=-55)
    ax.set_axis_off()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


def load_model(checkpoint, device):
    loaded = torch.load(checkpoint, map_location=device, weights_only=False)
    net = loaded["net"] if isinstance(loaded, dict) and "net" in loaded else loaded
    net.to(device)
    if device.type == "cpu":
        enable_mamba_cpu_fallback(net.fusion_mamba.mixer)
    net.eval()
    return net


def enable_mamba_cpu_fallback(mixer):
    """Replace the CUDA-only Mamba fast path with its reference equations."""
    def cpu_forward(hidden_states, inference_params=None):
        if inference_params is not None:
            raise ValueError("The visualization CPU fallback does not support inference cache.")
        batch, sequence_length, _ = hidden_states.shape
        xz = rearrange(
            mixer.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=sequence_length,
        )
        if mixer.in_proj.bias is not None:
            xz = xz + rearrange(mixer.in_proj.bias, "d -> d 1")

        x, z = xz.chunk(2, dim=1)
        x = F.silu(mixer.conv1d(x)[..., :sequence_length])
        x_dbl = mixer.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, b_state, c_state = torch.split(
            x_dbl, [mixer.dt_rank, mixer.d_state, mixer.d_state], dim=-1
        )
        dt = rearrange(mixer.dt_proj.weight @ dt.t(), "d (b l) -> b d l", l=sequence_length)
        b_state = rearrange(b_state, "(b l) s -> b s l", b=batch, l=sequence_length).contiguous()
        c_state = rearrange(c_state, "(b l) s -> b s l", b=batch, l=sequence_length).contiguous()
        a_state = -torch.exp(mixer.A_log.float())
        output = selective_scan_ref(
            x,
            dt,
            a_state,
            b_state,
            c_state,
            mixer.D.float(),
            z=z,
            delta_bias=mixer.dt_proj.bias.float(),
            delta_softplus=True,
        )
        return mixer.out_proj(rearrange(output, "b d l -> b l d"))

    mixer.forward = cpu_forward


def configure_project():
    config = ExperimentConfig(
        experiment_name="acfnet_architecture",
        dataset_type=0,
        model_name="ACFNet",
        channels=30,
        window_size=11,
        batch_size=128,
        num_workers=0,
        enable_training=False,
        enable_testing=False,
        enable_visualization=False,
    )
    ExperimentRunner(config)
    return config


def make_batch(hsi_pca, hsi, aux, labels, device):
    return {
        "hsi_pca": hsi_pca.to(device),
        "hsi": hsi.to(device),
        "aux": aux.to(device),
        "label": labels.to(device),
    }


def select_sample(net, test_loader, device, requested_index, search_samples):
    if requested_index is not None:
        if not 0 <= requested_index < len(test_loader.dataset):
            raise IndexError(f"sample-index must be in [0, {len(test_loader.dataset) - 1}]")
        hsi_pca, hsi, aux, label = test_loader.dataset[requested_index]
        batch = make_batch(
            hsi_pca.unsqueeze(0), hsi.unsqueeze(0), aux.unsqueeze(0), label.unsqueeze(0), device
        )
        with torch.no_grad():
            probabilities = net(batch["hsi_pca"].squeeze(1), batch["aux"]).softmax(dim=1)
        prediction = int(probabilities.argmax(dim=1).item())
        confidence = float(probabilities.max().item())
        return requested_index, batch, prediction, confidence

    best = None
    offset = 0
    with torch.no_grad():
        for hsi_pca, hsi, aux, labels in test_loader:
            batch = make_batch(hsi_pca, hsi, aux, labels, device)
            probabilities = net(batch["hsi_pca"].squeeze(1), batch["aux"]).softmax(dim=1)
            confidence, prediction = probabilities.max(dim=1)
            correct = prediction.eq(batch["label"])
            for local_index in torch.where(correct)[0].tolist():
                score = float(confidence[local_index].item())
                if best is None or score > best[0]:
                    best = (score, offset + local_index, int(prediction[local_index].item()))
            offset += labels.shape[0]
            if offset >= search_samples:
                break

    if best is None:
        raise RuntimeError("No correctly classified sample was found in the search range.")
    confidence, sample_index, prediction = best
    hsi_pca, hsi, aux, label = test_loader.dataset[sample_index]
    batch = make_batch(
        hsi_pca.unsqueeze(0), hsi.unsqueeze(0), aux.unsqueeze(0), label.unsqueeze(0), device
    )
    return sample_index, batch, prediction, confidence


def capture_features(net, batch):
    layer_names = ["pca_stem", "x_stem", "fusion_mamba", "pca_model", "x_model", "fusion"]
    captured = {}
    handles = []

    def hook(name):
        def save_output(_module, _inputs, output):
            captured[name] = output.detach().cpu()
        return save_output

    modules = dict(net.named_modules())
    for name in layer_names:
        handles.append(modules[name].register_forward_hook(hook(name)))
    try:
        with torch.no_grad():
            logits = net(batch["hsi_pca"].squeeze(1), batch["aux"])
    finally:
        for handle in handles:
            handle.remove()
    return logits, captured


def save_overview(items, path, dpi):
    columns = 4
    rows = int(np.ceil(len(items) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.0 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, (title, array, cmap) in zip(axes, items):
        ax.imshow(array, cmap=cmap, interpolation="bicubic")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = configure_project()
    set_random_seed(config.random_seed)

    print("Loading Houston2013 and fitting the same PCA used by training...")
    _, test_loader, _, _, _ = getMyData(
        datasetType=0,
        channels=30,
        windowSize=11,
        batch_size=128,
        num_workers=0,
    )
    net = load_model(args.checkpoint, device)
    sample_index, batch, prediction, confidence = select_sample(
        net, test_loader, device, args.sample_index, args.search_samples
    )
    row, col = map(int, test_loader.dataset.pos[sample_index])
    label = int(batch["label"].item())

    logits, features = capture_features(net, batch)
    pca_feat = features["pca_stem"].to(device)
    x_feat = features["x_stem"].to(device)
    with torch.no_grad():
        spatial_attention = net.pca_enhance.attention(pca_feat, x_feat)[0].cpu().numpy()
        channel_attention = net.x_enhance.attention(x_feat, pca_feat)[0].cpu().numpy()

    raw_hsi = loadmat(HOUSTON_DATA / "houston_hsi.mat")["houston_hsi"]
    raw_aux = loadmat(HOUSTON_DATA / "houston_lidar.mat")["houston_lidar"]
    raw_patch = crop_raw_patch(raw_hsi, row, col, config.window_size)
    raw_aux_patch = crop_raw_patch(raw_aux, row, col, config.window_size)
    pca_patch = batch["hsi_pca"][0, 0].detach().cpu().numpy().transpose(1, 2, 0)
    aux_patch = batch["aux"][0].detach().cpu().numpy().transpose(1, 2, 0)
    pca_rgb = np.stack([normalize_image(pca_patch[..., i]) for i in range(3)], axis=-1)
    aux_image = normalize_image(aux_patch[..., 0]) if aux_patch.shape[-1] == 1 else normalize_image(aux_patch[..., :3])

    raw_bands = raw_patch.shape[-1]
    rgb_indices = [int(raw_bands * 0.72), int(raw_bands * 0.50), int(raw_bands * 0.25)]
    raw_hsi_rgb = np.stack(
        [normalize_image(raw_patch[..., index]) for index in rgb_indices], axis=-1
    )
    raw_aux_image = normalize_image(
        raw_aux_patch[..., 0] if raw_aux_patch.shape[-1] == 1 else raw_aux_patch[..., :3]
    )

    save_image(raw_hsi_rgb, args.output_dir / "00a_hsi_original_patch.png", dpi=args.dpi)
    save_image(
        raw_aux_image,
        args.output_dir / "00b_x_modality_original_patch.png",
        cmap="gray" if raw_aux_patch.shape[-1] == 1 else None,
        dpi=args.dpi,
    )

    save_hsi_cube(raw_patch, args.output_dir / "01_hsi_cube.png", args.dpi)
    save_image(pca_rgb, args.output_dir / "02_pca_hsi_patch.png", dpi=args.dpi)
    save_image(aux_image, args.output_dir / "03_x_modality_patch.png", cmap="gray", dpi=args.dpi)

    overview_items = [
        ("Raw HSI patch", raw_hsi_rgb, None),
        ("Raw X patch", raw_aux_image, "gray" if raw_aux_patch.shape[-1] == 1 else None),
        ("PCA-HSI input", pca_rgb, None),
        ("X-modality input", aux_image, "gray"),
    ]
    for index, name in enumerate(["pca_stem", "x_stem", "fusion_mamba", "pca_model", "x_model", "fusion"], start=4):
        heatmap = features[name][0].abs().mean(dim=0).numpy()
        save_heatmap(heatmap, args.output_dir / f"{index:02d}_{name}.png", dpi=args.dpi)
        overview_items.append((name, normalize_image(heatmap, 0.0, 100.0), "turbo"))

    patch_size = config.window_size
    center_token = (patch_size * patch_size) // 2
    center_spatial = spatial_attention[center_token].reshape(patch_size, patch_size)
    save_heatmap(center_spatial, args.output_dir / "10_spatial_attention_center.png", dpi=args.dpi)
    save_matrix(spatial_attention, args.output_dir / "11_spatial_attention_matrix.png", dpi=args.dpi)
    save_matrix(channel_attention, args.output_dir / "12_channel_attention_matrix.png", dpi=args.dpi)
    overview_items.extend([
        ("Spatial attention", normalize_image(center_spatial, 0.0, 100.0), "turbo"),
        ("Channel attention", channel_attention, "magma"),
    ])
    save_overview(overview_items, args.output_dir / "13_acfnet_sample_overview.png", args.dpi)

    metadata = (
        f"dataset=Houston2013\n"
        f"test_sample_index={sample_index}\n"
        f"row={row}\ncol={col}\n"
        f"ground_truth_zero_based={label}\n"
        f"prediction_zero_based={prediction}\n"
        f"confidence={confidence:.6f}\n"
        f"logits={logits[0].detach().cpu().tolist()}\n"
    )
    (args.output_dir / "sample_info.txt").write_text(metadata, encoding="utf-8")
    print(f"Saved ACFNet architecture assets to: {args.output_dir}")
    print(f"Sample: test_index={sample_index}, coordinate=({row}, {col}), "
          f"label={label}, prediction={prediction}, confidence={confidence:.4f}")


if __name__ == "__main__":
    main()
