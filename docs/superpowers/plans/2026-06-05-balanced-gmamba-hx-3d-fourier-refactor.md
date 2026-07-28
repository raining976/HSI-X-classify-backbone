# BalancedGMambaHX 3D Fourier Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `BalancedGMambaHX` so Fourier modeling is concentrated on the raw HSI spectral-spatial cube, auxiliary modality frequency decomposition is removed, abstract token FFT is removed, and HSI-X fusion uses a lightweight HSI-guided gate.

**Architecture:** Add a `SpectralSpatialFourierEnhancement` module before the HSI 3D stem to perform 3D FFT over `(spectral, height, width)` on raw HSI and reconstruct a gated low/high-frequency enhanced HSI cube. Replace the current `SpectralGuidedFrequencyFusion` with a lightweight `HSIGuidedGatedFusion` that locally enhances `F_hsi/F_fusion` and `F_x`, then uses an HSI-derived spatial gate to inject auxiliary features. Keep the existing `SpatialSpectralStateModeling`, `SpatialMamba2D`, stage/downsampling pattern, and classification head for continuity.

**Tech Stack:** Python, PyTorch, `torch.fft`, optional `mamba_ssm` fallback already present in [src/models/balanced_gmamba_hx/model.py](../../../../src/models/balanced_gmamba_hx/model.py), pytest-style smoke tests.

---

## File Structure

- Modify: `src/models/balanced_gmamba_hx/model.py`
  - Remove token-level FFT and auxiliary/spatial FFT decomposition classes from the active forward path.
  - Add `SpectralSpatialFourierEnhancement` for raw HSI 3D Fourier enhancement.
  - Add `HSIGuidedGatedFusion` for lightweight HSI-guided multimodal fusion.
  - Simplify `BalancedGMambaBlock`, `BalancedGMambaStage`, `DownsampleStage`, and `BalancedGMambaHX.forward` signatures so stage blocks receive only `fusion_spatial`, `x_spatial`, and `spec_tokens`.
  - Keep `SpatialSpectralStateModeling`, `MambaTokenMixer`, `SpatialMamba2D`, and `BalancedFusionHead` unchanged unless a shape fix is required.
- Modify: `src/models/balanced_gmamba_hx/adapter.py`
  - Keep the adapter call signature compatible with the project batch contract. It may continue passing `hsi_pca`, but the refactored model will ignore it.
- Create: `tests/test_balanced_gmamba_hx_refactor.py`
  - Contains CPU-only smoke tests for the new Fourier module, gated fusion module, and full model forward pass.

---

### Task 1: Add CPU smoke tests for the refactored modules

**Files:**
- Create: `tests/test_balanced_gmamba_hx_refactor.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_balanced_gmamba_hx_refactor.py` with this content:

```python
import torch

from src.models.balanced_gmamba_hx.model import (
    BalancedGMambaHX,
    HSIGuidedGatedFusion,
    SpectralSpatialFourierEnhancement,
)


def test_spectral_spatial_fourier_enhancement_preserves_hsi_cube_shape_and_dtype():
    module = SpectralSpatialFourierEnhancement(cutoff=0.45)
    hsi = torch.randn(2, 1, 30, 11, 11)

    out = module(hsi)

    assert out.shape == hsi.shape
    assert out.dtype == hsi.dtype
    assert torch.isfinite(out).all()


def test_hsi_guided_gated_fusion_preserves_spatial_shape():
    module = HSIGuidedGatedFusion(embed_dim=16)
    hsi_spatial = torch.randn(2, 16, 11, 11)
    x_spatial = torch.randn(2, 16, 11, 11)

    out = module(hsi_spatial, x_spatial)

    assert out.shape == hsi_spatial.shape
    assert torch.isfinite(out).all()


def test_balanced_gmamba_hx_forward_without_token_or_aux_fft():
    model = BalancedGMambaHX(
        hsi_channels=30,
        pca_channels=30,
        aux_channels=1,
        num_classes=6,
        embed_dim=16,
        stem_dim=8,
        spec_tokens=8,
        stage_depths=(1, 1),
    )
    model.eval()
    hsi = torch.randn(2, 1, 30, 11, 11)
    hsi_pca = torch.randn(2, 30, 11, 11)
    aux = torch.randn(2, 1, 11, 11)

    with torch.no_grad():
        logits = model(hsi, hsi_pca, aux)

    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()
```

- [ ] **Step 2: Run the new tests and verify they fail before implementation**

Run:

```bash
python -m pytest tests/test_balanced_gmamba_hx_refactor.py -v
```

Expected result before implementation:

```text
ImportError: cannot import name 'HSIGuidedGatedFusion'
```

or:

```text
ImportError: cannot import name 'SpectralSpatialFourierEnhancement'
```

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_balanced_gmamba_hx_refactor.py
git commit -m "test: add balanced gmamba hx refactor smoke tests"
```

---

### Task 2: Add raw HSI 3D Fourier enhancement

**Files:**
- Modify: `src/models/balanced_gmamba_hx/model.py`
- Test: `tests/test_balanced_gmamba_hx_refactor.py`

- [ ] **Step 1: Add `SpectralSpatialFourierEnhancement` after `LearnableChannelFusion`**

In `src/models/balanced_gmamba_hx/model.py`, insert the following class immediately after `LearnableChannelFusion`:

```python
class SpectralSpatialFourierEnhancement(nn.Module):
    def __init__(self, cutoff=0.45):
        super().__init__()
        self.cutoff = cutoff
        self.low_weight = nn.Parameter(torch.zeros(1))
        self.high_weight = nn.Parameter(torch.zeros(1))

    def _frequency_masks(self, depth, height, width_freq, device, dtype):
        spectral_axis = torch.linspace(0.0, 1.0, steps=depth, device=device, dtype=dtype).view(depth, 1, 1)
        y_axis = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype).view(1, height, 1)
        x_axis = torch.linspace(0.0, 1.0, steps=width_freq, device=device, dtype=dtype).view(1, 1, width_freq)
        radius = torch.sqrt(spectral_axis * spectral_axis + y_axis * y_axis + x_axis * x_axis)
        low_mask = (radius <= self.cutoff).to(dtype)
        high_mask = 1.0 - low_mask
        return low_mask.view(1, 1, depth, height, width_freq), high_mask.view(1, 1, depth, height, width_freq)

    def forward(self, hsi):
        depth, height, width = hsi.shape[-3:]
        hsi_freq = torch.fft.rfftn(hsi, dim=(-3, -2, -1), norm="ortho")
        low_mask, high_mask = self._frequency_masks(
            depth,
            height,
            hsi_freq.shape[-1],
            hsi.device,
            hsi_freq.real.dtype,
        )
        low = torch.fft.irfftn(hsi_freq * low_mask, s=(depth, height, width), dim=(-3, -2, -1), norm="ortho")
        high = torch.fft.irfftn(hsi_freq * high_mask, s=(depth, height, width), dim=(-3, -2, -1), norm="ortho")
        low_weight = torch.sigmoid(self.low_weight).view(1, 1, 1, 1, 1)
        high_weight = torch.sigmoid(self.high_weight).view(1, 1, 1, 1, 1)
        return hsi + low_weight * low + high_weight * high
```

- [ ] **Step 2: Run the Fourier module test**

Run:

```bash
python -m pytest tests/test_balanced_gmamba_hx_refactor.py::test_spectral_spatial_fourier_enhancement_preserves_hsi_cube_shape_and_dtype -v
```

Expected result:

```text
PASSED
```

- [ ] **Step 3: Commit the Fourier module**

```bash
git add src/models/balanced_gmamba_hx/model.py
git commit -m "feat: add raw hsi 3d fourier enhancement"
```

---

### Task 3: Replace complex SGFF with HSI-guided gated fusion

**Files:**
- Modify: `src/models/balanced_gmamba_hx/model.py`
- Test: `tests/test_balanced_gmamba_hx_refactor.py`

- [ ] **Step 1: Add `HSIGuidedGatedFusion`**

In `src/models/balanced_gmamba_hx/model.py`, insert the following class after `DepthwiseResidual2d`:

```python
class HSIGuidedGatedFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.hsi_local = DepthwiseResidual2d(embed_dim)
        self.x_local = DepthwiseResidual2d(embed_dim)
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, hsi_spatial, x_spatial):
        hsi_local = self.hsi_local(hsi_spatial)
        x_local = self.x_local(x_spatial)
        gate = self.gate(torch.cat([hsi_local, x_local], dim=1))
        fused = hsi_local + gate * x_local
        return self.out(fused)
```

- [ ] **Step 2: Replace `BalancedGMambaBlock` with the simplified fusion block**

In `src/models/balanced_gmamba_hx/model.py`, replace the existing `BalancedGMambaBlock` class with:

```python
class BalancedGMambaBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fusion = HSIGuidedGatedFusion(embed_dim)
        self.s3m = SpatialSpectralStateModeling(embed_dim)

    def forward(self, hsi_spatial, x_spatial, spec_tokens):
        residual_spatial = hsi_spatial
        residual_tokens = spec_tokens
        fusion_spatial = self.fusion(hsi_spatial, x_spatial)
        return self.s3m(residual_spatial, fusion_spatial, spec_tokens, residual_tokens)
```

- [ ] **Step 3: Replace `BalancedGMambaStage` with the simplified stage signature**

In `src/models/balanced_gmamba_hx/model.py`, replace the existing `BalancedGMambaStage` class with:

```python
class BalancedGMambaStage(nn.Module):
    def __init__(self, embed_dim, depth):
        super().__init__()
        self.blocks = nn.ModuleList([BalancedGMambaBlock(embed_dim) for _ in range(depth)])

    def forward(self, hsi_spatial, x_spatial, spec_tokens):
        for block in self.blocks:
            hsi_spatial, spec_tokens = block(hsi_spatial, x_spatial, spec_tokens)
        return hsi_spatial, spec_tokens
```

- [ ] **Step 4: Run the gated fusion module test**

Run:

```bash
python -m pytest tests/test_balanced_gmamba_hx_refactor.py::test_hsi_guided_gated_fusion_preserves_spatial_shape -v
```

Expected result:

```text
PASSED
```

- [ ] **Step 5: Commit the gated fusion refactor**

```bash
git add src/models/balanced_gmamba_hx/model.py
git commit -m "feat: simplify hsi guided multimodal fusion"
```

---

### Task 4: Remove PCA token dependency from stages and wire 3D Fourier into the model forward path

**Files:**
- Modify: `src/models/balanced_gmamba_hx/model.py`
- Modify: `src/models/balanced_gmamba_hx/adapter.py`
- Test: `tests/test_balanced_gmamba_hx_refactor.py`

- [ ] **Step 1: Replace `DownsampleStage` so it no longer projects PCA tokens**

In `src/models/balanced_gmamba_hx/model.py`, replace the existing `DownsampleStage` class with:

```python
class DownsampleStage(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.spatial_down = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.x_down = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.token_spec = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))

    def forward(self, fusion_spatial, x_spatial, spec_tokens):
        return (
            self.spatial_down(fusion_spatial),
            self.x_down(x_spatial),
            self.token_spec(spec_tokens),
        )
```

- [ ] **Step 2: Replace `BalancedGMambaHX.__init__`**

In `src/models/balanced_gmamba_hx/model.py`, replace the existing `BalancedGMambaHX.__init__` method with:

```python
    def __init__(
        self,
        hsi_channels,
        pca_channels,
        aux_channels,
        num_classes,
        embed_dim=96,
        stem_dim=24,
        spec_tokens=16,
        stage_depths=(2, 2, 2),
    ):
        super().__init__()
        _ = hsi_channels
        _ = pca_channels
        self.hsi_fourier = SpectralSpatialFourierEnhancement()
        self.hsi_stem = HSIBalancedStem(embed_dim=embed_dim, stem_dim=stem_dim, spec_tokens=spec_tokens)
        self.x_stem = XBalancedStem(in_channels=aux_channels, embed_dim=embed_dim, token_count=spec_tokens)
        stage_dims = [embed_dim * (2 ** idx) for idx in range(len(stage_depths))]
        self.stages = nn.ModuleList([
            BalancedGMambaStage(embed_dim=dim, depth=depth)
            for dim, depth in zip(stage_dims, stage_depths)
        ])
        self.downsamples = nn.ModuleList([
            DownsampleStage(stage_dims[idx], stage_dims[idx + 1])
            for idx in range(len(stage_dims) - 1)
        ])
        self.head = BalancedFusionHead(embed_dim=stage_dims[-1], num_classes=num_classes)
```

- [ ] **Step 3: Replace `BalancedGMambaHX.forward`**

In `src/models/balanced_gmamba_hx/model.py`, replace the existing `BalancedGMambaHX.forward` method with:

```python
    def forward(self, hsi, hsi_pca, aux):
        _ = hsi_pca
        hsi = self.hsi_fourier(hsi)
        hsi_spatial, spec_tokens = self.hsi_stem(hsi)
        x_spatial, _ = self.x_stem(aux)

        fusion_spatial = hsi_spatial
        for idx, stage in enumerate(self.stages):
            fusion_spatial, spec_tokens = stage(fusion_spatial, x_spatial, spec_tokens)
            if idx < len(self.downsamples):
                fusion_spatial, x_spatial, spec_tokens = self.downsamples[idx](
                    fusion_spatial,
                    x_spatial,
                    spec_tokens,
                )
        return self.head(fusion_spatial, spec_tokens)
```

- [ ] **Step 4: Keep adapter compatibility and document ignored PCA input**

In `src/models/balanced_gmamba_hx/adapter.py`, replace `forward_train` and `forward_eval` with:

```python
def forward_train(bundle, batch):
    return bundle["net"](
        batch["hsi"],
        batch["hsi_pca"].squeeze(1),
        batch["aux"],
    )


def forward_eval(bundle, batch):
    return bundle["net"](
        batch["hsi"],
        batch["hsi_pca"].squeeze(1),
        batch["aux"],
    )
```

This is intentionally the same call shape as before. The model keeps `hsi_pca` in its public signature so trainer/evaluator integration does not change, but the refactored model ignores `hsi_pca` internally.

- [ ] **Step 5: Run the full model smoke test**

Run:

```bash
python -m pytest tests/test_balanced_gmamba_hx_refactor.py::test_balanced_gmamba_hx_forward_without_token_or_aux_fft -v
```

Expected result:

```text
PASSED
```

- [ ] **Step 6: Commit model wiring**

```bash
git add src/models/balanced_gmamba_hx/model.py src/models/balanced_gmamba_hx/adapter.py
git commit -m "feat: wire 3d fourier hsi refactor"
```

---

### Task 5: Remove inactive frequency-decomposition classes from `model.py`

**Files:**
- Modify: `src/models/balanced_gmamba_hx/model.py`
- Test: `tests/test_balanced_gmamba_hx_refactor.py`

- [ ] **Step 1: Delete unused classes from the active model file**

In `src/models/balanced_gmamba_hx/model.py`, remove these classes entirely after the new forward path passes tests:

```python
class PCABalancedTokenStem(nn.Module):
    ...

class HSISpectralGuide(nn.Module):
    ...

class DualModalSpatialFrequencyDecomposer(nn.Module):
    ...

class SpectralGuidedFrequencyFusion(nn.Module):
    ...
```

Keep these classes because they are still used:

```python
ConvBNGELU2d
ConvBNGELU3d
SimpleEncoder
DepthwiseResidual2d
HSIGuidedGatedFusion
LearnableChannelFusion
SpectralSpatialFourierEnhancement
FrequencyAwareResidualFusion
MambaTokenMixer
HSIBalancedStem
XBalancedStem
SpatialMamba2D
SpatialSpectralStateModeling
BalancedGMambaBlock
BalancedGMambaStage
DownsampleStage
BalancedFusionHead
BalancedGMambaHX
```

- [ ] **Step 2: Run the complete refactor test file**

Run:

```bash
python -m pytest tests/test_balanced_gmamba_hx_refactor.py -v
```

Expected result:

```text
3 passed
```

- [ ] **Step 3: Commit cleanup**

```bash
git add src/models/balanced_gmamba_hx/model.py
git commit -m "refactor: remove inactive token and auxiliary fft modules"
```

---

### Task 6: Run repository smoke verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Run the focused pytest smoke tests**

Run:

```bash
python -m pytest tests/test_balanced_gmamba_hx_refactor.py -v
```

Expected result:

```text
3 passed
```

- [ ] **Step 2: Run a direct adapter-level smoke forward**

Run:

```bash
python - <<'PY'
import torch
from src.models.balanced_gmamba_hx.model import BalancedGMambaHX

model = BalancedGMambaHX(
    hsi_channels=30,
    pca_channels=30,
    aux_channels=1,
    num_classes=6,
    embed_dim=16,
    stem_dim=8,
    spec_tokens=8,
    stage_depths=(1, 1),
)
model.eval()
with torch.no_grad():
    logits = model(
        torch.randn(2, 1, 30, 11, 11),
        torch.randn(2, 30, 11, 11),
        torch.randn(2, 1, 11, 11),
    )
print(tuple(logits.shape))
PY
```

Expected output:

```text
(2, 6)
```

- [ ] **Step 3: Run the project smoke training entrypoint if data and environment are available**

Run:

```bash
python scripts/strain.py
```

Expected result:

```text
The script starts training and completes the configured smoke run without a model shape/runtime error.
```

If this fails because external dataset files are missing, record the missing-data error exactly and keep the pytest/direct-forward verification as the completed local verification.

- [ ] **Step 4: Commit verification notes only if files changed**

If no files changed during verification, do not commit. If a verification artifact or note file was intentionally updated, run:

```bash
git add <changed-file>
git commit -m "docs: record balanced gmamba hx verification"
```

---

## Self-Review

**Spec coverage:**
- Raw HSI 3D Fourier modeling is implemented in Task 2 and wired in Task 4.
- Auxiliary modality FFT removal is implemented by replacing `SpectralGuidedFrequencyFusion` with `HSIGuidedGatedFusion` in Task 3 and deleting inactive FFT classes in Task 5.
- Abstract token FFT removal is implemented by deleting `HSISpectralGuide` and removing PCA token calibration from stages in Tasks 4-5.
- Lightweight HSI-guided auxiliary fusion is implemented in Task 3.
- Existing Mamba/state modeling continuity is preserved by keeping `SpatialSpectralStateModeling` and the classification head.

**Placeholder scan:** No TBD/TODO placeholders remain. Each implementation step includes concrete code or an exact command.

**Type consistency:** The refactored data flow is `hsi -> hsi_fourier -> hsi_stem -> fusion_spatial/spec_tokens`, `aux -> x_stem -> x_spatial`, `stage(fusion_spatial, x_spatial, spec_tokens)`, `downsample(fusion_spatial, x_spatial, spec_tokens)`, `head(fusion_spatial, spec_tokens)`. Function signatures are consistent across tasks.
