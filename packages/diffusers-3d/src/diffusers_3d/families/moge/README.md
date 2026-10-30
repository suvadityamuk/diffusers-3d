# MoGe and MoGe-2

Microsoft MoGe monocular geometry: from one RGB image, an affine-invariant point map, a validity mask, the camera
intrinsics and depth. Two variants, each matching the upstream code its consumer runs. MIT license.

| Class | Upstream | Weights | Used by |
| --- | --- | --- | --- |
| `MoGeModel` | `moge.model.v1.MoGeModel` at `a8c37341bc0325ca99b9d57981cc3bb2bd3e255b` | `Ruicheng/moge-vitl` at `ad326bfb61facd6c52b5a825bc1e34d7c97d9672` | SAM 3D Objects (pins that commit) |
| `MoGe2Model` | `moge.model.v2.MoGeModel` at `74fbce054ebed49800de42d0ad0e83495065719a` | `Ruicheng/moge-2-vitl` at `39c4d5e957afe587e04eec59dc2bcc3be5ecd968` | Pixal3D (installs unpinned HEAD; 74fbce0 at Pixal3D `f7cf384`) |

```python
import math

from diffusers_3d.families.moge import MoGe2Model, convert_moge_checkpoint

model = convert_moge_checkpoint("Ruicheng/moge-2-vitl", "moge-2-vitl")  # once; then MoGe2Model.from_pretrained
output = model.infer(image)  # image: (3, H, W) or (B, 3, H, W) RGB in [0, 1]
fov_x = 2 * math.atan(1 / (2 * output["intrinsics"][0, 0]))
```

`infer` takes upstream's arguments and defaults: `fov_x` (degrees; estimated when None), `resolution_level` (0-9),
`num_tokens`, `apply_mask`, `force_projection`, `use_fp16=True`. It returns:

- `points` (..., H, W, 3): OpenCV camera space, x right, y down, z forward. Masked pixels are `inf` unless
  `apply_mask=False`. MoGe-2 multiplies points and depth by its predicted metric scale (meters).
- `depth` (..., H, W): the z coordinate.
- `intrinsics` (..., 3, 3): normalized by image size: `fx = fx_pixels / W`, `fy = fy_pixels / H`, principal point
  `(0.5, 0.5)`.
- `mask` (..., H, W) bool; `MoGeModel` also returns `mask_prob`.

## Matching upstream

- The DINOv2 ViT-L/14 backbone is the stock Transformers `Dinov2Model`. Only the position-embedding interpolation is
  replaced: torch-hub DINOv2 resizes with `scale_factor=(grid + 0.1) / 37`, Transformers with `size=grid`, and the
  two differ on every grid MoGe picks.
- Focal and shift come from SciPy's `least_squares(method="lm", ftol=1e-3)` upstream. Its MINPACK solver is ported
  for the single unknown, so SciPy is not needed. It follows SciPy 1.16 and later (`x_scale="jac"`), which a fresh
  install of either consumer resolves to. SciPy 1.15 and earlier scaled differently and can stop elsewhere.
- `use_fp16=True` autocasts the network to float16 on the image's device, as upstream does, also on CPU. There,
  upstream MoGe-2 and MoGe with `force_projection=True` fail with a dtype error because CPU autocast cannot run
  float32. This port fails the same way, so pass `use_fp16=False` off CUDA.

## Conversion

```bash
python -m diffusers_3d.families.moge.conversion Ruicheng/moge-vitl moge-vitl
python -m diffusers_3d.families.moge.conversion Ruicheng/moge-2-vitl moge-2-vitl
```

`model.pt` is a pickle. It is downloaded at the pinned revision into a fresh directory and read with
`torch.load(weights_only=True)`. Tensors stay float32, the precision upstream loads.
