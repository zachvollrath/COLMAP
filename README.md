# COLMAP

Python implementation of the incremental Structure-from-Motion pipeline from
Johannes L. Schönberger and Jan-Michael Frahm, **Structure-from-Motion Revisited**,
CVPR 2016 ([paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)).

This is an independent implementation of that algorithm. It is not the official
C++ COLMAP system and it does not include vocabulary-tree matching, GPU feature
extraction, or dense multi-view stereo.

## Pipeline

The code follows the paper's two stages.

**Correspondence search (Section 2.1).** SIFT features, exhaustive Lowe-ratio
matching, then multi-model geometric verification.

**Incremental reconstruction (Sections 4.1–4.5).**

| Piece | What it does |
| --- | --- |
| Scene graph augmentation | Estimates F, H, and E, labels each pair general / planar / panoramic, and drops watermark-style border similarities. |
| Initialization | Seeds the model from a non-panoramic, preferably calibrated pair with a wide baseline and many shared neighbors. |
| Next best view | Ranks unregistered images with the multi-resolution occupancy score (`L` levels, `K_l = 2^l`, weight `K_l^2`). |
| Registration | RANSAC Perspective-n-Point against already triangulated points, then nonlinear pose refinement. |
| Triangulation | Recursive unique-sample RANSAC on feature tracks. Panoramic pairs are not triangulated. A consensus smaller than three ends the recursion. |
| Bundle adjustment | Cauchy reprojection loss on a simple camera (one focal length, one radial coefficient, principal point fixed at the image center). Levenberg-Marquardt uses PyTorch forward-mode Jacobians. Local BA runs after every registration. Global BA runs when the model has grown by the configured ratio, with re-triangulation and filtering before and after, iterated until those updates shrink. |
| Redundant view mining | Unaffected cameras whose visibility sets have Jaccard overlap above `V` are collapsed into one rigid group for global BA. |

The paper leaves several thresholds symbolic. Defaults match the numbers given
in its experiments (`α = 2°`, `t = 8 px`, `ε_0 = 0.03`, `η = 0.999`, `L = 6`)
and the public COLMAP defaults for the remaining tests (`N_F = 15`, homography
and essential inlier ratios `0.8`, initialization angle `16°`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install opencv-python-headless   # only needed to run on image files
```

Install a CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/) instead of the CPU wheel when reconstructions should run on a GPU.

Reconstruct a folder of photographs:

```bash
colmap-sfm /path/to/images --output sparse --device auto
```

`--device` is `auto`, `cpu`, or `cuda`. `auto` uses CUDA when the installed PyTorch can see a GPU, and CPU otherwise.

`--focal` sets a shared focal length in pixels. Without it, each image uses the
prior `1.2 * max(width, height)`. The text model in `sparse/` uses COLMAP's
`SIMPLE_RADIAL` camera convention, and `sparse/points.ply` is the point cloud.

Run the synthetic check (no photographs required):

```bash
python examples/synthetic_scene.py
python -m unittest discover -s tests
```

## Layout

- `colmap/scene_graph.py` — matching and Section 4.1 verification
- `colmap/next_view.py` — Section 4.2 score
- `colmap/triangulation.py` — Section 4.3
- `colmap/bundle.py` — Sections 4.4 and 4.5
- `colmap/mapper.py` — incremental loop
- `colmap/options.py` — thresholds
