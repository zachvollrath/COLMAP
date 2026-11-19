"""Run the mapper on a synthetic calibrated scene and print the pose error."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colmap.geometry import camera_center, umeyama
from colmap.io import write_ply, write_reconstruction
from colmap.mapper import Mapper
from colmap.options import MapperOptions, VerificationOptions
from colmap.scene_graph import build_scene_graph
from tests.synthetic import correspondences, make_scene


def main() -> None:
    world, cameras, images, true_poses = make_scene()
    matches = correspondences(images, min_matches=15)
    graph = build_scene_graph(
        cameras,
        images,
        matches,
        VerificationOptions(min_num_inliers=15),
        seed=0,
    )
    options = MapperOptions(
        init_min_num_inliers=15,
        abs_pose_min_num_inliers=12,
        ba_global_images_ratio=1.5,
        ba_max_refinement_iterations=2,
    )
    reconstruction = Mapper(graph, options).reconstruct()
    if reconstruction is None:
        raise SystemExit("reconstruction failed")
    output = ROOT / "workspace" / "synthetic"
    write_reconstruction(reconstruction, output)
    write_ply(reconstruction, output / "points.ply")
    estimated = []
    truth = []
    for image_id in reconstruction.registration_order:
        rotation, translation = reconstruction.poses[image_id]
        estimated.append(camera_center(rotation, translation))
        truth.append(true_poses[image_id][2])
    scale, rotation, translation = umeyama(np.asarray(estimated), np.asarray(truth))
    aligned = scale * (np.asarray(estimated) @ rotation.T) + translation
    error = np.linalg.norm(aligned - np.asarray(truth), axis=1)
    print(
        f"registered {reconstruction.num_images} images, {reconstruction.num_points} points, "
        f"mean reprojection {reconstruction.mean_reprojection_error():.3f} px, "
        f"median camera error {np.median(error):.4f}"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
