"""Command line entry for the incremental SfM pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from colmap.device import configure_device
from colmap.features import load_image_folder
from colmap.io import write_ply, write_reconstruction
from colmap.mapper import Mapper
from colmap.options import MapperOptions, VerificationOptions
from colmap.scene_graph import build_scene_graph, pairwise_matches_from_descriptors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Incremental Structure-from-Motion (Schönberger & Frahm, CVPR 2016).")
    parser.add_argument("images", type=Path, help="Directory of input photographs")
    parser.add_argument("--output", type=Path, default=Path("sparse"), help="Directory for the COLMAP text model")
    parser.add_argument("--focal", type=float, default=None, help="Focal length in pixels, shared by every image")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap used while debugging a subset")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="PyTorch device. auto uses CUDA when this build can see a GPU.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = configure_device(args.device)
    if device.type == "cuda":
        logging.info("Using %s (%s).", device, torch.cuda.get_device_name(device))
    else:
        logging.info("Using %s.", device)
    cameras, images = load_image_folder(args.images, focal=args.focal)
    if args.max_images is not None:
        keep = set(sorted(images)[: args.max_images])
        cameras = {key: value for key, value in cameras.items() if key in keep}
        images = {key: value for key, value in images.items() if key in keep}
    logging.info("Extracted features for %d images.", len(images))
    matches = pairwise_matches_from_descriptors(images)
    logging.info("Matched %d image pairs.", len(matches))
    graph = build_scene_graph(cameras, images, matches, VerificationOptions())
    logging.info("Verified %d pairs.", len(graph.pairs))
    reconstruction = Mapper(graph, MapperOptions()).reconstruct()
    if reconstruction is None:
        raise SystemExit("Reconstruction failed: no initial pair produced a model.")
    write_reconstruction(reconstruction, args.output)
    write_ply(reconstruction, args.output / "points.ply")
    logging.info(
        "Registered %d / %d images and %d points (mean reprojection %.3f px). Wrote %s",
        reconstruction.num_images,
        len(images),
        reconstruction.num_points,
        reconstruction.mean_reprojection_error(),
        args.output,
    )


if __name__ == "__main__":
    main()
