"""Tests for the CVPR 2016 incremental SfM components and the full mapper."""

from __future__ import annotations

import unittest

import torch

from colmap.bundle import BACamera, BAImage, BAObservation, bundle_adjust, group_redundant_views, overlap_ratio
from colmap.device import configure_device, generator, get_device
from colmap.geometry import (
    DTYPE,
    camera_center,
    dlt_pnp,
    fundamental_from_points,
    project_points,
    projection_matrix,
    sample_is_planar,
    sampson_error,
    triangulate_dlt,
    umeyama,
)
from colmap.mapper import Mapper
from colmap.ransac import adaptive_iterations, ransac
from colmap.next_view import NextBestViewScore
from colmap.options import MapperOptions, VerificationOptions
from colmap.scene_graph import GENERAL, Camera, ImageNode, build_scene_graph, verify_pair
from colmap.triangulation import ViewMeasurement, triangulate_track
from tests.synthetic import correspondences, look_at, make_scene


class GeometryTests(unittest.TestCase):
    def test_pnp_recovers_pose(self) -> None:
        generator = torch.Generator().manual_seed(1)
        world = torch.rand((20, 3), generator=generator, dtype=DTYPE) * 2 - 1
        center = torch.tensor([2.5, -0.4, 3.0], dtype=DTYPE)
        rotation, translation = look_at(center)
        focal, width, height = 600.0, 640, 480
        pixels, depth = project_points(world, rotation, translation, focal, width, height)
        self.assertTrue(torch.all(depth > 0))
        normalized = torch.column_stack([(pixels[:, 0] - width / 2) / focal, (pixels[:, 1] - height / 2) / focal])
        estimated_R, estimated_t = dlt_pnp(world, normalized)
        self.assertIsNotNone(estimated_R)
        uv, _ = project_points(world, estimated_R, estimated_t, focal, width, height)
        self.assertLess(torch.linalg.norm(uv - pixels, dim=1).mean().item(), 1e-4)

    def test_fundamental_inliers_are_small(self) -> None:
        generator = torch.Generator().manual_seed(2)
        world = torch.rand((30, 3), generator=generator, dtype=DTYPE) * 0.8 - 0.4
        r1, t1 = look_at(torch.tensor([4.0, 0.2, 0.3], dtype=DTYPE))
        r2, t2 = look_at(torch.tensor([0.2, 0.1, 4.0], dtype=DTYPE))
        focal, width, height = 700.0, 640, 480
        p1, _ = project_points(world, r1, t1, focal, width, height)
        p2, _ = project_points(world, r2, t2, focal, width, height)
        F = fundamental_from_points(p1, p2)
        error = sampson_error(F, p1, p2)
        self.assertLess(error.median().item(), 1e-6)

    def test_adaptive_iterations_survive_a_tiny_inlier_ratio(self) -> None:
        # ε^8 underflows, which used to make log(1 - ε^s) zero.
        self.assertEqual(adaptive_iterations(0.999, 0.01, 8, 10000, trial=3), 10000)
        self.assertEqual(adaptive_iterations(0.999, 1.0, 8, 10000, trial=4), 4)

        def fit(_sample):
            return torch.zeros(3)

        def residuals(_model):
            error = torch.ones(500)
            error[:2] = 0.0
            return error

        model, _inliers = ransac(
            500,
            8,
            fit,
            residuals,
            threshold=0.5,
            rng=torch.Generator().manual_seed(0),
            confidence=0.999,
            max_iters=20,
            min_inliers=1,
        )
        self.assertIsNone(model)

    def test_device_choice_places_tensors_and_generators(self) -> None:
        try:
            cpu = configure_device("cpu")
            self.assertEqual(cpu.type, "cpu")
            self.assertEqual(get_device().type, "cpu")
            self.assertEqual(torch.zeros(3).device.type, "cpu")
            self.assertEqual(generator(2).device.type, "cpu")
            if torch.cuda.is_available():
                gpu = configure_device("cuda")
                self.assertEqual(gpu.type, "cuda")
                self.assertEqual(torch.zeros(3).device.type, "cuda")
                self.assertEqual(generator(2).device.type, "cuda")
            else:
                with self.assertRaises(RuntimeError):
                    configure_device("cuda")
            auto = configure_device("auto")
            self.assertEqual(auto.type, "cuda" if torch.cuda.is_available() else "cpu")
            with self.assertRaises(ValueError):
                configure_device("tpu")
        finally:
            configure_device("cpu")

    def test_dominant_plane_does_not_yield_a_bogus_pose(self) -> None:
        generator = torch.Generator().manual_seed(3)
        plane = torch.rand((80, 3), generator=generator, dtype=DTYPE) - 0.5
        plane[:, 2] = 0.0
        volume = torch.rand((16, 3), generator=generator, dtype=DTYPE) - 0.5
        world = torch.cat([plane, volume], dim=0)
        focal, width, height = 700.0, 640, 480
        r1, t1 = look_at(torch.tensor([0.15, -0.2, 4.0], dtype=DTYPE))
        r2, t2 = look_at(torch.tensor([1.1, 0.25, 3.6], dtype=DTYPE))
        p1, depth1 = project_points(world, r1, t1, focal, width, height)
        p2, depth2 = project_points(world, r2, t2, focal, width, height)
        self.assertTrue(torch.all(depth1 > 0) and torch.all(depth2 > 0))
        plane_threshold = 4.0**2
        self.assertTrue(sample_is_planar(p1[:8], p2[:8], plane_threshold))
        self.assertFalse(sample_is_planar(p1[-8:], p2[-8:], plane_threshold))

        images = {
            1: ImageNode(1, "a", 1, p1),
            2: ImageNode(2, "b", 2, p2),
        }
        cameras = {
            1: Camera(1, width, height, focal),
            2: Camera(2, width, height, focal),
        }
        matches = torch.stack([torch.arange(len(world)), torch.arange(len(world))], dim=1)
        geometry = verify_pair(
            images[1],
            images[2],
            cameras[1],
            cameras[2],
            matches,
            VerificationOptions(min_num_inliers=15, ransac_max_iters=2000),
            torch.Generator().manual_seed(0),
        )
        self.assertIsNotNone(geometry)
        self.assertIsNotNone(geometry.R)
        identity = torch.eye(3, dtype=DTYPE)
        zero = torch.zeros(3, dtype=DTYPE)
        P1 = projection_matrix(identity, zero, focal, width, height)
        P2 = projection_matrix(geometry.R, geometry.t, focal, width, height)
        errors = []
        for feature_a, feature_b in geometry.matches[:40]:
            point = triangulate_dlt([(P1, p1[int(feature_a)]), (P2, p2[int(feature_b)])])
            self.assertIsNotNone(point)
            uv, _ = project_points(point.reshape(1, 3), geometry.R, geometry.t, focal, width, height)
            errors.append(torch.linalg.norm(uv[0] - p2[int(feature_b)]))
        self.assertLess(torch.stack(errors).median().item(), 1.0)


class NextViewTests(unittest.TestCase):
    def test_uniform_points_outrank_a_cluster(self) -> None:
        width, height, levels = 640, 480, 6
        uniform = NextBestViewScore(levels, width, height)
        clustered = NextBestViewScore(levels, width, height)
        generator = torch.Generator().manual_seed(3)
        for _ in range(80):
            uniform.add(float(torch.rand((), generator=generator) * width), float(torch.rand((), generator=generator) * height))
            clustered.add(float(torch.randn((), generator=generator) * 12 + 80), float(torch.randn((), generator=generator) * 12 + 60))
        self.assertGreater(uniform.score, clustered.score)
        self.assertEqual(uniform.num_points, clustered.num_points)

    def test_occupied_cells_are_not_counted_twice(self) -> None:
        score = NextBestViewScore(3, 100, 100)
        score.add(10, 10)
        once = score.score
        score.add(12, 11)
        self.assertEqual(score.score, once)


class TriangulationTests(unittest.TestCase):
    def test_recursive_ransac_splits_a_merged_track(self) -> None:
        focal, width, height = 700.0, 640, 480
        camera = Camera(1, width, height, focal)
        centers = [
            torch.tensor(center, dtype=DTYPE)
            for center in (
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
                [-4.0, 0.2, 0.0],
                [0.0, 0.3, -4.0],
                [3.0, 1.0, 3.0],
                [-3.0, -0.5, 3.0],
            )
        ]
        point_a = torch.tensor([0.1, 0.0, 0.0], dtype=DTYPE)
        point_b = torch.tensor([-0.2, 0.3, 0.4], dtype=DTYPE)
        measurements = []
        for index, center in enumerate(centers):
            rotation, translation = look_at(center)
            target = point_a if index < 3 else point_b
            pixels, depth = project_points(target.reshape(1, 3), rotation, translation, focal, width, height)
            self.assertGreater(float(depth[0]), 0)
            measurements.append(
                ViewMeasurement(index + 1, 0, pixels[0], rotation, translation, camera, set())
            )
        recovered = triangulate_track(
            measurements,
            min_angle_deg=2.0,
            max_error_px=8.0,
            confidence=0.999,
            min_inlier_ratio=0.03,
            min_consensus=3,
            rng=torch.Generator().manual_seed(4),
        )
        self.assertGreaterEqual(len(recovered), 2)
        found = torch.stack([point for point, _ in recovered])
        self.assertTrue(torch.linalg.norm(found - point_a, dim=1).min().item() < 1e-3)
        self.assertTrue(torch.linalg.norm(found - point_b, dim=1).min().item() < 1e-3)


class GroupingTests(unittest.TestCase):
    def test_overlap_groups_redundant_cameras(self) -> None:
        from colmap.options import MapperOptions

        images = []
        visibility = {}
        for image_id in range(1, 7):
            rotation, translation = look_at(torch.tensor([4.0, 0.0, 0.2 * image_id], dtype=DTYPE))
            images.append(BAImage(image_id, image_id, rotation, translation, 640, 480))
            visibility[image_id] = set(range(30))
        # A second cluster looks the other way and shares no points.
        for image_id in range(7, 11):
            rotation, translation = look_at(torch.tensor([-4.0, 0.0, 0.2 * image_id], dtype=DTYPE))
            images.append(BAImage(image_id, image_id, rotation, translation, 640, 480))
            visibility[image_id] = set(range(100, 130))
        self.assertGreater(overlap_ratio(visibility[1], visibility[2]), 0.9)
        self.assertEqual(overlap_ratio(visibility[1], visibility[7]), 0.0)
        options = MapperOptions(group_min_overlap=0.4, group_max_size=10, group_max_view_angle_deg=60)
        groups = group_redundant_views(images, visibility, affected={1}, options=options)
        flat = [image_id for group in groups for image_id in group]
        self.assertCountEqual(flat, list(range(1, 11)))
        self.assertIn([1], groups)
        mixed = [group for group in groups if set(group) & set(range(1, 7)) and set(group) & set(range(7, 11))]
        self.assertEqual(mixed, [])

    def test_grouped_bundle_preserves_a_consistent_scene(self) -> None:
        generator = torch.Generator().manual_seed(0)
        cloud = torch.rand((20, 3), generator=generator, dtype=DTYPE) * 0.8 - 0.4
        world = {i: point for i, point in enumerate(cloud)}
        focal, width, height = 700.0, 640, 480
        images = []
        observations = []
        for image_id, center in enumerate(
            [
                torch.tensor([4.0, 0.1, 0.2], dtype=DTYPE),
                torch.tensor([3.6, 0.2, 0.8], dtype=DTYPE),
                torch.tensor([3.7, -0.1, 1.0], dtype=DTYPE),
                torch.tensor([0.2, 0.2, 4.0], dtype=DTYPE),
            ],
            start=1,
        ):
            rotation, translation = look_at(center)
            images.append(BAImage(image_id, image_id, rotation, translation, width, height))
            for point_id, point in world.items():
                pixels, depth = project_points(point.reshape(1, 3), rotation, translation, focal, width, height)
                if depth[0] > 0:
                    observations.append(BAObservation(image_id, point_id, pixels[0]))
        cameras = [BACamera(image_id, focal, 0.0, focal) for image_id in range(1, 5)]
        result = bundle_adjust(
            images,
            cameras,
            {point_id: point.clone() for point_id, point in world.items()},
            observations,
            MapperOptions(),
            groups=[[1], [2, 3], [4]],
            refine_intrinsics=False,
            max_iterations=12,
            anchor_image=1,
            fixed_images={1},
        )
        self.assertTrue(result.success)
        errors = []
        for obs in observations:
            pixels, _ = project_points(
                result.points[obs.point_id].reshape(1, 3),
                result.rotations[obs.image_id],
                result.translations[obs.image_id],
                focal,
                width,
                height,
            )
            errors.append(torch.linalg.norm(pixels[0] - obs.xy))
        self.assertLess(torch.stack(errors).mean().item(), 1e-4)


class ReconstructionTests(unittest.TestCase):
    def test_incremental_scene_recovers_cameras_and_points(self) -> None:
        world, cameras, images, true_poses = make_scene(n_cameras=5, n_points=36, noise_px=0.2, seed=5)
        matches = correspondences(images, min_matches=20)
        verification = VerificationOptions(min_num_inliers=15, ransac_max_iters=2000)
        graph = build_scene_graph(cameras, images, matches, verification, seed=5)
        self.assertGreaterEqual(len(graph.pairs), 4)
        self.assertTrue(any(pair.model == GENERAL and pair.calibrated for pair in graph.pairs))
        options = MapperOptions(
            init_min_num_inliers=20,
            init_min_tri_angle_deg=8.0,
            abs_pose_min_num_inliers=12,
            abs_pose_min_inlier_ratio=0.2,
            ba_local_max_iterations=8,
            ba_global_max_iterations=12,
            ba_global_images_ratio=1.6,
            ba_global_points_ratio=1.4,
            ba_max_refinement_iterations=2,
            refine_focal_length=True,
            refine_extra_params=True,
            group_redundant_views=False,
            random_seed=5,
        )
        reconstruction = Mapper(graph, options).reconstruct()
        self.assertIsNotNone(reconstruction)
        self.assertGreaterEqual(reconstruction.num_images, 4)
        self.assertGreaterEqual(reconstruction.num_points, 20)
        self.assertLess(reconstruction.mean_reprojection_error(), 1.5)

        estimated = []
        truth = []
        for image_id in reconstruction.registration_order:
            R, t = reconstruction.poses[image_id]
            estimated.append(camera_center(R, t))
            truth.append(true_poses[image_id][2])
        estimated = torch.stack(estimated)
        truth = torch.stack(truth)
        scale, rotation, translation = umeyama(estimated, truth)
        aligned = scale * (estimated @ rotation.T) + translation
        median_error = torch.linalg.norm(aligned - truth, dim=1).median().item()
        self.assertLess(median_error, 0.15)

        recovered = torch.stack([point.xyz for point in reconstruction.points.values()])
        mapped = scale * (recovered @ rotation.T) + translation
        distances = torch.linalg.norm(mapped[:, None, :] - world[None, :, :], dim=2).min(dim=1).values
        self.assertLess(distances.median().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
