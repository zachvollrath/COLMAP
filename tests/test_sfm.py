"""Tests for the CVPR 2016 incremental SfM components and the full mapper."""

from __future__ import annotations

import unittest

import numpy as np

from colmap.bundle import BACamera, BAImage, BAObservation, bundle_adjust, group_redundant_views, overlap_ratio
from colmap.geometry import (
    camera_center,
    dlt_pnp,
    fundamental_from_points,
    project_points,
    sampson_error,
    umeyama,
)
from colmap.mapper import Mapper
from colmap.next_view import NextBestViewScore
from colmap.options import MapperOptions, VerificationOptions
from colmap.scene_graph import GENERAL, Camera, build_scene_graph
from colmap.triangulation import ViewMeasurement, triangulate_track
from tests.synthetic import correspondences, look_at, make_scene


class GeometryTests(unittest.TestCase):
    def test_pnp_recovers_pose(self) -> None:
        rng = np.random.default_rng(1)
        world = rng.uniform(-1.0, 1.0, size=(20, 3))
        center = np.array([2.5, -0.4, 3.0])
        rotation, translation = look_at(center)
        focal, width, height = 600.0, 640, 480
        pixels, depth = project_points(world, rotation, translation, focal, width, height)
        self.assertTrue(np.all(depth > 0))
        normalized = np.column_stack([(pixels[:, 0] - width / 2) / focal, (pixels[:, 1] - height / 2) / focal])
        estimated_R, estimated_t = dlt_pnp(world, normalized)
        self.assertIsNotNone(estimated_R)
        uv, _ = project_points(world, estimated_R, estimated_t, focal, width, height)
        self.assertLess(np.mean(np.linalg.norm(uv - pixels, axis=1)), 1e-4)

    def test_fundamental_inliers_are_small(self) -> None:
        rng = np.random.default_rng(2)
        world = rng.uniform(-0.4, 0.4, size=(30, 3))
        r1, t1 = look_at(np.array([4.0, 0.2, 0.3]))
        r2, t2 = look_at(np.array([0.2, 0.1, 4.0]))
        focal, width, height = 700.0, 640, 480
        p1, _ = project_points(world, r1, t1, focal, width, height)
        p2, _ = project_points(world, r2, t2, focal, width, height)
        F = fundamental_from_points(p1, p2)
        error = sampson_error(F, p1, p2)
        self.assertLess(np.median(error), 1e-6)


class NextViewTests(unittest.TestCase):
    def test_uniform_points_outrank_a_cluster(self) -> None:
        width, height, levels = 640, 480, 6
        uniform = NextBestViewScore(levels, width, height)
        clustered = NextBestViewScore(levels, width, height)
        rng = np.random.default_rng(3)
        for _ in range(80):
            uniform.add(float(rng.uniform(0, width)), float(rng.uniform(0, height)))
            clustered.add(float(rng.normal(80, 12)), float(rng.normal(60, 12)))
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
            np.array([4.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 4.0]),
            np.array([-4.0, 0.2, 0.0]),
            np.array([0.0, 0.3, -4.0]),
            np.array([3.0, 1.0, 3.0]),
            np.array([-3.0, -0.5, 3.0]),
        ]
        point_a = np.array([0.1, 0.0, 0.0])
        point_b = np.array([-0.2, 0.3, 0.4])
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
            rng=np.random.default_rng(4),
        )
        self.assertGreaterEqual(len(recovered), 2)
        found = np.stack([point for point, _ in recovered])
        self.assertTrue(np.min(np.linalg.norm(found - point_a, axis=1)) < 1e-3)
        self.assertTrue(np.min(np.linalg.norm(found - point_b, axis=1)) < 1e-3)


class GroupingTests(unittest.TestCase):
    def test_overlap_groups_redundant_cameras(self) -> None:
        from colmap.options import MapperOptions

        images = []
        visibility = {}
        for image_id in range(1, 7):
            rotation, translation = look_at(np.array([4.0, 0.0, 0.2 * image_id]))
            images.append(BAImage(image_id, image_id, rotation, translation, 640, 480))
            visibility[image_id] = set(range(30))
        # A second cluster looks the other way and shares no points.
        for image_id in range(7, 11):
            rotation, translation = look_at(np.array([-4.0, 0.0, 0.2 * image_id]))
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
        rng = np.random.default_rng(0)
        world = {i: point for i, point in enumerate(rng.uniform(-0.4, 0.4, size=(20, 3)))}
        focal, width, height = 700.0, 640, 480
        images = []
        observations = []
        for image_id, center in enumerate(
            [np.array([4.0, 0.1, 0.2]), np.array([3.6, 0.2, 0.8]), np.array([3.7, -0.1, 1.0]), np.array([0.2, 0.2, 4.0])],
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
            {point_id: point.copy() for point_id, point in world.items()},
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
            errors.append(np.linalg.norm(pixels[0] - obs.xy))
        self.assertLess(float(np.mean(errors)), 1e-4)


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
        estimated = np.asarray(estimated)
        truth = np.asarray(truth)
        scale, rotation, translation = umeyama(estimated, truth)
        aligned = scale * (estimated @ rotation.T) + translation
        median_error = float(np.median(np.linalg.norm(aligned - truth, axis=1)))
        self.assertLess(median_error, 0.15)

        # Points, up to the same similarity, should land on the synthetic scene.
        recovered = np.stack([point.xyz for point in reconstruction.points.values()])
        mapped = scale * (recovered @ rotation.T) + translation
        distances = np.min(np.linalg.norm(mapped[:, None, :] - world[None, :, :], axis=2), axis=1)
        self.assertLess(float(np.median(distances)), 0.05)


if __name__ == "__main__":
    unittest.main()
