"""Incremental reconstruction driver, Sections 2.2 and 4.

The scene graph is seeded from a calibrated, non-panoramic pair with a wide
baseline and redundant overlap. Images are then registered in next-best-view
order, new tracks are triangulated, and local bundle adjustment runs after
every registration. Global bundle adjustment, re-triangulation, and filtering
repeat whenever the model has grown enough that another full refinement pays
for itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from colmap.bundle import (
    BACamera,
    BAImage,
    BAObservation,
    bundle_adjust,
    group_redundant_views,
)
from colmap.device import generator
from colmap.geometry import (
    DTYPE,
    as_tensor,
    camera_center,
    dlt_pnp,
    pixels_to_normalized,
    project_points,
    refine_pose,
    triangulation_angle_deg,
)
from colmap.next_view import NextBestViewScore, rank_candidates
from colmap.options import MapperOptions
from colmap.ransac import ransac
from colmap.scene_graph import PANORAMIC, WATERMARK, Camera, ImageNode, SceneGraph
from colmap.triangulation import (
    ViewMeasurement,
    panoramic_pairs,
    triangulate_track,
    two_view_points,
)

logger = logging.getLogger("colmap")


@dataclass
class Observation:
    image_id: int
    feature_idx: int


@dataclass
class Point3D:
    xyz: torch.Tensor
    observations: list[Observation] = field(default_factory=list)
    color: torch.Tensor = field(default_factory=lambda: torch.tensor([200, 200, 200], dtype=torch.uint8))
    error: float = 0.0


@dataclass
class Reconstruction:
    cameras: dict[int, Camera]
    images: dict[int, ImageNode]
    poses: dict[int, tuple[torch.Tensor, torch.Tensor]]
    points: dict[int, Point3D]
    registration_order: list[int]

    @property
    def num_images(self) -> int:
        return len(self.poses)

    @property
    def num_points(self) -> int:
        return len(self.points)

    def mean_reprojection_error(self) -> float:
        if not self.points:
            return 0.0
        errors = [point.error for point in self.points.values()]
        return float(torch.tensor(errors, dtype=DTYPE).mean())


class Mapper:
    def __init__(self, graph: SceneGraph, options: MapperOptions | None = None) -> None:
        self.graph = graph
        self.options = options or MapperOptions()
        self.rng = generator(int(self.options.random_seed))
        self.poses: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.points: dict[int, Point3D] = {}
        self.registration_order: list[int] = []
        self._next_point_id = 1
        self._feature_to_track: dict[tuple[int, int], int] = {}
        self._tracks: dict[int, list[tuple[int, int]]] = {}
        self._track_points: dict[int, list[int]] = {}
        self._obs_to_point: dict[tuple[int, int], int] = {}
        self._scores: dict[int, NextBestViewScore] = {}
        self._trials: dict[int, int] = {}
        self._exhausted: set[int] = set()
        self._panoramic: dict[int, set[int]] = {}
        self._last_global_images = 0
        self._last_global_points = 0

    def reconstruct(self) -> Reconstruction | None:
        self._build_tracks()
        self._panoramic = panoramic_pairs(self.graph.pairs)
        seeds = self._select_seeds()
        if not seeds:
            logger.warning("No suitable initial pair.")
            return None
        if not any(self._try_initialize(seed) for seed in seeds):
            logger.warning("Reconstruction failed: no initial pair produced a model.")
            return None
        self._local_bundle(self.registration_order[-1])
        self._filter_observations()
        self._refresh_scores()
        while True:
            image_id = self._next_image()
            if image_id is None:
                break
            if not self._register(image_id):
                self._trials[image_id] = self._trials.get(image_id, 0) + 1
                if self._trials[image_id] >= self.options.max_registration_trials:
                    self._exhausted.add(image_id)
                continue
            self._triangulate_image(image_id)
            self._local_bundle(image_id)
            self._filter_observations()
            self._refresh_scores()
            if self._global_ba_due():
                self._iterative_global_refinement()
        if len(self.poses) >= 2 and self.points:
            self._iterative_global_refinement()
        self._update_point_errors()
        return self._result()

    def _result(self) -> Reconstruction:
        return Reconstruction(
            cameras=self.graph.cameras,
            images=self.graph.images,
            poses={image_id: (R.clone(), t.clone()) for image_id, (R, t) in self.poses.items()},
            points=self.points,
            registration_order=list(self.registration_order),
        )

    def _build_tracks(self) -> None:
        parent: dict[tuple[int, int], tuple[int, int]] = {}

        def add(node: tuple[int, int]) -> None:
            parent.setdefault(node, node)

        def find(node: tuple[int, int]) -> tuple[int, int]:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        images_of: dict[tuple[int, int], set[int]] = {}
        members: dict[tuple[int, int], list[tuple[int, int]]] = {}

        def root_state(node: tuple[int, int]) -> None:
            if node not in images_of:
                images_of[node] = {node[0]}
                members[node] = [node]

        for pair in self.graph.pairs:
            if pair.model == WATERMARK:
                continue
            for feature_a, feature_b in pair.matches:
                a = (pair.image_id1, int(feature_a))
                b = (pair.image_id2, int(feature_b))
                add(a)
                add(b)
                ra, rb = find(a), find(b)
                root_state(ra)
                root_state(rb)
                if ra == rb:
                    continue
                if images_of[ra] & images_of[rb]:
                    continue
                if len(members[ra]) < len(members[rb]):
                    ra, rb = rb, ra
                parent[rb] = ra
                images_of[ra] |= images_of[rb]
                members[ra].extend(members[rb])
                del images_of[rb]
                del members[rb]

        next_id = 0
        for root, obs in members.items():
            if find(root) != root:
                continue
            self._tracks[next_id] = list(obs)
            for image_id, feature_idx in obs:
                self._feature_to_track[(image_id, feature_idx)] = next_id
            next_id += 1

    def _select_seeds(self):
        """Seed candidates, best first. A failed triangulation tries the next pair."""
        calibrated = []
        fallback = []
        for pair in self.graph.pairs:
            if pair.model in (PANORAMIC, WATERMARK) or pair.R is None:
                continue
            if pair.num_inliers < self.options.init_min_num_inliers:
                continue
            if pair.median_angle_deg <= 0.0:
                continue
            if pair.calibrated:
                calibrated.append(pair)
            else:
                fallback.append(pair)
        pool = calibrated if self.options.prefer_calibrated_seed and calibrated else calibrated + fallback
        # A wide baseline is preferred, but a facade can be a perfectly valid
        # seed whose median angle is below the 16° target. Dropping every such
        # pair leaves the reconstruction with nothing to grow from.
        wide = [pair for pair in pool if pair.median_angle_deg >= self.options.init_min_tri_angle_deg]
        chosen = wide if wide else pool

        def score(pair) -> float:
            shared = self.graph.neighbors(pair.image_id1) & self.graph.neighbors(pair.image_id2)
            return pair.num_inliers * (1 + len(shared)) * pair.median_angle_deg

        return sorted(chosen, key=score, reverse=True)

    def _try_initialize(self, pair) -> bool:
        self.poses.clear()
        self.points.clear()
        self.registration_order.clear()
        self._next_point_id = 1
        self._track_points.clear()
        self._obs_to_point.clear()
        logger.info(
            "Initial pair (%s, %s) with %d inliers, median angle %.1f deg.",
            pair.image_id1,
            pair.image_id2,
            pair.num_inliers,
            pair.median_angle_deg,
        )
        if self._initialize(pair):
            return True
        logger.warning("Seed triangulation produced no model for pair (%s, %s).", pair.image_id1, pair.image_id2)
        return False

    def _initialize(self, pair) -> bool:
        self.poses[pair.image_id1] = (torch.eye(3, dtype=DTYPE), torch.zeros(3, dtype=DTYPE))
        self.poses[pair.image_id2] = (pair.R.clone(), pair.t.clone())
        self.registration_order.extend([pair.image_id1, pair.image_id2])
        left = []
        right = []
        for feature_a, feature_b in pair.matches:
            measurement_a = self._measurement(pair.image_id1, int(feature_a))
            measurement_b = self._measurement(pair.image_id2, int(feature_b))
            if measurement_a is None or measurement_b is None:
                continue
            left.append(measurement_a)
            right.append(measurement_b)
        created = two_view_points(
            left,
            right,
            self.options.tri_min_angle_deg,
            self.options.tri_max_error_px,
        )
        for xyz, first, second in created:
            self._add_point(xyz, [(first.image_id, first.feature_idx), (second.image_id, second.feature_idx)])
        logger.info("Seed triangulation produced %d points.", len(self.points))
        return len(self.points) >= self.options.tri_min_consensus

    def _measurement(self, image_id: int, feature_idx: int) -> ViewMeasurement | None:
        if image_id not in self.poses:
            return None
        image = self.graph.images[image_id]
        if feature_idx < 0 or feature_idx >= len(image.keypoints):
            return None
        R, t = self.poses[image_id]
        camera = self.graph.cameras[image.camera_id]
        return ViewMeasurement(
            image_id,
            feature_idx,
            image.keypoints[feature_idx],
            R,
            t,
            camera,
            self._panoramic.get(image_id, set()),
        )

    def _add_point(self, xyz: torch.Tensor, observations: list[tuple[int, int]]) -> int | None:
        unique = []
        seen_images = set()
        for image_id, feature_idx in observations:
            key = (image_id, feature_idx)
            if image_id in seen_images or key in self._obs_to_point:
                continue
            seen_images.add(image_id)
            unique.append(Observation(image_id, feature_idx))
        if len(unique) < 2:
            return None
        point_id = self._next_point_id
        self._next_point_id += 1
        point = Point3D(xyz=as_tensor(xyz).clone(), observations=unique)
        self.points[point_id] = point
        for obs in unique:
            self._obs_to_point[(obs.image_id, obs.feature_idx)] = point_id
            track_id = self._feature_to_track.get((obs.image_id, obs.feature_idx))
            if track_id is not None:
                bucket = self._track_points.setdefault(track_id, [])
                if point_id not in bucket:
                    bucket.append(point_id)
        return point_id

    def _refresh_scores(self) -> None:
        self._scores = {}
        for image_id, image in self.graph.images.items():
            if image_id in self.poses or image_id in self._exhausted:
                continue
            camera = self.graph.cameras[image.camera_id]
            score = NextBestViewScore(self.options.next_view_levels, camera.width, camera.height)
            _, pixels, _, _ = self._correspondences(image_id)
            for xy in pixels:
                score.add(float(xy[0]), float(xy[1]))
            self._scores[image_id] = score

    def _next_image(self) -> int | None:
        ranked = rank_candidates(self._scores, self.options.min_visible_points)
        for image_id in ranked:
            if image_id not in self._exhausted and image_id not in self.poses:
                return image_id
        return None

    def _register(self, image_id: int) -> bool:
        world, pixels, feature_indices, point_ids = self._correspondences(image_id)
        minimum = self.options.abs_pose_min_num_inliers
        if len(world) < max(6, minimum):
            return False
        image = self.graph.images[image_id]
        camera = self.graph.cameras[image.camera_id]
        normalized = pixels_to_normalized(pixels, camera.focal, camera.width, camera.height, camera.k1)

        def fit(sample: torch.Tensor):
            pose = dlt_pnp(world[sample], normalized[sample])
            if pose is None:
                return None
            _, depth = project_points(world[sample], pose[0], pose[1], camera.focal, camera.width, camera.height, camera.k1)
            if int((depth > 0).sum()) < 4:
                return None
            return pose

        def residual(model: object) -> torch.Tensor:
            R, t = model  # type: ignore[misc]
            uv, depth = project_points(world, R, t, camera.focal, camera.width, camera.height, camera.k1)
            error = torch.linalg.norm(uv - pixels, dim=1)
            return torch.where(depth > 0, error, torch.full_like(error, 1e6))

        model, inliers = ransac(
            len(world),
            6,
            fit,
            residual,
            self.options.abs_pose_max_error_px,
            self.rng,
            confidence=self.options.abs_pose_ransac_confidence,
            max_iters=self.options.abs_pose_ransac_max_iters,
            min_inliers=minimum,
        )
        if model is None or inliers is None:
            return False
        count = int(inliers.sum().item())
        if count < minimum or count / len(world) < self.options.abs_pose_min_inlier_ratio:
            return False
        R, t = refine_pose(
            model[0],
            model[1],
            world[inliers],
            pixels[inliers],
            camera.focal,
            camera.width,
            camera.height,
            camera.k1,
        )
        uv, depth = project_points(world, R, t, camera.focal, camera.width, camera.height, camera.k1)
        error = torch.linalg.norm(uv - pixels, dim=1)
        refined_inliers = (depth > 0) & (error <= self.options.abs_pose_max_error_px)
        if int(refined_inliers.sum().item()) < minimum:
            return False
        self.poses[image_id] = (R.detach(), t.detach())
        self.registration_order.append(image_id)
        for keep, feature_idx, point_id in zip(refined_inliers.tolist(), feature_indices.tolist(), point_ids.tolist()):
            if not keep:
                continue
            self._attach_observation(int(point_id), image_id, int(feature_idx))
        logger.info("Registered image %s (%d inliers).", image_id, int(refined_inliers.sum().item()))
        return True

    def _correspondences(self, image_id: int):
        image = self.graph.images[image_id]
        world = []
        pixels = []
        features = []
        point_ids = []
        used_points = set()
        used_features = set()

        def consider(feature_idx: int, point_id: int) -> None:
            if point_id not in self.points or point_id in used_points or feature_idx in used_features:
                return
            if any(obs.image_id == image_id for obs in self.points[point_id].observations):
                return
            if feature_idx < 0 or feature_idx >= len(image.keypoints):
                return
            used_points.add(point_id)
            used_features.add(feature_idx)
            world.append(self.points[point_id].xyz)
            pixels.append(image.keypoints[feature_idx])
            features.append(feature_idx)
            point_ids.append(point_id)

        for feature_idx in range(len(image.keypoints)):
            track_id = self._feature_to_track.get((image_id, feature_idx))
            if track_id is None:
                continue
            candidates = [pid for pid in self._track_points.get(track_id, []) if pid in self.points]
            if candidates:
                consider(feature_idx, candidates[0])

        # A conflicting earlier match can keep a verified correspondence out of the
        # track. The inlier itself still identifies the 3D point in a registered image.
        for pair in self.graph.pairs:
            if pair.image_id1 == image_id and pair.image_id2 in self.poses:
                feature_col, other_col, other_image = 0, 1, pair.image_id2
            elif pair.image_id2 == image_id and pair.image_id1 in self.poses:
                feature_col, other_col, other_image = 1, 0, pair.image_id1
            else:
                continue
            for row in pair.matches:
                point_id = self._obs_to_point.get((other_image, int(row[other_col])))
                if point_id is None:
                    continue
                consider(int(row[feature_col]), point_id)

        if not world:
            empty = torch.zeros(0, dtype=torch.long)
            return torch.zeros((0, 3), dtype=DTYPE), torch.zeros((0, 2), dtype=DTYPE), empty, empty
        return (
            torch.stack([as_tensor(point) for point in world]),
            torch.stack([as_tensor(pixel) for pixel in pixels]),
            torch.tensor(features, dtype=torch.long),
            torch.tensor(point_ids, dtype=torch.long),
        )

    def _attach_observation(self, point_id: int, image_id: int, feature_idx: int) -> bool:
        key = (image_id, feature_idx)
        if key in self._obs_to_point or point_id not in self.points:
            return False
        if any(obs.image_id == image_id for obs in self.points[point_id].observations):
            return False
        self.points[point_id].observations.append(Observation(image_id, feature_idx))
        self._obs_to_point[key] = point_id
        track_id = self._feature_to_track.get(key)
        if track_id is not None and point_id not in self._track_points.get(track_id, []):
            self._track_points.setdefault(track_id, []).append(point_id)
        return True

    def _triangulate_image(self, image_id: int) -> int:
        self._continue_tracks(only_image=image_id)
        created = 0
        seen = set()
        image = self.graph.images[image_id]
        for feature_idx in range(len(image.keypoints)):
            track_id = self._feature_to_track.get((image_id, feature_idx))
            if track_id is None or track_id in seen:
                continue
            seen.add(track_id)
            if (image_id, feature_idx) in self._obs_to_point:
                continue
            created += self._triangulate_remaining(track_id)
        return created

    def _continue_tracks(self, only_image: int | None = None) -> int:
        """Add observations that now fall inside the filtering threshold."""
        added = 0
        track_items = list(self._tracks.items())
        for track_id, members in track_items:
            point_ids = [pid for pid in self._track_points.get(track_id, []) if pid in self.points]
            if not point_ids:
                continue
            for image_id, feature_idx in members:
                if only_image is not None and image_id != only_image:
                    continue
                if image_id not in self.poses or (image_id, feature_idx) in self._obs_to_point:
                    continue
                measurement = self._measurement(image_id, feature_idx)
                if measurement is None:
                    continue
                best_id = None
                best_error = self.options.filter_max_reproj_error_px
                for point_id in point_ids:
                    uv, depth = project_points(
                        self.points[point_id].xyz.reshape(1, 3),
                        measurement.R,
                        measurement.t,
                        measurement.camera.focal,
                        measurement.camera.width,
                        measurement.camera.height,
                        measurement.camera.k1,
                    )
                    if float(depth[0]) <= 0:
                        continue
                    error = float(torch.linalg.norm(uv[0] - measurement.xy))
                    if error <= best_error:
                        best_error = error
                        best_id = point_id
                if best_id is not None and self._attach_observation(best_id, image_id, feature_idx):
                    added += 1
        return added

    def _triangulate_remaining(self, track_id: int) -> int:
        measurements = []
        for image_id, feature_idx in self._tracks.get(track_id, []):
            if image_id not in self.poses or (image_id, feature_idx) in self._obs_to_point:
                continue
            measurement = self._measurement(image_id, feature_idx)
            if measurement is not None:
                measurements.append(measurement)
        if len(measurements) < self.options.tri_min_consensus:
            return 0
        recovered = triangulate_track(
            measurements,
            self.options.tri_min_angle_deg,
            self.options.tri_max_error_px,
            self.options.tri_ransac_confidence,
            self.options.tri_min_inlier_ratio,
            self.options.tri_min_consensus,
            self.rng,
        )
        created = 0
        for xyz, consensus in recovered:
            observations = [(m.image_id, m.feature_idx) for m in consensus]
            if self._add_point(xyz, observations) is not None:
                created += 1
        return created

    def _retriangulate(self) -> int:
        added = self._continue_tracks()
        created = 0
        for track_id in list(self._tracks):
            created += self._triangulate_remaining(track_id)
        merged = self._merge_track_points()
        return added + created + merged

    def _merge_track_points(self) -> int:
        merged = 0
        for point_ids in list(self._track_points.values()):
            live = [pid for pid in point_ids if pid in self.points]
            if len(live) < 2:
                continue
            primary = max(live, key=lambda pid: len(self.points[pid].observations))
            for other in live:
                if other == primary or other not in self.points:
                    continue
                if self._points_consistent(primary, other):
                    for obs in list(self.points[other].observations):
                        self._obs_to_point.pop((obs.image_id, obs.feature_idx), None)
                        self._attach_observation(primary, obs.image_id, obs.feature_idx)
                    self._delete_point(other)
                    merged += 1
        return merged

    def _points_consistent(self, primary: int, other: int) -> bool:
        point = self.points[primary]
        for obs in self.points[other].observations:
            if any(existing.image_id == obs.image_id for existing in point.observations):
                return False
            measurement = self._measurement(obs.image_id, obs.feature_idx)
            if measurement is None:
                return False
            uv, depth = project_points(
                point.xyz.reshape(1, 3),
                measurement.R,
                measurement.t,
                measurement.camera.focal,
                measurement.camera.width,
                measurement.camera.height,
                measurement.camera.k1,
            )
            if float(depth[0]) <= 0:
                return False
            if float(torch.linalg.norm(uv[0] - measurement.xy)) > self.options.filter_max_reproj_error_px:
                return False
        return True

    def _delete_point(self, point_id: int) -> None:
        point = self.points.pop(point_id, None)
        if point is None:
            return
        for obs in point.observations:
            self._obs_to_point.pop((obs.image_id, obs.feature_idx), None)
            track_id = self._feature_to_track.get((obs.image_id, obs.feature_idx))
            if track_id is None:
                continue
            bucket = self._track_points.get(track_id, [])
            if point_id in bucket:
                bucket.remove(point_id)

    def _local_bundle(self, new_image: int) -> None:
        local = self._most_connected(new_image, self.options.ba_local_num_images)
        if len(local) < 2:
            return
        point_ids = self._points_visible_to(set(local))
        if len(point_ids) < 3:
            return
        image_ids = self._images_observing(point_ids)
        self._run_bundle(
            image_ids,
            point_ids,
            refine_intrinsics=False,
            max_iterations=self.options.ba_local_max_iterations,
            anchor=local[0] if local[0] != new_image else local[-1],
            fixed=set(image_ids) - set(local),
            groups=None,
        )

    def _most_connected(self, image_id: int, limit: int) -> list[int]:
        counts: dict[int, int] = {}
        for point in self.points.values():
            images = [obs.image_id for obs in point.observations]
            if image_id not in images:
                continue
            for other in images:
                if other != image_id and other in self.poses:
                    counts[other] = counts.get(other, 0) + 1
        ordered = sorted(counts, key=lambda other: counts[other], reverse=True)
        # Keep an already-stable camera first so local BA has a fixed gauge.
        connected = [image_id] + ordered[: max(limit - 1, 0)]
        if len(connected) == 1:
            return connected
        anchor = connected[1]
        return [anchor] + [image for image in connected if image != anchor]

    def _points_visible_to(self, image_ids: set[int]) -> list[int]:
        found = []
        for point_id, point in self.points.items():
            if any(obs.image_id in image_ids for obs in point.observations):
                found.append(point_id)
        return found

    def _images_observing(self, point_ids: list[int]) -> list[int]:
        images = set()
        for point_id in point_ids:
            for obs in self.points[point_id].observations:
                if obs.image_id in self.poses:
                    images.add(obs.image_id)
        return [image_id for image_id in self.registration_order if image_id in images]

    def _global_ba_due(self) -> bool:
        if self._last_global_images == 0:
            self._last_global_images = len(self.poses)
            self._last_global_points = max(len(self.points), 1)
            return False
        images_grown = len(self.poses) >= self._last_global_images * self.options.ba_global_images_ratio
        points_grown = len(self.points) >= self._last_global_points * self.options.ba_global_points_ratio
        return images_grown and points_grown

    def _iterative_global_refinement(self) -> None:
        previous_filtered = None
        previous_new = None
        for _ in range(self.options.ba_max_refinement_iterations):
            self._retriangulate()
            self._global_bundle()
            filtered = self._filter_observations()
            new_points = self._retriangulate()
            self._filter_observations()
            if filtered == 0 and new_points == 0:
                break
            if previous_filtered is not None and filtered >= previous_filtered and new_points >= previous_new:
                break
            previous_filtered = filtered
            previous_new = new_points
        self._last_global_images = len(self.poses)
        self._last_global_points = max(len(self.points), 1)
        self._refresh_scores()

    def _global_bundle(self) -> None:
        if len(self.poses) < 2 or len(self.points) < 3:
            return
        image_ids = list(self.registration_order)
        point_ids = list(self.points)
        groups = None
        if self.options.group_redundant_views and len(image_ids) >= self.options.group_min_images:
            groups = self._redundant_groups(image_ids)
        self._run_bundle(
            image_ids,
            point_ids,
            refine_intrinsics=self.options.refine_focal_length or self.options.refine_extra_params,
            max_iterations=self.options.ba_global_max_iterations,
            anchor=image_ids[0],
            fixed={image_ids[0]},
            groups=groups,
        )
        self._filter_degenerate_cameras()

    def _redundant_groups(self, image_ids: list[int]) -> list[list[int]]:
        images = []
        visibility = {}
        affected = {image_ids[0], image_ids[-1]}
        for image_id in image_ids:
            image = self.graph.images[image_id]
            R, t = self.poses[image_id]
            camera = self.graph.cameras[image.camera_id]
            images.append(BAImage(image_id, image.camera_id, R, t, camera.width, camera.height))
            point_ids = set()
            bad = 0
            total = 0
            for point_id, point in self.points.items():
                for obs in point.observations:
                    if obs.image_id != image_id:
                        continue
                    point_ids.add(point_id)
                    total += 1
                    uv, depth = project_points(
                        point.xyz.reshape(1, 3),
                        R,
                        t,
                        camera.focal,
                        camera.width,
                        camera.height,
                        camera.k1,
                    )
                    error = float(torch.linalg.norm(uv[0] - image.keypoints[obs.feature_idx]))
                    if float(depth[0]) <= 0 or error > self.options.group_affected_error_px:
                        bad += 1
            visibility[image_id] = point_ids
            if total > 0 and bad / total > self.options.group_affected_error_ratio:
                affected.add(image_id)
        return group_redundant_views(images, visibility, affected, self.options)

    def _run_bundle(
        self,
        image_ids: list[int],
        point_ids: list[int],
        refine_intrinsics: bool,
        max_iterations: int,
        anchor: int,
        fixed: set[int],
        groups: list[list[int]] | None,
    ) -> None:
        images = []
        camera_ids = set()
        for image_id in image_ids:
            image = self.graph.images[image_id]
            R, t = self.poses[image_id]
            camera = self.graph.cameras[image.camera_id]
            images.append(BAImage(image_id, camera.camera_id, R, t, camera.width, camera.height))
            camera_ids.add(camera.camera_id)
        cameras = []
        for camera_id in camera_ids:
            camera = self.graph.cameras[camera_id]
            cameras.append(BACamera(camera_id, camera.focal, camera.k1, camera.prior_focal))
        xyz = {point_id: self.points[point_id].xyz.clone() for point_id in point_ids}
        observations = []
        active_points = set(point_ids)
        active_images = set(image_ids)
        for point_id in point_ids:
            for obs in self.points[point_id].observations:
                if obs.image_id not in active_images:
                    continue
                xy = self.graph.images[obs.image_id].keypoints[obs.feature_idx]
                observations.append(BAObservation(obs.image_id, point_id, as_tensor(xy)))
        if anchor not in active_images:
            anchor = image_ids[0]
        result = bundle_adjust(
            images,
            cameras,
            xyz,
            observations,
            self.options,
            groups=groups,
            refine_intrinsics=refine_intrinsics,
            max_iterations=max_iterations,
            anchor_image=anchor,
            fixed_images=fixed | {anchor},
        )
        for image_id, R in result.rotations.items():
            self.poses[image_id] = (R, result.translations[image_id])
        for point_id, point_xyz in result.points.items():
            if point_id in self.points:
                self.points[point_id].xyz = point_xyz
        if refine_intrinsics:
            for camera_id, focal in result.focals.items():
                self.graph.cameras[camera_id].focal = float(focal)
                self.graph.cameras[camera_id].k1 = float(result.distortions[camera_id])
        del active_points

    def _filter_observations(self) -> int:
        removed = 0
        for point_id, point in list(self.points.items()):
            kept = []
            for obs in point.observations:
                measurement = self._measurement(obs.image_id, obs.feature_idx)
                if measurement is None:
                    removed += 1
                    self._obs_to_point.pop((obs.image_id, obs.feature_idx), None)
                    continue
                uv, depth = project_points(
                    point.xyz.reshape(1, 3),
                    measurement.R,
                    measurement.t,
                    measurement.camera.focal,
                    measurement.camera.width,
                    measurement.camera.height,
                    measurement.camera.k1,
                )
                error = float(torch.linalg.norm(uv[0] - measurement.xy))
                if float(depth[0]) <= 0 or error > self.options.filter_max_reproj_error_px:
                    removed += 1
                    self._obs_to_point.pop((obs.image_id, obs.feature_idx), None)
                    continue
                kept.append(obs)
            point.observations = kept
            if len(kept) < 2 or not self._angle_ok(point):
                self._delete_point(point_id)
                removed += len(kept)
        return removed

    def _angle_ok(self, point: Point3D) -> bool:
        centers = []
        for obs in point.observations:
            R, t = self.poses[obs.image_id]
            centers.append(camera_center(R, t))
        best = 0.0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                best = max(best, triangulation_angle_deg(centers[i], centers[j], point.xyz))
        return best >= self.options.filter_min_tri_angle_deg

    def _filter_degenerate_cameras(self) -> None:
        if not (self.options.refine_focal_length or self.options.refine_extra_params):
            return
        for image_id in list(self.poses):
            image = self.graph.images[image_id]
            camera = self.graph.cameras[image.camera_id]
            ratio = camera.focal / max(camera.prior_focal, 1e-6)
            if ratio < self.options.min_focal_length_ratio or ratio > self.options.max_focal_length_ratio:
                self._unregister(image_id)
                continue
            if abs(camera.k1) > self.options.max_extra_param:
                self._unregister(image_id)

    def _unregister(self, image_id: int) -> None:
        self.poses.pop(image_id, None)
        if image_id in self.registration_order:
            self.registration_order.remove(image_id)
        self._exhausted.add(image_id)
        for point in self.points.values():
            point.observations = [obs for obs in point.observations if obs.image_id != image_id]
        for key in [key for key in self._obs_to_point if key[0] == image_id]:
            self._obs_to_point.pop(key, None)
        for point_id, point in list(self.points.items()):
            if len(point.observations) < 2 or not self._angle_ok(point):
                self._delete_point(point_id)

    def _update_point_errors(self) -> None:
        for point in self.points.values():
            errors = []
            for obs in point.observations:
                measurement = self._measurement(obs.image_id, obs.feature_idx)
                if measurement is None:
                    continue
                uv, _ = project_points(
                    point.xyz.reshape(1, 3),
                    measurement.R,
                    measurement.t,
                    measurement.camera.focal,
                    measurement.camera.width,
                    measurement.camera.height,
                    measurement.camera.k1,
                )
                errors.append(float(torch.linalg.norm(uv[0] - measurement.xy)))
            point.error = float(torch.tensor(errors, dtype=DTYPE).mean()) if errors else 0.0
