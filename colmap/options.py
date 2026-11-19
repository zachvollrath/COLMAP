"""Thresholds for verification and incremental reconstruction.

The CVPR 2016 paper writes several decision thresholds symbolically
(N_F, ε_HF, ε_EF, ε_SF, ε_SE, t, α, η, ε_0). Numeric defaults below follow
the values stated in the paper's experiments and the COLMAP defaults that
implement the same tests.
"""

from dataclasses import dataclass


@dataclass
class VerificationOptions:
    """Scene-graph augmentation, Section 4.1."""

    # A pair is geometrically verified when the fundamental matrix has at least
    # this many inliers (N_F).
    min_num_inliers: int = 15
    max_error_px: float = 4.0
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 10000
    # Moving camera in a general scene when N_H / N_F is below ε_HF.
    max_homography_inlier_ratio: float = 0.8
    # Calibration is accepted when N_E / N_F exceeds ε_EF.
    min_essential_inlier_ratio: float = 0.8
    # Median triangulation angle below this (degrees) is treated as pure rotation.
    panoramic_max_angle_deg: float = 1.0
    # Watermarks, timestamps, and frames: N_S / N_F > ε_SF or N_S / N_E > ε_SE.
    watermark_border_fraction: float = 0.1
    max_watermark_inlier_ratio: float = 0.3
    min_border_matches: int = 8


@dataclass
class MapperOptions:
    """Incremental reconstruction, Sections 4.2–4.5."""

    # Initialization. Seeds must be non-panoramic; calibrated pairs are preferred.
    init_min_num_inliers: int = 100
    init_min_tri_angle_deg: float = 16.0
    prefer_calibrated_seed: bool = True

    # Next-best-view pyramid, Section 4.2. Experiments in the paper use L = 6.
    next_view_levels: int = 6
    # Candidates must see N_t > 0 triangulated points.
    min_visible_points: int = 1

    # Absolute pose. Outlier threshold is in pixels.
    abs_pose_max_error_px: float = 12.0
    abs_pose_min_num_inliers: int = 30
    abs_pose_min_inlier_ratio: float = 0.25
    abs_pose_ransac_confidence: float = 0.999
    abs_pose_ransac_max_iters: int = 10000
    max_registration_trials: int = 3

    # Robust triangulation, Section 4.3. Experiment settings: α = 2°, t = 8 px,
    # ε_0 = 0.03. η = 0.999 is the stricter of the two reported operating points.
    tri_min_angle_deg: float = 2.0
    tri_max_error_px: float = 8.0
    tri_ransac_confidence: float = 0.999
    tri_min_inlier_ratio: float = 0.03
    tri_min_consensus: int = 3

    # Bundle adjustment, Section 4.4.
    ba_local_num_images: int = 6
    ba_local_max_iterations: int = 25
    ba_global_max_iterations: int = 50
    # Global BA after the model grows by this factor (amortized linear runtime).
    ba_global_images_ratio: float = 1.1
    ba_global_points_ratio: float = 1.1
    ba_loss_scale_px: float = 1.0
    ba_max_refinement_iterations: int = 5
    # Refine the simple camera (focal length and one radial coefficient) during
    # global BA. Principal point stays fixed at the image center.
    refine_focal_length: bool = True
    refine_extra_params: bool = True
    min_focal_length_ratio: float = 0.1
    max_focal_length_ratio: float = 10.0
    max_extra_param: float = 1.0

    # Filtering.
    filter_max_reproj_error_px: float = 4.0
    filter_min_tri_angle_deg: float = 1.5

    # Redundant view mining, Section 4.5.
    group_redundant_views: bool = True
    group_min_images: int = 30
    group_max_size: int = 10
    group_min_overlap: float = 0.4
    group_affected_error_px: float = 4.0
    group_affected_error_ratio: float = 0.02
    group_neighbor_count: int = 50
    group_max_view_angle_deg: float = 60.0

    random_seed: int = 0
