"""Multi-resolution next-best-view score, Section 4.2.

The image is partitioned at levels l = 1..L into K_l = 2^l bins along each
axis. The first visible point that falls into an empty cell of level l adds
w_l = K_l^2 to the image score. Uniform coverage therefore outranks a cluster
with the same point count, and the score is dominated by the point count when
points are scarce.
"""

from __future__ import annotations


class NextBestViewScore:
    def __init__(self, levels: int, width: int, height: int) -> None:
        if levels < 1:
            raise ValueError("levels must be positive")
        self.levels = levels
        self.width = max(int(width), 1)
        self.height = max(int(height), 1)
        self.occupied: list[set[tuple[int, int]]] = [set() for _ in range(levels)]
        self.score = 0
        self.num_points = 0

    def add(self, x: float, y: float) -> None:
        self.num_points += 1
        for level in range(1, self.levels + 1):
            bins = 2**level
            weight = bins * bins
            col = min(bins - 1, max(0, int(x / self.width * bins)))
            row = min(bins - 1, max(0, int(y / self.height * bins)))
            cells = self.occupied[level - 1]
            key = (col, row)
            if key not in cells:
                cells.add(key)
                self.score += weight

    def copy_empty(self) -> "NextBestViewScore":
        return NextBestViewScore(self.levels, self.width, self.height)


def rank_candidates(scores: dict[int, NextBestViewScore], min_points: int) -> list[int]:
    """Highest score first. Images with fewer than N_t visible points are omitted."""
    ranked = [image_id for image_id, score in scores.items() if score.num_points >= min_points and score.score > 0]
    ranked.sort(key=lambda image_id: (scores[image_id].score, scores[image_id].num_points, -image_id), reverse=True)
    return ranked
