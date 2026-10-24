"""Portable ensemble1 math; no robot, simulator, or coordinate conversion."""
from cpu_budget import CPU_BUDGET
import numpy as np

CHUNK_SIZE, COEFFICIENT = 100, .01

class SparseTemporalEnsembler:
    """Keep complete overlapping chunks, addressed by absolute action frame."""
    def __init__(self):
        self.chunks = []
        self.last_start = self.last_frame = -1
        self.next_query_index = 0

    def add(self, start_frame, chunk):
        if (not isinstance(start_frame, (int, np.integer)) or isinstance(start_frame, bool)
                or start_frame < 0 or start_frame <= self.last_start or start_frame < self.last_frame):
            raise ValueError('Chunk starts must be nonnegative, increasing action-frame integers')
        value = np.asarray(chunk)
        if (value.shape != (CHUNK_SIZE, 7) or value.dtype.kind not in 'fiu'
                or not np.isfinite(value).all()):
            raise ValueError('Expected a finite real 100-by-7 public action chunk')
        self.chunks.append((int(start_frame), self.next_query_index, value.astype(np.float64, copy=True)))
        self.next_query_index += 1
        self.last_start = int(start_frame)

    def action(self, frame):
        if (not isinstance(frame, (int, np.integer)) or isinstance(frame, bool)
                or frame < 0 or frame < self.last_frame):
            raise ValueError('Action frames must be nonnegative and cannot run backwards')
        self.last_frame = int(frame)
        self.chunks = [item for item in self.chunks if item[0] + CHUNK_SIZE > frame]
        active = [item for item in self.chunks if item[0] <= frame]
        if not active:
            raise ValueError('No ACT prediction covers this action frame')
        weights = np.exp(-COEFFICIENT * np.arange(len(active), dtype=np.float64))
        weights /= weights.sum()
        rows = np.stack([chunk[frame - start] for start, _, chunk in active])
        action = np.sum(rows * weights[:, None], axis=0)
        sources = [{'query_index': query, 'start_frame': start, 'chunk_row': int(frame - start),
                    'normalized_weight': float(weight)}
                   for (start, query, _), weight in zip(active, weights, strict=True)]
        return action, sources
