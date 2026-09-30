from enum import IntEnum
from typing import List, Tuple, Union

import torch
from transformers import LogitsProcessor

from flowar.data.sequence import (
    BBOX_TOKEN_COUNT,
    COORD_TOKEN_MAX,
    COORD_TOKEN_MIN,
    EDGE_ID_MAX,
    EDGE_ID_MIN,
    BRepTokenType,
    is_complexity_token,
    is_coord_token,
    is_edge_id_token,
)

BBOX_THRESHOLD = 1 / 2 ** (10 - 1)
# ============================================================================
# Geometric validation config & utilities
# ============================================================================


class BRepStateMachineLogitsProcessor(LogitsProcessor):
    def __init__(self, reuse_edge_ids: bool = True):
        if not reuse_edge_ids:
            raise ValueError("Edge-ID recycling is required")
        self.r_coord = (COORD_TOKEN_MIN, COORD_TOKEN_MAX + 1)
        self.r_edge_id = (EDGE_ID_MIN, EDGE_ID_MAX + 1)
        self.reuse_edge_ids = reuse_edge_ids

        self._active_edges_caches: List[set] = []
        self._used_edges_caches: List[set] = []
        self._last_seq_lens: List[int] = []

    @staticmethod
    def _count_trailing(seq: List[int], condition) -> int:
        for i, token in enumerate(reversed(seq)):
            if not condition(token):
                return i
        return len(seq)

    def _get_edge_cmds(self, active_edges: set, allow_end: bool = True) -> List[int]:
        cmds = [BRepTokenType.EDGE_NEW]
        if active_edges:
            cmds.append(BRepTokenType.EDGE_REF)
        if allow_end:
            cmds.append(BRepTokenType.LOOP_END)
        return cmds

    def _update_active_edges(self, batch_idx: int, seq: List[int]) -> set:
        while len(self._active_edges_caches) <= batch_idx:
            self._active_edges_caches.append(set())
            self._used_edges_caches.append(set())
            self._last_seq_lens.append(0)
        seq_len = len(seq)
        expected_len = self._last_seq_lens[batch_idx] + 1

        # Reset the cache if the sequence does not grow one token at a time as expected
        if seq_len != expected_len:
            self._active_edges_caches[batch_idx].clear()
            self._used_edges_caches[batch_idx].clear()
            self._last_seq_lens[batch_idx] = 0

        edges = self._active_edges_caches[batch_idx]
        used_edges = self._used_edges_caches[batch_idx]
        start_idx = max(1, self._last_seq_lens[batch_idx])

        # Process only newly added tokens incrementally
        for i in range(start_idx, seq_len):
            cmd, tok = seq[i - 1], seq[i]
            if self.r_edge_id[0] <= tok < self.r_edge_id[1]:
                if cmd == BRepTokenType.EDGE_NEW:
                    edges.add(tok)
                    used_edges.add(tok)
                elif cmd == BRepTokenType.EDGE_REF:
                    edges.discard(tok)

        self._last_seq_lens[batch_idx] = seq_len
        return edges

    def _get_transitions(
        self, seq: List[int], active_edges: set, batch_idx: int = 0
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        if not seq:
            return [BRepTokenType.BOS], []

        last = seq[-1]

        # ==========================================
        # 1. Dynamic sequence handling (coordinates and edge IDs)
        # ==========================================
        if is_coord_token(last):
            # Keep forcing coordinate tokens until all six coordinates are present
            if self._count_trailing(seq, is_coord_token) < BBOX_TOKEN_COUNT:
                return [], [self.r_coord]

            ## This branch is not expected to be reached
            ctx_token = seq[-(BBOX_TOKEN_COUNT + 1)]
            return (
                [BRepTokenType.SURFACE_GEOM]
                if ctx_token == BRepTokenType.FACE_START
                else [BRepTokenType.CURVE_GEOM]
            ), []

        if is_edge_id_token(last):
            # An EdgeID must be immediately followed by coordinates
            if len(seq) >= 2 and seq[-2] == BRepTokenType.EDGE_NEW:
                return [], [self.r_coord]
            # Only an EdgeID following EDGE_REF transitions directly to the next edge instruction
            return self._get_edge_cmds(active_edges), []

        # ==========================================
        # 2. Static state routing (state machine)
        # ==========================================
        if is_complexity_token(last):
            return [BRepTokenType.FACE_START, BRepTokenType.EOS], []

        match last:
            case BRepTokenType.BOS:
                return [
                    BRepTokenType.COMPLEXITY_L1,
                    BRepTokenType.COMPLEXITY_L2,
                    BRepTokenType.COMPLEXITY_L3,
                    BRepTokenType.FACE_START,
                    BRepTokenType.EOS,
                ], []

            case BRepTokenType.FACE_END:
                return [BRepTokenType.FACE_START, BRepTokenType.EOS], []

            case BRepTokenType.FACE_START:
                return [], [self.r_coord]

            # Placeholder transition: end face geometry -> start an inner loop or end the current face
            case BRepTokenType.SURFACE_GEOM | BRepTokenType.LOOP_END:
                return [BRepTokenType.LOOP_START, BRepTokenType.FACE_END], []

            # Placeholder transition: end edge geometry -> continue to the next edge
            case BRepTokenType.CURVE_GEOM:
                return self._get_edge_cmds(active_edges), []

            case BRepTokenType.LOOP_START:
                return self._get_edge_cmds(active_edges, allow_end=False), []

            case BRepTokenType.EDGE_NEW:
                blocked = (
                    active_edges
                    if self.reuse_edge_ids
                    else self._used_edges_caches[batch_idx]
                    if len(self._used_edges_caches) > batch_idx
                    else active_edges
                )
                allowed = [t for t in range(*self.r_edge_id) if t not in blocked]
                return (allowed, []) if allowed else ([BRepTokenType.PAD], [])

            case BRepTokenType.EDGE_REF:
                return (list(active_edges), []) if active_edges else ([BRepTokenType.PAD], [])

            case _:
                return [BRepTokenType.PAD], []

    def reset(self):
        """Discard incremental state; the next mask rebuilds it from token history."""
        self._active_edges_caches.clear()
        self._used_edges_caches.clear()
        self._last_seq_lens.clear()

    def mask_rows(self, logits, histories, rows):
        """Apply grammar constraints to selected rows without exposing cache internals."""
        mask = torch.full_like(logits, -float("inf"))
        for output_row, sample_row in enumerate(rows):
            active = self._update_active_edges(sample_row, histories[sample_row])
            tokens, ranges = self._get_transitions(histories[sample_row], active, sample_row)
            if tokens:
                mask[output_row, tokens] = 0
            for start, end in ranges:
                mask[output_row, start:end] = 0
        return logits + mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        bsz = scores.shape[0]
        seqs = input_ids.tolist()

        # 1. Dynamically align the cache length
        while len(self._active_edges_caches) < bsz:
            self._active_edges_caches.append(set())
            self._used_edges_caches.append(set())
            self._last_seq_lens.append(0)

        mask = torch.full_like(scores, -float("inf"))

        for i, seq in enumerate(seqs):
            active_edges = self._update_active_edges(i, seq)
            tokens, ranges = self._get_transitions(seq, active_edges, i)

            if tokens:
                mask[i, tokens] = 0.0
            for r_start, r_end in ranges:
                mask[i, r_start:r_end] = 0.0

        return scores + mask


class TokenValidationStatus(IntEnum):
    SUCCESS = 0
    UNCLOSED_FACE = 1
    UNCLOSED_LOOP = 2
    NESTED_FACE = 3
    NESTED_LOOP = 4
    LOOP_OUTSIDE_FACE = 5
    EDGE_OUTSIDE_LOOP = 6
    FACE_END_WITHOUT_START = 7
    LOOP_END_WITHOUT_START = 8
    TRUNCATED_STREAM = 9
    INVALID_BBOX = 10
    INVALID_GEOM_PLACEHOLDER = 11
    INVALID_EDGE_ID = 12
    UNDEFINED_EDGE_REF = 13
    DUPLICATE_EDGE_ID = 14
    UNEXPECTED_TOKEN = 15
    MISSING_EOS = 16


class IncrementalTokenValidator:
    def __init__(self, ordering: str = "interleaved"):
        if ordering != "interleaved":
            raise ValueError("Only interleaved ordering is supported")
        self.in_face = False
        self.in_loop = False
        self.known_edges: set[int] = set()
        self.done = False

        self._ordering = ordering
        self._phase = "token"
        self._coords_left = 0
        self._pending_edge_id: int | None = None

    @property
    def at_token_boundary(self) -> bool:
        return self._phase == "token" and not self.done

    def copy(self) -> "IncrementalTokenValidator":
        v = IncrementalTokenValidator.__new__(IncrementalTokenValidator)
        v.in_face = self.in_face
        v.in_loop = self.in_loop
        v.known_edges = set(self.known_edges)
        v.done = self.done
        v._ordering = self._ordering
        v._phase = self._phase
        v._coords_left = self._coords_left
        v._pending_edge_id = self._pending_edge_id
        return v

    def step(self, tok: int) -> TokenValidationStatus:
        if self.done:
            return TokenValidationStatus.SUCCESS

        if tok in (BRepTokenType.BOS, BRepTokenType.PAD) or is_complexity_token(tok):
            return TokenValidationStatus.SUCCESS

        if self._phase == "face_bbox":
            if not is_coord_token(tok):
                return TokenValidationStatus.INVALID_BBOX
            self._coords_left -= 1
            if self._coords_left == 0:
                self._phase = "face_geom"
            return TokenValidationStatus.SUCCESS

        if self._phase == "face_geom":
            if tok != BRepTokenType.SURFACE_GEOM:
                return TokenValidationStatus.INVALID_GEOM_PLACEHOLDER
            self._phase = "token"
            return TokenValidationStatus.SUCCESS

        if self._phase == "edge_id_new":
            if not is_edge_id_token(tok):
                return TokenValidationStatus.INVALID_EDGE_ID
            if tok in self.known_edges:
                return TokenValidationStatus.DUPLICATE_EDGE_ID
            self._pending_edge_id = tok
            self._phase = "edge_bbox"
            self._coords_left = BBOX_TOKEN_COUNT
            return TokenValidationStatus.SUCCESS

        if self._phase == "edge_id_ref":
            if not is_edge_id_token(tok):
                return TokenValidationStatus.INVALID_EDGE_ID
            if tok not in self.known_edges:
                return TokenValidationStatus.UNDEFINED_EDGE_REF
            self.known_edges.remove(tok)
            self._phase = "token"
            return TokenValidationStatus.SUCCESS

        if self._phase == "edge_bbox":
            if not is_coord_token(tok):
                return TokenValidationStatus.INVALID_BBOX
            self._coords_left -= 1
            if self._coords_left == 0:
                self._phase = "edge_geom"
            return TokenValidationStatus.SUCCESS

        if self._phase == "edge_geom":
            if tok != BRepTokenType.CURVE_GEOM:
                return TokenValidationStatus.INVALID_GEOM_PLACEHOLDER
            if self._pending_edge_id is not None:
                self.known_edges.add(self._pending_edge_id)
            self._pending_edge_id = None
            self._phase = "token"
            return TokenValidationStatus.SUCCESS

        if tok == BRepTokenType.EOS:
            if self.in_loop:
                return TokenValidationStatus.UNCLOSED_LOOP
            if self.in_face:
                return TokenValidationStatus.UNCLOSED_FACE
            self.done = True
            return TokenValidationStatus.SUCCESS

        if tok == BRepTokenType.FACE_START:
            if self.in_face:
                return TokenValidationStatus.NESTED_FACE
            self.in_face = True
            self._phase = "face_bbox"
            self._coords_left = BBOX_TOKEN_COUNT
            return TokenValidationStatus.SUCCESS

        if tok == BRepTokenType.FACE_END:
            if not self.in_face:
                return TokenValidationStatus.FACE_END_WITHOUT_START
            if self.in_loop:
                return TokenValidationStatus.UNCLOSED_LOOP
            self.in_face = False
            return TokenValidationStatus.SUCCESS

        if tok == BRepTokenType.LOOP_START:
            if not self.in_face:
                return TokenValidationStatus.LOOP_OUTSIDE_FACE
            if self.in_loop:
                return TokenValidationStatus.NESTED_LOOP
            self.in_loop = True
            return TokenValidationStatus.SUCCESS

        if tok == BRepTokenType.LOOP_END:
            if not self.in_loop:
                return TokenValidationStatus.LOOP_END_WITHOUT_START
            self.in_loop = False
            return TokenValidationStatus.SUCCESS

        if tok == BRepTokenType.EDGE_NEW:
            if not self.in_loop:
                return TokenValidationStatus.EDGE_OUTSIDE_LOOP
            self._phase = "edge_id_new"
            return TokenValidationStatus.SUCCESS

        if tok == BRepTokenType.EDGE_REF:
            if not self.in_loop:
                return TokenValidationStatus.EDGE_OUTSIDE_LOOP
            self._phase = "edge_id_ref"
            return TokenValidationStatus.SUCCESS

        return TokenValidationStatus.UNEXPECTED_TOKEN


def validate_ids(
    ids: torch.Tensor | list[int],
    ordering: str = "interleaved",
) -> tuple[TokenValidationStatus, str, int]:
    if ordering != "interleaved":
        raise ValueError("Only interleaved ordering is supported")
    tokens = ids.tolist() if hasattr(ids, "tolist") else ids
    idx, n = 0, len(tokens)
    in_face, in_loop = False, False
    known_edges: set[int] = set()

    def check_coords(start: int, count: int) -> bool:
        return all(is_coord_token(t) for t in tokens[start : start + count])

    while idx < n:
        tok = tokens[idx]

        if tok in (BRepTokenType.BOS, BRepTokenType.PAD) or is_complexity_token(tok):
            idx += 1
            continue

        if tok == BRepTokenType.EOS:
            if in_loop:
                return TokenValidationStatus.UNCLOSED_LOOP, "EOS with open LOOP", idx
            if in_face:
                return TokenValidationStatus.UNCLOSED_FACE, "EOS with open FACE", idx
            return TokenValidationStatus.SUCCESS, "Success", idx

        if tok == BRepTokenType.FACE_START:
            if in_face:
                return TokenValidationStatus.NESTED_FACE, "Nested FACE_START", idx
            if idx + 8 > n:
                return (
                    TokenValidationStatus.TRUNCATED_STREAM,
                    "Truncated FACE_START",
                    idx,
                )
            if not check_coords(idx + 1, 6):
                return TokenValidationStatus.INVALID_BBOX, "Invalid FACE bbox", idx + 1
            if tokens[idx + 7] != BRepTokenType.SURFACE_GEOM:
                return (
                    TokenValidationStatus.INVALID_GEOM_PLACEHOLDER,
                    "Missing SURFACE_GEOM",
                    idx + 7,
                )
            in_face = True
            idx += 8
            continue

        elif tok == BRepTokenType.FACE_END:
            if not in_face:
                return (
                    TokenValidationStatus.FACE_END_WITHOUT_START,
                    "FACE_END without START",
                    idx,
                )
            if in_loop:
                return (
                    TokenValidationStatus.UNCLOSED_LOOP,
                    "FACE_END with open LOOP",
                    idx,
                )

            in_face = False
            idx += 1

        elif tok == BRepTokenType.LOOP_START:
            if not in_face:
                return (
                    TokenValidationStatus.LOOP_OUTSIDE_FACE,
                    "LOOP_START outside FACE",
                    idx,
                )
            if in_loop:
                return TokenValidationStatus.NESTED_LOOP, "Nested LOOP_START", idx

            in_loop = True
            idx += 1

        elif tok == BRepTokenType.LOOP_END:
            if not in_loop:
                return (
                    TokenValidationStatus.LOOP_END_WITHOUT_START,
                    "LOOP_END without START",
                    idx,
                )

            in_loop = False
            idx += 1

        elif tok == BRepTokenType.EDGE_NEW:
            if not in_loop:
                return (
                    TokenValidationStatus.EDGE_OUTSIDE_LOOP,
                    "EDGE_NEW outside LOOP",
                    idx,
                )
            if idx + 2 > n:
                return TokenValidationStatus.TRUNCATED_STREAM, "Truncated EDGE_NEW", idx

            edge_id = tokens[idx + 1]
            if not is_edge_id_token(edge_id):
                return (
                    TokenValidationStatus.INVALID_EDGE_ID,
                    f"Invalid EDGE_ID: {edge_id}",
                    idx + 1,
                )
            if edge_id in known_edges:
                return (
                    TokenValidationStatus.DUPLICATE_EDGE_ID,
                    f"Duplicate EDGE_ID: {edge_id}",
                    idx + 1,
                )

            if idx + 9 > n:
                return (
                    TokenValidationStatus.TRUNCATED_STREAM,
                    "Truncated EDGE_NEW bbox",
                    idx,
                )
            if not check_coords(idx + 2, 6):
                return TokenValidationStatus.INVALID_BBOX, "Invalid EDGE bbox", idx + 2
            if tokens[idx + 8] != BRepTokenType.CURVE_GEOM:
                return (
                    TokenValidationStatus.INVALID_GEOM_PLACEHOLDER,
                    "Missing CURVE_GEOM",
                    idx + 8,
                )
            known_edges.add(edge_id)
            idx += 9

        elif tok == BRepTokenType.EDGE_REF:
            if not in_loop:
                return (
                    TokenValidationStatus.EDGE_OUTSIDE_LOOP,
                    "EDGE_REF outside LOOP",
                    idx,
                )
            if idx + 2 > n:
                return TokenValidationStatus.TRUNCATED_STREAM, "Truncated EDGE_REF", idx

            edge_id = tokens[idx + 1]
            if not is_edge_id_token(edge_id):
                return (
                    TokenValidationStatus.INVALID_EDGE_ID,
                    f"Invalid EDGE_ID: {edge_id}",
                    idx + 1,
                )
            if edge_id not in known_edges:
                return (
                    TokenValidationStatus.UNDEFINED_EDGE_REF,
                    f"Undefined EDGE_REF: {edge_id}",
                    idx + 1,
                )

            # Release references to enable ID recycling as required by the specification
            known_edges.remove(edge_id)
            idx += 2

        else:
            return (
                TokenValidationStatus.UNEXPECTED_TOKEN,
                f"Unexpected token: {tok}",
                idx,
            )

    if in_loop:
        return TokenValidationStatus.UNCLOSED_LOOP, "Missing LOOP_END", idx
    if in_face:
        return TokenValidationStatus.UNCLOSED_FACE, "Missing FACE_END", idx

    return TokenValidationStatus.MISSING_EOS, "Missing EOS", idx


def count_faces_edges(
    ids: Union[torch.Tensor, List[int]],
) -> Tuple[int, int]:
    """Count faces and unique edges from a token id sequence.

    Works on any ids regardless of validity.
    Faces = number of FACE_START tokens.
    Edges = number of EDGE_NEW tokens (unique edges only).

    Returns:
        (num_faces, num_edges)
    """
    tokens = ids.tolist() if hasattr(ids, "tolist") else ids
    num_faces = 0
    num_edges = 0
    for tok in tokens:
        if tok == BRepTokenType.FACE_START:
            num_faces += 1
        elif tok == BRepTokenType.EDGE_NEW:
            num_edges += 1
    return num_faces, num_edges
