"""Convert a mixed token/geometry prefix once, for decoding and result assembly."""

from dataclasses import dataclass

import numpy as np
import torch

from flowar.data.sequence import BRepSequence, BRepTokenType
from flowar.generation.constraints import IncrementalTokenValidator, TokenValidationStatus


@dataclass
class SequencePrefix:
    ids: list[int]
    positions: np.ndarray
    surface_indexes: np.ndarray
    surfaces: np.ndarray
    curve_indexes: np.ndarray
    curves: np.ndarray
    shared_prefill: bool = False

    @classmethod
    def from_tokens(cls, tokens):
        if not tokens:
            raise ValueError("A generation prefix cannot be empty")
        empty = np.empty(0, dtype=np.int64)
        return cls(list(tokens), np.arange(len(tokens)), empty, np.empty(0), empty, np.empty(0))

    @classmethod
    def from_sequence(cls, seq: BRepSequence):
        ids = np.asarray(seq.text_ids)
        indexes = np.asarray(seq.text_indexes, dtype=np.int64)
        eos = np.flatnonzero(ids == BRepTokenType.EOS)
        length = int(indexes[eos[0]]) if len(eos) else seq.total_length
        if length == 0:
            return cls.from_tokens([BRepTokenType.BOS])
        dense = np.full(length, -1, dtype=np.int64)
        text = indexes < length
        dense[indexes[text]] = ids[text]
        geometry = []
        for ix, vectors, placeholder in (
            (seq.surface_geom_indexes, seq.surface_geom_vectors, BRepTokenType.SURFACE_GEOM),
            (seq.curve_geom_indexes, seq.curve_geom_vectors, BRepTokenType.CURVE_GEOM),
        ):
            ix, vectors = np.asarray(ix, dtype=np.int64), np.asarray(vectors)
            mask = ix < length
            dense[ix[mask]] = placeholder
            geometry.extend((ix[mask], vectors[mask] if len(vectors) else vectors))
        if np.any(dense < 0):
            raise ValueError("Prefix contains unassigned token positions")
        return cls(
            dense.tolist(), np.asarray(seq.position_ids)[:length], *geometry, shared_prefill=True
        )

    def validator(self):
        validator = IncrementalTokenValidator()
        for token in self.ids:
            if validator.step(token) != TokenValidationStatus.SUCCESS:
                raise ValueError("Prefix violates the token grammar")
        if not validator.at_token_boundary:
            raise ValueError("Prefix must end at a complete token block, without EOS")
        return validator

    def embed(self, model, device, dtype):
        result = torch.zeros(1, len(self.ids), model.config.hidden_size, device=device, dtype=dtype)
        text = np.ones(len(self.ids), dtype=bool)
        text[self.surface_indexes] = False
        text[self.curve_indexes] = False
        indexes = torch.as_tensor(np.flatnonzero(text), device=device)
        ids = torch.tensor(self.ids, device=device)[indexes]
        result[0, indexes] = model.text_embed(ids).to(dtype)
        for ix, values, projection in (
            (self.surface_indexes, self.surfaces, model.surface_embed),
            (self.curve_indexes, self.curves, model.curve_embed),
        ):
            if len(ix):
                vectors = torch.as_tensor(values, device=device, dtype=dtype).flatten(1)
                if vectors.shape[-1] != projection.net[0].in_features:
                    raise ValueError(
                        "Prefix geometry dimensions do not match the checkpoint representation"
                    )
                result[0, torch.as_tensor(ix, device=device)] = projection(vectors).to(dtype)
        return result
