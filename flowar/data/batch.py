"""The explicit contract between sequence packing and the model."""

from dataclasses import dataclass, fields, replace

import torch


@dataclass
class PackedBatch:
    total_len: int
    max_seqlen: int
    doc_num: int
    cu_seqlens: torch.Tensor
    packed_text_ids: torch.Tensor
    packed_text_indexes: torch.Tensor
    packed_position_ids: torch.Tensor
    packed_surface_geom_indexes: torch.Tensor
    packed_curve_geom_indexes: torch.Tensor
    ce_loss_indexes: torch.Tensor
    packed_label_ids: torch.Tensor
    surface_loss_indexes: torch.Tensor
    curve_loss_indexes: torch.Tensor
    surface_vectors: torch.Tensor
    curve_vectors: torch.Tensor
    condition_indexes: torch.Tensor | None = None
    condition_pc: torch.Tensor | None = None
    pixel_values: torch.Tensor | None = None
    condition_input_ids: torch.Tensor | None = None
    condition_attention_mask: torch.Tensor | None = None

    def __post_init__(self):
        supplied = sum(
            value is not None
            for value in (self.condition_pc, self.pixel_values, self.condition_input_ids)
        )
        if supplied > 1:
            raise ValueError("A packed batch can contain only one conditioning modality")
        if (self.condition_indexes is not None) != (supplied == 1):
            raise ValueError("Condition indexes and condition data must be supplied together")
        if (self.condition_input_ids is None) != (self.condition_attention_mask is None):
            raise ValueError("Text conditioning requires both input IDs and an attention mask")

    def _map_tensors(self, transform):
        return replace(
            self,
            **{
                f.name: transform(value)
                for f in fields(self)
                if isinstance(value := getattr(self, f.name), torch.Tensor)
            },
        )

    def to(self, device):
        return self._map_tensors(lambda tensor: tensor.to(device, non_blocking=True))

    def pin_memory(self):
        return self._map_tensors(lambda tensor: tensor.pin_memory())

    def get_token_num(self):
        return (
            len(self.packed_text_indexes)
            + len(self.packed_surface_geom_indexes)
            + len(self.packed_curve_geom_indexes)
        )

    def sample_num(self):
        return self.doc_num

    def model_inputs(self, vae=None) -> dict:
        surfaces, curves = self.surface_vectors, self.curve_vectors
        if vae is not None:
            surfaces = (
                vae.encode_surf(surfaces.float()) if surfaces.numel() else surfaces.reshape(0, 48)
            )
            curves = vae.encode_edge(curves.float()) if curves.numel() else curves.reshape(0, 12)
        result = {
            name: getattr(self, name)
            for name in (
                "total_len",
                "max_seqlen",
                "cu_seqlens",
                "packed_text_ids",
                "packed_text_indexes",
                "packed_position_ids",
                "packed_surface_geom_indexes",
                "packed_curve_geom_indexes",
                "ce_loss_indexes",
                "packed_label_ids",
                "surface_loss_indexes",
                "curve_loss_indexes",
            )
        }
        result.update(
            packed_surface_vectors=surfaces.flatten(1), packed_curve_vectors=curves.flatten(1)
        )
        if self.condition_indexes is not None:
            if self.condition_pc is not None:
                condition = {"pts": self.condition_pc}
            elif self.pixel_values is not None:
                condition = {"pixel_values": self.pixel_values}
            else:
                condition = {
                    "input_ids": self.condition_input_ids,
                    "attention_mask": self.condition_attention_mask,
                }
            result.update(
                condition_inputs=condition, packed_condition_indexes=self.condition_indexes
            )
        return result


def collate_packed(batch) -> PackedBatch:
    if len(batch) != 1:
        raise ValueError("PackedDataset already batches tokens; use DataLoader(batch_size=1)")
    return PackedBatch(**batch[0])
