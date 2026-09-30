import json
import logging
import multiprocessing
import random
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from flowar.data.ordering import _bfs_reorder_faces
from flowar.data.sequence import (
    BBOX_THRESHOLD,
    BRepSequence,
    arrays_to_sequence,
    num_faces_to_complexity_token,
)

logger = logging.getLogger(__name__)


def load_paths(source: str | Path | list[str | Path], ext: str) -> list[Path]:
    if isinstance(source, list):
        return sorted([p for p in (Path(x) for x in source) if p.is_file()])
    source = Path(source)
    if source.is_dir():
        return sorted(source.rglob(f"*{ext}"))
    return [source] if source.is_file() else []


def next_multiple_of_n(v: float | int, *, n: int) -> int:
    return ((int(v) + n - 1) // n) * n


class PackedDataset(torch.utils.data.IterableDataset):
    @property
    def global_step(self) -> int:
        return self._shared_step.value

    @global_step.setter
    def global_step(self, v: int) -> None:
        self._shared_step.value = int(v)

    def __init__(
        self,
        data_source: str | Path | list[str | Path],
        max_num_tokens: int = 32768,
        file_ext: str = ".npz",
        shuffle: bool = True,
        seed: int = 42,
        augment: bool = False,
        resolution: int = 1024,
        max_face: int = 80,
        rank: int = 0,
        world_size: int = 1,
        edge_id_range: int = 980,
        cond_num_tokens: int = 0,
        cond_pc_points: int = 2048,
        cond_pc_extra_dir: str = "",
        cond_image_dir: str = "",
        cond_caption_json: str = "",
        caption_dropout_prob: float = 0.1,
        random_rotation_prob: float = 0.0,
        face_reorder: str = "bfs",
        ordering: str = "interleaved",
        reuse_edge_ids: bool = True,
        geom_repr: str = "bezier",
        random_edgeid_start: bool = True,
        cond_clip_model: str = "openai/clip-vit-large-patch14",
    ):
        if face_reorder != "bfs":
            raise ValueError("Only BFS face ordering is supported")
        if ordering != "interleaved":
            raise ValueError("Only interleaved ordering is supported")
        if not reuse_edge_ids:
            raise ValueError("Edge-ID recycling is required")
        if max_num_tokens <= cond_num_tokens or cond_num_tokens < 0:
            raise ValueError("Token budget must leave room after the condition prefix")
        self.max_num_tokens, self.shuffle, self.seed = max_num_tokens, shuffle, seed
        self.augment, self.resolution, self.max_face = augment, resolution, max_face
        self.edge_id_range = edge_id_range
        self.rank, self.world_size = rank, world_size
        self.face_reorder = face_reorder
        self.ordering = ordering
        self.reuse_edge_ids = reuse_edge_ids
        self.geom_repr = geom_repr
        self.random_edgeid_start = random_edgeid_start
        self._shared_step = multiprocessing.Value("l", 0)
        self.global_step = 0
        self.cond_num_tokens = cond_num_tokens
        self.cond_pc_points = cond_pc_points
        self.random_rotation_prob = random_rotation_prob
        self.cube_rotations = Rotation.create_group("O").as_matrix()
        self.file_paths = load_paths(data_source, file_ext)
        self._pc_index: dict[str, Path] = (
            {p.stem[:33]: p for p in Path(cond_pc_extra_dir).rglob("*.ply")}
            if cond_pc_extra_dir and cond_num_tokens > 0
            else {}
        )
        self._img_index: dict[str, list] = {}
        self._img_transform = None
        if cond_image_dir and cond_num_tokens > 0:
            for p in Path(cond_image_dir).rglob("*.png"):
                self._img_index.setdefault(p.stem[:33], []).append(p)
            from torchvision.transforms import v2

            base = [
                v2.ToImage(),
                v2.Resize((256, 256), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
            if augment:
                aug = [
                    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
                    v2.RandomGrayscale(p=0.2),
                ]
                self._img_transform = v2.Compose(aug + base)
            else:
                self._img_transform = v2.Compose(base)
        self._caption_tokens: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._caption_null: tuple[np.ndarray, np.ndarray] | None = None
        self.caption_dropout_prob = caption_dropout_prob
        if cond_caption_json and cond_num_tokens > 0:
            from transformers import CLIPTokenizer

            tokenizer = CLIPTokenizer.from_pretrained(cond_clip_model)
            with open(cond_caption_json, encoding="utf-8") as f:
                raw_captions: dict[str, str] = json.load(f)

            def tokenize(text):
                encoded = tokenizer(
                    text, max_length=77, padding="max_length", truncation=True, return_tensors="np"
                )
                return tuple(
                    encoded[key][0].astype(np.int32) for key in ("input_ids", "attention_mask")
                )

            self._caption_tokens = {key: tokenize(text) for key, text in raw_captions.items()}
            self._caption_null = tokenize("")
        if self.rank == 0:
            print(
                f"PackedDataset: {len(self.file_paths)} samples, max_tokens={max_num_tokens}, augment={augment}"
            )
            if self._pc_index:
                print(
                    f"PackedDataset: PLY index {len(self._pc_index)} entries from {cond_pc_extra_dir}"
                )
            if self._img_index:
                covered = sum(1 for p in self.file_paths if p.stem[:33] in self._img_index)
                print(
                    f"PackedDataset: PNG index {len(self._img_index)} entries from {cond_image_dir} "
                    f"— covers {covered}/{len(self.file_paths)} samples "
                    f"({100.0 * covered / max(len(self.file_paths), 1):.1f}%)"
                )
            if self._caption_tokens:
                covered = sum(1 for p in self.file_paths if p.stem[:33] in self._caption_tokens)
                print(
                    f"PackedDataset: caption index {len(self._caption_tokens)} entries from {cond_caption_json} "
                    f"— covers {covered}/{len(self.file_paths)} samples "
                    f"({100.0 * covered / max(len(self.file_paths), 1):.1f}%) "
                    f"dropout={caption_dropout_prob}"
                )

    def _process_sample(
        self,
        path: Path,
    ) -> BRepSequence | None:
        try:
            with np.load(path) as data:
                f_ctrl, e_ctrl = data["face_controls"], data["edge_controls"]
                o_edge, f_o_off = data["outer_edge_indices"], data["face_outer_offsets"]
                i_edge, i_l_off, f_i_off = (
                    data["inner_edge_indices"],
                    data["inner_loop_offsets"],
                    data["face_inner_offsets"],
                )
            if len(f_ctrl) == 0 or len(e_ctrl) == 0 or len(o_edge) == 0 or len(f_o_off) == 0:
                logger.warning(f"Empty face or edge controls in {path}, skipping sample")
                return None
            f_ctrl, o_edge, f_o_off, i_edge, i_l_off, f_i_off = _bfs_reorder_faces(
                f_ctrl, o_edge, f_o_off, i_edge, i_l_off, f_i_off, random_start_num=0
            )

            R_total = None
            if self.random_rotation_prob > 0 and random.random() < self.random_rotation_prob:
                R_total = self.cube_rotations[np.random.randint(24)]
            if R_total is not None:
                f_ctrl[..., :3] = f_ctrl[..., :3] @ R_total.T
                e_ctrl[..., :3] = e_ctrl[..., :3] @ R_total.T
                all_pts = np.concatenate(
                    [f_ctrl[..., :3].reshape(-1, 3), e_ctrl[..., :3].reshape(-1, 3)]
                )
                pt_min, pt_max = all_pts.min(axis=0), all_pts.max(axis=0)
                center = (pt_max + pt_min) / 2.0
                scale = 2.0 / np.max(pt_max - pt_min)

                f_ctrl[..., :3] = (f_ctrl[..., :3] - center) * scale
                e_ctrl[..., :3] = (e_ctrl[..., :3] - center) * scale

            if self.augment and (self.cond_num_tokens == 0 or self._caption_tokens):
                s = np.random.uniform(0.75, 1.0)
                max_t = 1.0 - s
                t = np.random.uniform(-max_t, max_t, size=(3,)).astype(np.float32)
                f_ctrl[..., :3] = f_ctrl[..., :3] * s + t
                e_ctrl[..., :3] = e_ctrl[..., :3] * s + t

            if self.cond_num_tokens > 0:
                complexity_token = None
            else:
                complexity_token = num_faces_to_complexity_token(len(f_ctrl))

            seq = arrays_to_sequence(
                face_controls=f_ctrl,
                edge_controls=e_ctrl,
                outer_edge_indices=o_edge,
                face_outer_offsets=f_o_off,
                inner_edge_indices=i_edge,
                inner_loop_offsets=i_l_off,
                face_inner_offsets=f_i_off,
                resolution=self.resolution,
                edge_id_range=self.edge_id_range,
                complexity_token=complexity_token,
                bbox_threshold=BBOX_THRESHOLD,
                ordering=self.ordering,
                reuse_edge_ids=self.reuse_edge_ids,
                geom_repr=self.geom_repr,
                random_edgeid_start=self.random_edgeid_start,
            )

            if self.cond_num_tokens and not self._attach_condition(seq, path, R_total):
                return None
            return seq
        except (OSError, BadZipFile, ValueError, KeyError, IndexError) as exc:
            logger.warning("Skipping sample %s: %s", path, exc)
            return None

    def _attach_condition(self, seq: BRepSequence, path: Path, rotation) -> bool:
        """Load the matching condition and reserve its leading sequence positions."""
        if self._img_index:
            png_candidates = self._img_index.get(path.stem[:33])
            if png_candidates is None:
                return False
            png_path = png_candidates[np.random.randint(len(png_candidates))]
            from PIL import Image

            with Image.open(png_path) as image:
                seq.condition_pixel_values = self._img_transform(image.convert("RGB")).numpy()
        elif self._caption_tokens:
            tokens = self._caption_tokens.get(path.stem[:33])
            if tokens is None:
                return False
            if self.caption_dropout_prob > 0 and random.random() < self.caption_dropout_prob:
                tokens = self._caption_null
            seq.condition_text_ids, seq.condition_text_mask = tokens
        elif self._pc_index:
            ply = self._pc_index.get(path.stem[:33])
            if ply is None:
                return False
            geom = trimesh.load(str(ply), process=False)
            pts = np.asarray(geom.vertices, dtype=np.float32)
            n = len(pts)
            if n >= self.cond_pc_points:
                pts = pts[np.random.choice(n, self.cond_pc_points, replace=False)]
            elif 0 < n < self.cond_pc_points:
                pts = pts[np.random.choice(n, self.cond_pc_points, replace=True)]
            if rotation is not None:
                pts = pts @ rotation.T
            ctr = (pts.max(0) + pts.min(0)) / 2.0
            sc = np.max(pts.max(0) - pts.min(0))
            seq.condition_pc = ((pts - ctr) * (2.0 / (sc + 1e-8))).astype(np.float32)
        else:
            logger.warning("No conditional inputs configured for %s", path)
            return False

        count = self.cond_num_tokens
        seq.condition_indexes = np.arange(count, dtype=np.int64)
        seq.text_indexes += count
        seq.surface_geom_indexes += count
        seq.curve_geom_indexes += count
        seq.position_ids = np.concatenate([seq.condition_indexes, seq.position_ids + count])
        seq.total_length += count
        return True

    def _pack_sequences(self, samples: list["BRepSequence"]):
        if not samples:
            return {}

        lens = np.array([s.total_length for s in samples], dtype=np.int32)
        offsets = np.concatenate([[0], np.cumsum(lens[:-1])])
        total_len = int(np.sum(lens))

        p_text_ids = np.concatenate([s.text_ids for s in samples])
        p_pos_ids = np.concatenate([s.position_ids for s in samples])
        surf_vecs = np.concatenate([s.surface_geom_vectors for s in samples])
        curve_vecs = np.concatenate([s.curve_geom_vectors for s in samples])

        p_text_idx = np.concatenate([s.text_indexes + o for s, o in zip(samples, offsets)])
        p_surf_geom = np.concatenate([s.surface_geom_indexes + o for s, o in zip(samples, offsets)])
        p_curve_geom = np.concatenate([s.curve_geom_indexes + o for s, o in zip(samples, offsets)])
        has_cond = len(samples[0].condition_indexes) > 0
        if has_cond:
            p_cond_idx = np.concatenate([s.condition_indexes + o for s, o in zip(samples, offsets)])

        target_idx = np.concatenate([s.text_indexes[1:] + o for s, o in zip(samples, offsets)])
        target_labels = np.concatenate([s.text_ids[1:] for s in samples])

        pred_idx = target_idx - 1

        is_valid_src = np.zeros(total_len + 1, dtype=bool)
        is_valid_src[p_text_idx] = True
        is_valid_src[p_surf_geom] = True
        is_valid_src[p_curve_geom] = True

        mask = is_valid_src[pred_idx]
        ce_loss_idx = pred_idx[mask]
        p_label_ids = target_labels[mask]

        if total_len == 0:
            return {}

        # Tail truncation logic (No Padding)
        if total_len > self.max_num_tokens:
            total_len = self.max_num_tokens
            if has_cond:
                # A condition is one encoder output: never keep only some of its slots.
                cond_n = len(samples[0].condition_indexes)
                partial = offsets[(offsets < total_len) & (offsets + cond_n > total_len)]
                if len(partial):
                    total_len = int(partial[0])
            sample_lens = np.diff(np.minimum(np.cumsum(lens), total_len), prepend=0).tolist()

            p_pos_ids = p_pos_ids[:total_len]

            t_mask = p_text_idx < total_len
            p_text_ids, p_text_idx = p_text_ids[t_mask], p_text_idx[t_mask]

            s_mask, c_mask = p_surf_geom < total_len, p_curve_geom < total_len
            p_surf_geom, surf_vecs = p_surf_geom[s_mask], surf_vecs[s_mask]
            p_curve_geom, curve_vecs = p_curve_geom[c_mask], curve_vecs[c_mask]

            l_mask = ce_loss_idx < (total_len - 1)
            ce_loss_idx, p_label_ids = ce_loss_idx[l_mask], p_label_ids[l_mask]
        else:
            sample_lens = lens.tolist()

        sample_lens = [length for length in sample_lens if length > 0]
        max_docs = next_multiple_of_n(len(sample_lens), n=128)
        if has_cond:
            p_cond_idx = p_cond_idx[p_cond_idx < total_len]
            cond_n = len(samples[0].condition_indexes)
            n_complete = len(p_cond_idx) // cond_n
            p_cond_idx = p_cond_idx[: n_complete * cond_n]

        actual_cu_seqlens = np.cumsum([0] + sample_lens, dtype=np.int32)
        cu_seqlens = np.full(max_docs + 1, actual_cu_seqlens[-1], dtype=np.int32)
        cu_seqlens[: len(sample_lens) + 1] = actual_cu_seqlens

        def to_ts(arr, dtype=None):
            return torch.from_numpy(arr).to(dtype) if dtype else torch.from_numpy(arr)

        result = {
            "total_len": total_len,
            "doc_num": len(sample_lens),
            "max_seqlen": max(sample_lens, default=0),
            "cu_seqlens": to_ts(cu_seqlens),
            "packed_text_ids": to_ts(p_text_ids),
            "packed_text_indexes": to_ts(p_text_idx),
            "packed_position_ids": to_ts(p_pos_ids),
            "packed_surface_geom_indexes": to_ts(p_surf_geom),
            "packed_curve_geom_indexes": to_ts(p_curve_geom),
            "surface_vectors": to_ts(surf_vecs, torch.float32),
            "curve_vectors": to_ts(curve_vecs, torch.float32),
            "ce_loss_indexes": to_ts(ce_loss_idx),
            "packed_label_ids": to_ts(p_label_ids),
            "surface_loss_indexes": to_ts(p_surf_geom - 1),
            "curve_loss_indexes": to_ts(p_curve_geom - 1),
        }
        if has_cond:
            result["condition_indexes"] = to_ts(p_cond_idx)
            cond_pcs = [s.condition_pc for s in samples[:n_complete] if s.condition_pc is not None]
            if cond_pcs:
                result["condition_pc"] = torch.from_numpy(np.stack(cond_pcs))
            cond_imgs = [
                s.condition_pixel_values
                for s in samples[:n_complete]
                if s.condition_pixel_values is not None
            ]
            if cond_imgs:
                result["pixel_values"] = torch.from_numpy(np.stack(cond_imgs))
            cond_text_ids = [
                s.condition_text_ids
                for s in samples[:n_complete]
                if s.condition_text_ids is not None
            ]
            if cond_text_ids:
                result["condition_input_ids"] = torch.from_numpy(np.stack(cond_text_ids)).long()
                result["condition_attention_mask"] = torch.from_numpy(
                    np.stack(
                        [
                            s.condition_text_mask
                            for s in samples[:n_complete]
                            if s.condition_text_mask is not None
                        ]
                    )
                ).long()
        return result

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rank, world_size = self.rank, self.world_size

        if worker_info is not None:
            rank = rank * worker_info.num_workers + worker_info.id
            world_size *= worker_info.num_workers

        file_paths = self.file_paths[rank::world_size]
        epoch_seed = self.seed + self.global_step + rank
        np.random.seed(epoch_seed)
        random.seed(epoch_seed)

        indices = (
            np.random.permutation(len(file_paths)) if self.shuffle else np.arange(len(file_paths))
        )
        current_batch, current_tokens = [], 0

        for idx in indices:
            sample = self._process_sample(file_paths[idx])
            if (
                not sample
                or not len(sample.surface_geom_indexes)
                or not len(sample.curve_geom_indexes)
            ):
                continue

            current_batch.append(sample)
            current_tokens += sample.total_length

            if current_tokens >= self.max_num_tokens:
                packed_data = self._pack_sequences(current_batch)
                if packed_data:
                    yield packed_data
                    self.global_step += 1
                current_batch, current_tokens = [], 0

        if current_batch:
            packed_data = self._pack_sequences(current_batch)
            if packed_data:
                yield packed_data
                self.global_step += 1

    def __len__(self):
        return len(self.file_paths)

    def state_dict(self) -> dict:
        return {"global_step": self.global_step, "seed": self.seed}

    def load_state_dict(self, state: dict):
        self.global_step = state.get("global_step", 0)
        self.seed = state.get("seed", self.seed)
        if self.rank == 0:
            print(f"Dataset state restored: global_step={self.global_step}")

    def set_global_step(self, step: int):
        self.global_step = step
