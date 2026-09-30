"""A single batched decoding loop for fresh generation and prefix completion."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from flowar.data.sequence import BRepSequence, BRepTokenType
from flowar.generation.constraints import (
    BRepStateMachineLogitsProcessor,
    IncrementalTokenValidator,
    TokenValidationStatus,
)
from flowar.generation.prefix import SequencePrefix
from flowar.generation.types import MODE_AWAIT_EDGE as MODE_AWAIT_EDGE
from flowar.generation.types import MODE_BBOX_CURV as MODE_BBOX_CURV
from flowar.generation.types import MODE_BBOX_SURF as MODE_BBOX_SURF
from flowar.generation.types import MODE_DIFF_CURV as MODE_DIFF_CURV
from flowar.generation.types import MODE_DIFF_SURF as MODE_DIFF_SURF
from flowar.generation.types import MODE_DONE as MODE_DONE
from flowar.generation.types import MODE_TEXT as MODE_TEXT
from flowar.generation.types import SamplingOptions
from flowar.generation.types import StepResult as StepResult
from flowar.models.embeddings import KVCache


@torch.inference_mode()
def sample_next_text_token(logits, rng, temperature=1.0, top_k=None, top_p=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        thresh = logits.topk(k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < thresh, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)


@torch.inference_mode()
def sample_next_geom_vec(diff_head, cond, num_sampling_steps=20, return_with_trajectory=False):
    with torch.autocast(
        device_type=cond.device.type,
        dtype=cond.dtype,
        enabled=cond.device.type == "cuda",
    ):
        if return_with_trajectory:
            return diff_head.sample_with_logprob(cond, num_sampling_steps=num_sampling_steps)
        else:
            return diff_head.sample(cond, num_sampling_steps=num_sampling_steps)


@dataclass
class DecodeState:
    cache: KVCache
    hidden: torch.Tensor
    modes: torch.Tensor
    countdowns: torch.Tensor
    validators: list[IncrementalTokenValidator]
    histories: list[list[int]]
    grammar: BRepStateMachineLogitsProcessor | None

    def advance(self, ids: torch.Tensor, invalid: torch.Tensor) -> None:
        """Advance the interleaved grammar after accepting one token per row."""
        old = self.modes
        new = old.clone()
        new[(old == MODE_DIFF_SURF) | (old == MODE_DIFF_CURV)] = MODE_TEXT
        bbox = (old == MODE_BBOX_SURF) | (old == MODE_BBOX_CURV)
        self.countdowns[bbox] -= 1
        for bbox_mode, geom_mode in (
            (MODE_BBOX_SURF, MODE_DIFF_SURF),
            (MODE_BBOX_CURV, MODE_DIFF_CURV),
        ):
            new[(old == bbox_mode) & (self.countdowns == 0)] = geom_mode
        face = (old == MODE_TEXT) & (ids == BRepTokenType.FACE_START)
        new[face], self.countdowns[face] = MODE_BBOX_SURF, 6
        new[(old == MODE_TEXT) & (ids == BRepTokenType.EDGE_NEW)] = MODE_AWAIT_EDGE
        edge = old == MODE_AWAIT_EDGE
        new[edge], self.countdowns[edge] = MODE_BBOX_CURV, 6
        new[((old == MODE_TEXT) & (ids == BRepTokenType.EOS)) | invalid] = MODE_DONE
        self.modes = new


class Engine:
    def __init__(self, model):
        self.model = model

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return torch.bfloat16 if self.device.type == "cuda" else torch.float32

    def autocast(self):
        return torch.autocast(
            self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"
        )

    def _cache(self, batch_size, length):
        config = self.model.config
        return KVCache(
            batch_size=batch_size,
            seq_len=length,
            dtype=self.dtype,
            device=self.device,
            num_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            num_layers=config.num_hidden_layers,
        )

    @staticmethod
    def _check_method(ordering, reuse_edge_ids):
        if ordering != "interleaved" or not reuse_edge_ids:
            raise ValueError("Generation requires interleaved ordering and edge-ID recycling")

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        num_samples=1,
        max_tokens=2048,
        temperature=1.0,
        top_k=None,
        top_p=None,
        seed=78,
        use_validation=True,
        use_fsm=False,
        return_with_trajectory=False,
        condition_embed=None,
        num_sampling_steps=20,
        ordering="interleaved",
        reuse_edge_ids=True,
    ):
        """Generate at most `max_tokens` new tokens after a text-only prefix."""
        self._check_method(ordering, reuse_edge_ids)
        options = SamplingOptions(
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            use_validation,
            use_fsm,
            num_sampling_steps,
            return_with_trajectory,
        )
        prefix = SequencePrefix.from_tokens(tokens)
        yield from self._generate(prefix, num_samples, options, condition_embed)

    @torch.inference_mode()
    def generate_from_prefix(
        self,
        seq: BRepSequence,
        num_samples=1,
        max_new_tokens=2048,
        temperature=1.0,
        top_k=None,
        top_p=None,
        seed=78,
        use_validation=True,
        use_fsm=False,
        return_with_trajectory=False,
        condition_embed=None,
        ordering="interleaved",
        reuse_edge_ids=True,
        num_sampling_steps=20,
    ):
        """Complete a mixed prefix ending between faces/loops/edges; strip its terminal EOS."""
        self._check_method(ordering, reuse_edge_ids)
        options = SamplingOptions(
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            use_validation,
            use_fsm,
            num_sampling_steps,
            return_with_trajectory,
        )
        prefix = SequencePrefix.from_sequence(seq)
        yield from self._generate(prefix, num_samples, options, condition_embed)

    def _generate(self, prefix, num_samples, options, condition):
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        validator = prefix.validator()
        if options.max_new_tokens == 0:
            return
        rng = torch.Generator(device=self.device).manual_seed(options.seed)
        torch.manual_seed(options.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(options.seed)
        with self.autocast():
            state = self._prefill(prefix, num_samples, options, condition, validator)
        for _ in range(options.max_new_tokens):
            if bool((state.modes == MODE_DONE).all()):
                break
            # Context ends before yielding: early generator close cannot leak autocast.
            with self.autocast():
                result = self._step(state, options, rng)
            yield result

    def _prefill(self, prefix, count, options, condition, validator):
        prefill_count = 1 if prefix.shared_prefill else count
        embedded = prefix.embed(self.model, self.device, self.dtype).expand(prefill_count, -1, -1)
        positions = torch.tensor(prefix.positions, device=self.device).unsqueeze(0)
        if condition is not None:
            if condition.ndim != 3 or condition.shape[0] not in (1, prefill_count):
                raise ValueError(
                    "Condition prefix must have batch size 1 or match generation batch"
                )
            cond = condition.to(device=self.device, dtype=self.dtype).expand(prefill_count, -1, -1)
            embedded = torch.cat((cond, embedded), dim=1)
            positions = torch.cat(
                (torch.arange(cond.size(1), device=self.device)[None], positions + cond.size(1)),
                dim=1,
            )
        cache = self._cache(prefill_count, embedded.size(1) + options.max_new_tokens)
        hidden = self.model.transformer.forward_inference(
            embedded, position_ids=positions, kv_cache=cache
        )[:, -1:]
        if prefill_count != count:
            expanded = self._cache(count, cache.max_seq_len)
            expanded.prefill(cache)
            cache, hidden = expanded, hidden.expand(count, -1, -1)
        return DecodeState(
            cache,
            hidden,
            torch.full((count,), MODE_TEXT, device=self.device),
            torch.zeros(count, dtype=torch.long, device=self.device),
            [validator.copy() for _ in range(count)],
            [list(prefix.ids) for _ in range(count)],
            BRepStateMachineLogitsProcessor() if options.constrain_tokens else None,
        )

    def _geometry(self, head, hidden, active, options, count):
        vectors = torch.zeros(count, 1, head.ch_target, device=self.device, dtype=self.dtype)
        trajectory = None
        if active.any():
            sampled = sample_next_geom_vec(
                head, hidden[active, -1], options.steps, options.trajectory
            )
            if options.trajectory:
                sampled, history, _ = sampled
                history = torch.stack(history, dim=1).to(self.dtype)
                trajectory = history.new_zeros(count, *history.shape[1:])
                trajectory[active] = history
            vectors[active] = sampled.to(self.dtype).unsqueeze(1)
        return vectors, trajectory

    def _step(self, state, options, rng):
        modes = state.modes
        count = len(modes)
        surface, curve, done = modes == MODE_DIFF_SURF, modes == MODE_DIFF_CURV, modes == MODE_DONE
        text = ~(surface | curve | done)
        ids = torch.full((count, 1), BRepTokenType.PAD, device=self.device)
        if text.any():
            logits = self.model.text_head(state.hidden[text, -1])
            if state.grammar is not None:
                logits = state.grammar.mask_rows(
                    logits, state.histories, text.nonzero()[:, 0].tolist()
                )
            ids[text] = sample_next_text_token(
                logits, rng, options.temperature, options.top_k, options.top_p
            )
        surf, surf_traj = self._geometry(
            self.model.surface_diff_head, state.hidden, surface, options, count
        )
        curv, curv_traj = self._geometry(
            self.model.curve_diff_head, state.hidden, curve, options, count
        )
        ids[surface], ids[curve] = BRepTokenType.SURFACE_GEOM, BRepTokenType.CURVE_GEOM
        invalid = torch.zeros(count, dtype=torch.bool, device=self.device)
        for i in (~done).nonzero()[:, 0].tolist():
            if (
                options.validate
                and state.validators[i].step(int(ids[i, 0])) != TokenValidationStatus.SUCCESS
            ):
                invalid[i], ids[i, 0] = True, BRepTokenType.EOS
            state.histories[i].append(int(ids[i, 0]))
        result = StepResult(
            modes.tolist(),
            ids[:, 0].tolist(),
            surf.float().cpu().numpy(),
            curv.float().cpu().numpy(),
            invalid.tolist(),
            None if surf_traj is None else surf_traj.float().cpu().numpy(),
            None if curv_traj is None else curv_traj.float().cpu().numpy(),
        )
        embedded = torch.empty(
            count, 1, self.model.config.hidden_size, device=self.device, dtype=self.dtype
        )
        if text.any():
            embedded[text] = self.model.text_embed(ids[text]).to(self.dtype)
        if surface.any():
            embedded[surface] = self.model.surface_embed(surf[surface]).to(self.dtype)
        if curve.any():
            embedded[curve] = self.model.curve_embed(curv[curve]).to(self.dtype)
        if done.any():
            embedded[done] = self.model.text_embed(
                torch.full_like(ids[done], BRepTokenType.EOS)
            ).to(self.dtype)
        state.advance(ids[:, 0], invalid)
        state.hidden = self.model.transformer(embedded, kv_cache=state.cache)
        return result
