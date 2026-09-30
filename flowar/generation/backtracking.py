from dataclasses import dataclass

import numpy as np
import torch

from flowar.data.sequence import (
    BBOX_TOKEN_COUNT,
    BRepTokenType,
    denormalize_points_with_bbox,
    tokens_to_bbox,
)
from flowar.generation.constraints import (
    BRepStateMachineLogitsProcessor,
    IncrementalTokenValidator,
    TokenValidationStatus,
)
from flowar.generation.engine import (
    MODE_AWAIT_EDGE,
    MODE_BBOX_CURV,
    MODE_BBOX_SURF,
    MODE_DIFF_CURV,
    MODE_DIFF_SURF,
    MODE_DONE,
    MODE_TEXT,
    Engine,
    StepResult,
    sample_next_geom_vec,
    sample_next_text_token,
)
from flowar.generation.prefix import SequencePrefix
from flowar.generation.types import SamplingOptions
from flowar.geometry.validation import (
    GeomChecker,
    GeomCheckResult,
)
from flowar.models.embeddings import KVCache
from flowar.visualization.debug import _plot_failed_state

_LEVEL_FACE, _LEVEL_LOOP, _LEVEL_EDGE = 0, 1, 2


@dataclass
class SampleCheckpoint:
    """Full per-sample generation state snapshot used for backtracking."""

    level: int
    kv_seqlen: int
    seq_snapshot: list[int]
    last_hidden: torch.Tensor  # (1, 1, hidden_size)
    engine_state: int
    countdown: int
    validator_snapshot: IncrementalTokenValidator
    geom_snapshot: dict
    surf_vecs_snapshot: list[np.ndarray]
    curv_vecs_snapshot: list[np.ndarray]
    budget: int


@dataclass
class _Proposal:
    mode: int
    token: int
    surface: torch.Tensor
    curve: torch.Tensor
    invalid: bool = False


class GeomRejectionEngine:
    """Single-sample decoding with a bounded hierarchy of geometry retries."""

    def __init__(self, model):
        self.model = model.eval()

    @torch.inference_mode()
    def generate_single(
        self,
        tokens,
        max_tokens=2048,
        temperature=1.0,
        top_k=None,
        top_p=None,
        seed=78,
        use_fsm=True,
        geom_config=None,
        condition_embed=None,
        num_sampling_steps=20,
        failed_plot_path=None,
        ordering="interleaved",
        reuse_edge_ids=True,
    ):
        Engine._check_method(ordering, reuse_edge_ids)
        if self.model.config.surface_latent_dim != 64 or self.model.config.curve_latent_dim != 16:
            raise ValueError("Geometry backtracking requires Bézier control points")
        runtime = Engine(self.model)
        prefix = SequencePrefix.from_tokens(tokens)
        prefix.validator()
        SamplingOptions(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            steps=num_sampling_steps,
        )
        parameters = dict(
            tokens=list(tokens),
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            use_fsm=use_fsm,
            geom_config=geom_config,
            condition_embed=condition_embed,
            num_sampling_steps=num_sampling_steps,
            failed_plot_path=failed_plot_path,
            reuse_edge_ids=reuse_edge_ids,
        )
        with runtime.autocast():
            session = _RejectionSession(self.model, runtime, **parameters)
        yield from session.run()


class _RejectionSession:
    def __init__(
        self,
        model,
        runtime,
        condition_embed,
        failed_plot_path,
        geom_config,
        max_tokens,
        num_sampling_steps,
        reuse_edge_ids,
        seed,
        temperature,
        tokens,
        top_k,
        top_p,
        use_fsm,
    ):
        self.model = model
        self.runtime = runtime
        self.condition_embed = condition_embed
        self.failed_plot_path = failed_plot_path
        self.geom_config = geom_config
        self.max_tokens = max_tokens
        self.num_sampling_steps = num_sampling_steps
        self.reuse_edge_ids = reuse_edge_ids
        self.seed = seed
        self.temperature = temperature
        self.tokens = tokens
        self.top_k = top_k
        self.top_p = top_p
        self.use_fsm = use_fsm

        self.cfg = self.geom_config
        self.device, self.dtype = (next(self.model.parameters()).device, self.runtime.dtype)
        self.rng = torch.Generator(device=self.device).manual_seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        n_cond = self.condition_embed.size(1) if self.condition_embed is not None else 0
        self.kv_cache = KVCache(
            batch_size=1,
            seq_len=len(self.tokens) + self.max_tokens + n_cond,
            dtype=self.dtype,
            device=self.device,
            num_heads=self.model.config.num_key_value_heads,
            head_dim=self.model.config.hidden_size // self.model.config.num_attention_heads,
            num_layers=self.model.config.num_hidden_layers,
        )
        ids_t = torch.tensor([self.tokens], device=self.device)
        init_emb = self.model.text_embed(ids_t).to(self.dtype)
        if self.condition_embed is not None:
            cond = self.condition_embed.to(device=self.device, dtype=self.dtype)
            init_emb = torch.cat([cond, init_emb], dim=1)
        self.last_hidden = self.model.transformer.forward_inference(
            init_emb, kv_cache=self.kv_cache
        )[:, -1:]
        self.state = MODE_TEXT
        self.countdown = 0
        self.validator = SequencePrefix.from_tokens(self.tokens).validator()
        self.fsm_proc = (
            BRepStateMachineLogitsProcessor(reuse_edge_ids=self.reuse_edge_ids)
            if self.use_fsm
            else None
        )
        self.seq: list[int] = list(self.tokens)
        self.surf_vecs: list[np.ndarray] = []
        self.curv_vecs: list[np.ndarray] = []
        self.geom_checker = GeomChecker(self.cfg) if self.cfg is not None else None
        self.checkpoints: list[SampleCheckpoint] = []
        self._edge_id_tok: int | None = None
        self._await_ref_id: bool = False
        self._bt_budget = self.cfg.budget_total_face if self.cfg else 0
        self._gs: dict = {}
        self.failed = False
        self._ycur = len(self.seq)
        self._plen = len(self.seq)
        self._buf_surf = torch.zeros(
            1, 1, self.model.config.surface_latent_dim, device=self.device, dtype=self.dtype
        )
        self._buf_curv = torch.zeros(
            1, 1, self.model.config.curve_latent_dim, device=self.device, dtype=self.dtype
        )
        self._np_zero_surf = self._buf_surf.float().cpu().numpy()
        self._np_zero_curv = self._buf_curv.float().cpu().numpy()
        self._text_id_buf = torch.zeros(1, 1, device=self.device, dtype=torch.long)

    def _snap(self, level: int, budget: int) -> SampleCheckpoint:
        return SampleCheckpoint(
            level=level,
            kv_seqlen=int(self.kv_cache.cache_seqlens[0].item()),
            seq_snapshot=list(self.seq),
            last_hidden=self.last_hidden.clone(),
            engine_state=self.state,
            countdown=self.countdown,
            validator_snapshot=self.validator.copy(),
            geom_snapshot=self.geom_checker.snapshot() if self.geom_checker else {},
            surf_vecs_snapshot=list(self.surf_vecs),
            curv_vecs_snapshot=list(self.curv_vecs),
            budget=budget,
        )

    def _restore(self, ck: SampleCheckpoint) -> None:
        self.kv_cache.cache_seqlens.fill_(ck.kv_seqlen)
        self.seq, self.last_hidden = (list(ck.seq_snapshot), ck.last_hidden.clone())
        self.state, self.countdown = (ck.engine_state, ck.countdown)
        self.validator = ck.validator_snapshot.copy()
        if self.geom_checker:
            self.geom_checker.restore(ck.geom_snapshot)
        self.surf_vecs, self.curv_vecs = (list(ck.surf_vecs_snapshot), list(ck.curv_vecs_snapshot))
        self._edge_id_tok, self._await_ref_id = (None, False)
        if self.fsm_proc is not None:
            self.fsm_proc.reset()

    def _gs_inc(self, key: str) -> None:
        self._gs[key] = self._gs.get(key, 0) + 1

    def _backtrack(self, target_level: int) -> bool:
        """Pop stack to *target_level* or coarser. Returns success."""
        for lvl in [v for v in (_LEVEL_EDGE, _LEVEL_LOOP, _LEVEL_FACE) if v <= target_level]:
            while self.checkpoints:
                ck = self.checkpoints[-1]
                if ck.level > lvl:
                    self.checkpoints.pop()
                    continue
                if ck.level == lvl and ck.budget > 0:
                    ck.budget -= 1
                    if lvl == _LEVEL_FACE:
                        self._bt_budget -= 1
                        if self._bt_budget < 0:
                            return False
                    self._restore(ck)
                    return True
                if ck.level == lvl:
                    self.checkpoints.pop()
                    continue
                break
        return False

    def _geom_ok_new(self, raw: np.ndarray) -> GeomCheckResult:
        if not self.geom_checker:
            return GeomCheckResult.OK
        try:
            ctrl = denormalize_points_with_bbox(
                raw.reshape(4, 4), tokens_to_bbox(self.seq[-BBOX_TOKEN_COUNT:])
            )
        except Exception:
            return GeomCheckResult.ENDPOINT_MISMATCH
        r = self.geom_checker.validate_new_edge(ctrl)
        if r == GeomCheckResult.OK:
            self.geom_checker.commit_edge(ctrl)
            if self._edge_id_tok is not None:
                self.geom_checker.register_edge_token(self._edge_id_tok, ctrl)
        return r

    def _geom_ok_ref(self, tok: int) -> GeomCheckResult:
        if not self.geom_checker:
            return GeomCheckResult.OK
        ctrl = self.geom_checker.get_edge_by_token(tok)
        if ctrl is None:
            return GeomCheckResult.OK
        r = self.geom_checker.validate_ref_edge(ctrl)
        if r == GeomCheckResult.OK:
            self.geom_checker.commit_edge(ctrl)
            self.geom_checker.release_edge_token(tok)
        return r

    def _emit_rollback(self):
        """Rollback StepResult if consumer is ahead of seq, else None."""
        rb = self._ycur - len(self.seq)
        if rb > 0:
            self._ycur = len(self.seq)
            return StepResult(
                modes=[self.state],
                text_ids=[BRepTokenType.PAD],
                surf_vecs=np.zeros((1, 1, 1)),
                curv_vecs=np.zeros((1, 1, 1)),
                invalid_flags=[False],
                rollback_count=rb,
            )
        return None

    def _step(self):
        proposal = self._sample()
        if proposal is None or not self._accept(proposal):
            return
        self._feed(proposal)
        if not self._checkpoint_events(proposal):
            return
        self._emit(proposal)

    def _sample(self) -> "_Proposal | None":
        proposal = _Proposal(self.state, BRepTokenType.PAD, self._buf_surf, self._buf_curv)
        self._buf_surf.zero_()
        self._buf_curv.zero_()
        if proposal.mode == MODE_DIFF_SURF:
            x = sample_next_geom_vec(
                self.model.surface_diff_head, self.last_hidden[0, -1:], self.num_sampling_steps
            )
            proposal.surface[0], proposal.token = (x.to(self.dtype), BRepTokenType.SURFACE_GEOM)
            raw_surf = x.float().cpu().numpy().ravel()
            try:
                _surf_ctrl = denormalize_points_with_bbox(
                    raw_surf.reshape(4, 4, 4), tokens_to_bbox(self.seq[-BBOX_TOKEN_COUNT:])
                )
            except Exception:
                self._gs_inc("rej_surf_denorm")
                self._gs_inc("bt_face")
                if self._backtrack(_LEVEL_FACE):
                    rb = self._emit_rollback()
                    if rb:
                        self.events.append(rb)
                    return
                self.failed, self.state, proposal.token = (True, MODE_DONE, BRepTokenType.EOS)
                return
            self.surf_vecs.append(raw_surf)
        elif proposal.mode == MODE_DIFF_CURV:
            x = sample_next_geom_vec(
                self.model.curve_diff_head, self.last_hidden[0, -1:], self.num_sampling_steps
            )
            proposal.curve[0], proposal.token = (x.to(self.dtype), BRepTokenType.CURVE_GEOM)
            raw = x.float().cpu().numpy().ravel()
            _curv_result = self._geom_ok_new(raw)
            if _curv_result != GeomCheckResult.OK:
                self._gs_inc(_curv_result.name)
                self._gs_inc("bt_edge")
                if self._backtrack(_LEVEL_EDGE):
                    rb = self._emit_rollback()
                    if rb:
                        self.events.append(rb)
                    return
                self.failed, self.state, proposal.token = (True, MODE_DONE, BRepTokenType.EOS)
                return
            self.curv_vecs.append(raw)
        else:
            logits = self.model.text_head(self.last_hidden[0, -1:])
            if self.fsm_proc is not None:
                logits = self.fsm_proc.mask_rows(logits, [self.seq], [0])
            proposal.token = int(
                sample_next_text_token(
                    logits, self.rng, self.temperature, self.top_k, self.top_p
                ).item()
            )
            if self._await_ref_id:
                self._await_ref_id = False
                _ref_result = self._geom_ok_ref(proposal.token)
                if _ref_result != GeomCheckResult.OK:
                    self._gs_inc(_ref_result.name + "_ref")
                    self._gs_inc("bt_loop")
                if _ref_result != GeomCheckResult.OK:
                    if self._backtrack(_LEVEL_LOOP):
                        rb = self._emit_rollback()
                        if rb:
                            self.events.append(rb)
                    else:
                        self.failed = True
                    return
        return proposal

    def _accept(self, proposal) -> bool:
        proposal.invalid = False
        if self.validator.step(proposal.token) != TokenValidationStatus.SUCCESS:
            self.failed = True
            proposal.invalid, proposal.token, self.state = (True, BRepTokenType.EOS, MODE_DONE)
        self.seq.append(proposal.token)
        ns = proposal.mode
        if proposal.mode in (MODE_DIFF_SURF, MODE_DIFF_CURV):
            ns = MODE_TEXT
        elif proposal.mode in (MODE_BBOX_SURF, MODE_BBOX_CURV):
            self.countdown -= 1
            if self.countdown <= 0:
                if self.cfg is not None and self.cfg.bbox_min_span > 0:
                    try:
                        bbox = tokens_to_bbox(self.seq[-BBOX_TOKEN_COUNT:])
                        span = bbox[3:] - bbox[:3]
                        if span.max() < self.cfg.bbox_min_span:
                            bt_level = (
                                _LEVEL_FACE if proposal.mode == MODE_BBOX_SURF else _LEVEL_EDGE
                            )
                            self._gs_inc(
                                "rej_bbox_surf"
                                if proposal.mode == MODE_BBOX_SURF
                                else "rej_bbox_curv"
                            )
                            self._gs_inc("bt_face" if bt_level == _LEVEL_FACE else "bt_edge")
                            if self._backtrack(bt_level):
                                rb = self._emit_rollback()
                                if rb:
                                    self.events.append(rb)
                                return
                            self.failed, proposal.token, ns = (True, BRepTokenType.EOS, MODE_DONE)
                    except Exception:
                        self.failed, proposal.token, ns = (True, BRepTokenType.EOS, MODE_DONE)
                if not self.failed:
                    ns = MODE_DIFF_SURF if proposal.mode == MODE_BBOX_SURF else MODE_DIFF_CURV
        elif proposal.mode == MODE_TEXT:
            if proposal.token == BRepTokenType.FACE_START:
                ns, self.countdown = (MODE_BBOX_SURF, 6)
            if proposal.token == BRepTokenType.EDGE_NEW:
                ns = MODE_AWAIT_EDGE
            elif proposal.token == BRepTokenType.EOS:
                ns = MODE_DONE
        elif proposal.mode == MODE_AWAIT_EDGE:
            ns, self.countdown = (MODE_BBOX_CURV, 6)
        if proposal.invalid:
            ns = MODE_DONE
        self.state = ns
        return True

    def _feed(self, proposal):
        if proposal.token == BRepTokenType.SURFACE_GEOM:
            emb = self.model.surface_embed(proposal.surface).to(self.dtype)
        elif proposal.token == BRepTokenType.CURVE_GEOM:
            emb = self.model.curve_embed(proposal.curve).to(self.dtype)
        else:
            self._text_id_buf[0, 0] = proposal.token
            emb = self.model.text_embed(self._text_id_buf).to(self.dtype)
        self.last_hidden = self.model.transformer(emb, kv_cache=self.kv_cache)

    def _checkpoint_events(self, proposal) -> bool:
        if proposal.mode == MODE_TEXT and self.cfg is not None:
            if proposal.token == BRepTokenType.FACE_START:
                self.checkpoints = [c for c in self.checkpoints if c.level == _LEVEL_FACE]
                self.checkpoints.append(self._snap(_LEVEL_FACE, self.cfg.budget_face))
            elif proposal.token == BRepTokenType.LOOP_START:
                self.checkpoints = [c for c in self.checkpoints if c.level <= _LEVEL_LOOP]
                if self.geom_checker:
                    self.geom_checker.begin_loop()
                self.checkpoints.append(self._snap(_LEVEL_LOOP, self.cfg.budget_loop))
            elif proposal.token == BRepTokenType.EDGE_NEW:
                self.checkpoints.append(self._snap(_LEVEL_EDGE, self.cfg.budget_edge))
                self._edge_id_tok = None
            elif proposal.token == BRepTokenType.EDGE_REF:
                self._await_ref_id = True
            elif proposal.token == BRepTokenType.LOOP_END:
                if self.geom_checker and not self.geom_checker.validate_loop_closure():
                    self._gs_inc("rej_loop_closure")
                    self._gs_inc("bt_loop")
                    if self._backtrack(_LEVEL_LOOP):
                        rb = self._emit_rollback()
                        if rb:
                            self.events.append(rb)
                    else:
                        self.failed = True
                    return
                if self.geom_checker:
                    self.geom_checker.end_loop()
                self.checkpoints = [c for c in self.checkpoints if c.level <= _LEVEL_LOOP]
        if proposal.mode == MODE_AWAIT_EDGE:
            self._edge_id_tok = proposal.token
        return True

    def _emit(self, proposal):
        self._ycur = len(self.seq)
        self.events.append(
            StepResult(
                modes=[self.state],
                text_ids=[proposal.token],
                surf_vecs=proposal.surface.float().cpu().numpy()
                if proposal.token == BRepTokenType.SURFACE_GEOM
                else self._np_zero_surf,
                curv_vecs=proposal.curve.float().cpu().numpy()
                if proposal.token == BRepTokenType.CURVE_GEOM
                else self._np_zero_curv,
                invalid_flags=[proposal.invalid or self.failed],
            )
        )

    def run(self):
        while (
            self.state != MODE_DONE
            and not self.failed
            and len(self.seq) - self._plen < self.max_tokens
        ):
            self.events = []
            with self.runtime.autocast():
                self._step()
            yield from self.events
        self.failed |= self.state != MODE_DONE
        yield StepResult(
            modes=[self.state],
            text_ids=[BRepTokenType.PAD],
            surf_vecs=self._np_zero_surf,
            curv_vecs=self._np_zero_curv,
            invalid_flags=[self.failed],
            final_seq=list(self.seq),
            final_surf_vecs=list(self.surf_vecs),
            final_curv_vecs=list(self.curv_vecs),
            geom_stats=dict(self._gs) or None,
        )
        if self.failed and self.failed_plot_path:
            _plot_failed_state(self.seq, self.surf_vecs, self.curv_vecs, self.failed_plot_path)
