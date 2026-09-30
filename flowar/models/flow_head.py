import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    half = dim // 2
    t = time_factor * t.float()
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def time_shift_func(t, flow_shift=1.0):
    return (1 / flow_shift) / ((1 / flow_shift) + (1 / t - 1))


# ── Sampling ────────────────────────────────────────────────────────────────


def _get_score_from_velocity(velocity, x, t):
    sigma_t = 1 - t
    var = sigma_t**2 - t * (-1) * sigma_t
    return (t * velocity - x) / var


def _euler_maruyama_step(x, v, t, dt, cfg, cfg_mult):
    with torch.amp.autocast("cuda", enabled=False):
        v = v.to(torch.float32)
        if cfg_mult == 2:
            cond_v, uncond_v = torch.chunk(v, 2, dim=0)
            v = uncond_v + cfg * (cond_v - uncond_v)
        score = _get_score_from_velocity(v, x, t)
        drift = v + (1 - t) * score
        noise_scale = (2.0 * (1.0 - t) * dt) ** 0.5
        return x + drift * dt + noise_scale * torch.randn_like(x)


def _euler_step(x, v, dt, cfg, cfg_mult):
    with torch.amp.autocast("cuda", enabled=False):
        v = v.to(torch.float32)
        if cfg_mult == 2:
            cond_v, uncond_v = torch.chunk(v, 2, dim=0)
            v = uncond_v + cfg * (cond_v - uncond_v)
        return x + v * dt


def euler_maruyama(
    input_dim,
    forward_fn,
    c,
    cfg=1.0,
    num_sampling_steps=20,
    last_step_size=0.04,
    time_shift=1.0,
    prediction_type="x_prev",
):
    if prediction_type != "x_prev":
        raise ValueError("Only x-prediction is supported")
    cfg_mult = 2 if cfg > 1.0 else 1
    x_shape = list(c.shape)
    x_shape[0] = x_shape[0] // cfg_mult
    x_shape[-1] = input_dim
    x = torch.randn(x_shape, device=c.device, dtype=c.dtype)

    t_all = torch.linspace(0, 1 - last_step_size, num_sampling_steps + 1, device=c.device).to(
        dtype=c.dtype
    )
    if time_shift != 1.0:
        t_all = time_shift_func(t_all, time_shift)
    dt = t_all[1:] - t_all[:-1]

    t = torch.tensor(0.0, device=c.device, dtype=c.dtype)
    t_batch = torch.zeros(c.shape[0], device=c.device, dtype=c.dtype)

    for i in range(num_sampling_steps):
        t_batch[:] = t
        combined = torch.cat([x] * cfg_mult, dim=0)
        output = forward_fn(combined, t_batch, c)
        denom = 1 - t_batch.view(-1, 1)
        if output.dim() == 3:
            denom = denom.unsqueeze(-1)
        v = (output - combined) / denom
        x = _euler_maruyama_step(x, v, t, dt[i], cfg, cfg_mult)
        t = t + dt[i]

    # last step: pure euler
    t_batch[:] = 1 - last_step_size
    combined = torch.cat([x] * cfg_mult, dim=0)
    output = forward_fn(combined, t_batch, c)
    denom = 1 - t_batch.view(-1, 1)
    if output.dim() == 3:
        denom = denom.unsqueeze(-1)
    v = (output - combined) / denom
    x = _euler_step(x, v, last_step_size, cfg, cfg_mult)

    return torch.cat([x] * cfg_mult, dim=0)


def sde_step_forward_with_logprob(
    v: torch.Tensor,  # Predicted forward velocity (from pure noise to a clean image)
    t: torch.Tensor,  # Current time (0 to 1)
    t_next: torch.Tensor,  # Next time step (t + dt)
    dt: float,  # Positive step size
    x: torch.Tensor,  # Current state x_t
    x_next_target: torch.Tensor = None,  # Recorded next state (None during rollout; recorded state for RL loss)
    generator: torch.Generator = None,
    sigma_max: float = 0.999,  # Safety threshold to prevent division by zero
    noise_level: float = 0.8,  # SDE exploration strength / \eta in CPS
    sde_type: str = "cps",
):

    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    v, x = v.float(), x.float()
    if x_next_target is not None:
        x_next_target = x_next_target.float()

    tau = 1.0 - t
    tau_next = 1.0 - t_next

    if sde_type == "sde":
        tau_safe = torch.where(tau == 1.0, torch.tensor(sigma_max, device=tau.device), tau)
        sigma_t = torch.sqrt(tau_safe / (1.0 - tau_safe)) * noise_level

        mean = (
            x * (1.0 - (sigma_t**2) / (2.0 * tau_safe) * dt)
            + v * (1.0 + (sigma_t**2) * (1.0 - tau_safe) / (2.0 * tau_safe)) * dt
        )
        std_dev = sigma_t * math.sqrt(dt)

    elif sde_type == "cps":
        std_dev = tau_next * math.sin(noise_level * math.pi / 2)
        x_data = x + v * tau
        x_noise = x - v * t

        mean = x_data * t_next + x_noise * torch.sqrt(tau_next**2 - std_dev**2)

    else:
        raise ValueError(f"Unsupported sde_type: {sde_type}")

    # ==========================================
    # Sample an action
    # ==========================================
    # During inference, sample the next state using the mean and variance; during training, use the supplied target
    if x_next_target is None:
        variance_noise = torch.randn_like(v, generator=generator, device=v.device, dtype=v.dtype)
        x_next = mean + std_dev * variance_noise
    else:
        x_next = x_next_target

    # ==========================================
    # Compute the log probability
    # ==========================================
    if sde_type == "sde":
        # Standard SDE log probability: includes the denominator 2\sigma^2
        variance = std_dev**2
        log_prob = -((x_next.detach() - mean) ** 2) / (2 * variance)
    else:  # cps
        # CPS log probability: the paper authors intentionally omit the denominator 2\sigma^2
        # This weights RL optimization toward early, high-exploration steps and avoids numerical instability from dividing by tiny values at later steps
        log_prob = -((x_next.detach() - mean) ** 2)

    log_prob = log_prob.flatten(start_dim=1).sum(dim=1)

    return x_next, log_prob, mean, std_dev


def euler_maruyama_with_logprob(
    input_dim,
    forward_fn,
    c,
    cfg=1.0,
    num_sampling_steps=20,
    last_step_size=0.04,
    time_shift=1.0,
    prediction_type="x_prev",
):
    if prediction_type != "x_prev":
        raise ValueError("Only x-prediction is supported")
    cfg_mult = 2 if cfg > 1.0 else 1
    x_shape = list(c.shape)
    x_shape[0] = x_shape[0] // cfg_mult
    x_shape[-1] = input_dim
    x = torch.randn(x_shape, device=c.device, dtype=c.dtype)

    t_all = torch.linspace(0, 1 - last_step_size, num_sampling_steps + 1, device=c.device).to(
        dtype=c.dtype
    )
    if time_shift != 1.0:
        t_all = time_shift_func(t_all, time_shift)
    dt = t_all[1:] - t_all[:-1]

    t = torch.tensor(0.0, device=c.device, dtype=c.dtype)
    t_batch = torch.zeros(c.shape[0], device=c.device, dtype=c.dtype)
    vecs_trajectory = [x]
    log_prob = []
    for i in range(num_sampling_steps):
        t_batch[:] = t
        combined = torch.cat([x] * cfg_mult, dim=0)
        output = forward_fn(combined, t_batch, c)
        denom = 1 - t_batch.view(-1, 1)
        if output.dim() == 3:
            denom = denom.unsqueeze(-1)
        v = (output - combined) / denom
        x, logprob, _, _ = sde_step_forward_with_logprob(v, t, t + dt[i], dt[i], x)
        t = t + dt[i]
        vecs_trajectory.append(x)
        log_prob.append(logprob)
    # last step
    t_batch[:] = 1 - last_step_size
    combined = torch.cat([x] * cfg_mult, dim=0)
    output = forward_fn(combined, t_batch, c)
    denom = 1 - t_batch.view(-1, 1)
    if output.dim() == 3:
        denom = denom.unsqueeze(-1)
    v = (output - combined) / denom
    x, logprob, _, _ = sde_step_forward_with_logprob(v, t, t + last_step_size, last_step_size, x)
    vecs_trajectory.append(x)
    log_prob.append(logprob)
    return torch.cat([x] * cfg_mult, dim=0), vecs_trajectory, log_prob


# ── Modules ─────────────────────────────────────────────────────────────────


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size))


# ── MLP Encoder (for single-vector geometry) ────────────────────────────────


class MlpResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        hidden_dim = int(channels * 1.5)
        self.w1 = nn.Linear(channels, hidden_dim * 2)
        self.w2 = nn.Linear(hidden_dim, channels)

    def forward(self, x, scale, shift, gate):
        h = self.norm(x) * (1 + scale) + shift
        h1, h2 = self.w1(h).chunk(2, dim=-1)
        h = self.w2(F.silu(h1) * h2)
        return x + h * gate


class MlpFinalLayer(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.ada_ln = nn.Linear(channels, channels * 2)
        self.linear = nn.Linear(channels, out_channels)

    def forward(self, x, y):
        scale, shift = self.ada_ln(y).chunk(2, dim=-1)
        x = self.norm(x) * (1.0 + scale) + shift
        return self.linear(x)


class MlpEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        z_channels,
        num_res_blocks,
        num_ada_ln_blocks=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList(
            [MlpResBlock(model_channels) for _ in range(num_res_blocks)]
        )
        self.ada_ln_blocks = nn.ModuleList(
            [nn.Linear(model_channels, model_channels * 3) for _ in range(num_ada_ln_blocks)]
        )
        self.ada_ln_switch_freq = max(1, num_res_blocks // num_ada_ln_blocks)
        self.final_layer = MlpFinalLayer(model_channels, self.out_channels)
        self._init_weights()

    def _init_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        for block in self.ada_ln_blocks:
            nn.init.constant_(block.weight, 0)
            nn.init.constant_(block.bias, 0)
        nn.init.constant_(self.final_layer.ada_ln.weight, 0)
        nn.init.constant_(self.final_layer.ada_ln.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = F.silu(t + c)
        scale, shift, gate = self.ada_ln_blocks[0](y).chunk(3, dim=-1)
        for i, block in enumerate(self.res_blocks):
            if i > 0 and i % self.ada_ln_switch_freq == 0:
                scale, shift, gate = self.ada_ln_blocks[i // self.ada_ln_switch_freq](y).chunk(
                    3, dim=-1
                )
            x = block(x, scale, shift, gate)
        return self.final_layer(x, y)


# ── MlpDiffHead ─────────────────────────────────────────────────────────────


class MlpDiffHead(nn.Module):
    def __init__(
        self,
        ch_target,
        ch_cond,
        ch_latent=1024,
        depth=6,
        depth_adaln=2,
        time_shift=1.0,
        time_schedule="logit_normal",
        P_mean=0.0,
        P_std=1.0,
        prediction_type="x_prev",
    ):
        if prediction_type != "x_prev":
            raise ValueError("Only x-prediction is supported")
        super().__init__()
        self.ch_target = ch_target
        self.time_shift = time_shift
        self.time_schedule = time_schedule
        self.P_mean = P_mean
        self.P_std = P_std
        self.prediction_type = prediction_type
        self.net = MlpEncoder(
            in_channels=ch_target,
            model_channels=ch_latent,
            z_channels=ch_cond,
            num_res_blocks=depth,
            num_ada_ln_blocks=depth_adaln,
        )

    def forward(self, x, cond):
        with torch.autocast(device_type="cuda", enabled=False):
            with torch.no_grad():
                if self.time_schedule == "logit_normal":
                    t = (
                        torch.randn(x.shape[0], device=x.device) * self.P_std + self.P_mean
                    ).sigmoid()
                else:
                    t = torch.rand(x.shape[0], device=x.device)
                if self.time_shift != 1.0:
                    t = time_shift_func(t, self.time_shift)
                e = torch.randn_like(x)
                ti = t.view(-1, 1)
                z = (1.0 - ti) * e + ti * x
                v = (x - z) / (1 - ti).clamp_min(0.05)

        x_pred = self.net(z, t, cond)
        v_pred = (x_pred - z) / (1 - ti).clamp_min(0.05)

        with torch.autocast(device_type="cuda", enabled=False):
            loss = ((v - v_pred.float()) ** 2).mean(dim=-1)
        return loss

    def sample(self, cond, cfg=1.0, num_sampling_steps=20):
        return euler_maruyama(
            self.ch_target,
            self.net,
            cond,
            cfg=cfg,
            num_sampling_steps=num_sampling_steps,
            time_shift=self.time_shift,
            prediction_type=self.prediction_type,
        )

    def sample_with_logprob(self, cond, cfg=1.0, num_sampling_steps=20):
        return euler_maruyama_with_logprob(
            self.ch_target,
            self.net,
            cond,
            cfg=cfg,
            num_sampling_steps=num_sampling_steps,
            time_shift=self.time_shift,
            prediction_type=self.prediction_type,
        )


# ── DiffHead (Transformer) ──────────────────────────────────────────────────
