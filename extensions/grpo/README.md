# GRPO extension

Experimental policy optimization, separate from supervised training and the paper's reference path. It imports the core model, sampler and CAD builder; the core never imports this extension.

Run from the repository root:

```bash
python -m extensions.grpo.train --help
```

Supply your own initial checkpoint. This extension requires the CAD dependencies for validity rewards and is GPU-oriented. Its reward design and hyperparameters are research code, not a validated release recipe. Rollout/log-probability helpers shared with trajectory visualization remain in `flowar/models/flow_head.py`.

The current experimental update optimizes topology-token log probabilities. Geometry trajectory replay helpers are included, but geometry-head policy-gradient terms are not enabled. This refactor preserves that objective; it does not claim a complete PPO-style GRPO implementation or paper-quality refinement results.
