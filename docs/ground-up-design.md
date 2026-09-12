# Ground-up design review

Status: superseded on 2026-09-12 by [Inference engine design](inference-engine-design.md).

This file previously captured a ground-up review while Ribn was still framed mainly
as a text-generation engine. The broader audit found that framing itself was too
narrow: the existing `ribn` runtime is an autoregressive token runtime, while the
project target is a general inference engine covering multimodal, encoder/pooling,
encoder-decoder, diffusion/media, and other current model execution regimes.

Use these documents instead:

- [Inference engine design](inference-engine-design.md) — current target architecture,
  public surfaces, model/package boundaries, specialized runtimes, model-support
  strategy, external-engine lessons, and pressure tests.
- [Architecture](architecture.md) — current code versus target architecture.
- [Roadmap](roadmap.md) — ordered redesign, qualification, and implementation gates.
- [Runtime redesign history](runtime-redesign.md) — history of the earlier AR runtime
  refactor and its ownership decisions.

The useful conclusions from the earlier review are carried into those documents;
do not treat its former text-only decomposition as an architectural constraint.
