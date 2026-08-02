# Open-Weights Still-Image Generators for the DGX Spark (GB10)

**Researched:** 2026-08-01
**Target hardware:** NVIDIA DGX Spark / GB10 Grace-Blackwell, aarch64, CUDA `sm_121`, 121.7 GB unified memory (no discrete VRAM), ~273 GB/s memory bandwidth.
**Consumer:** this project's proxy, which already drives a local ComfyUI over HTTP for `/imagine`. ComfyUI-native models are therefore the cheapest to adopt.

> **How to read the numbers in this document.**
> Every timing is tagged with one of three confidence levels:
> - **[MEASURED-SPARK]** — someone published a number taken on a real DGX Spark / GB10. Trust these.
> - **[MEASURED-OTHER]** — measured, but on a different GPU (named). Useful only for ratios, never for absolute planning.
> - **[COMPUTED]** — derived by me from a measured number (e.g. scaling a 50-step figure to 8 steps) or from parameter count. Directionally right, not a promise.
>
> Nothing in this document is a number I invented. Where no data exists I say "no Spark measurement exists".

---

## 1. Executive summary

**Recommended default for this box: `ERNIE-Image-Turbo` (Baidu, 8B, Apache 2.0).**

One-line justification: it is the only model that is simultaneously top-3 in open-weights image quality, **Apache 2.0** (unrestricted commercial use of weights *and* outputs), natively supported in ComfyUI with an official day-0 workflow template, and **actually measured on a DGX Spark at ~11 s per 1024x1024 image** (8 steps) — every other candidate fails at least one of those four.

Runner-up / fallback: **`Z-Image-Turbo`** (6B, Apache 2.0) — smallest download, fastest measured non-distilled-Flux option (7.2 s compiled), slightly lower quality ceiling.
Quality ceiling to evaluate later: **`HiDream-O1-Image-Dev`** (8B, MIT) — currently #1 on the open-weights arena, but **no Spark measurement exists** and it needs 28 steps.

Full reasoning in §7.

---

## 2. The single most important hardware fact

**Image diffusion on GB10 is compute-bound, not bandwidth-bound.** This is the opposite of LLM token generation on the same box, and it inverts most of the tuning advice you will read.

Evidence: FLUX.1-dev is 12B params ≈ 24 GB of bf16 weights and takes ~2.2 s per step at 1024x1024 ([haruni.net, Jan 2026](https://www.haruni.net/en/blog/dgx-spark)). If it were bandwidth-bound at 273 GB/s a step would cost ~0.09 s. It costs 25x that. At 1024x1024 a DiT is chewing ~4k tokens of self-attention per step, and GB10's dense bf16 throughput is the limit.

Three practical consequences:

1. **Step count is the dominant cost driver, not model size.** A 4-step distilled 9B model beats a 50-step 6B model by an order of magnitude. Prefer distilled/Turbo checkpoints. This is why the "biggest model that fits in 121 GB" framing is a trap here.
2. **Quantization that requires dequant-to-bf16 makes things *slower*.** haruni measured `FLUX.2-dev` in bnb **4-bit** as the *slowest of all 11 models tested* — 397 s — while the same model in **fp8_scaled** ran 50.14 s warm in ComfyUI ([NVIDIA forums, "My ComfyUI setup and patches"](https://forums.developer.nvidia.com/t/my-comfyui-setup-and-patches/368344)). Quantization only helps when it hits a hardware matmul path.
3. **But NVFP4 and fp8_scaled *do* help**, because Blackwell has native tensor-core paths for them. Measured NVFP4 gains on Spark are 20–33% ([NVIDIA forums, "Image diffusion speeds", May–Jul 2026](https://forums.developer.nvidia.com/t/image-diffusion-speeds/369095)). Do not confuse this with the LLM-side finding that ["NVFP4 is 1.5x FP8 on a DGX Spark — but it's compression, not the FP4 cores"](https://ai-muninn.com/en/blog/dgx-spark-nvfp4-compression-not-compute); that analysis is about single-stream LLM decode and explicitly warns its conclusions should not be carried over to image diffusion without separate validation.

Corollary: on this box, **do not download 30 GB+ of a big undistilled model expecting the unified memory to save you.** It will fit and it will be slow.

---

## 3. Spark-measured timings (the load-bearing table)

### 3.1 Primary source — NVIDIA Developer Forums, "Image diffusion speeds" (thread 369095)

Reported by user `ijontichy` (with contributions from `vasimv`, `vr8vr8`), **May 5 – July 9 2026**, all at **1024x1024**, HuggingFace `diffusers` pipeline (not ComfyUI). This is the best dataset that exists for this hardware.
Source: <https://forums.developer.nvidia.com/t/image-diffusion-speeds/369095>

| Model | Steps | Precision / backend | **Seconds** | Confidence |
|---|---|---|---|---|
| FLUX.2-klein-9B (distilled) | 4 | bf16 + `torch.compile` | **4.4** | [MEASURED-SPARK] |
| FLUX.2-klein-9B (distilled) | 4 | NVFP4 | **3.3** | [MEASURED-SPARK] |
| Z-Image-Turbo | 9 | bf16, default attention | **12.1** | [MEASURED-SPARK] |
| Z-Image-Turbo | 9 | bf16 + SageAttention | 13.2 | [MEASURED-SPARK] |
| Z-Image-Turbo | 9 | bf16 + "Flash" | 12.9 | [MEASURED-SPARK] |
| Z-Image-Turbo | 9 | bf16 + `torch.compile` | **8.1**, later **7.2** | [MEASURED-SPARK] |
| Z-Image-Turbo | 9 | GGUF Q8_0 + compile | 9.1 | [MEASURED-SPARK] |
| Z-Image-Turbo | 9 | NVFP4 | **5.6** | [MEASURED-SPARK] |
| ERNIE-Image-Turbo | 8 | compiled | **11.2** | [MEASURED-SPARK] |
| ERNIE-Image-Turbo | 8 | NVFP4 | **6.4** | [MEASURED-SPARK] |
| Krea-2-Turbo | 8 | fp16, cuDNN attn backend | **13.9** | [MEASURED-SPARK] |
| Krea-2-Turbo | 8 | NVFP4 | 12.4 | [MEASURED-SPARK] |
| SDXL 1.0 | 30 | bf16, compiled | **11.3** | [MEASURED-SPARK] |
| Qwen-Image-2512 | 50 | (unstated) | **61.0** | [MEASURED-SPARK] |

Tuning findings from the same thread, all Spark-measured:
- `torch.compile()` took Z-Image-Turbo from 12.1 s → 8.1 s. **`torch.compile` works on sm_121 as of ~May 2026** — this contradicts older Spark guides (see §6).
- `DIFFUSERS_ATTN_BACKEND=_native_cudnn` took Krea-2-Turbo from 32 s → 13.9 s. Big win; worth trying everywhere.
- `torch.set_float32_matmul_precision('high')` took Krea-2-Turbo 39.3 s → 32 s.
- SageAttention and Flash were **not faster** than default SDPA for Z-Image (13.2 / 12.9 vs 12.1). Do not spend a day source-building SageAttention for still images on the strength of video-model advice.

### 3.2 Second source — haruni.net, "Image Generation on the DGX Spark: 11 Models Compared"

Published **January 2026**. `diffusers`, bf16, **1024x1024, 50 steps for every model, guidance 4.0, seed 42**.
Source: <https://www.haruni.net/en/blog/dgx-spark>

> **Read this table with care.** Forcing 50 steps on distilled/Turbo models (Z-Image-Turbo, SD3.5-large-turbo) massively misrepresents them — those models are designed for 4–9 steps. The table is best used as a **per-step cost** measurement, which is what the right-hand column gives you.

| Model | Load (s) | Gen (s) @50 steps | Peak mem (GB) | **s/step** [COMPUTED] |
|---|---|---|---|---|
| SD 3.5 medium | 66 | **34** | 22 | 0.68 |
| SD 3.5 large | 171 | 82 | 29 | 1.64 |
| SD 3.5 large turbo | 171 | 82 | 29 | 1.64 |
| FLUX.2-klein-9B | 266 | 95 | 66 | 1.90 |
| FLUX.1-schnell | 291 | 110 | 72 | 2.20 |
| FLUX.1-dev | 284 | 111 | **95** | 2.22 |
| FLUX.1-Kontext-dev | 289 | 111 | 66 | 2.22 |
| Z-Image-Turbo | 63 | 178 | 22 | 3.56 |
| Z-Image (base) | 120 | 179 | 22 | 3.58 |
| Qwen-Image-2512 | 427 | 212 | 63 | 4.24 |
| FLUX.2-dev (bnb 4-bit) | 265 | **397** | 34 | 7.94 |

**Discrepancy flagged:** haruni's Z-Image-Turbo implies 3.56 s/step, while thread 369095 measures 12.1 s / 9 steps = 1.34 s/step — a 2.7x gap. Most likely explanation is the DGX Spark software stack improving between January and May 2026 (a widely-reported January driver/stack update, e.g. [HackerNoon, Jan 2026](https://hackernoon.com/i-was-ready-to-return-my-dgx-spark-then-nvidias-january-update-changed-everything)). **Treat the May–July thread as authoritative and haruni's January absolute numbers as a pessimistic floor.** The *relative* ordering in haruni's table is still informative.

### 3.3 Third source — ComfyUI-specific timings (NVIDIA forums, thread 368344)

These are the only **ComfyUI** (not diffusers) numbers I found, which matters because that is the runtime this project actually calls. Step counts are **not stated** in the post, so treat them as indicative only.
Source: <https://forums.developer.nvidia.com/t/my-comfyui-setup-and-patches/368344>

| Workflow | Res | Cold (s) | Warm (s) | Mem |
|---|---|---|---|---|
| FLUX.1-dev full + T5XXL fp16 | 1024² | 113.17 | **32.61** | 32.16 GB |
| FLUX.2-dev fp8-mixed | 1024² | 300.38 | **50.14** | 68 GB |
| FLUX.2-dev full bf16 + Mistral-3-Small bf16 | 1024² | 407.52 | **80.25** | 93.80 GB |
| Z-Image T2I bf16 | 1024² | 96.17 | 43.73 | 43.5 GB |
| Wan 2.2 14B "T2I" fp8 | 640² | 644.75 | 565.24 | 18 GB |

**Cold-start cost is brutal and is the thing that will actually annoy `/imagine` users.** Model load is 100–400 s on this box (see haruni's "Load" column too — 427 s for Qwen-Image). Keep ComfyUI resident with the model cached; the difference between cold and warm is 3–8x. The Z-Image ComfyUI figure (43.73 s) is inconsistent with the diffusers figure (12.1 s) — steps were not reported, so I would not plan against it.

---

## 4. Model-by-model comparison

Sizes: "[CONFIRMED]" means I found the figure published; "[COMPUTED]" means I derived it as params x bytes-per-weight and it may be off by the VAE/encoder overhead. **Total download** includes the text encoder and VAE, which are frequently forgotten and are often *bigger than the DiT*.

---

### ERNIE-Image / ERNIE-Image-Turbo — Baidu
- **Params:** 8B single-stream DiT (36 layers, hidden 4096, FFN 12288, 32 heads).
- **Released:** 2026-04-15. <https://github.com/baidu/ernie-image>
- **Licence:** **Apache 2.0** — commercial use of weights and outputs permitted, no revenue cap, no filtering obligation. Cleanest licence of any top-tier model.
- **Size:** DiT bf16 ~16 GB [COMPUTED]. Plus `ministral-3-3b` text encoder (~6 GB [COMPUTED]) + a small prompt-enhancer LM + `flux2-vae`. **Total ~23 GB** [COMPUTED]. Base (SFT) variant stated to "run on 24 GB VRAM" [CONFIRMED].
- **Quality:** Elo 1166 (base) / 1163 (Turbo) on the Artificial Analysis arena as mirrored by [theopenweights.com, updated 2026-08-01](https://theopenweights.com/leaderboards/text-to-image) — **#2 and #3 among open weights**, above FLUX.2 [dev] (1154). GenEval 0.8667 for Turbo, vs FLUX.2-klein-9B 0.8481 and Qwen-Image 0.8683 ([HF model card](https://huggingface.co/baidu/ERNIE-Image-Turbo)). Standout strength is **legible in-image text** — multi-line copy, bilingual signage, labelled diagrams.
- **ComfyUI:** **Native, day-0.** Official tutorial at <https://docs.comfy.org/tutorials/image/ernie-image/ernie-image>; official templates `image_ernie_image.json` and `image_ernie_image_turbo.json` in Comfy-Org/workflow_templates; repackaged weights at `Comfy-Org/ERNIE-Image`.
  - `ernie-image-turbo.safetensors` → `models/diffusion_models/`
  - `ministral-3-3b.safetensors` → `models/text_encoders/`
  - `ernie-image-prompt-enhancer.safetensors` → `models/text_encoders/`
  - `flux2-vae.safetensors` → `models/vae/`
- **Steps/CFG:** Turbo = 8 steps, CFG 1.0. Base = ~50 steps.
- **Spark speed:** **11.2 s compiled / 6.4 s NVFP4** at 8 steps 1024² [MEASURED-SPARK]. Base variant at 50 steps ≈ 70 s [COMPUTED from the same per-step cost].

---

### Z-Image-Turbo / Z-Image-Base — Alibaba Tongyi Lab
- **Params:** 6B, S3-DiT (Scalable Single-Stream DiT).
- **Released:** Turbo 2025-11-27; **Base (undistilled) 2026-01-28** ([ComfyUI Wiki](https://comfyui-wiki.com/en/news/2026-01-28-alibaba-z-image-base-release)). Z-Image-Edit announced but **not released** as of Aug 2026.
- **Licence:** **Apache 2.0** for Turbo [CONFIRMED]. Base licence not separately confirmed — assume Apache 2.0 but **verify on the model card before commercial use**.
- **Size:** `z_image_turbo_bf16.safetensors` ~12 GB [COMPUTED] + `qwen_3_4b.safetensors` text encoder ~8 GB [COMPUTED] + `ae.safetensors` VAE ~0.3 GB. **Total ~20 GB** — the smallest credible full stack here. fp8 runs in 8 GB, GGUF down to 6 GB [CONFIRMED, consumer-GPU context].
- **Quality:** best-in-class *photorealism per parameter*; well below ERNIE/HiDream on the arena. GenEval: Z-Image 0.8400, Z-Image-Turbo 0.8233.
- **ComfyUI:** **Native.** <https://docs.comfy.org/tutorials/image/z-image/z-image-turbo>, repo `Comfy-Org/z_image_turbo`, files as listed above into `diffusion_models/`, `text_encoders/`, `vae/`.
- **Steps/CFG:** Turbo = 8 NFE (docs) / 9 steps as benchmarked. Base = 30–50 steps at CFG 3–5.
- **Spark speed:** **12.1 s default, 7.2 s with `torch.compile`, 5.6 s NVFP4** at 9 steps 1024² [MEASURED-SPARK]. A separate Spark report ([thread 371788](https://forums.developer.nvidia.com/t/z-image-turbo-nvfp4/371788), planoform, 2026-05-30) gives NVFP4 at ~19 s gen / 47 s cold load, peak 10.1 GB vs bf16's ~23 GB — slower than ijontichy's 5.6 s, so **the NVFP4 win is implementation-sensitive**; the reliable claim is the memory drop (23 GB → 10.1 GB, quality intact), not a specific speed.

---

### FLUX.2 [klein] 4B / 9B — Black Forest Labs
- **Params:** 4B and 9B rectified-flow transformers, each in **base** (50-step) and **distilled** (4-step) form. 9B pairs with an 8B Qwen3 embedder.
- **Released:** **2026-01-15.** <https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence>
- **Licence:** **4B = Apache 2.0. 9B = FLUX Non-Commercial Licence.** This split is the single most important licensing fact in this document — the fast, good one is the one you cannot sell output from. Confirmed on the BFL blog and HF cards.
- **Size:** 9B fp8 ≈ 9–10 GB (Q8_0 GGUF ~10 GB vs "18 GB non-quantized") [CONFIRMED-ish, from community guides]. 4B fp8 ≈ 4 GB [COMPUTED]. Text encoders: 4B uses `qwen_3_4b.safetensors`; 9B uses `qwen_3_8b_fp8mixed.safetensors`. Shared `flux2-vae.safetensors`.
- **Quality:** BFL claim the 9B "matches or exceeds models 5x its size". Spark-measured GenEval puts klein-9B (0.8481) slightly *below* ERNIE-Turbo and Qwen-Image.
- **ComfyUI:** **Native.** <https://docs.comfy.org/tutorials/flux/flux-2-klein>. Files: `flux-2-klein-9b-fp8.safetensors` (or `-4b-`, or `-base-` variants) → `diffusion_models/`; encoder → `text_encoders/`; `flux2-vae.safetensors` (from `Comfy-Org/flux2-dev`) → `vae/`.
- **Spark speed:** **9B distilled: 4.4 s compiled, 3.3 s NVFP4** at 4 steps 1024² [MEASURED-SPARK]. **This is the fastest good model measured on this hardware.** 9B base at 50 steps ≈ 95 s [MEASURED-SPARK, haruni].
- **4B speed: no Spark measurement exists.** Nearest comparable: ComfyUI docs quote **~1.2 s distilled / ~17 s base on an RTX 5090** [MEASURED-OTHER]. Scaling by the klein-9B ratio suggests roughly 2–4 s on Spark [COMPUTED, low confidence].

---

### FLUX.2 [dev] — Black Forest Labs
- **Params:** 32B flow-matching transformer, paired with a Mistral-3-Small text encoder.
- **Released:** 2025-11-25 ([MarkTechPost](https://www.marktechpost.com/2025/11/25/black-forest-labs-releases-flux-2-a-32b-flow-matching-transformer-for-production-image-pipelines/)).
- **Licence:** **FLUX [dev] Non-Commercial Licence v2.0** — <https://bfl.ai/legal/non-commercial-license-terms>. Commercial use requires a paid/royalty licence from BFL.
- **Size:** bf16 ≈ 64 GB DiT [COMPUTED] + Mistral-3-Small encoder; **measured 93.8 GB resident** in ComfyUI at bf16, 68 GB at fp8-mixed [MEASURED-SPARK].
- **Quality:** Elo 1154, #4 open weights. Excellent multi-reference / editing.
- **ComfyUI:** Native.
- **Spark speed:** **80.25 s warm at bf16, 50.14 s warm at fp8-mixed** [MEASURED-SPARK, ComfyUI]. In bnb 4-bit it is *worse*: 397 s [MEASURED-SPARK].
- **Verdict:** technically runnable here thanks to unified memory, and this is the model that shows off the box — but it is non-commercial, it eats 68–94 GB (colliding with your resident 90 GB llama.cpp model), and it is 10x slower than klein-9B for a small quality gain.

---

### FLUX.1 family — dev / schnell / Krea / Kontext
- **Params:** 12B for all four.
- **Licence:** **schnell = Apache 2.0.** **dev, Krea-dev, Kontext-dev = FLUX.1 Non-Commercial Licence.** ([HF card](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md))
- **Size:** DiT bf16 ~23.8 GB / fp8 ~11.9 GB [COMPUTED]; + T5-XXL (~9.8 GB fp16 / 4.9 GB fp8) + CLIP-L (~0.25 GB) + VAE (~0.33 GB).
- **Variant roles:** `dev` = general; `schnell` = 4-step distilled, Apache 2.0; `Krea-dev` = opinionated photographic aesthetic, kills the "AI look" (non-commercial); `Kontext-dev` = instruction-based *image editing*, not really a T2I model (non-commercial).
- **ComfyUI:** Native, extremely mature — the largest LoRA/ControlNet ecosystem of anything here. That ecosystem is now FLUX.1's main remaining advantage.
- **Spark speed:** dev at 50 steps = 111 s / 95 GB [MEASURED-SPARK, haruni]; dev in ComfyUI warm = **32.61 s** at 32.16 GB [MEASURED-SPARK]. schnell at 50 steps = 110 s → **~8.8 s at its native 4 steps** [COMPUTED]. Kontext-dev = 111 s @ 50 steps / 66 GB [MEASURED-SPARK].
- **Verdict:** superseded on quality-per-second by FLUX.2 klein and by ERNIE/Z-Image. Keep FLUX.1-dev/Kontext installed only if you need the LoRA ecosystem or instruction-editing.

---

### Qwen-Image / Qwen-Image-2512 — Alibaba
- **Params:** 20B MMDiT.
- **Released:** original Aug 2025; **Qwen-Image-2512 on 2025-12-31** ([Qwen blog](https://qwen.ai/blog?id=qwen-image-2512)) — a substantial realism and text-rendering upgrade.
- **Licence:** **Apache 2.0.** Commercially clean.
- **Size:** bf16 ≈ 40 GB (needs 40 GB+ VRAM) [CONFIRMED]; **fp8 download ~26.7 GB** [CONFIRMED]; GGUF ladder Q2_K ~7 GB, Q3_K ~9 GB, Q4 12–13 GB, Q5 14–15 GB, Q6 16.8 GB, Q8 21.8 GB [CONFIRMED]. Plus a Qwen2.5-VL text encoder (~16 GB bf16) and VAE.
- **Quality:** the reference model for *dense multilingual text in images*; GenEval 0.8683 — the highest raw GenEval in this document. Qwen claimed strongest open-source model on AI Arena at 2512 launch.
- **ComfyUI:** **Native**, plus GGUF and Nunchaku paths. <https://docs.comfy.org/tutorials/image/qwen/qwen-image-2512>
- **Spark speed:** **61.0 s at 50 steps** [MEASURED-SPARK, thread 369095]; haruni measured 212 s at 50 steps in January [MEASURED-SPARK] — again the January/May software gap. **Load time 427 s** [MEASURED-SPARK] — the worst cold start of anything tested.
- **Verdict:** best Apache-2.0 *quality ceiling*, but 5x slower than ERNIE-Turbo and with a punishing cold start. Good "high-effort" second model, bad default.

---

### HiDream-O1-Image / -Dev — HiDream.ai
- **Params:** 8B unified transformer, **pixel-native (no VAE, no external latent space)**.
- **Released:** 2026-05-08; `Dev-2604` checkpoint 2026-05-14.
- **Licence:** **MIT.** The most permissive licence in this entire comparison.
- **Size:** bf16 ~16 GB [COMPUTED]; fp8_scaled and mxfp8 variants published by Comfy-Org. Needs a `gemma4_e4b_it_fp8_scaled` text encoder (from `Comfy-Org/gemma-4`).
- **Quality:** **#1 open-weights on the arena** — Elo 1188 vs FLUX.2 [dev]'s 1154 ([theopenweights.com, 2026-08-01](https://theopenweights.com/leaderboards/text-to-image)). Self-reported GenEval 0.90 (vs FLUX.2 0.87, Qwen-Image 0.87), DPG-Bench 89.83, HPSv3 10.37, English long-text rendering 0.979.
- **ComfyUI:** **Native.** <https://docs.comfy.org/tutorials/image/hidream/hidream-o1> — checkpoints go in `models/checkpoints/` (note: *checkpoints*, not `diffusion_models/`), encoder in `text_encoders/`, optional distillation LoRAs from Kijai in `loras/`.
- **Steps/CFG:** full = 50 steps; **Dev = 28 steps, CFG 1.0**. Native up to 2048x2048.
- **Spark speed:** **NO SPARK MEASUREMENT EXISTS.** I could not find a single published GB10 timing for HiDream-O1. Nearest usable anchor: it is an 8B DiT like ERNIE-Image, which costs ~1.4 s/step on Spark, so 28 steps ≈ **~39 s** [COMPUTED, medium confidence — the pixel-native no-VAE architecture means attention cost may not scale like a latent DiT, and this estimate could easily be off by 2x in either direction].
- **Verdict:** the most interesting model to *test*, and licence-wise the best. Not the default until someone times it here.

---

### Krea 2 (Raw / Turbo) — Krea AI
- **Params:** 12B DiT.
- **Released:** 2026-06-22, weights on HF 2026-06-23.
- **Licence:** **Krea 2 Community Licence** — *not* a plain open licence. It obliges deployers to implement content filtering or an equivalent review process. Commercially usable but with a compliance string attached; read it before shipping (<https://www.krea.ai/krea-2-licensing>).
- **Size:** `krea2_turbo_fp8_scaled.safetensors` ~12 GB [COMPUTED] + `qwen3vl_4b_fp8_scaled.safetensors` encoder + `qwen_image_vae.safetensors`. Repo `Comfy-Org/Krea-2`.
- **Quality:** top-10 on Artificial Analysis, 2nd among independent labs. Strong aesthetic/"non-AI-looking" reputation.
- **ComfyUI:** **Native** — first mainstream turbo model with native ComfyUI node support. <https://docs.comfy.org/tutorials/image/krea/krea-2>. Turbo default 8 steps; Raw supports 52 steps. A `krea2_style_reference.safetensors` LoRA enables style-reference workflows.
- **Spark speed:** **13.9 s** at 8 steps fp16 with the cuDNN attention backend; 12.4 s NVFP4 [MEASURED-SPARK]. Note it was **32 s until the cuDNN backend was set** — a 2.3x config trap.

---

### Ideogram 4.0 — Ideogram
- **Params:** 9.3B single-stream DiT + vision-language text encoder; supports structured JSON prompts for layout/colour/text placement; native 2K.
- **Released:** 2026-06-03. Ranks #1 on DesignArena among open-weight models.
- **Licence:** **Ideogram Non-Commercial Model Agreement — no revenue-generating use at any scale.** (Inference *code* is Apache 2.0; the weights are not.) Gated on HF.
- **ComfyUI:** **Not natively supported as local weights.** ComfyUI's Ideogram integration is an **API partner node** (JSON prompting via the Bounding Box Canvas node), i.e. it calls Ideogram's servers, not your GPU.
- **Spark speed:** no measurement, and largely moot.
- **Verdict:** **exclude.** Non-commercial *and* not locally wired into ComfyUI is a double disqualification for this project.

---

### Stable Diffusion 3.5 (Medium / Large / Large-Turbo) — Stability AI
- **Params:** Medium 2.5B, Large 8B.
- **Licence:** **Stability AI Community Licence** — free for commercial use *only* while the organisation is under **$1M annual revenue** (from any source); above that requires a paid Enterprise licence. <https://stability.ai/news-updates/license-update>
- **Size:** Medium ~5 GB fp16, Large ~16 GB fp16 [COMPUTED]; peak resident measured at **22 GB (Medium) / 29 GB (Large)** [MEASURED-SPARK].
- **Quality:** SD3 Large sits at the **bottom** of the current open-weights leaderboard (Elo 1027, rank 20 of 20). Comprehensively obsolete on quality.
- **ComfyUI:** Native, mature.
- **Spark speed:** Medium **34 s** and Large **82 s** at 50 steps [MEASURED-SPARK, haruni] — the *fastest per-step* of anything haruni tested (0.68 s/step for Medium). Large-Turbo at its native 4 steps ≈ **6.6 s** [COMPUTED].
- **Verdict:** only interesting as a latency floor. The licence revenue cap plus 2024-era quality rules it out as a default.

---

### SDXL 1.0 — Stability AI
- **Params:** 3.5B UNet.
- **Licence:** **CreativeML OpenRAIL++-M** — permissive, **no revenue cap**, use-restriction clauses only. Notably *more* permissive commercially than SD3.5.
- **Size:** ~6.9 GB fp16 single-file checkpoint (UNet + CLIP + VAE).
- **Quality:** far behind everything modern on prompt adherence and text rendering, but has by far the deepest fine-tune/LoRA/ControlNet ecosystem in existence (Civitai etc.).
- **ComfyUI:** Native, the original target.
- **Spark speed:** **11.3 s at 30 steps 1024², bf16 compiled** [MEASURED-SPARK].
- **Verdict:** keep as a cheap stylistic/LoRA workhorse; not a default.

---

### Also-rans and things deliberately not recommended
- **HunyuanImage 3.0 (Tencent, 80B MoE):** frequently listed among the 2026 leaders, but the Tencent community licence carries territorial restrictions (notably excluding the EU) and the size is impractical against your resident 90 GB LLM. Not investigated further.
- **NVIDIA Sana:** fast and small, but quality is well below the 2026 pack.
- **"Cosmos3-Super-Text2Image":** appeared in one secondary SEO source claiming a 1,219 Elo lead in July 2026. **I could not corroborate this** against the leaderboard I actually fetched (which is topped by HiDream-O1 at 1,188 as of 2026-08-01) or against any first-party NVIDIA source. **Treat as unverified; do not plan a download around it.**
- **Qwen-Image-Max:** appears on leaderboards but is an API/closed offering, not open weights.

---

## 5. Consolidated comparison table

Times are 1024x1024 on **DGX Spark** at each model's *native* step count, best measured configuration.

| Model | Params | Licence | Commercial? | ComfyUI native | Native steps | **Spark s/img** | Confidence | Total download |
|---|---|---|---|---|---|---|---|---|
| **ERNIE-Image-Turbo** | 8B | Apache 2.0 | **Yes, unrestricted** | Yes (day-0) | 8 | **11.2** (6.4 NVFP4) | MEASURED-SPARK | ~23 GB |
| **Z-Image-Turbo** | 6B | Apache 2.0 | **Yes, unrestricted** | Yes | 8–9 | **7.2** compiled (5.6 NVFP4) | MEASURED-SPARK | ~20 GB |
| FLUX.2-klein-9B distilled | 9B | FLUX NCL | **No** | Yes | 4 | **4.4** (3.3 NVFP4) | MEASURED-SPARK | ~18 GB |
| FLUX.2-klein-4B distilled | 4B | Apache 2.0 | **Yes** | Yes | 4 | ~2–4 | COMPUTED (5090: 1.2 s) | ~12 GB |
| HiDream-O1-Image-Dev | 8B | **MIT** | **Yes, unrestricted** | Yes | 28 | ~39 | **COMPUTED — no Spark data** | ~24 GB |
| Krea-2-Turbo | 12B | Krea Community | Yes, w/ filtering duty | Yes | 8 | **13.9** (12.4 NVFP4) | MEASURED-SPARK | ~18 GB |
| SDXL 1.0 | 3.5B | OpenRAIL++-M | **Yes** | Yes | 30 | **11.3** | MEASURED-SPARK | ~7 GB |
| ERNIE-Image (base SFT) | 8B | Apache 2.0 | **Yes** | Yes | 50 | ~70 | COMPUTED | ~23 GB |
| Qwen-Image-2512 | 20B | Apache 2.0 | **Yes** | Yes | 50 | **61.0** | MEASURED-SPARK | ~27 GB fp8 |
| FLUX.1-schnell | 12B | Apache 2.0 | **Yes** | Yes | 4 | ~8.8 | COMPUTED from 110 s@50 | ~35 GB |
| FLUX.1-dev | 12B | FLUX NCL | **No** | Yes | 20–50 | **32.6** (ComfyUI warm) | MEASURED-SPARK | ~35 GB |
| FLUX.1-Kontext-dev | 12B | FLUX NCL | **No** | Yes | 50 | **111** | MEASURED-SPARK | ~35 GB |
| SD 3.5 Medium | 2.5B | Stability Community | Under $1M rev only | Yes | 28 | ~19 | COMPUTED from 34 s@50 | ~11 GB |
| SD 3.5 Large-Turbo | 8B | Stability Community | Under $1M rev only | Yes | 4 | ~6.6 | COMPUTED from 82 s@50 | ~20 GB |
| FLUX.2-dev | 32B | FLUX NCL | **No** | Yes | 50 | **50.1** fp8 / **80.3** bf16 | MEASURED-SPARK | ~64 GB+ |
| Ideogram 4.0 | 9.3B | Ideogram NC | **No** | API node only | — | n/a | — | gated |

---

## 6. Operational notes specific to this box

**Conflicting advice on ComfyUI flags — resolve by testing, don't assume.**
Earlier research for this project concluded ComfyUI needs `--disable-mmap` to address more than 64 GB. The Spark-specific tuning kit [Triplany/comfyui-dgx-spark](https://github.com/Triplany/comfyui-dgx-spark) explicitly says to **avoid** `--disable-mmap`, along with `--gpu-only` and global `--fp16-unet/vae/text-enc`, because forcing global precision breaks per-model auto-detection (it reports `--fp16-vae` producing all-black LTX 2.3 video). **These two claims are not reconciled.** Since every model recommended below stays under ~25 GB resident, the >64 GB question does not arise for the recommended default — which is a good reason to pick a small model and sidestep the issue entirely.

**Attention backends.** flash-attn has no sm_121 kernels and is excluded by every Spark-specific build. SageAttention **2.2** builds against sm_121 and works; SageAttention **3** reportedly produces mosaic visual artifacts on Spark (Triplany kit, citing upstream issue #321) — though [AEON-7/comfyui-aeon-spark](https://github.com/AEON-7/comfyui-aeon-spark) does compile SA3 for `sm_121a`. **For still images this is moot:** Spark measurements show Sage (13.2 s) and Flash (12.9 s) were *slower* than default SDPA (12.1 s) for Z-Image. Skip the source build unless you are doing video.

**`torch.compile` status changed during 2026.** The AEON kit disables it, stating "Triton doesn't yet emit working SASS for sm_121a". But thread 369095 reports working `torch.compile` gains from May 2026 onward (12.1 s → 8.1 s on Z-Image). **Try it; it is the single biggest free win available** (~40%). Budget for a long first-run compile.

**Attention backend env var.** `DIFFUSERS_ATTN_BACKEND=_native_cudnn` gave a 2.3x improvement on Krea-2. Worth testing per-model.

**Quantization decision rule for this box:**
- fp8_scaled / mxfp8 / NVFP4 → **use** (native Blackwell matmul, 20–33% faster, big memory savings).
- bnb 4-bit / other dequant-to-bf16 schemes → **avoid** (measured 8x slower on FLUX.2-dev).
- bitsandbytes is excluded from Spark builds anyway (no ARM64 sm_121 kernels).

**Cold start is the real UX problem.** 100–430 s to load. If `/imagine` is interactive, keep ComfyUI warm with one model pinned rather than swapping models per request. This argues strongly for a *single small default* plus explicit opt-in to heavier models, not a model picker that thrashes.

**Memory budget conflict.** llama.cpp already holds a 90 GB model on this box. With 121.7 GB unified, anything above ~25 GB resident for image gen forces you to evict the LLM. FLUX.2-dev (68–94 GB) and FLUX.1-dev (95 GB at bf16 in diffusers) are effectively mutually exclusive with the LLM. Every model recommended below fits in the ~25 GB headroom.

---

## 7. Recommendation

### Best default image model for this box: **ERNIE-Image-Turbo (8B, Apache 2.0)**

**Justification (one line):** it is the only option that is Apache-2.0 unrestricted, top-3 in open-weights quality, natively wired into ComfyUI with an official template, *and* has a real DGX Spark measurement (11.2 s at 1024² / 8 steps, 6.4 s with NVFP4) — everything faster is either non-commercial (FLUX.2-klein-9B) or measurably lower quality (Z-Image-Turbo).

**Concrete install** (~23 GB, all from `Comfy-Org/ERNIE-Image`):
```
models/diffusion_models/ernie-image-turbo.safetensors
models/text_encoders/ministral-3-3b.safetensors
models/text_encoders/ernie-image-prompt-enhancer.safetensors
models/vae/flux2-vae.safetensors
```
Load the `image_ernie_image_turbo.json` template. 8 steps, CFG 1.0, 1024x1024 (also supports 848x1264, 1264x848, 768x1376, 896x1200, 1376x768, 1200x896).

### Secondary picks, in priority order

1. **Z-Image-Turbo (6B, Apache 2.0)** — install alongside; ~20 GB, 7.2 s compiled. Use it as the low-latency path for `/imagine` when the user does not need text-in-image. It is the best-documented model on this hardware (most Spark measurements of any model) so it is also the right thing to benchmark your own setup against.
2. **HiDream-O1-Image-Dev (8B, MIT)** — the highest-quality open-weights model available and the most permissive licence. **Worth an evaluation run precisely because nobody has timed it on a Spark.** If it lands near the ~39 s estimate it is a good "high quality" tier; if `torch.compile` + fp8_scaled bring it under 20 s it should displace ERNIE as the default.
3. **FLUX.2-klein-4B distilled (Apache 2.0)** — the licence-clean member of the fastest family. No Spark number exists (1.2 s on a 5090); if it lands at 2–4 s it becomes the obvious "fast draft" tier.
4. **SDXL 1.0** — keep installed at 7 GB purely for the LoRA/ControlNet ecosystem.

### Explicitly not recommended
- Anything in the **FLUX NCL** family (FLUX.1-dev/Krea/Kontext, FLUX.2-dev, FLUX.2-klein-9B) if outputs might ever be commercial — the licence is genuinely restrictive and BFL charge for the commercial grant.
- **FLUX.2-dev** on capacity grounds regardless of licence: 68–94 GB collides with the resident LLM.
- **Ideogram 4.0** — non-commercial *and* API-only in ComfyUI.
- **SD 3.5 / SD 3** — bottom of the quality leaderboard plus a revenue-capped licence.

### Open questions worth an hour of local benchmarking
1. Time HiDream-O1-Image-Dev on this box. It is the biggest unknown and potentially the best answer.
2. Confirm whether `torch.compile` currently works in *your* ComfyUI (not diffusers) on sm_121 — the ~40% win depends on it and sources disagree by date.
3. Time FLUX.2-klein-4B distilled; if it is ~2 s it changes the default.
4. Resolve the `--disable-mmap` conflict empirically, though the recommended stack avoids needing it.

---

## Sources

- NVIDIA Developer Forums — "Image diffusion speeds" (May–Jul 2026): <https://forums.developer.nvidia.com/t/image-diffusion-speeds/369095>
- NVIDIA Developer Forums — "My ComfyUI setup and patches": <https://forums.developer.nvidia.com/t/my-comfyui-setup-and-patches/368344>
- NVIDIA Developer Forums — "Z-Image Turbo NVFP4" (2026-05-30): <https://forums.developer.nvidia.com/t/z-image-turbo-nvfp4/371788>
- haruni.net — "Image Generation on the DGX Spark: 11 Models Compared" (Jan 2026): <https://www.haruni.net/en/blog/dgx-spark>
- The Open Weights — text-to-image leaderboard (updated 2026-08-01): <https://theopenweights.com/leaderboards/text-to-image>
- Baidu ERNIE-Image (2026-04-15): <https://github.com/baidu/ernie-image> · <https://huggingface.co/baidu/ERNIE-Image-Turbo>
- ComfyUI docs — ERNIE-Image: <https://docs.comfy.org/tutorials/image/ernie-image/ernie-image>
- ComfyUI docs — Z-Image-Turbo: <https://docs.comfy.org/tutorials/image/z-image/z-image-turbo>
- ComfyUI docs — HiDream-O1: <https://docs.comfy.org/tutorials/image/hidream/hidream-o1>
- ComfyUI docs — FLUX.2 klein: <https://docs.comfy.org/tutorials/flux/flux-2-klein>
- ComfyUI docs — Krea 2: <https://docs.comfy.org/tutorials/image/krea/krea-2>
- ComfyUI docs — Qwen-Image-2512: <https://docs.comfy.org/tutorials/image/qwen/qwen-image-2512>
- Black Forest Labs — FLUX.2 [klein] announcement (2026-01-15): <https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence>
- Black Forest Labs — FLUX [dev] Non-Commercial Licence v2.0: <https://bfl.ai/legal/non-commercial-license-terms>
- HiDream-O1-Image-Dev model card: <https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev>
- Qwen — Qwen-Image-2512 (2025-12-31): <https://qwen.ai/blog?id=qwen-image-2512>
- Alibaba Z-Image-Base release (2026-01-28): <https://comfyui-wiki.com/en/news/2026-01-28-alibaba-z-image-base-release>
- Stability AI Community Licence: <https://stability.ai/news-updates/license-update>
- Triplany/comfyui-dgx-spark (Spark tuning kit): <https://github.com/Triplany/comfyui-dgx-spark>
- AEON-7/comfyui-aeon-spark (Spark CUDA13/SA3/NVFP4 build): <https://github.com/AEON-7/comfyui-aeon-spark>
- ai-muninn — "NVFP4 is 1.5x FP8 on a DGX Spark — but it's compression, not the FP4 cores": <https://ai-muninn.com/en/blog/dgx-spark-nvfp4-compression-not-compute>
- MarkTechPost — FLUX.2 32B release (2025-11-25): <https://www.marktechpost.com/2025/11/25/black-forest-labs-releases-flux-2-a-32b-flow-matching-transformer-for-production-image-pipelines/>
