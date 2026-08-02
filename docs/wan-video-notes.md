# Wan 2.2 — Video Model Notes (companion to `image-generation-options.md`)

**Researched:** 2026-08-01
**Hardware:** NVIDIA DGX Spark / GB10, aarch64, `sm_121`, 121.7 GB unified memory, ~273 GB/s.
**Scope:** this file covers only the two things that were still open — (1) whether the Wan2.2-Lightning "20x speedup" claim survives scrutiny, and (2) the exact ComfyUI file layout with repo ids and sizes. For still-image model selection see [`image-generation-options.md`](./image-generation-options.md).

**Prior findings carried forward and now confirmed:**
- Open weights stop at Wan 2.2. ✅ Confirmed (§3).
- No true text-to-image checkpoint. ✅ Confirmed (§4).
- 14B fp16 ≈ 61 GB / ~370 s for 81 frames at 640x640 on a Spark. ✅ **Confirmed against the primary source** — Triplany, NVIDIA Dev Forums, 2026-04-28/29: "cold start: 557.53 s, **warm: 370.40 s**", 61 GB cold / 65 GB at higher res, 640x640, 81 frames, 96% GPU utilisation. <https://forums.developer.nvidia.com/t/dgx-spark-comfyui/368179>

---

## 1. The Lightning "20x" claim — verdict

**Verdict: arithmetically honest, but measured by nobody, and against a baseline nobody actually runs. Budget 5–10x, not 20x.**

### 1.1 Where the number comes from

Verbatim from the official README ([ModelTC/Wan2.2-Lightning](https://github.com/ModelTC/Wan2.2-Lightning), [lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning), first release 2025-08-04):

> "Video generation now requires only **4 steps** without the need of CFG trick, leading to **x20 speed-up**"

**There is no measured timing, no GPU named, and no benchmark table anywhere in the official repo or model card.** The 20x is a pure number-of-function-evaluations ratio. It was never substantiated with wall-clock data by the authors. That is the single most important thing to know about it.

### 1.2 The baseline is 40 steps, not 20 — confirmed from source

Read directly out of the official Wan2.2 config files:

| Config file | `sample_steps` | `sample_guide_scale` (CFG) | `boundary` | VAE |
|---|---|---|---|---|
| [`wan_t2v_A14B.py`](https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/wan/configs/wan_t2v_A14B.py) | **40** | (3.0, 4.0) → **CFG on** | 0.875 | Wan2.1 |
| [`wan_i2v_A14B.py`](https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/wan/configs/wan_i2v_A14B.py) | **40** | (3.5, 3.5) → **CFG on** | 0.900 | Wan2.1 |
| [`wan_ti2v_5B.py`](https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/wan/configs/wan_ti2v_5B.py) | **50** | 5.0 | — | Wan2.2 |

### 1.3 Correction: the MoE design does **not** double the passes

The working hypothesis was that the high-noise/low-noise two-expert design means two passes over the timeline. **It does not.** The experts *split* the denoising schedule; they do not each traverse it. From the Wan2.2 README ([Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)):

> "We define a threshold step t_moe corresponding to half of the SNR_min, and **switch to the low-noise expert when t < t_moe**."

That is what `boundary = 0.875` encodes. 40 steps total, split at the boundary, with only **14B active per step** out of ~27B total. **The MoE costs disk and memory, not compute.** Any arithmetic that multiplies steps by 2 for "two experts" is double-counting.

### 1.4 The honest NFE arithmetic

| Configuration | Steps | CFG | Fwd/step | **NFE** | Speedup vs 80 |
|---|---|---|---|---|---|
| Official baseline T2V-A14B | 40 | 3.0 / 4.0 (on) | 2 | **80** | 1x |
| Official baseline I2V-A14B | 40 | 3.5 / 3.5 (on) | 2 | **80** | 1x |
| **Lightning as published (NFE4)** | 4 (2 high + 2 low) | 1.0 (off) | 1 | **4** | **20x** ✅ |
| Lightning as commonly run in ComfyUI | 8 (4 high + 4 low) | 1.0 (off) | 1 | **8** | 10x |
| Community 20-step baseline | 20 (10+10) | ~3.5 (on) | 2 | **40** | 10x |
| Community 20-step baseline vs 4+4 Lightning | — | — | — | — | **5x** |

**80 ÷ 4 = 20. The claim survives on its own terms.** The `NFE4` in the model names (`Wan2.2-T2V-A14B-NFE4-V1`) is the authors explicitly telling you they are counting function evaluations.

**Why you will not see 20x in practice — three reasons:**

1. **Most ComfyUI Lightning workflows run 4+4 = 8 steps, not 2+2 = 4.** Independently confirmed: [earngenix](https://www.earngenix.com/workflows/wan-2-2-fast-comfyui) (2026-07-17) documents "High-noise expert: 4 steps / Low-noise expert: 4 steps / Total: 8"; the [Civitai GGUF workflow](https://civitai.com/models/1822764/wan-22-i2v-gguf-compact-speed-wf-or-lightning-lora-44-steps) is titled "Lightning Lora **4+4 steps**"; [diffusiondoodles](https://diffusiondoodles.substack.com/p/wan22-text-2-image-generation) (2025-08-08) says "4 (**8 in total** across the high and low noise models)". A true 2+2 workflow does exist ([RunComfy](https://www.runcomfy.com/comfyui-workflows/wan-2-2-lightning-t2v-i2v-comfyui-ultra-fast-video-gen), "4 total steps—2 steps for each KSampler") but it is the minority.
2. **Nobody runs the 40-step baseline in ComfyUI.** The community default is ~20 steps. Against that, the ceiling is 10x.
3. **NFE ratio is not wall-clock.** Text encoding (umt5-xxl, 6.7–11.4 GB), VAE decode, and the high→low expert swap are fixed costs that do not shrink with step count. At 4 NFE they become a large fraction of total runtime. ⚠️ *Inference, not a published measurement.*

### 1.5 Published before/after timings (all with the GPU named)

| Source | Date | GPU | Baseline | With Lightning | Assessment |
|---|---|---|---|---|---|
| [earngenix](https://www.earngenix.com/workflows/wan-2-2-fast-comfyui) | 2026-07-17 | RTX 4090 24GB | 40–50 min for 5 s @ 832x480, 121 frames, "20+ steps" | **1–3 min** (4+4, CFG 1.0, euler) | ⚠️ **Do not trust.** 40–50 min for 20 steps at 480p on a 4090 is implausible; the implied 20–40x contradicts its own step counts. Blog-grade, not instrumented. |
| [diffusiondoodles](https://diffusiondoodles.substack.com/p/wan22-text-2-image-generation) | 2025-08-08 | unnamed 16 GB card, Q8 GGUF | **22 s/step** | **170–200 s** total at 4+4 | ✅ **The most internally consistent data point found.** 22 x 8 = 176 s matches. Extrapolating the baseline (20 steps + CFG = 40 NFE x 22 s) gives ~880 s → **~5x**. |
| Search-surfaced, unverified | — | RTX 4090 | ~9 min per 5 s clip, base Wan 2.2 | — | Could not trace to a primary source. Soft. |

**Bottom line: there is no rigorous instrumented A/B benchmark of Wan 2.2 Lightning published anywhere.** Every circulating number is either the unmeasured 20x marketing figure or a blog estimate.

### 1.6 ⚠️ No DGX Spark / GB10 measurement of Lightning exists

Searched specifically. **Zero** published Wan 2.2 *Lightning* timings on GB10. The original "~20x extrapolated from RTX-4090 data" caveat stands — and it is worse than that, because the 4090 data itself is not instrumented.

What does exist for **base** Wan 2.2 on Spark:

| Reporter | Date | Config | Result |
|---|---|---|---|
| Triplany | 2026-04-28/29 | 14B **fp16**, 640x640, 81 frames | **cold 557.53 s, warm 370.40 s**, 61 GB cold / 65 GB higher-res, 96% GPU util. [link](https://forums.developer.nvidia.com/t/dgx-spark-comfyui/368179) |
| Triplany | 2026-04-29 | 14B **t2i fp8**, 640x640, "duration 5" | **cold 644.75 s, warm 565.24 s**, 18 GB. [link](https://forums.developer.nvidia.com/t/my-comfyui-setup-and-patches/368344) |
| bettiwessam296 | — | Wan 2.2 | "15 to 25 minutes"; also a `torch.OutOfMemoryError` trying to allocate **269.47 GiB** against 121.69 GiB, then crash |
| whitezombie2000 | 2026-05-13 | I2V, 5 s clip | DGX Spark native **"30+ minutes"** vs RTX 5070 + ComfyUI **"~3 minutes"**. Others in-thread call 7–10 min the bandwidth-limited expectation, so this run was likely misconfigured. [link](https://forums.developer.nvidia.com/t/image-to-video-generation-using-the-spark-vs-pc/370077) |

⚠️ **Note the anomaly:** the fp8 run (565 s warm, 18 GB) is *slower* than the fp16 run (370 s warm, 61 GB) despite using a third of the memory. The two posts are not directly comparable — something differs in configuration — but it is consistent with the finding in `image-generation-options.md` that on GB10 quantization buys memory, not speed, unless it lands on a native Blackwell matmul path.

⚠️ **My inference, flagged as such:** 121.7 GB unified memory is a genuine structural advantage here. Both fp16 experts (28.6 + 28.6 = 57.2 GB) plus the fp16 text encoder fit resident simultaneously, eliminating the high→low expert swap that bottlenecks 24 GB cards. Triplany's 61 GB resident figure is consistent with exactly that. **Not a published finding.**

### 1.7 Quality tradeoffs — confirmed, and they are the real cost

All primary, from lightx2v HF discussions:

- **["bad motion"](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/5):** "less character and camera movement than the original model, sometimes even **no movement**"; "the new lightning LoRA **kills the motion** and is worse than the one from Wan 2.1."
- **[Discussion 20](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/20):** "The generated character shots are all in **slow motion**."
- **["Blurry movements"](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/25):** "the 2.2 lightning model, specially in the **high noise model**, kills all the complex motions… like a **live wallpaper**."
- **Developer acknowledgement:** the T2V LoRA slows motion and they were working on it; **"the I2V LoRA is better than the T2V LoRA, still worse than the 40-step base model though."**

Note the direct contradiction with the official README, which claims Lightning "retains excellent motion dynamics" and is "on par with the base model, sometimes even better." Consistent user reports say otherwise.

**Community mitigation:** LoRA strength **0.6–0.8 on high**, **1.0 on low**; **CFG 2–3.5 on high**, **CFG 1 on low**. ⚠️ Note this re-enables CFG on the high-noise pass, which **doubles the NFE for those steps** — yet another reason real speedup lands well under 20x.

### 1.8 Newer Lightning versions — and the Comfy-Org mirror is nine months stale

| Version | Date | Folder |
|---|---|---|
| T2V-A14B-NFE4-V1 | 2025-08-04 | `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1` |
| T2V-A14B-NFE4-V1.1 | 2025-08-07 | `…-Seko-V1.1` ("slightly better than V1") |
| I2V-A14B-NFE4-V1 | 2025-08-07 | `Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1` |
| T2V 250928 preview | 2025-09-28 | `…-lora-250928`, `…-250928-dyno` |
| **T2V-A14B-NFE4-V2.0** | **2025-11-08** | `…-Seko-V2.0` |

**V2.0 (2025-11-08) is the current best for T2V** — reported to improve **camera controllability and motion dynamics**, i.e. it directly targets the complaints in §1.7. Phased DMD underpins all lightx2v Wan2.2 distills from 2025-09-28 onward.

- ❌ **No 8-step variant is published.** The "8 steps" in workflows is 4+4 of the *4-step* LoRA, not a distinct model.
- ❌ **No TI2V-5B Lightning LoRA** — still a TODO in the repo.
- ❌ **No 2026 Lightning release.** Latest is 2025-11-08; the line appears dormant.
- ⚠️ **The Comfy-Org repackaged `loras/` folder carries only T2V v1.1 and I2V v1 — NOT V2.0.** For V2.0 you must pull from `lightx2v/Wan2.2-Lightning` directly.

---

## 2. ComfyUI file layout

Primary sources: [docs.comfy.org/tutorials/video/wan/wan2_2](https://docs.comfy.org/tutorials/video/wan/wan2_2), [ComfyUI_examples/wan22](https://comfyanonymous.github.io/ComfyUI_examples/wan22/), plus HuggingFace file listings.

### `models/diffusion_models/`

Repo **`Comfy-Org/Wan_2.2_ComfyUI_Repackaged`**, path `split_files/diffusion_models/` — [listing](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models)

| Filename | Size |
|---|---|
| `wan2.2_t2v_high_noise_14B_fp16.safetensors` | **28.6 GB** |
| `wan2.2_t2v_low_noise_14B_fp16.safetensors` | **28.6 GB** |
| `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | **14.3 GB** |
| `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | **14.3 GB** |
| `wan2.2_i2v_high_noise_14B_fp16.safetensors` | **28.6 GB** |
| `wan2.2_i2v_low_noise_14B_fp16.safetensors` | **28.6 GB** |
| `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | **14.3 GB** |
| `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | **14.3 GB** |
| `wan2.2_ti2v_5B_fp16.safetensors` | **10 GB** |

**You need BOTH the high- and low-noise files — they are not alternatives.** Full fp16 T2V set = **57.2 GB**; fp8_scaled set = **28.6 GB**.

⚠️ **Do not clone this repo.** The folder also holds `wan2.2_animate_14B_bf16` (34.5 GB), `wan2.2_animate_14B_int8_convrot` (18.4 GB), `wan2.2_fun_14B_control_high_noise_bf16` (28.6 GB), `chrono_edit_14B_fp16` (32.8 GB) and more — **the total folder is 702 GB.** Fetch individual files:

```
https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/<filename>
```

### `models/text_encoders/`

⚠️ **These live in the Wan _2.1_ repo, not 2.2** — a common tripwire.
Repo **`Comfy-Org/Wan_2.1_ComfyUI_repackaged`**, path `split_files/text_encoders/` — [listing](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders)

| Filename | Size | Notes |
|---|---|---|
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | **6.74 GB** | ✅ The one the official ComfyUI docs specify, for every Wan 2.2 variant |
| `umt5_xxl_fp16.safetensors` | **11.4 GB** | ⚠️ *Inference:* affordable on 121.7 GB, but **no published quality delta for Wan 2.2** — do not assume it helps |

### `models/vae/`

Repo **`Comfy-Org/Wan_2.2_ComfyUI_Repackaged`**, path `split_files/vae/` — [listing](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/vae)

| Filename | Size | Used by |
|---|---|---|
| `wan_2.1_vae.safetensors` | **254 MB** | ✅ **ALL 14B models** — T2V-A14B, I2V-A14B, FLF2V. Confirmed by `wan_i2v_A14B.py` → `vae_checkpoint = 'Wan2.1_VAE.pth'` |
| `wan2.2_vae.safetensors` | **1.41 GB** | ✅ **5B TI2V only.** Confirmed by `wan_ti2v_5B.py` → `vae_checkpoint = 'Wan2.2_VAE.pth'` |

Why: the new Wan2.2-VAE uses 4x16x16 compression (4x32x32 = 64x with patchification), which is what lets the 5B model exist at all. The 14B experts were trained against the old 2.1 VAE. **Mixing these produces garbage.**

### `models/loras/`

**Option A — Comfy-Org mirror.** Convenient, but **V1.1/V1 only, no V2.0**. Path `split_files/loras/` — [listing](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/loras)

| Filename | Size |
|---|---|
| `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` | **1.23 GB** |
| `wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors` | **1.23 GB** |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | **1.23 GB** |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | **1.23 GB** |

**Option B — upstream [`lightx2v/Wan2.2-Lightning`](https://huggingface.co/lightx2v/Wan2.2-Lightning/tree/main).** Required for V2.0. Every version folder contains exactly two files with **identical generic names**:

- `high_noise_model.safetensors` — **1.23 GB**
- `low_noise_model.safetensors` — **1.23 GB**

⚠️ **Rename on download.** Every folder ships the same two filenames; pulling multiple versions into `models/loras/` will collide silently.

Recommended pulls:
- **T2V (best):** `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/` (2025-11-08) → 2.45 GB
- **I2V (only option):** `Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/` (2025-08-07) → 2.46 GB

### Lightning wiring

Two KSamplers in series — high-noise LoRA on the first, low-noise on the second. **CFG 1.0**, scheduler **euler**, LoRA strength 1.0. Use **2+2 steps** to actually realise the NFE4 claim, or **4+4** for the common quality-biased config. If motion dies, drop the high-noise LoRA to 0.6–0.8 and/or raise CFG on the high pass to 2–3.5 (accepting the NFE cost).

---

## 3. Is there anything newer than Wan 2.2 in open weights?

**No. Wan 2.2 is still the newest open-weights Wan generation as of August 2026.**

Confirmed from the [Wan-AI HuggingFace org listing](https://huggingface.co/Wan-AI) itself: the complete published set is `Wan2.2-T2V-A14B`, `Wan2.2-I2V-A14B`, `Wan2.2-TI2V-5B` (2025-08-07/09), `Wan2.2-S2V-14B` (2025-09-17), `Wan2.2-Animate-14B` (2025-11-05/13). **No Wan 2.5, 2.6, 2.7 or 3.0 weights exist.**

Alibaba changed strategy after 2.2:
- **Wan 2.5-Preview** shipped **September 2025** as **API-only via Alibaba Cloud Bailian**. It was pre-announced as open; the weights never appeared.
- **Wan 2.6 / 2.7** continued the API-only pattern. The monetised capabilities are native audio-video sync, multi-character consistency, and multi-shot narrative.
- ⚠️ The 2.5/2.6/2.7 details come from SEO-flavoured secondary sites ([spheron](https://www.spheron.network/blog/deploy-wan-2-5-gpu-cloud/), [wan27.org](https://wan27.org/blog/wan-2-6-open-source-guide), [flaq.ai](https://flaq.ai/blog/detail/Is-Wan-2-7-Open-Source-API-Only-or-Platform-First-What-to-Expect-Next-f8a95542d6cc/)) — **treat the version numbering as soft.** The load-bearing fact, that nothing past 2.2 is on Wan-AI's HuggingFace, is confirmed from the org listing.

**One genuinely new open-weights release:** [`Wan-AI/Wan-Dancer-14B`](https://huggingface.co/Wan-AI/Wan-Dancer-14B), **2026-07-13** — music-to-dance video, weights + inference code released. Specialised; **not** a Wan 3.0 and not a general T2V/I2V upgrade.

---

## 4. Text-to-image via Wan 2.2 — confirmed dead end

**No true text-to-image checkpoint exists. Single-frame generation is the only route, and it is both slow and unreliable on this box.**

- The Wan-AI org listing contains **no text-to-image model**; every entry is a video model.
- [diffusiondoodles](https://diffusiondoodles.substack.com/p/wan22-text-2-image-generation) (2025-08-08): **"No dedicated T2I checkpoint exists"** — you run the standard T2V workflow with frame count set to 1, still loading *both* experts.
- [Stable Diffusion Art's "Wan 2.2 text-to-image" workflow](https://stable-diffusion-art.com/wan-2-2-text-to-image/) is likewise the T2V graph at length 1.

⚠️ **Quality caveat, worth knowing before spending the download:** the diffusiondoodles author, testing this directly, reported **"inconsistent results with composition and background integration issues"** and a **"low 'good to bad ratio' of output images"**, concluding it was problematic for regular use — despite widespread online claims that Wan 2.2 beats Flux at stills.

⚠️ **Speed caveat:** Triplany's Spark "t2i" measurement is **644.75 s cold / 565.24 s warm at 640x640** — roughly **50x slower than ERNIE-Image-Turbo's measured 11.2 s at 1024x1024** on the same machine. Wan 2.2 is not a still-image option on this hardware under any framing.

---

## 5. Practical takeaways for this box

1. **Budget 5–10x from Lightning, not 20x.** The 20x is real NFE arithmetic against the 40-step official baseline, but it is an unmeasured claim and the config that achieves it (2+2 steps) is the one people abandon for quality reasons.
2. **Pull T2V V2.0 from lightx2v directly**, not the Comfy-Org mirror — the mirror is nine months stale and V2.0 specifically targets the motion complaints.
3. **fp16 is defensible here.** 57.2 GB for both experts fits in 121.7 GB alongside the encoder, avoids fp8 uncertainty on sm_121, and Triplany's 61 GB resident measurement supports it. ⚠️ But note this collides with the resident 90 GB llama.cpp model — you cannot have both.
4. **Plan on SageAttention 2.2 or v3 for sm_121.** flash-attn has no sm_121 wheel and will not get one (FA2/FA3 stop at sm_90, FA4 does sm_100 only). Build paths: [AEON-7/comfyui-aeon-spark](https://github.com/AEON-7/comfyui-aeon-spark) (SA3 for `sm_121a`) or [Triplany/comfyui-dgx-spark](https://github.com/Triplany/comfyui-dgx-spark) (SA2.2 native sm_121). Unlike still images — where Sage measured *slower* than SDPA — attention backend does matter for video's much longer sequences.
5. ⚠️ **`torch.compile` may be unavailable for video.** [ecarmen16/SparkyUI](https://github.com/ecarmen16/SparkyUI/) sets `TORCHDYNAMO_DISABLE=1` because "Triton doesn't support sm_121a". Note this conflicts with still-image reports from May 2026 of working `torch.compile` gains — the situation is version-dependent and worth re-testing.
6. **Do not force `--fp16-vae` globally** (Triplany: caused all-black output on another model). Let ComfyUI auto-detect dtype per model.
7. **Nobody has benchmarked Wan 2.2 Lightning on GB10.** If you run it, you would be producing the first public number.
