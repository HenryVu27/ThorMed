# Mid- to High-Effort Mask Refinement Options (Steps 2 – 4)

This companion document captures the more resource-intensive strategies that can push segmentation accuracy beyond what the lightweight Step-1 pipeline delivers.  Keep it for reference when you have access to extra compute time, additional clinical reviewers, or a research lab environment.

---
## 2 Learnable Refinement Modules

### 2-A RefineNet-lite (U-Net Residual Corrector)
* **Inputs** `{ultrasound_image, coarse_mask}` concatenated along the channel axis.
* **Architecture** 
  * Encoder: frozen weights from a U-Net trained on public abdominal ultrasound (e.g. BUSI, KARMEN).
  * Decoder: 4 up-sampling blocks (bilinear + conv) predicting a *residual* to be added to the coarse mask logits.
* **Loss Combo**
  * Focal-Tversky (λ=0.7)
  * Signed Distance Map (boundary loss)
  * Log Hausdorff (95 %)
* **Training Recipe**
  * 128 × 128 random crops centred on the bladder.
  * Heavy rotations ±25°, elastic deformation, speckle noise.
  * Freeze encoder, train decoder for 5–10 epochs (batch-16, Adam 1e-4).
  * Early-stop on validation HD95.
* **Typical Gain** +1 – 4 pp Dice, −10 – 20 % HD95 against the SAM baseline.

### 2-B Shape-Aware Adversarial Refinement
* Append a patch-GAN discriminator that classifies between expert and refined masks.
* Objective: enforce smooth curvature and reject anatomically implausible angles.
* Requires balanced batches (expert vs refined masks) and careful learning-rate scheduling to avoid mode collapse.

---
## 3 Multi-Observer & Uncertainty-Driven Clean-up

### 3-A Targeted Double Review + STAPLE Fusion
1. Run a refinement net with Monte-Carlo dropout (32 forward passes).
2. Build a per-pixel entropy map; threshold top 10 % most uncertain pixels.
3. Ask **two** additional clinicians to correct *only* those hotspot regions.
4. Fuse the original and both corrected masks via STAPLE → probabilistic gold.
5. Re-threshold at posterior ≥ 0.9 to obtain a conservative binary mask.

**Benefit** Cuts manual correction time by ~70 % while matching full-slice re-annotation HD95.

### 3-B Active Learning Loop
* After the first round of corrections, retrain the refinement net and repeat the uncertainty sampling.
* Typically converges within 2 rounds for organs with simple topology (bladder, LV, prostate).

---
## 4 Ultrasound-Specific Pre-conditioning

| Pre-Filter | Purpose | Library | Parameters |
|------------|---------|---------|-------------|
| **SRAD** (Speckle Reducing Anisotropic Diffusion) | Denoise speckle while keeping edges | `medpy` or custom | 20 iterations, Δt = 0.125 |
| **NLM** (Non-Local Means, fast) | Complement SRAD for low-contrast scans | `skimage.restoration` | h = 0.8 × σ, patch=5, search=11 |
| **CLAHE** | Stabilise brightness / contrast across probes | OpenCV | clip-limit = 2.0, tile = 8×8 |
| **Depth Crop** | Remove reverberations beyond bladder depth | Simple slice | keep first 75 % rows |

Apply these filters **before** feeding images to the refinement net *and* the CRF; otherwise histogram shifts may invalidate learned weights.

---
## When to Revisit These Options
* Regulatory submission demanding sub-millimetre boundary error.
* Moving from 2-D slices to 3-D ultrasound volumes (RefineNet can be upgraded to 3-D U-Net).
* Access to a PACS server for large-scale pre-training.

---
### References
1. Oktay et al., “Anatomically Constrained Neural Networks (ACNN),” MICCAI 2017.
2. Milletari et al., “V-net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation,” 3DV 2016.
3. Warfield et al., “Simultaneous Truth and Performance Level Estimation (STAPLE),” IEEE TMI 2004.

Keep this file version-controlled alongside your code so you can quickly ramp up once additional resources become available.