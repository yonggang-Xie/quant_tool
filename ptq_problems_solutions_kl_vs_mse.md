# Post-Training Quantization: Problems & Solutions (+ KL vs. MSE Calibration)

This document combines the earlier notes into a single, math-checked reference.

---

## 1) Distribution Errors → **Calibration**

**Problem (what goes wrong):**  
Post-training quantization (PTQ) maps real-valued activations/weights to a small discrete set. If the clipping range is too tight → heavy saturation; too loose → too much quantization noise. Either way, the quantized distribution no longer matches the FP32 distribution ⇒ degraded accuracy.

**Setup (uniform affine quantization):**  
For a tensor \(x\), bitwidth \(b\), integer range \([q_{\min}, q_{\max}]\) (e.g., \([-127,127]\) for signed 8‑bit), choose scale \(s>0\) and zero‑point \(z\):
\[
q=\operatorname{clip}\!\big(\operatorname{round}(x/s) + z,\ q_{\min},\ q_{\max}\big),\qquad
\hat{x}=s\,\big(q-z\big).
\]
Often we clip \(x\in[-\alpha,\alpha]\) (symmetric), so with \(Q = q_{\max} = 2^{b-1}-1\) we use \(s=\alpha / Q\) and \(z=0\).

**Goal (calibration):**  
Pick \(\alpha\) (or \((s,z)\)) to best preserve the FP32 distribution using a small *calibration* set \(\mathcal{D}\).

**Common optimality criteria:**
- **MSE (min reconstruction error):**
\[
\alpha^\star=\arg\min_{\alpha}\ \mathbb{E}_{x\sim\mathcal{D}}\big[(x-\hat{x}_\alpha)^2\big].
\]
- **KL divergence (match histograms):** Let \(P\) be the FP32 histogram; \(Q_\alpha\) the dequantized histogram under \(\alpha\).
\[
\alpha^\star=\arg\min_{\alpha}\ \mathrm{KL}\!\left(P\;\|\;Q_\alpha\right)=\arg\min_{\alpha}\ \sum_i P(i)\log\frac{P(i)}{Q_\alpha(i)}.
\]
- **Percentile / ACIQ / entropy-aware:** Closed-form thresholds from assumed distributions (Gaussian/Laplace), using sample estimates (\(\sigma\), \(\mathbb{E}|x|\)).

**Why this works:**  
Calibration explicitly trades off clipping error vs. rounding error to minimize total distortion between \(x\) and \(\hat{x}\), so the quantized layer statistics track the FP32 ones much better.

---

## 2) Layer-wise Accumulated Errors → **AdaQuant / Layer-wise Reconstruction**

**Problem:**  
Quantization error in early layers propagates and *compounds* through the network; a locally “okay” range can still cause a big end‑to‑end drift.

**Key idea (optimize one layer at a time against FP targets):**  
Freeze a FP32 *teacher* network \(F^{\mathrm{fp}}\). Quantize layers progressively. For layer \(\ell\), learn that layer’s quantization parameters (and sometimes its rounding) by matching its FP32 *outputs* given real activations.

**Objective (layer-wise reconstruction):**  
Let \(h_{\ell-1}\) be the (already‑quantized) input activations into layer \(\ell\). Define
\[
y_\ell^{\mathrm{fp}} = F_\ell^{\mathrm{fp}}(h_{\ell-1}),\qquad
y_\ell^{q} = F_\ell^{q}(h_{\ell-1};\,\theta_\ell^q),
\]
where \(\theta_\ell^q\) are the scales/zero‑points (and possibly rounding parameters) for layer \(\ell\). Solve
\[
\theta_\ell^{q\star}=\arg\min_{\theta_\ell^q}\ \mathbb{E}_{h_{\ell-1}\sim\mathcal{D}}\big[\|y_\ell^{\mathrm{fp}}-y_\ell^{q}\|_2^2\big] + \lambda\,\Omega(\theta_\ell^q).
\]
In **AdaRound/AdaQuant‑style** methods, weights are parameterized with *learnable rounding offsets*. A simplified form uses a per‑weight offset \(r\in[0,1]\) (obtained via a squashing function) so that
\[
Q(w)=\lfloor w \rfloor + r,\quad r=\sigma(\alpha),
\]
and \(\alpha\) (hence \(r\)) is learned by backprop through the (de)quantized forward (with STE), again using only a small calibration set.

**(Nice to have) Second‑order / importance weighting:**  
Some variants weight errors by local curvature (Hessian‑aware) so changes in “sensitive” channels cost more:
\[
\min_{\theta_\ell^q}\ (y_\ell^{q}-y_\ell^{\mathrm{fp}})^\top H_\ell \,(y_\ell^{q}-y_\ell^{\mathrm{fp}}),
\]
with \(H_\ell\) a Gauss‑Newton/Hessian proxy from calibration batches.

**Why this works:**  
By directly minimizing the *post‑layer* mismatch (not just per‑tensor MSE), you account for how each layer’s noise flows through its nonlinearity and next layers. Doing this progressively (block/greedy coordinate descent) contains error accumulation.

---

## 3) Batch Normalization Errors → **BatchNorm Reconstruction (BN Re‑estimation / Affine Patch)**

**Problem:**  
Quantization changes the effective scale/offset of the preceding convolution’s outputs. If BN is *folded* into Conv and then quantized, stale BN statistics \((\mu,\sigma^2)\) and quantization of the fused weights/bias can shift activations, hurting accuracy.

**BN folding (what we typically do before PTQ):**  
For a Conv/Linear \(y=W*x+b\) followed by BN
\[
\mathrm{BN}(y)=\gamma\cdot \frac{y-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta,
\]
we fold BN into Conv:
\[
W' = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}\,W,\qquad
b' = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}\,(b-\mu)+\beta.
\]
Then we quantize \(W'\) and \(b'\).

**Reconstruction (fix after quantization):**  
After quantizing to \(\hat{W}',\hat{b}'\), the actual layer output drifts. BN reconstruction re‑estimates correction factors (either re‑estimating BN statistics on calibration data or directly solving a small regression for an affine fix) so that the *quantized* layer output matches the FP32 target.

One practical variant adds a post‑conv affine “patch” \((a,c)\) per output channel:
\[
\hat{y} = a\odot (\hat{W}'*x+\hat{b}') + c,
\]
and solves
\[
(a^\star,c^\star)=\arg\min_{a,c}\ \mathbb{E}\big[\|y^{\mathrm{fp}} - (a\odot \hat{y} + c)\|_2^2\big],
\]
which has a closed‑form least‑squares solution per channel. Equivalently, re‑compute BN running \(\mu,\sigma^2\) on calibration data with the *quantized* weights (BN “re‑estimation”) and refold.

**Why this works:**  
BN is an affine normalization; small multiplicative/additive mismatches introduced by quantization can be countered by recalibrating those same affine degrees of freedom using real activation statistics.

---

## 4) Partial Quantization (Selective / Mixed‑Precision)

**Problem:**  
Some layers are far more sensitive to quantization (e.g., first/last layers, attention projections, depthwise convs, LayerNorm‑adjacent blocks). Forcing everything to low bitwidth can cause disproportionate accuracy loss.

**Solution:**  
Keep sensitive pieces at higher precision (FP16/FP32) or use higher bitwidth (e.g., 8‑bit for most layers, 16‑bit for embeddings / final classifier). Optionally choose bitwidth per layer.

**How to choose (principled selection):**
- **Sensitivity via loss increase:** For each layer \(\ell\), measure \(\Delta \mathcal{L}_\ell\) when quantizing only that layer (others FP). Rank layers by \(\Delta \mathcal{L}_\ell\); keep top‑\(k\) sensitive layers in higher precision.
- **Hessian/curvature proxy:** Let \(H_\ell\) be a diagonal/trace proxy. Prefer higher precision for layers with large curvature (changes hurt more).
- **Budgeted mixed‑precision (knapsack):**
\[
\min_{\{b_\ell\}} \sum_\ell \mathrm{Err}_\ell(b_\ell)\quad
\text{s.t.}\ \sum_\ell \mathrm{Cost}_\ell(b_\ell)\le B,\ \ b_\ell\in\{4,8,16,32\}.
\]
Greedy or dynamic programming works well using per‑layer error/cost curves measured on calibration data.

**Why this works:**  
You spend precision where it *buys* you the most (high‑sensitivity layers), achieving most of the throughput/memory gains while avoiding the worst accuracy cliffs.

---

## 5) KL Divergence vs. MSE for Calibration

### **Problem:**  
When deciding the clipping threshold \(\alpha\) for uniform quantization, we need an objective that makes the quantized distribution **match** the FP32 distribution as closely as possible.  
- **MSE** focuses on minimizing the average *value* error per sample.  
- **KL divergence** focuses on preserving the *probability distribution* shape of activations.

### **MSE (Mean Squared Error)**
**Math:** For FP32 tensor \(x\) and dequantized \(\hat{x}_\alpha\),
\[
\mathrm{MSE}(\alpha) = \mathbb{E}_{x\sim\mathcal{D}}\!\left[ \big(x - \hat{x}_\alpha\big)^2 \right],\quad
\alpha^\star = \arg\min_{\alpha} \mathrm{MSE}(\alpha).
\]
**Why it works:** Directly minimizes L2 reconstruction error; effective when the model is locally linear in the region of errors and when activations are symmetric/unimodal.  
**Limitations:** Ignores distributional shape; rare but important large values may be sacrificed.

### **KL Divergence**
**Math:** Using histograms \(P\) (FP32) and \(Q_\alpha\) (quantized‑dequantized under \(\alpha\)),
\[
\mathrm{KL}\big(P\ \|\ Q_\alpha\big)=\sum_{i} P(i)\log \frac{P(i)}{Q_\alpha(i)},\quad
\alpha^\star=\arg\min_{\alpha}\ \mathrm{KL}\big(P\ \|\ Q_\alpha\big).
\]
**Why it works:** Preserves *information* and relative frequencies across bins; better for skewed, heavy‑tailed, or multi‑modal activations and often aligns better with preserving decision boundaries.  
**Limitations:** Needs reliable histograms (more calibration data), smoothing for zero bins, and is more compute‑intensive.

### **When KL can be better than MSE in PTQ**
1. **Distribution shape matters:** KL maintains tails and multi‑modality; MSE may over‑favor dense central mass.  
2. **Decision boundary sensitivity:** KL tends to keep high‑magnitude rare activations in the right relative scale.  
3. **Empirical practice:** For many activation tensors (esp. post‑nonlinearity), KL‑calibrated ranges frequently yield higher post‑PTQ accuracy than pure MSE.

### **Rule of thumb**
- Use **MSE** for well‑behaved, near‑Gaussian weights or middle layers with unimodal, symmetric activations.  
- Use **KL** for activations with **skew/heavy tails/multi‑modality**, especially right after ReLU/GELU or attention blocks.  
- **Pragmatic:** Try both and keep per‑tensor best on the calibration set.

---

## 6) Minimal, Practical Recipe

1. **Collect a small calibration set** representative of inference data.  
2. **Per‑tensor calibration** (activations & weights): search \(\alpha\) by MSE or KL.  
3. **BN re‑estimation / reconstruction:** run calibration data with quantized weights to recompute BN stats or fit a per‑channel affine patch; refold.  
4. **Layer‑wise reconstruction (AdaQuant/AdaRound):** greedily minimize \(\|y_\ell^{\mathrm{fp}}-y_\ell^q\|^2\) on calibration batches.  
5. **Partial quantization / mixed precision:** keep top‑sensitivity layers in higher precision; re‑run steps 2–4 quickly after changes.

---

## 7) Quick Math Intuition (Error Budget)

For symmetric uniform quantization with clipping at \(\pm \alpha\), a common approximation decomposes total distortion as
\[
\underbrace{\mathbb{E}\big[(x-\operatorname{clip}(x,\!-\alpha,\alpha))^2\big]}_{\text{clipping error}}
\;+\;
\underbrace{\frac{s^2}{12}}_{\text{rounding error (high‑resolution approx.)}},\quad s=\alpha/Q,\ Q=2^{b-1}-1.
\]
Calibration picks \(\alpha\) that (approximately) minimizes this sum; layer‑wise reconstruction reduces the *post‑layer* mismatch, BN reconstruction cancels affine shifts, and mixed precision caps the worst contributors to propagated error.

---

### Sanity‑check expectations
- Many models recover to within ~0.1–1.0% top‑1 (or analogous metric) with INT8 PTQ after **Calibration + BN reconstruction + Layer‑wise reconstruction**.  
- If a gap persists, **partial quantization** of a handful of sensitive layers typically closes it with minimal throughput loss.
