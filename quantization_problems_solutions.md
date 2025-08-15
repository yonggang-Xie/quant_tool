# Quantization Problems & Solutions + Calibration Objective Comparison

## 1. Distribution Errors → Calibration

**Problem:**  
Post-training quantization (PTQ) maps real-valued activations/weights to a small discrete set. If your clipping range is too tight → heavy saturation; too loose → too much quantization noise. Either way, the quantized distribution no longer matches the FP32 distribution ⇒ degraded accuracy.

**Math (uniform affine quantization):**  
For a tensor \(x\), bitwidth \(b\), integer range \([q_{\min}, q_{\max}]\), choose scale \(s>0\) and zero-point \(z\):
\[
q=\mathrm{clip}\!\left(\left\lfloor \tfrac{x}{s}\right\rceil + z,\ q_{\min}, q_{\max}\right),\quad
\hat{x}=s\,(q-z).
\]
Often clip \(x\in[-\alpha,\alpha]\) so \(s=\alpha/q_{\max}\), \(z=0\).

**Goal:**  
Pick \(\alpha\) (or \((s,z)\)) to best preserve the FP32 distribution using a small *calibration* set \(\mathcal{D}\).

**Optimality criteria:**
- **MSE:**  
\(\alpha^\star = \arg\min_{\alpha} \mathbb{E}[(x-\hat{x}_\alpha)^2]\)
- **KL divergence:**  
\(\alpha^\star = \arg\min_{\alpha} \mathrm{KL}(P\,\|\,Q_\alpha)\)

**Why it works:**  
Balances clipping vs rounding error to minimize total distortion.

---

## 2. Layer-wise Accumulated Errors → AdaQuant

**Problem:**  
Quantization error in early layers propagates and compounds through the network.

**Solution:**  
Freeze FP32 *teacher* network. Quantize layers progressively. For layer \(\ell\):
\[
\theta_\ell^{q\star} = \arg\min_{\theta_\ell^q} \mathbb{E}[\|y_\ell^{\text{fp}} - y_\ell^q\|_2^2] + \lambda\,\Omega(\theta_\ell^q).
\]
Here \(y_\ell^{\text{fp}}\) is FP32 output, \(y_\ell^q\) is quantized output given quantized inputs.

**Why it works:**  
Minimizes post-layer mismatch, containing error accumulation.

---

## 3. Batch Normalization Errors → BatchNorm Reconstruction

**Problem:**  
Quantization changes effective scale/offset of preceding conv outputs. BN folding + quantization can yield mismatched activation distributions.

**BN folding:**  
\[
W' = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}} W,\quad
b' = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}(b-\mu) + \beta.
\]

**Reconstruction:**  
After quantization, re-estimate BN parameters or fit affine patch:
\[
(a^\star,c^\star) = \arg\min_{a,c} \mathbb{E}[\|y^{\text{fp}} - (a\odot \hat{y} + c)\|_2^2].
\]

**Why it works:**  
BN is affine; we can re-tune its parameters to cancel quantization shifts.

---

## 4. Partial Quantization → Mixed Precision

**Problem:**  
Some layers are more sensitive to quantization. Uniform low bitwidth may hurt accuracy.

**Solution:**  
Keep sensitive layers at higher precision (FP16/FP32) or use higher bitwidth selectively.

**Selection methods:**
- Measure loss increase when quantizing only one layer.
- Use Hessian-based sensitivity.
- Solve knapsack for error vs cost.

**Why it works:**  
Spends precision budget where it matters most.

---

# MSE vs KL Divergence for Calibration

## MSE (Mean Squared Error)

**Math:**  
\[
\text{MSE}(\alpha) = \mathbb{E}[(x - \hat{x}_\alpha)^2].
\]

**Why it works:**  
- Minimizes average reconstruction error.  
- Effective for symmetric, unimodal distributions.

**Limitations:**  
- Ignores distributional shape.  
- May sacrifice rare but important large values.

---

## KL Divergence

**Math:**  
Let \(P\) = FP32 histogram, \(Q_\alpha\) = quantized histogram:  
\[
\text{KL}(P\,\|\,Q_\alpha) = \sum_i P(i) \log \frac{P(i)}{Q_\alpha(i)}.
\]

**Why it works:**  
- Preserves probability distribution shape.  
- Good for skewed, heavy-tailed, multi-modal data.

**Limitations:**  
- Needs good histogram estimation (more calibration data).  
- More computationally expensive.

---

## When KL is Better

- **Distribution shape matters:** KL preserves tails and multi-modal structure.  
- **Decision boundary sensitivity:** KL keeps relative scaling of important activations.  
- **Empirical:** Often better for activation calibration in vision models.

---

## Rule of Thumb

- **MSE:** Symmetric, Gaussian-like distributions.  
- **KL:** Skewed, heavy-tailed, non-Gaussian distributions.  
- **Hybrid:** Try both, pick best per tensor.

