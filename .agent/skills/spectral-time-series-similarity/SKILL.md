---
name: spectral-time-series-similarity
description: >
  Guidance for implementing spectral (frequency-domain) similarity measures and 
  retrieval-augmented forecasting on time series. Trigger this skill when the user 
  asks to compute similarity between time-series segments in the frequency domain, 
  implement STFT-based matching, or integrate retrieval-augmented forecasting (like SpecReTF).
---

# Spectral Time Series Similarity & Retrieval-Augmented Forecasting

This skill provides the mathematical workflows and implementation patterns for comparing time-series segments in the frequency domain and projecting retrieval-augmented forecasts.

## Core Workflow

### 1. Z-Score Normalization (DC Offset Removal)
Before performing spectral decomposition (STFT), the input time-series window must be Z-score normalized. Without this, the DC component (absolute price level/offset) will completely dominate the amplitude and phase spectra, making distinct upward and downward trends look identical.

```python
std = np.std(x)
x_norm = (x - np.mean(x)) / std if std > 1e-8 else x - np.mean(x)
```

### 2. Short-Time Fourier Transform (STFT)
Partition the normalized sequence into $W$ overlapping frames of size $M$ with hop size $B$. Apply a window function (like a Hann window) to reduce spectral leakage before computing the real FFT (`rfft`).

```python
window = np.hanning(M)
coefs = np.fft.rfft(frame * window)
```

### 3. Amplitude Similarity via Jensen-Shannon Divergence (JSD)
1. Convert the absolute amplitude spectrum of each frame into a probability distribution:
   $$p(f) = \frac{A(f)}{\sum_f A(f)}$$
2. Compute the JSD (base 2) between the query and candidate distributions to bound the divergence in $[0, 1]$:
   $$d_{JS} = \text{JSD}(p_Q \parallel p_X)$$
3. Derive the amplitude similarity:
   $$s_{amp} = 1.0 - d_{JS}$$

### 4. Amplitude-Weighted Phase Coherence
Phase angle difference can be noisy in frequency bins with negligible energy. To prevent numerical noise from diluting the coherence score, weight the wrapped phase differences by the product of the amplitudes at each frequency:

$$\Delta \Phi = \frac{\sum_f [\Phi_Q(f) - \Phi_X(f)]_{\text{wrapped}} \cdot (A_Q(f) A_X(f))}{\sum_f (A_Q(f) A_X(f))}$$
$$s_{phase} = \cos(\Delta \Phi)$$

### 5. Mean-Based Future Path Alignment
When aligning retrieved historical future continuations to the current query price, do not scale based solely on the starting price point (which can introduce massive discontinuous leaps if there is a gap). Instead, scale the future path using the ratio of the mean of the query window to the mean of the candidate's historical window:

$$\text{scale\_factor} = \frac{\text{mean}(Q)}{\text{mean}(X_k)}$$
$$\text{aligned\_path} = Y_k \cdot \text{scale\_factor}$$

---

## Known Pitfalls

- **DC Offset Domination**: Skipping Z-score normalization before STFT causes the DC component ($f=0$) to drown out all AC components, leading to identical similarity scores for opposite trends.
- **Phase Dilution**: Averaging phase differences uniformly across all frequency bins allows empty bins containing only numerical noise to dominate the average phase difference. Always use amplitude-weighted phase coherence.
- **Discontinuous Boundary Leaps**: Scaling retrieved future segments using only the boundary price ratio ($Q[-1] / Y_k[0]$) is highly sensitive to noise and gaps. Scaling by the ratio of the window means is much more stable.
