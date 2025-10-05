
1. which of your front-ends **follow your Hartley–cosine discrete generalized convolution** idea, i.e.,
   [
   (f*g)(nh)=\frac{h}{2}\sum_{m=-\infty}^{\infty} f(mh),\big[g(nh{+}mh)+g(nh{-}mh)\big],
   ]
2. a tidy list of **all algorithms you developed** (plus the spectral baseline) with short algorithms, properties, and how to run them, and
3. a quick comparison you can read off at a glance.

---

# FRONT-ENDS for TCNs — What’s in this repo

## 0) Your mathematical template (what “follows your method” means)

The expression
[
(f*g)(nh)=\frac{h}{2}\sum_{m=-\infty}^{\infty} f(mh),[g(nh{+}mh)+g(nh{-}mh)]
]
is the **even (cosine/Hartley) branch** of a generalized discrete convolution on grid step (h).
If we use an **odd-length, even-symmetric FIR** (w[m]=w[-m]) (Type-I), then:

* the operation collapses to a **linear-phase** FIR with **constant group delay** ((K!-!1)/2),
* correlation equals convolution (because of even symmetry), and
* in code this is exactly a **symmetric Conv1D** with mirrored taps (your `SymmetricConv1d`).

So, any front-end that is a **time-domain, even-symmetric, odd-length FIR** (or a cascade of such FIRs) **follows your method.**

---

## 1) Methods in this repo (and whether they follow your formula)

| Front-end `--front_end`           | In code                                        | Follows your Hartley–cosine (Type-I symmetric FIR)? | Domain |
| --------------------------------- | ---------------------------------------------- | --------------------------------------------------- | ------ |
| **Baseline**                      | none                                           | —                                                   | —      |
| **LPS-Conv** (your original)      | `lpsconv` via `HartleyTCN` + `SymmetricConv1d` | **Yes** (single Type-I FIR; optional residual)      | Time   |
| **LPS-Conv-Plus** (sharper)       | `lpsconv_plus`                                 | **Yes** (two cascaded Type-I FIRs + gate + 1×1 mix) | Time   |
| **Sinc-LPF1d** (learnable cutoff) | `lpsconv_sinc`                                 | **Yes** (windowed-sinc Type-I FIR; cutoff learned)  | Time   |
| **Spectral Pooling**              | `spectral`                                     | **No** (hard truncation in **frequency**; not FIR)  | Freq   |

> TL;DR: **`lpsconv` / `lpsconv_plus` / `lpsconv_sinc`** are all faithful to your (h)–Hartley–cosine linear-phase prefiltering idea. **`spectral`** is a strong baseline but it works in the Fourier domain (ideal LPF by bin truncation), not your time-domain HCDC form.

---

## 2) Algorithm summaries (your implementations)

### A) LPS-Conv (original linear-phase symmetric front-end)

**Idea.** Single **even-symmetric** odd-length FIR before the TCN; optionally residual add.

**Algorithm.**

1. Learn the **half kernel** ({w[m]}_{m=0}^{k}), mirror to get (w[-m]!=!w[m]) (Type-I).
2. (Optional) normalize to **unity DC** (\sum_m w[m]\approx1).
3. **Causal** or **non-causal** padding (non-causal for offline tasks).
4. Output (y = x) **if residual** is used, else (y = \text{SymmetricConv1d}(x)).
5. Feed (y) to the TCN.

**Properties.** Linear phase; ~½ parameter count vs unconstrained conv (mirroring); very cheap.

**CLI (FordA, fast):**

```bash
python ecg_ucr_test.py --ucr_name FordA --front_end lpsconv --sym_kernel 11 \
  --epochs 20 --batch_size 32 --levels 3 --hidden 16
```

---

### B) LPS-Conv-Plus (sharper linear-phase, still time-domain)

**Idea.** Two cascaded symmetric FIR stages (sharper transition) → 1×1 channel mix → **gated residual** blend with identity.

**Algorithm.**

1. Stage-1: Type-I FIR (`SymmetricConv1d`, length (k_1)), ReLU (mild),
2. Stage-2: Type-I FIR (`SymmetricConv1d`, length (k_2)) *(optionally set **dilation=2** for extra sharpness)*,
3. 1×1 pointwise **mix** (re-combine channels),
4. **Unity-DC projection** (match mean to input),
5. Learnable gate (\alpha=\sigma(\beta)), output (y = x + \alpha,(F(x)-x)).

**Why it’s sharper.** Cascading linear-phase FIRs approximates a higher-order low-pass with a narrower transition band than one short K=11 filter.

**CLI (FordA, fast):**

```bash
python ecg_ucr_test.py --ucr_name FordA --front_end lpsconv_plus \
  --epochs 20 --batch_size 32 --levels 3 --hidden 16
```

*(You can make it even sharper by letting the factory pass a bigger `k` → stages 13+13, and/or using `dilation=2` in stage-2.)*

---

### C) Sinc-LPF1d (learnable cutoff linear-phase)

**Idea.** Start from the **ideal LPF** (sinc), apply a **Kaiser window** (β controls sidelobes), and **learn the cutoff** (and optionally β). Always odd (K) ⇒ linear phase.

**Algorithm.**

1. Build windowed-sinc kernel (h[n] = 2f_c,\text{sinc}(2f_c n)\cdot \text{Kaiser}_\beta[n]) for odd (K),
2. If enabled, **learn** (f_c) (via sigmoid parameterization) and/or **β**,
3. Normalize to **unity-DC**,
4. Depthwise Conv1d with that kernel (same kernel on each channel) + optional 1×1 mix,
5. (Optional) re-center mean to match input.

**Why it’s sharp.** You initialize at (or near) the **optimal** low-pass shape; learning tunes only a few scalars (cutoff, β), giving excellent stopband attenuation per tap.

**CLI (FordA, fast):**

```bash
python ecg_ucr_test.py --ucr_name FordA --front_end lpsconv_sinc \
  --epochs 20 --batch_size 32 --levels 3 --hidden 16
```

---

### D) Spectral Pooling (baseline; frequency-domain)

**Idea.** **rFFT → hard truncate** high-frequency bins → **irFFT**. Acts like an ideal, zero-phase LPF. Offline only (non-causal).

**Algorithm.**

1. (X=\text{rFFT}(x)),
2. Keep first (\lfloor \rho F\rfloor) bins, zero the rest ((\rho=) `--spec_cut`),
3. (y=\text{irFFT}(X)).
4. (We do FFT in fp32 for AMP safety; cast back to input dtype.)

**CLI (FordA, fast):**

```bash
python ecg_ucr_test.py --ucr_name FordA --front_end spectral --spec_cut 0.5 \
  --epochs 20 --batch_size 32 --levels 3 --hidden 16
```

---

## 3) Quick comparison (what to use when)

| Property                   |          LPS-Conv |              LPS-Conv-Plus |                             Sinc-LPF1d |                       Spectral |
| -------------------------- | ----------------: | -------------------------: | -------------------------------------: | -----------------------------: |
| Linear phase (time-domain) |                 ✅ |                          ✅ |                                      ✅ |                 ✅ (zero-phase) |
| Causal-capable (streaming) |                 ✅ |                          ✅ |                                      ✅ |                              ❌ |
| Learnable sharpness        |    kernel weights |      cascade+dilation+gate |                     **cutoff (and β)** |  cutoff by hyperparam ((\rho)) |
| Params (per channel)       | (\frac{K{+}1}{2}) |       ~(K_1/2 + K_2/2 + C) | (K) (computed each fwd), + few scalars |                              0 |
| Compute                    |         (O(BCTK)) | ~2× LPS-Conv (still cheap) |                              (O(BCTK)) |                 (O(BCT\log T)) |
| Best for                   |    fast/simple AA |  **sharper AA w/ control** |          **very sharp AA w/ tiny DOF** | strongest offline LPF baseline |

**Rules of thumb**

* Want **offline SOTA-ish**? `spectral` is a tough baseline to beat because it’s an ideal LPF.
* Want **time-domain & causal-ready**? `lpsconv_sinc` or `lpsconv_plus`.
* Want **smallest code delta**? `lpsconv` (your original) is the simplest.

---

## 4) How to run (your exact commands)

Baseline:

```bash
python ecg_ucr_test.py --ucr_name FordA --front_end none \
  --epochs 20 --batch_size 32 --levels 3 --hidden 16
```

Spectral:

```bash
python ecg_ucr_test.py --ucr_name FordA --front_end spectral --spec_cut 0.5 \
  --epochs 20 --batch_size 32 --levels 3 --hidden 16
```

LPS-Conv (original):

```bash
python ecg_ucr_test.py --ucr_name FordA --front_end lpsconv --sym_kernel 11 \
  --epochs 30 --batch_size 32 --levels 3 --hidden 16
```

LPS-Conv-Plus (sharper):

```bash
python ecg_ucr_test.py --ucr_name FordA --front_end lpsconv_plus \
  --epochs 20 --batch_size 32 --levels 3 --hidden 16
```

Sinc-LPF1d (learnable cutoff):

```bash
python ecg_ucr_test.py --ucr_name FordA --front_end lpsconv_sinc \
  --epochs 20 --batch_size 32 --levels 3 --hidden 16
```

---

## 5) Practical notes

* **Odd kernels only.** Your Type-I linear-phase constraint requires **odd** lengths. The factory’s `_odd()` guard keeps it valid.
* **Unity-DC** helps stabilize training (no average-level shift). Keep it on (or initialize close to it).
* For **extra sharpness** in `lpsconv_plus`: raise `k` in the factory (e.g., 27 → stages 13+13) and/or set stage-2 **dilation=2**.
* For **classification (offline)** you can wrap any time-domain front-end with a **zero-phase (forward–backward)** wrapper at **inference** to square the magnitude response and cancel delay.
