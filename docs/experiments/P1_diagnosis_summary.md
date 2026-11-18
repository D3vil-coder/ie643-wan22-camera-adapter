# P1 diagnosis summary (v1 borders)

> Pre-recalibration output (gray-band gate; includes the two bogus NO-DATA cross-product cases). See the recalibration addendum in P1_RESULTS.md.

Generated 2026-11-16 14:57:38. Metrics are means over frames with R fraction > 1% unless marked @120.


## heldout__zoom_out (mode `zoom_out`)

| variant | fill | min_sigma | R@120 | gray score | gray@120 | color ratio | hf ratio | seam | lat std R/V | lat mean-abs R/V | x0hat std R/V | x0hat hf R/V |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_baseline | zeros | 0.600 | 0.204 | 0.000 | 0.000 | 0.668 | 0.386 | 0.999 | 1.057 | 1.027 | 1.046 | 0.139 |
| fill_replicate | replicate | 0.600 | 0.204 | -0.004 | -0.004 | 0.974 | 0.715 | 0.943 | 1.090 | 1.109 | 1.088 | 0.676 |
| min_sigma_0.3 | zeros | 0.300 | 0.204 | 0.002 | 0.004 | 0.632 | 0.296 | 1.005 | 1.091 | 1.055 | 1.101 | 0.121 |
| min_sigma_0.0 | zeros | 0.000 | 0.204 | 0.003 | 0.005 | 0.619 | 0.283 | 1.006 | 1.107 | 1.072 | 1.101 | 0.121 |
| frozen_base | zeros | 0.000 | 0.204 | -0.005 | -0.005 | 0.960 | 0.874 | 0.997 | 1.052 | 1.066 | 1.055 | 0.781 |

- **H1 (zero-guide shortcut): NOT APPLICABLE.** baseline shows no gray-band problem (R frames=30, gray_band_score=0.000 <= 0.15)
- **H2 (deterministic regression target): NOT APPLICABLE.** baseline shows no gray-band problem (R frames=30, gray_band_score=0.000 <= 0.15)
- **H3 (adapter hand-off at sigma=0.6): NOT APPLICABLE.** baseline shows no gray-band problem (R frames=30, gray_band_score=0.000 <= 0.15)

## heldout__pan_right (mode `pan_right`)

| variant | fill | min_sigma | R@120 | gray score | gray@120 | color ratio | hf ratio | seam | lat std R/V | lat mean-abs R/V | x0hat std R/V | x0hat hf R/V |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_baseline | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| fill_replicate | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| min_sigma_0.3 | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| min_sigma_0.0 | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| frozen_base | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |

- **H1 (zero-guide shortcut): NO DATA.** v1_baseline missing
- **H2 (deterministic regression target): NO DATA.** v1_baseline missing
- **H3 (adapter hand-off at sigma=0.6): NO DATA.** v1_baseline missing

## motorcycle__zoom_out (mode `zoom_out`)

| variant | fill | min_sigma | R@120 | gray score | gray@120 | color ratio | hf ratio | seam | lat std R/V | lat mean-abs R/V | x0hat std R/V | x0hat hf R/V |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_baseline | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| fill_replicate | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| min_sigma_0.3 | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| min_sigma_0.0 | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| frozen_base | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |

- **H1 (zero-guide shortcut): NO DATA.** v1_baseline missing
- **H2 (deterministic regression target): NO DATA.** v1_baseline missing
- **H3 (adapter hand-off at sigma=0.6): NO DATA.** v1_baseline missing

## motorcycle__pan_right (mode `pan_right`)

| variant | fill | min_sigma | R@120 | gray score | gray@120 | color ratio | hf ratio | seam | lat std R/V | lat mean-abs R/V | x0hat std R/V | x0hat hf R/V |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_baseline | zeros | 0.600 | 0.107 | 0.017 | 0.032 | 0.817 | 0.416 | 0.943 | 0.940 | 0.945 | 0.931 | 0.371 |
| fill_replicate | replicate | 0.600 | 0.107 | -0.042 | -0.015 | 0.934 | 0.420 | 0.971 | 0.980 | 0.997 | 0.968 | 0.391 |
| min_sigma_0.3 | zeros | 0.300 | 0.107 | 0.027 | 0.034 | 0.803 | 0.399 | 0.940 | 0.961 | 0.968 | 0.966 | 0.345 |
| min_sigma_0.0 | zeros | 0.000 | 0.107 | 0.021 | 0.022 | 0.791 | 0.393 | 0.939 | 0.970 | 0.978 | 0.966 | 0.345 |
| frozen_base | zeros | 0.000 | 0.107 | -0.039 | -0.039 | 1.233 | 0.419 | 0.941 | 0.991 | 1.001 | 0.992 | 0.428 |

- **H1 (zero-guide shortcut): NOT APPLICABLE.** baseline shows no gray-band problem (R frames=28, gray_band_score=0.017 <= 0.15)
- **H2 (deterministic regression target): NOT APPLICABLE.** baseline shows no gray-band problem (R frames=28, gray_band_score=0.017 <= 0.15)
- **H3 (adapter hand-off at sigma=0.6): NOT APPLICABLE.** baseline shows no gray-band problem (R frames=28, gray_band_score=0.017 <= 0.15)

## Overall

- H1: heldout__zoom_out=NOT APPLICABLE; heldout__pan_right=NO DATA; motorcycle__zoom_out=NO DATA; motorcycle__pan_right=NOT APPLICABLE
- H2: heldout__zoom_out=NOT APPLICABLE; heldout__pan_right=NO DATA; motorcycle__zoom_out=NO DATA; motorcycle__pan_right=NOT APPLICABLE
- H3: heldout__zoom_out=NOT APPLICABLE; heldout__pan_right=NO DATA; motorcycle__zoom_out=NO DATA; motorcycle__pan_right=NOT APPLICABLE

## Thresholds

- gray problem present: baseline gray_band_score > 0.15
- gray reduction counts: >= 25% relative drop of gray_band_score
- latent collapse (H1): R/valid ratio of std AND mean|z| of the final latent < 0.5
- x0-hat smooth (H2): R/valid Laplacian-variance ratio < 0.25 or latent std ratio < 0.5
- H4 (data scarcity) is not testable here.
- `frozen_base` is a control: the frozen model ignores the camera path, so its R region is only the area that WOULD be revealed; it shows natural statistics there.
