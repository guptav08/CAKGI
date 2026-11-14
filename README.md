# CAKGI: Context-Aware Kernel-Guided Imputation

Code for the paper **"Context-Aware Kernel-Guided Imputation for Physiological Time Series"**, accepted at Machine Learning for Health (ML4H) 2026.

*Vaibhav Gupta, Simon Burbach, Beyza Cinar, Maria Maleshkova*

CAKGI is a diffusion-based method for imputing long gaps (30 min to 8 h) in continuous glucose monitoring (CGM) and heart-rate (HR) data. Unlike methods that rely only on the observed signal window, CAKGI conditions on structured physiological context around each gap:
- pre-gap trends and post-gap recovery;
- the patient's own history at the same time of day 1, 3 and 7 days earlier.

Learned position weights balance the pre- and post-gap context at every step of the gap. The core denoiser is the **Position-Interpolated Dual Context Kernel (PIDCK)**.

## Repository contents

| Notebook | What it does | Paper |
|---|---|---|
| `impute_cakgi.ipynb` | Imputes the full dataset with CAKGI. Short gaps (< 30 min) are linearly interpolated, long gaps (30 min–8 h) are imputed by CAKGI, and longer gaps are left missing. Also writes the files the other notebooks need. | Method, imputed dataset |
| `CAKGI_MonteCarlo_Evaluation.ipynb` | Monte Carlo benchmark on held-out patients against MICE, BRITS, CATSI, GP-VAE and CSDI-adapted, with patient-level statistics, duration analysis and all ablations. | Table 1, appendix tables, ablation table |
| `impute_brits.ipynb` | Imputes the full dataset with BRITS, using the same short/long-gap rule. | Downstream benchmark |
| `impute_csdi.ipynb` | Imputes the full dataset with CSDI-adapted, using the same short/long-gap rule. | Downstream benchmark |
| `fcn_hypo_benchmark.ipynb` | 30-minute hypoglycemia prediction with an FCN, trained on original or imputed data. | Downstream results and statistics |

## Data

The experiments use **Subdataset II of DiaData**, which combines D1NAMO, HUPA-UCM and DDATSHR. It contains CGM and HR data of 51 people with diabetes, recorded under free-living conditions. The dataset is not included in this repository. Download it from Zenodo (see the DiaData reference in the paper) and save it as `SDBII.csv` in a data directory of your choice.

`SDBII.csv` needs these columns:

| Column | Content |
|---|---|
| `PtID` | patient ID |
| `ts` | timestamp, on a regular 5-minute grid |
| `GlucoseCGM` | glucose in mg/dL; missing values empty or 0 |
| `HR` | heart rate in bpm, matched to the CGM timestamps; missing values empty or 0 |

## Installation

Python ≥ 3.10 is required. The code runs on CUDA, Apple MPS or CPU; a GPU or Apple Silicon is strongly recommended for the diffusion models.

```bash
pip install torch numpy "pandas>=2.0" scipy scikit-learn matplotlib tqdm jupyter
```

Tested with PyTorch 2.14, pandas 2.2 and scikit-learn 1.9.

## How to run

In each notebook, set `BASE` in the first code cell to your data directory. Then run the notebooks in this order:

```
1. impute_cakgi.ipynb                  -> SDBII_gap_classified.csv, SDBII_context_vectors.csv,
                                          SDBII_imputed_pidck.csv
2. CAKGI_MonteCarlo_Evaluation.ipynb   -> Table 1, duration tables, paired statistics, ablations
3. impute_brits.ipynb                  -> SDBII_imputed_brits.csv
4. impute_csdi.ipynb                   -> SDBII_imputed_csdi.csv
5. fcn_hypo_benchmark.ipynb            -> downstream results and significance tests
```

Steps 3 and 4 are independent of step 2 and can run in any order. Step 5 needs steps 1, 3 and 4.

### 1. CAKGI imputation (`impute_cakgi.ipynb`)

1. Classifies every missing value by the length of its gap.
2. Builds a context vector for every long gap.
3. Trains CAKGI.
   - Training: 75 epochs; Adam (learning rate 10⁻³, batch 32); 5-epoch warm-up, then cosine decay.
   - Loss: diffusion loss + 0.1 × Cauchy regulariser + 0.3 × boundary loss.
4. Imputes the long gaps.
   - Context window: min(120 + 0.5·duration, 360) min on each side.
   - Interpolation initialisation, an adaptive number of diffusion steps, and boundary guidance.
   - Output: the median of 50 chains.

The output `SDBII_imputed_pidck.csv` contains:

| Column | Content |
|---|---|
| `GlucoseCGM_imputed_final`, `HR_imputed_final` | observed or imputed values |
| `imputation_method_glucose`, `imputation_method_hr` | `real`, `linear_short`, `PIDCK_long` or `missing` |
| interval columns | 10th–90th percentile range of the chains, for CAKGI-imputed values |
| `lambda_pre`, `lambda_post` | learned position weights, showing how much each imputed value relies on pre- vs post-gap context |

### 2. Monte Carlo evaluation (`CAKGI_MonteCarlo_Evaluation.ipynb`)

**Protocol**
- **Split:** patient-disjoint. 25% of patients are held out (13 in our data), and every model and all normalisation statistics use only the training patients.
- **Gaps:** 3 trials × 500 independent synthetic gaps of 1–8 h, drawn from originally observed stretches of the held-out patients. Each gap is masked and imputed on its own.
- **Training:** every learned model is trained for 75 epochs on the same windows and masking.
- **Metrics:** RMSE, MAE and CRPS, with 50 chains or samples for the probabilistic methods.
- **Statistics:** paired differences with a patient-level cluster bootstrap (10,000 resamples), plus patient counts and Wilcoxon tests.

**Baselines**
- MICE (scikit-learn);
- BRITS (Cao et al., 2018);
- CATSI: the authors' official code (github.com/ai4h-hkbu/CATSI, commit 239df11), with one fix — the series mean is the mean of the observed values, as defined in the CATSI paper;
- GP-VAE (Fortuin et al., 2020);
- CSDI-adapted (after Tashiro et al., 2021).

The baselines are implemented in the notebook so that every method sees the same window, masking and training budget. The notebook header describes each implementation.

The notebook has two parts:
- **Part A** (Cells 1–15): main results, figures and CAKGI-vs-baseline statistics.
- **Part B** (Cells 16–21): the ablations.
  - six retrained CAKGI variants;
  - the inference recipe switched off;
  - CSDI-adapted with CAKGI's recipe, with and without CAKGI's boundary loss;
  - a time-of-day profile and a ridge regression on CAKGI's context vector.

Checkpoints and results are saved to `BASE`, and the notebook resumes where it stopped.

**Tables without retraining.** Run Cell 1, place the released `mc_results_all.csv` in `BASE/mc_results/`, then run Cells 13, 15 and 21. This reproduces the paper's tables exactly.

### 3–4. BRITS and CSDI-adapted imputation

These notebooks impute the full dataset with the same rule as CAKGI, so the downstream benchmark compares the methods on equal terms:
- linear interpolation for gaps under 30 min;
- the respective method for gaps of 30 min–8 h;
- gaps longer than 8 h left missing.

Note that the downstream BRITS is a simplified variant: it uses a constant time step and a GRU cell. The Monte Carlo notebook uses the full BRITS formulation.

### 5. Downstream benchmark (`fcn_hypo_benchmark.ipynb`)

The FCN predicts hypoglycemia (any glucose < 70 mg/dL in the next 30 min) from 2 h of glucose and HR.
- **Folds:** 3 repeats × 5 patient-level folds, stratified by each patient's hypoglycemia burden, with identical splits for every method.
- **Inputs and labels:** imputed values are used only in training and validation inputs. All labels come from originally observed glucose.
- **Test windows:** the same originally observed windows for every method.
- **Statistics:** Nadeau–Bengio corrected repeated-CV t-tests with Holm correction.

## Reproducibility

- **Seeds** are fixed for the patient split, gap sampling, training and sampling.
- **GPU nondeterminism.** GPU kernels (CUDA/MPS) are not bit-deterministic, so retrained models and diffusion samples can differ slightly from the released numbers. For scale, retraining CAKGI with a second seed changed glucose RMSE by 0.8 mg/dL.
- **Runtime.** On CPU, the diffusion methods take about 30 s per gap. A full Monte Carlo run needs a GPU or Apple Silicon.
- **Smoke test.** `impute_cakgi.ipynb` and `CAKGI_MonteCarlo_Evaluation.ipynb` have a `SMOKE_TEST` switch in their first cell. It runs the notebook in minutes with tiny settings to check that everything works; its numbers are meaningless.

#

## Contact

Vaibhav Gupta: <guptav@hsu-hh.de>
