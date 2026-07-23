# Cross-Lingual Transfer of Sentiment Classification: Code Submission

**Samiya Mokhles Rahman**

**Report:** `Report_Samiya Mokhles Rahman.pdf` (submitted alongside this code)

This is the code submission for the report comparing monolingual and
multilingual BERT-family encoders on English/German sentiment
classification, including a zero-shot English-to-German transfer
condition, benchmarked against a TF-IDF baseline.

## Files in this submission

- `experiment.ipynb` -- the complete, runnable notebook containing all
  code used to produce every result reported in the paper.
- `results.csv` -- the raw output metrics for every condition and seed.
- `figure_overall.png` -- Figure 1 in the report (accuracy/macro-F1
  across all seven conditions, with error bars for reseeded conditions).
- `figure_zeroshot.png` -- Figure 2 in the report (the zero-shot
  transfer gap).
- `figure_cost.png` -- Figure 3 in the report (training time and peak
  GPU memory by condition).
- `README.md` -- this file.

## Dependency and environment information

- Platform: Kaggle Notebooks, GPU accelerator required (tested on a T4).
- Python 3.12.
- Key packages, installed by the notebook's first cell:
  - `torch` (CUDA-enabled)
  - `transformers` >= 4.35
  - `datasets` >= 2.14
  - `scikit-learn` >= 1.3
  - `pandas`
  - `matplotlib`

If running on Kaggle specifically: Settings -> Accelerator -> GPU;
Settings -> Internet -> ON (off by default -- every download fails
until this is enabled).

## Obtaining and preparing the data

Data is downloaded automatically by the notebook, which means no manual
download is needed. It uses the Multilingual Amazon Reviews Corpus
(MARC; Keung et al., 2020), accessed via the
`SetFit/amazon_reviews_multi_en` and `SetFit/amazon_reviews_multi_de`
mirrors on the Hugging Face Hub (the original `amazon_reviews_multi`
listing uses an old-style loading script no longer supported by current
`datasets` versions).

Preparation performed in-notebook:
1. Load raw reviews with 1-5 star ratings.
2. Binarise: 1-2 stars -> negative (0), 4-5 stars -> positive (1),
   3-star reviews dropped as ambiguous.
3. Stratified subsample to 2,000 train / 500 validation / 500 test
   examples per language, preserving the original class balance.

The corpus is distributed under Amazon's non-commercial research
licence; used here strictly for non-commercial academic coursework.

## Running training and evaluation

1. Open `experiment.ipynb` in Colab or Kaggle with a GPU runtime enabled.
2. Run all cells top to bottom (Run All, or Shift+Enter cell by cell).
3. The notebook, in order: installs dependencies, loads and prepares
   EN/DE data, trains the TF-IDF+LogReg baseline, fine-tunes and
   evaluates all five Transformer conditions (with extra seeds for the
   three conditions central to the research question), then generates
   the summary table, three figures, and the LaTeX table.

Total runtime: roughly 20-35 minutes for the initial single-seed pass
across all five Transformer conditions, plus approximately a further 10
minutes for the six additional reseeded runs (three core conditions x
two extra seeds), on a T4 GPU.

## Random seeds

Seed 42 is used for every condition. Seeds 43 and 44 are additionally
used for the three conditions most central to the research question:
monolingual German, in-language multilingual German, and multilingual
zero-shot transfer. This decision was made to prioritise variance
estimates where they matter most for the research question, given time
constraints. The remaining conditions (both baselines, monolingual
English, in-language multilingual English) use a single seed. Seeds are
set via `torch.manual_seed()` and `numpy`'s seed for model
initialisation and data shuffling, and via `random_state` for the
baseline's `LogisticRegression` and the stratified subsampling.

**Reproducibility check:** the full pipeline was independently rerun
twice; all reported metrics (accuracy, macro-F1, and their per-seed
standard deviations) matched exactly between runs.

## Important hyperparameters

| Hyperparameter | Value |
|---|---|
| Train / validation / test size (per language) | 2,000 / 500 / 500 |
| Max sequence length | 128 tokens |
| Batch size | 16 |
| Epochs | up to 2, early stopping (patience 1) on validation macro-F1 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Baseline vectorizer | TF-IDF, unigrams + bigrams, max 20,000 features |
| Baseline classifier | Logistic Regression, max_iter=1000 |

Model checkpoints:
- Monolingual English: `bert-base-uncased`
- Monolingual German: `dbmdz/bert-base-german-cased` (chosen over
  `deepset/gbert-base` specifically to keep WordPiece tokenization
  consistent across all three encoders -- `deepset/gbert-base` uses a
  SentencePiece-based tokenizer, which would have broken this
  consistency and required an extra dependency)
- Multilingual: `bert-base-multilingual-cased` (mBERT)

## Reproducible final results

`results.csv` contains the exact raw metrics for every condition and
seed from the runs used in the report. Headline results (test set):

| Condition | Accuracy | Macro-F1 |
|---|---|---|
| Baseline (TF-IDF+LogReg), EN | 0.874 | 0.874 |
| Baseline (TF-IDF+LogReg), DE | 0.818 | 0.818 |
| Monolingual EN (BERT) | 0.920 | 0.920 |
| Monolingual DE (German BERT) | 0.893 +/- 0.006 | 0.893 +/- 0.006 |
| Multilingual EN (mBERT) | 0.872 | 0.872 |
| Multilingual DE (mBERT) | 0.835 +/- 0.017 | 0.835 +/- 0.017 |
| Zero-shot transfer (mBERT) | 0.761 +/- 0.027 | 0.751 +/- 0.031 |

(Values with +/- are mean +/- standard deviation over 3 seeds; all
others are single-seed.)

## Connection between this code and the report

- Report Table 1 (dataset statistics) <- printed by the notebook's data
  loading cell.
- Report Table 2 (hyperparameters) <- the configuration cell near the
  top of the notebook.
- Report Table 3 (experimental conditions) <- the `conditions` list in
  the training cell.
- Report Table 4 (results) <- `results.csv` / `results_table.tex`.
- Report Table 5 (computational cost) <- the `train_seconds`,
  `peak_memory_mb`, `trainable_params` columns in `results.csv`.
- Report Figures 1-3 <- `figure_overall.png`, `figure_zeroshot.png`,
  `figure_cost.png`, generated directly by the notebook's final cells.
- Report Section 6.4 (Error Analysis) <- the misclassified-examples cell,
  which extracts and prints every zero-shot test-set misclassification
  along with the true/predicted label breakdown discussed in the report.
