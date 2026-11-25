# CV-Scan: AI-Powered Resume Screener

CV-Scan ranks candidate resumes against a job description (JD) and explains every score. It combines **TF-IDF keyword matching** with **Sentence-BERT semantic matching** into a calibrated **hybrid score**, compares each candidate against 2,361 historical resumes scored for the same job, and predicts each resume's professional domain. It ships as a Streamlit dashboard with custom light and dark themes.

## Features

- **Batch screening.** Upload several PDF, DOCX (including tables) or TXT resumes and get a ranked leaderboard you can export as CSV. You can also add held-out sample resumes and pick from 24 built-in sample job descriptions.
- **Two scan modes.**
  - *Basic*: TF-IDF keyword match only.
  - *Advanced*: hybrid of TF-IDF and Sentence-BERT, with an adjustable keyword weight (default 30 %).
- **Explainable keyword score.** Shows each matched term's exact share of the TF-IDF cosine similarity, the job's key requirements that were found, and the ones that are missing.
- **Pool benchmark.** Shows the candidate's percentile among historical resumes scored for *this* job, the median and top-10 % thresholds, the score distribution, and the closest historical profiles.
- **Semantic map.** Interactive 2D/3D PCA of the embedding space: the pool coloured by match score, plus the candidates and the job description.
- **Domain fit.** A calibrated SVM predicts the candidate's and the job's category out of 24 and flags mismatches.
- **Model evaluation page.** Real, reproducible results:
  - Table 3 and Figures 9–11 for TF-IDF vs BERT vs Hybrid classifiers
  - confusion matrices
  - a hyper-parameter (C) sweep showing training vs validation accuracy
  - a JD → resume ranking benchmark on held-out resumes

## How it works

| Step | What happens |
|---|---|
| 1. Parse | `pypdf` / `python-docx` / multi-encoding TXT extraction, with clear errors for scanned or corrupt files |
| 2. Clean | Lowercase; remove URLs, e-mails and non-ASCII; turn punctuation into spaces (so `Python,Java` → `python java`); remove NLTK and recruiting stop-words; WordNet lemmatisation. `c++` and `c#` survive. |
| 3. Keyword engine | TF-IDF with uni- and bi-grams and sub-linear tf, fitted per screening run on *pool + JD + candidates* so every JD term is in the vocabulary. Score = cosine(JD, resume) = Σ w<sub>JD</sub>·w<sub>CV</sub>, which makes per-term attribution exact. |
| 4. Semantic engine | `all-MiniLM-L6-v2` reads at most 256 word-pieces, so each document is split into 150-word chunks. The document vector is **[mean of all chunks ‖ first chunk] / √2**: the whole resume plus its title and summary. |
| 5. Calibrate & fuse | Both raw cosines are min-max normalised against the historical pool scored **for the same JD**. Hybrid = *w*·keyword + (1 − *w*)·semantic, with *w* = 0.30. |

The job's **key requirements** are its highest-weighted TF-IDF terms, filtered to vocabulary that real resumes use:

- the term appears in at least 2 historical resumes;
- bigrams must be cohesive phrases;
- the term must be concentrated in a few of the 24 categories (low normalised entropy), not spread evenly across all of them.

This keeps word pairs that only exist in the job text (e.g. "support full") out of the explanations.

## Evaluation results

All numbers come from `python Resume_Screening_Project/src/train_models.py` (or `python evaluate_classifiers.py`) on `dataset/Resume/Resume.csv`: 2,481 unique resumes in 24 categories. 120 of them (5 per category) are held out and never used for training.

**A · Category classification** (LinearSVC, 80 % stratified training split, TF-IDF fitted on the training part only)

| Measure | TF-IDF | BERT | Hybrid |
|---|---|---|---|
| Accuracy | 0.704 | 0.765 | **0.786** |
| Precision | 0.681 | 0.720 | **0.747** |
| Sensitivity | 0.659 | 0.709 | **0.735** |
| Specificity | 0.987 | 0.990 | **0.991** |
| F1-score | 0.661 | 0.704 | **0.734** |
| MCC | 0.690 | 0.755 | **0.777** |

The hybrid is best at every training split from 50 % to 90 % (accuracy 0.774 → 0.814).

**C · JD → resume ranking** (24 sample jobs × 120 held-out resumes; a resume is relevant when it is in the job's category)

| Engine | ROC-AUC | P@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| Keyword (TF-IDF) | 0.882 | 0.492 | 0.767 | 0.596 |
| Semantic (BERT) | 0.882 | 0.525 | 0.802 | 0.644 |
| **Hybrid (30/70)** | **0.892** | **0.575** | **0.832** | **0.666** |

A sweep of the hybrid weight shows that 30 % keyword gives the best Precision@5. The **Model evaluation** page also has the full per-split curves, confusion matrices and the C sweep.

## Project structure

```
.streamlit/config.toml         Light + dark custom themes (switch via ⋮ → System / Light / Dark)
requirements.txt               Runtime dependencies (CPU-only PyTorch) — used locally and on Streamlit Cloud
requirements-dev.txt           + pytest and matplotlib (tests, report figures)
Resume_Screening_Project/
├── src/
│   ├── app.py                 Streamlit dashboard (Home · Screen resumes · Model evaluation)
│   ├── engine.py              Scoring engine: TF-IDF, Sentence-BERT, calibration, hybrid, explanations
│   ├── evaluation.py          Classification experiment, C sweep, ranking benchmark
│   ├── train_models.py        Builds every artifact in models/ and runs the evaluation
│   ├── utils.py               Text extraction, cleaning, chunking
│   ├── ui.py, charts.py       Design system (CSS, components) and Plotly charts
│   └── sample_data.py         24 sample job descriptions (one per category)
├── models/                    Trained artifacts + metadata.json + evaluation_results.json
└── tests/                     pytest suite (utils, engine, headless UI tests)
evaluate_classifiers.py        Re-runs the evaluation, regenerates fig9–11.png, prints Table 3
generate_plots.py              Re-draws fig9–11.png from models/evaluation_results.json
dataset/                       Resume.csv + original PDFs by category (only needed for training)
```

## Installation and usage

Run every command from the repository root.

```bash
# 1. Create and activate a virtual environment (Python 3.10+; 3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies (requirements-dev.txt adds pytest + matplotlib)
pip install -r requirements-dev.txt

# 3. Launch the dashboard → http://localhost:8501
streamlit run Resume_Screening_Project/src/app.py

# Optional
python Resume_Screening_Project/src/train_models.py   # rebuild models/ (≈3–4 min on a laptop CPU; --skip-eval to skip the evaluation)
python -m pytest Resume_Screening_Project/tests       # 30 tests: cleaning, extraction, scoring, UI flows
python evaluate_classifiers.py                         # refresh evaluation results + report figures
```

The trained artifacts are committed in `Resume_Screening_Project/models/`, so step 3 works straight after installing. The first run downloads all-MiniLM-L6-v2 (~90 MB) and the NLTK stop-word and WordNet data.

**Quick demo:** open *Screen resumes*, load a sample job such as "HR Generalist", click **Add sample resumes** (three of the six match the job's category) and press **Run scan**.

## Deploying to Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and choose **Create app → Deploy a public app from GitHub**.
2. Repository `akshatsharma-x/cv-scan`, choose your branch, main file path `Resume_Screening_Project/src/app.py`.
3. Under **Advanced settings** pick **Python 3.12**, then **Deploy**.

Community Cloud installs `requirements.txt` from the repository root (CPU-only PyTorch keeps the install small) and runs the app from the root, so `.streamlit/config.toml` applies. The first boot takes a few minutes; the running app uses about 0.7–0.8 GB of memory.

## Notes and limitations

- Scores are *relative*: 100 % means "as close to this job as the best historical resume", and the pool percentile is the most robust number to compare across jobs.
- Category labels in the public dataset are noisy (e.g. *AUTOMOBILE* and *BPO* resumes span many roles), which caps classification accuracy and lowers the ranking scores for those jobs.
- Scanned, image-only PDFs have no text layer; the app reports them instead of scoring empty text.
