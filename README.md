# Reproduce & Extend: *Training a Playlist Curator Based on User Taste*

ML Mini Project (UE24CS352A) · PES University

**Team: 26**  Sarang S Nair : PES2UG24CS447  Shreya C Rao : PES2UG24CS481

This repository reproduces the results of Awadelkarim & Coelho (Stanford, 2018),
*"Training a Playlist Curator Based on User Taste"*, and extends it with one additional model
(**XGBoost**) evaluated under exactly the same data, split and metrics.

**Task:** given a song's 10 Spotify audio features and its genre tags, predict which of 13
Spotify-curated playlists it belongs to (13-class classification).

---

## 1. What is in this repository

```
.
├── playlist_curator.ipynb   # the whole project: data build, Part 1 (reproduce), Part 2 (XGBoost), comparison
├── README.md              
├── writeup.pdf              # one-page project write-up
└── results/                 # created when the notebook runs (CSV tables + summary.json)
```

The large CSV files (`tracks.csv` ≈ 111 MB, `artists.csv` ≈ 65 MB) are **not** committed
(GitHub rejects files over 100 MB). See Section 3 for where to get them.

## 2. Project overview

| Part | What it does | Notebook sections |
|---|---|---|
| **Part 1 – Reproduce** | Perceptron; Polynomial / Sigmoid / RBF SVMs (scaled vs. unscaled, 5-fold CV); tuned RBF SVM with per-class precision/recall/F1; 1- and 2-hidden-layer neural networks in PyTorch | 3 – 6 |
| **Part 2 – Extend** | One new model not in the paper: **XGBoost** (gradient-boosted trees), plus an audio-only vs. audio+tags ablation | 8 – 9 |
| **Comparison** | *Reported vs. Your Result* table, model comparison, discussion | 7, 9, 10 |

Protocol shared by every model: 80/20 stratified split (`random_state=42`), 5-fold CV,
standardised inputs for the scale-sensitive models, hyper-parameters tuned on the training
split only.

## 3. Data

* **Playlists and tracks:** the authors' exact 13 playlists (1,027 track entries), downloaded
  automatically by the notebook from
  [`kevin-coelho/playlistr-ml-v1`](https://github.com/kevin-coelho/playlistr-ml-v1)
  (`get_toy_set/results/toy_data_set_playlists_full.json`). One track that appears in two playlists
  is dropped → 1,025 tracks.
* **Audio features and genre tags:** the authors' feature database was never published, and Spotify's
  `audio-features` endpoint is closed to new apps, so these are re-attached:
  * audio features from the Kaggle **Spotify Dataset 1921–2020** (`tracks.csv`), matched by track ID, then by
    title + first artist; remaining tracks are filled from the free **ReccoBeats** API (estimates);
  * genre tags from the same dataset's `artists.csv`, matched by artist ID.
* Tracks without features are dropped, leaving **678 tracks × 434 features** (10 audio + 424 genre tags)
  (542 train / 136 test).

Download `tracks.csv` and `artists.csv` from Kaggle ("Spotify Dataset 1921-2020, 600k+ Tracks")
and place them as described below.

## 4. Setup

Requires Python 3.10+.

```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. How to run

### Option A – Google Colab (how the notebook was developed)

1. Create a Drive folder named **`ML Mini Project`** and upload `tracks.csv` and `artists.csv` into it.
2. Open `playlist_curator.ipynb` in Colab and choose **Runtime → Run all**.
3. Click through the Drive-authorisation popup in the first cell. It copies the two CSVs to `/content/`.

### Option B – Local Jupyter

1. Put `tracks.csv` and `artists.csv` in the repo folder (or any folder).
2. In the first cell, **skip the Google Drive code** (it needs `google.colab`).
3. In the config cell (Section 0), set:
   ```python
   FEATURE_CSV_PATHS = ["tracks.csv"]
   GENRE_CSV_PATH    = "artists.csv"
   ```
4. Run:
   ```bash
   jupyter notebook playlist_curator.ipynb      # then Run All
   ```

An internet connection is needed on the first run (the authors' JSON, ReccoBeats, and the public
Spotify Tracks Dataset used in Section 6). Downloads are cached in `results/`.
Expected runtime: a few minutes on CPU (XGBoost tuning ≈ 1 min).

### Optional configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `DATA_SOURCE` | `paper` | `paper` = authors' exact playlists; `proxy` = public-genre proxy playlists (fallback) |
| `PAPER_JSON` | – | path to a local copy of `toy_data_set_playlists_full.json` |
| `SPOTIFY_CSV` | – | local copy of the public Spotify Tracks Dataset (`dataset.csv`) |
| `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | – | optional free Spotify dev app, to read artist genres directly |

Other switches in the config cell: `USE_RECCOBEATS`, `ONLY_EXACT_FEATURES`, `NN_SEEDS`, `NN_EPOCHS`, `RUN_PROXY_USERS`.

## 6. Outputs

Running the notebook writes to `results/`:

| File | Contents |
|---|---|
| `reported_vs_yours.csv` | paper's reported numbers vs. ours, with the difference in percentage points |
| `table1_svm_perceptron.csv` | scaled / unscaled CV accuracy for Perceptron and the three SVMs |
| `table2_rbf_svm_per_class.csv` | per-playlist precision / recall / F1 for the tuned RBF SVM |
| `model_comparison.csv`, `per_class_f1.csv` | all models side by side, including XGBoost |
| `ablation_audio_vs_tags.csv` | audio-only vs. audio + tags |
| `summary.json` | chosen hyper-parameters and headline accuracies |

## 7. Headline results

| Model | Hold-out test acc. (20 %) | 5-fold CV |
|---|---|---|
| Perceptron (scaled) | 0.699 | 0.76 ± 0.02 |
| RBF SVM (tuned: C=100, γ=1e-4) | 0.699 | 0.74 ± 0.04 |
| NN, 1 hidden layer (mean of 5 seeds) | 0.750 ± 0.016 | – |
| NN, 2 hidden layers (mean of 5 seeds) | 0.732 ± 0.014 | – |
| **XGBoost (new)** | 0.735 | 0.77 ± 0.02 |

The paper reports 0.80 (tuned RBF SVM) and 0.82 (1-hidden-layer NN). 6 of 13 reported numbers are
reproduced within 5 percentage points, and all six qualitative claims hold on our data. XGBoost performs
similarly to the reference models; on audio features alone it is the best model (0.52 vs. 0.38 SVM, 0.44 NN).

## 8. Known limitations

* Audio features/genres are re-attached after the fact (a mix of exact Spotify values and ReccoBeats
  estimates), so exact numerical agreement with the paper is not expected.
* Only 678 of 1,025 tracks have features; the test set is small (136 songs, ≈ ±4 pp noise).
* The paper's real-user experiments (Table 4) use private playlists; Section 6 uses clearly labelled proxy users.
* Some hyper-parameters (polynomial degree, γ, learning rate, L2) are not stated in the paper and were tuned or defaulted.

## 9. References

* A. Awadelkarim and K. Coelho, *Training a Playlist Curator Based on User Taste*, Stanford, 2018.
* M. Pandya, *Spotify Tracks Dataset* (Hugging Face / Kaggle).
* Pedregosa et al., scikit-learn (JMLR 2011) · Paszke et al., PyTorch (2019) · Chen & Guestrin, XGBoost (KDD 2016).
