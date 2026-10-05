
# MRI QC Dashboard

A Streamlit dashboard for batch quality-control of MRI scans, built around
the feature/artifact pipeline from `Model1.ipynb`.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Two data inputs (sidebar)

1. **Reference (labeled) data** — a CSV with QC features *and* an `Artifact`
   label column. Used only to train the XGBoost classifier and to set
   normalization ranges. Defaults to the bundled `final_artifact_features.csv`;
   upload your own to retrain on different data.

2. **Scan batch to score** — three input methods, pick from the sidebar radio:

   - **Upload ZIP** — a ZIP of paired `.hdr`/`.img` volumes (or standalone
     `.nii`/`.nii.gz`), one pair per scan. Fine for smaller batches; the
     bundled `.streamlit/config.toml` raises the upload cap to 2GB, but
     browser upload gets impractical well before 10GB.
   - **Local folder path** — type a path already on the machine running
     the app (e.g. wherever you downloaded/extracted a big dataset). Reads
     straight from disk with no upload step and no size limit. **This is
     the recommended method for large (multi-GB) datasets.**
   - **Download from Kaggle** — give a dataset slug (`owner/dataset-name`,
     from the dataset's URL) and the app downloads it via `kagglehub` into
     a local cache, then reads it the same way as the local-folder method.
     Requires Kaggle API credentials on the machine: either a
     `~/.kaggle/kaggle.json` file, or `KAGGLE_USERNAME` / `KAGGLE_KEY`
     environment variables (create a key at kaggle.com → Account → Create
     New API Token).

   Any folder structure works — pairs are matched by matching filename stem
   within the same folder. If nothing is supplied yet, the dashboard scores
   the reference CSV itself as a demo.

   For each scan, the app runs the notebook's per-axial-slice feature
   pipeline (`Mean`, `Std`, `SNR`, `Entropy`, GLCM texture stats, etc.),
   averages across slices, and feeds the 6 model features into the trained
   XGBoost classifier. A **"max slices per scan"** slider trades extraction
   speed for thoroughness — GLCM computation is done per slice, so large
   batches of large volumes go faster if you sample fewer slices.

## Is the XGBoost model pretrained?

Not out of the box, but it's **persisted with joblib** so it only actually
trains once. The first time you run the app, it trains on the reference CSV
(a few seconds) and saves the result to `saved_models/xgb_artifact_classifier.joblib`.
Every run after that — even after closing VS Code and coming back later —
loads that file straight off disk instead of retraining, as long as the
reference CSV hasn't changed (checked via a content hash stored alongside
the model). Change the reference CSV and it automatically retrains and
overwrites the saved file. There's also a **"🔁 Force retrain"** button in
the sidebar if you want to retrain on demand regardless.

A 10GB batch of scans never touches training either way — only the small
reference CSV does. The 10GB only affects **feature extraction** time.

## What's included

- **XGBoost artifact classifier** — trained live (cached, and persisted to
  disk via joblib) on the reference data using the same hyperparameters as
  the notebook. Feature set: the notebook's original 6
  (`SNR`, `Entropy`, `LaplacianVariance`, `GLCMContrast`, `GLCMEnergy`,
  `GLCMHomogeneity`) plus two new bias-field-detection features
  (`BiasQuadrantRange`, `BiasGradientMagnitude`) when the reference data
  includes them — see "Improving accuracy" below.
- **QC Score (0–100) & PASS/REVIEW/FAIL status** — a probability-weighted
  "quality" score across all predicted classes (not just the top class),
  so an uncertain prediction lands between categories instead of snapping
  hard to one label. Class weights and PASS/REVIEW thresholds are
  adjustable from the sidebar.
- **Batch Scorecard** — KPI summary, color-coded results table, confusion
  matrix on the hold-out split, and a CSV download of the full QC report.
- **Analytics** — artifact distribution, PASS/REVIEW/FAIL split, per-feature
  box plots, average feature profile bars, and a normalized radar chart
  comparing predicted classes.
- **Scan Drill-down** — pick any scan to see its class probabilities, a
  radar of its feature profile vs. the dataset average, raw feature values,
  and — when the scan came from an uploaded volume — an **interactive
  axial slice viewer**.
- **Roadmap page** — the NIfTI/Analyze slice viewer is now live; still
  planned: skull-strip overlay and ABIDE site analysis, with notes on how
  to wire them in.

## Improving accuracy

The bundled reference CSV caps out around **92% accuracy**, and it's not a
tuning problem — `bias` and `original` are nearly statistically identical
in the notebook's original 6 whole-slice-averaged features (e.g. mean
intensity differs by <2%, well within 1 std dev). Everything else
(`blur`/`motion`/`noise`) already classifies at ~99–100%.

Two new features target this specifically — `BiasQuadrantRange` and
`BiasGradientMagnitude` — which measure *spatial* intensity trends (a 2x2
quadrant-mean spread, and a fitted intensity-gradient magnitude) rather
than whole-image averages, since bias-field artifacts are smooth spatial
gradients that whole-image stats wash out. Validated on synthetic
clean-vs-biased volumes, these separate the two classes cleanly with no
overlap across trials.

**Any scan you extract via ZIP / local folder / Kaggle already gets these
two new features automatically** — no setup needed. The catch: the
*reference/training* CSV bundled with this app was extracted before these
features existed, so the model can't use them until the reference data is
regenerated from the original raw volumes.

If you have (or can get) those raw volumes, use:

```bash
python regenerate_reference_csv.py \
    --volumes-dir /path/to/raw/volumes \
    --manifest /path/to/manifest.csv \
    --output reference_features_v2.csv \
    --max-slices 64
```

`manifest.csv` needs `filename,Artifact,Level` columns (filename = the
volume's filename stem, no extension). Drop the resulting CSV into the
"Reference (labeled) data" uploader in the sidebar — the app will
auto-detect the new columns and start using them, and the sidebar will
say so explicitly ("✅ Using 2 bias-detection feature(s)...").

Without the raw volumes, the model stays at the ~92% ceiling on this
particular dataset (a `🔁 Force retrain` won't change that — the
information genuinely isn't in the current features).

## Files

- `app.py` — the Streamlit app.
- `feature_extraction.py` — volume loading, hdr/img pairing, and the
  per-slice feature pipeline (ported from `Model1.ipynb`, plus the new
  bias-detection features).
- `regenerate_reference_csv.py` — rebuilds a labeled reference CSV (with
  the new bias features) from raw volumes + a label manifest.
- `final_artifact_features.csv` — bundled reference/demo dataset.
- `saved_models/` — created automatically; holds the joblib-persisted model.
- `requirements.txt`.

## Notes / assumptions

- The notebook's own `quality_map` only covered `original/blur/noise/motion`
  and didn't handle the `bias` class — this app replaces that hard map with
  configurable class-quality weights covering all 5 classes.
- `.hdr`/`.img` pairs are matched by identical filename stem within the same
  folder; mismatched or orphaned files are reported as warnings in the
  sidebar rather than silently dropped.
- Large batches with many/large volumes can take a while to extract since
  GLCM texture features are computed per slice — lower "max slices per scan"
  for faster turnaround, especially when just spot-checking a batch.
- The slice viewer only works for scans loaded from an uploaded ZIP (it
  needs the raw volume); scans scored from a precomputed feature CSV don't
  have a volume to display.
- The local-folder and Kaggle inputs cache extraction results keyed by
  path; if you add/change files in that folder, click **"🔄 (Re)scan
  folder"** in the sidebar to force a re-extract.

