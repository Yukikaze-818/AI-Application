# Project Running Guide

## 1. File Structure

- `data/raw/weatherAUS.csv`
  The real Kaggle Rain in Australia dataset if you have it.
- `data/raw/weatherAUS_demo.csv`
  Auto-generated demo dataset used for local testing.
- `scripts/generate_demo_data.py`
  Generates a demo dataset.
- `scripts/run_eda.py`
  Runs exploratory data analysis and exports plots.
- `scripts/train_pipeline.py`
  Trains, tunes, evaluates, and saves the best model.
- `app/streamlit_app.py`
  Streamlit application for probability prediction and what-if analysis.
- `src/rain_prediction/`
  Reusable project modules.
- `outputs/eda/`
  EDA plots and summary JSON.
- `outputs/metrics/`
  Evaluation metrics, ROC curve, PR curve, confusion matrix.
- `outputs/models/`
  Saved model and training summary.
- `outputs/reports/`
  Explainability artifacts and error analysis.

## 2. Recommended Execution Order

### Option A: If you do not have the Kaggle dataset yet

1. Generate demo data:

```powershell
python scripts/generate_demo_data.py
```

2. Run EDA:

```powershell
python scripts/run_eda.py --data data/raw/weatherAUS_demo.csv
```

3. Train and evaluate models:

```powershell
python scripts/train_pipeline.py --data data/raw/weatherAUS_demo.csv
```

4. Launch the web app:

```powershell
streamlit run app/streamlit_app.py
```

### Option B: If you already have the Kaggle dataset

1. Put the CSV file here:

```text
data/raw/weatherAUS.csv
```

2. Run EDA:

```powershell
python scripts/run_eda.py --data data/raw/weatherAUS.csv
```

3. Train and evaluate:

```powershell
python scripts/train_pipeline.py --data data/raw/weatherAUS.csv
```

4. Launch the web app:

```powershell
streamlit run app/streamlit_app.py
```


### Recommended Course Workflow

1. Small-sample multi-model comparison:

```powershell
python scripts/train_pipeline.py --data data/raw/weatherAUS.csv --max-rows 5000 --run-name baseline
```

2. Larger GPU-focused XGBoost run:

```powershell
python scripts/train_pipeline.py --data data/raw/weatherAUS.csv --max-rows 20000 --xgb-only --run-name xgb_gpu
```

3. Full-dataset XGBoost optimization:

```powershell
python scripts/train_pipeline.py --data data/raw/weatherAUS.csv --xgb-only --run-name xgb_full
```

## 3. What Each Step Produces

- `generate_demo_data.py`
  Creates a testable weather dataset.
- `run_eda.py`
  Produces missing-value plots, feature distributions, correlation heatmap, and `eda_summary.json`.
- `train_pipeline.py`
  Produces:
  - `outputs/models/best_model.joblib`
  - `outputs/models/training_summary.json`
  - `outputs/metrics/model_comparison.csv`
  - `outputs/metrics/roc_curve.png`
  - `outputs/metrics/pr_curve.png`
  - `outputs/metrics/confusion_matrix.png`
  - `outputs/reports/feature_importance.png`
  - `outputs/reports/partial_dependence.png`
  - `outputs/reports/error_analysis.csv`
- `streamlit_app.py`
  Loads the saved model and allows interactive what-if predictions.

## 4. Important Notes

- `xgboost` is optional in the current environment. If it is not installed, the pipeline still runs with Logistic Regression, Decision Tree, and Random Forest.
- The current local validation was completed with the auto-generated demo dataset.
- To use the real course project dataset, replace the demo CSV with the Kaggle file and rerun EDA plus training.
- For the project requirements, a good strategy is:
  - use `--max-rows 5000` for multi-model comparison,
  - then use `--xgb-only` on larger data to focus on GPU training.


## 5. Separate Output Folders

Each run can now be saved in its own subfolder under `outputs/` by using `--run-name`.

Examples:

- `outputs/baseline/metrics`
- `outputs/baseline/models`
- `outputs/xgb_gpu/metrics`
- `outputs/xgb_gpu/models`
- `outputs/xgb_full/metrics`
- `outputs/xgb_full/models`
