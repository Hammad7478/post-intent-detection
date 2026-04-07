# Group 27 Final Submission

This package contains the final report source and the code used for the Group 27 project, **Post Intent Detection: Social Media Communication**, for COMPSCI 4NL3.

## Contents

- `Final Project TeX/`
  - `main.tex`: final report source
  - `custom.bib`: bibliography file
  - `acl.sty`, `acl_natbib.bst`: ACL template files used to compile the report
- `data/data_splits/`
  - `train.json`
  - `val.json`
  - `test.json`
- `baseline_models/`
  - majority, random, and logistic-regression baselines
- `models/`
  - `model_a_tfidf.py`
  - `model_b_embeddings.py`
  - `model_c_transformer.py`
  - `model_d_roberta.py`
- `util/`
  - preprocessing, kappa computation, visualization, and Codabench packaging utilities

## Dataset

The final dataset contains **2,156** Reddit posts from:

- `r/cscareerquestions`
- `r/personalfinance`
- `r/careerguidance`
- `r/tifu`
- `r/unpopularopinion`

The standardized split used throughout the final project is:

- Train: 1,507
- Validation: 321
- Test: 328

## Omitted artifacts

Large trained model checkpoints were intentionally **not** included in this submission archive because of file-size limits. The repository therefore contains the full training and inference scripts, but not the saved transformer weights.

## Environment

Recommended Python version: **3.10+**

Core dependencies:
- `numpy`
- `scikit-learn`
- `joblib`
- `torch`
- `transformers`
- `datasets`
- `accelerate`

Example installation:

```bash
pip install numpy scikit-learn joblib torch transformers datasets accelerate
```

## Reproducing the models

Run commands from the project root (`Group27_Final_Submission/`).

### Baseline models

```bash
python baseline_models/logistic_regression_baseline.py
python baseline_models/majority_baseline.py
python baseline_models/random_baseline.py
```

### Model A: word-level TF-IDF + logistic regression

```bash
python models/model_a_tfidf.py train \
  --train data/data_splits/train.json \
  --val data/data_splits/val.json \
  --test data/data_splits/test.json \
  --artifacts_dir models/artifacts
```

### Model B: hybrid word/character TF-IDF

```bash
python models/model_b_embeddings.py train \
  --train data/data_splits/train.json \
  --val data/data_splits/val.json \
  --test data/data_splits/test.json \
  --artifacts_dir models/artifacts_b
```

### Model C: DistilBERT

```bash
python models/model_c_transformer.py train \
  --train data/data_splits/train.json \
  --val data/data_splits/val.json \
  --test data/data_splits/test.json \
  --output_dir models/artifacts_c
```

### Model D: RoBERTa-base

```bash
python models/model_d_roberta.py train \
  --train data/data_splits/train.json \
  --val data/data_splits/val.json \
  --test data/data_splits/test.json \
  --output_dir models/artifacts_d_roberta
```

## Codabench / unlabeled inference

Each model script contains a `predict` mode that can be used on unlabeled JSON files containing `id` and `text` fields. For example:

```bash
python models/model_d_roberta.py predict \
  --input test_unlabeled.json \
  --output submission.json \
  --model_path models/artifacts_d_roberta/best_model
```

## Report compilation

Compile the report from the `Final Project TeX/` directory:

```bash
pdflatex main.tex
pdflatex main.tex
```

The bibliography is embedded directly in `main.tex`, so no BibTeX step is required for the final version.

## Notes

- All file read/write operations use UTF-8.
- The report source is the authoritative description of the final system and dataset.
- Auxiliary logs or intermediate artifacts that may appear in development were not intended as the final summary of results.
