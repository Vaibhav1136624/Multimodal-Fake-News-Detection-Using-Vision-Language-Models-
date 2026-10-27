# Multimodal Fake News Detection Using Vision-Language Models

Initial research implementation: frozen CLIP encoders, multimodal feature fusion,
logistic classification baselines, and experimental Mixed Effects Random Forest (MERF).

## Windows setup

Use `.venv\Scripts\python.exe` directly; activation is optional.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe fake_news.py demo
.\.venv\Scripts\python.exe fake_news.py train artifacts/synthetic.npz --output artifacts/synthetic-models --merf-iterations 2
```

The demo contains fabricated embeddings and labels. Its metrics only check execution;
they are not fake-news detection results. No real dataset or pretrained weights ship here.

## Real data

Create a CSV with columns `id,text,image,label,group`. Image paths are relative to the CSV.
Labels: 0 = real, 1 = fake. Each group should identify a shared news event, story, or
source, chosen for the research question. Use repeated observations per group for MERF.
Keep related posts and near-duplicate images in the same group. Check duplicates across
events before training. At least five groups are required; substantially more are advised.

```powershell
.\.venv\Scripts\python.exe fake_news.py extract data/train.csv
.\.venv\Scripts\python.exe fake_news.py train artifacts/features.npz
.\.venv\Scripts\python.exe fake_news.py predict artifacts/models/interaction_merf.joblib artifacts/features.npz
```

Extraction downloads `openai/clip-vit-base-patch32` on first use. CPU is supported;
CUDA is selected when available. CLIP truncates text to 77 tokens: long claims require
a later chunking or long-text encoder experiment. Do not silently treat full articles
as fully represented.

## Experiments

Compare text-only, image-only, concatenation, and interaction fusion. Interaction fusion
uses normalized text/image embeddings, absolute difference, elementwise product and
cosine similarity. Logistic regression is the classification baseline. MERF uses a
random forest fixed effect and a per-group random intercept, fitted to binary labels
as an experimental regression formulation. Scores can fall outside [0,1] and are NOT
probabilities. New groups receive the fixed-effect prediction; this group-held-out
evaluation therefore measures generalization to unseen groups, not personalization.

Deterministic group splits reserve approximately 60/20/20 percent of groups for
train/validation/test. Thresholds maximize validation macro-F1; test labels are not
used to tune them. Metrics report binary accuracy, macro-F1, confusion matrices and
the fraction of scores within a heuristic abstention margin. `uncertain` is an
abstention policy, not a trained third class or verified misleading-content label.
Saved split IDs and per-example predictions support auditing. Joblib files must only
be loaded from trusted sources.

The predict command accepts cached features with the same encoder. For unlabeled
inference, omit the manifest label column. Extraction stores a sentinel -1 and
training rejects these unlabeled features; prediction does not use labels.
See DATASET_PLAN.md for the selected public benchmark and preparation requirements.

## Current scope and next work

Implemented: input validation, pretrained encoder extraction, feature fusion,
six baseline experiments, grouped evaluation, model artifacts, synthetic smoke test.
Next: select and license a real dataset (e.g. Fakeddit or a verified image-text corpus),
deduplicate it, measure class/group balance, train and evaluate on real examples,
add calibrated uncertainty, then evidence retrieval and a user interface.
Embedding similarity does not establish factual truth, image authenticity, or age.
Evidence-grounded explanations and external factual verification are not implemented.

References: https://huggingface.co/docs/transformers/model_doc/clip and
https://github.com/manifoldai/merf .
