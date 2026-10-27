# Multimodal-Fake-News-Detection-Using-Vision-Language-Models-

## Veriscope website

The website source is in [`website/index.html`](website/index.html). It is a responsive,
buildless HTML/CSS/JavaScript research interface for the multimodal fake news detection project.

Private hosted site: https://veriscope-multimodal-lab.dark-box-6331.chatgpt.site

Access to the hosted site is restricted to its owner. This repository does not configure
GitHub Pages or change the hosted site's access controls.

### Preview locally

Open `website/index.html` in a browser, or serve it with Python:

```sh
python -m http.server 8000 --directory website
```

Then visit http://localhost:8000.

### Included functionality

- Responsive project overview and methodology sections.
- Browser-local claim entry and image preview (JPG, PNG, WebP; up to 10 MB).
- Input validation, pair preparation, and reset controls.
- Research status and explanations of CLIP, fusion, and MERF.

### Current limitations

The website is a research prototype. Live model inference is not connected, and it
does not classify uploaded claims as fake or real. Inputs remain in the browser tab
and are not uploaded or saved. Google Fonts is used for typography.

The Python research pipeline is included in this repository. Its checks used synthetic
embeddings; real-data training and evaluation on Fakeddit are pending. No research
accuracy is claimed by this site.

## Python research pipeline

See [PIPELINE.md](PIPELINE.md) for the full methodology, dataset format, commands,
and limitations, and [DATASET_PLAN.md](DATASET_PLAN.md) for public dataset preparation.

Included: CLIP text/image feature extraction, concatenation and interaction fusion,
text-only and image-only baselines, logistic classification, Mixed Effects Random
Forest, grouped train/validation/test splits, model saving, prediction, and tests.

### Windows setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe fake_news.py demo
.\.venv\Scripts\python.exe fake_news.py train artifacts/synthetic.npz --output artifacts/synthetic-models --merf-iterations 2
.\.venv\Scripts\python.exe fake_news.py predict artifacts/synthetic-models/interaction_merf.joblib artifacts/synthetic.npz
```

`requirements-lock.txt` records the verified environment; `requirements.txt` contains
dependency ranges. Python 3.12 was used locally. CLIP weights download on first
feature extraction. Synthetic results only verify execution and must not be reported
as fake-news detection accuracy.

Virtual environments, caches, downloaded data, and generated model artifacts are
excluded from Git. Recreate them with the commands above. No real dataset or trained
real-data model is included.
