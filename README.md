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

The Python research pipeline is currently maintained locally and is not included in
this website source upload. Its checks used synthetic embeddings; real-data training
and evaluation on Fakeddit are pending. No research accuracy is claimed by this site.
