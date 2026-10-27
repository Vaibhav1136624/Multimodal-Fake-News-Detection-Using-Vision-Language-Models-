"""CLIP fusion baselines and experimental mixed-effects regression for fake news."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fusion(text, image, mode='interaction'):
    text, image = np.asarray(text), np.asarray(image)
    if text.ndim != 2 or text.shape != image.shape:
        raise ValueError('Text and image embeddings must have matching (rows, dimensions) shapes.')
    if not np.isfinite(text).all() or not np.isfinite(image).all():
        raise ValueError('Embeddings must be finite.')
    t = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    v = image / np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    if mode == 'text': return t
    if mode == 'image': return v
    if mode == 'concat': return np.concatenate([t, v], axis=1)
    if mode != 'interaction': raise ValueError('Unknown fusion mode')
    return np.concatenate([t, v, np.abs(t-v), t*v, np.sum(t*v, axis=1, keepdims=True)], axis=1)


def read_manifest(path):
    path = Path(path).resolve()
    df = pd.read_csv(path, dtype={'id': str, 'group': str})
    required = {'id', 'text', 'image', 'group'}
    if not required.issubset(df.columns):
        raise ValueError(f'Required columns: {sorted(required)}')
    if df[list(required)].isna().any().any() or df.id.duplicated().any():
        raise ValueError('Missing values or duplicate IDs in manifest.')
    if 'label' in df and not df.label.isin([0, 1]).all():
        raise ValueError('Labels must be 0 (real) or 1 (fake).')
    if 'label' not in df: df['label'] = -1
    for col in ['text', 'image', 'group']:
        if df[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'Empty {col} value.')
    df['image'] = [str((path.parent / p).resolve()) for p in df.image]
    missing = [p for p in df.image if not Path(p).is_file()]
    if missing: raise ValueError(f'Image not found: {missing[0]}')
    return df


def extract(args):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    df = read_manifest(args.manifest)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)
    texts, images = [], []
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start:start+args.batch_size]
        pictures = []
        for path in batch.image:
            with Image.open(path) as im: pictures.append(im.convert('RGB'))
        inputs = processor(text=batch.text.tolist(), images=pictures, return_tensors='pt',
                           padding=True, truncation=True, max_length=77).to(device)
        with torch.inference_mode():
            result = model(**inputs)
        texts.append(result.text_embeds.cpu().numpy())
        images.append(result.image_embeds.cpu().numpy())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, text=np.concatenate(texts), image=np.concatenate(images),
                        label=df.label.to_numpy(), group=df.group.to_numpy(dtype=str),
                        ids=df.id.to_numpy(dtype=str), encoder=np.array(args.model))


def split_groups(y, groups):
    if len(np.unique(groups)) < 5:
        raise ValueError('Use at least five independent event/source groups.')
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
    train, test = next(splitter.split(y, y, groups))
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=43).split(
        y[train], y[train], groups[train]))
    train, valid = train[a], train[b]
    for name, indices in [('train', train), ('validation', valid), ('test', test)]:
        if len(np.unique(y[indices])) != 2:
            raise ValueError(f'{name} split needs both classes; add representative groups.')
    return train, valid, test


def train(args):
    from merf.merf import MERF
    with np.load(args.features, allow_pickle=False) as data:
        t, v, y, groups = data['text'], data['image'], data['label'], data['group']
        ids, encoder = data['ids'], str(data['encoder'])
    if not (len(t) == len(v) == len(y) == len(groups) == len(ids)):
        raise ValueError('Feature rows are not aligned.')
    if not np.isin(y, [0, 1]).all() or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid labels or duplicate IDs.')
    tr, va, te = split_groups(y, groups)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in ['text', 'image', 'concat', 'interaction']:
        x = fusion(t, v, mode)
        for kind in (['logistic', 'merf'] if mode in ['concat', 'interaction'] else ['logistic']):
            name = f'{mode}_{kind}'
            if kind == 'merf':
                model = MERF(RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                             random_state=42, n_jobs=-1), max_iterations=args.merf_iterations)
                model.fit(x[tr], np.ones((len(tr), 1)), pd.Series(groups[tr]), y[tr].astype(float))
                def score(indices):
                    return model.predict(x[indices], np.ones((len(indices), 1)), pd.Series(groups[indices]))
            else:
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
                model.fit(x[tr], y[tr])
                def score(indices): return model.predict_proba(x[indices])[:, 1]
            valid_scores = score(va)
            thresholds = np.linspace(0, 1, 101)
            threshold = float(max(thresholds, key=lambda q: f1_score(y[va], valid_scores >= q, average='macro')))
            scores = score(te); predictions = scores >= threshold
            uncertain = np.abs(scores-threshold) < args.margin
            results[name] = {'accuracy': accuracy_score(y[te], predictions),
                             'macro_f1': f1_score(y[te], predictions, average='macro'),
                             'confusion_matrix_real_fake': confusion_matrix(y[te], predictions, labels=[0, 1]).tolist(),
                             'threshold': threshold, 'abstention_rate': float(uncertain.mean())}
            joblib.dump({'model': model, 'kind': kind, 'fusion': mode, 'encoder': encoder,
                         'threshold': threshold, 'margin': args.margin}, out / f'{name}.joblib')
            pd.DataFrame({'id': ids[te], 'label': y[te], 'score': scores,
                          'prediction': np.where(uncertain, 'uncertain', np.where(predictions, 'fake', 'real'))}).to_csv(out / f'{name}_predictions.csv', index=False)
    (out / 'metrics.json').write_text(json.dumps(results, indent=2))
    (out / 'splits.json').write_text(json.dumps({k: ids[idx].tolist() for k, idx in [('train',tr),('validation',va),('test',te)]}, indent=2))
    print(json.dumps(results, indent=2))


def demo(args):
    """Synthetic embedding smoke test, never a scientific benchmark."""
    rng = np.random.default_rng(42)
    groups = np.repeat(np.arange(20), 10).astype(str)
    y = np.tile([0, 1], 100)
    t = rng.normal(size=(200, 16)); v = t.copy()
    v[y == 1] = rng.normal(size=(int(y.sum()), 16))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, text=t, image=v, label=y, group=groups,
                        ids=np.array([f'synthetic-{i}' for i in range(200)]), encoder=np.array('SYNTHETIC-NOT-CLIP'))


def predict(args):
    bundle = joblib.load(args.model)
    with np.load(args.features, allow_pickle=False) as data:
        if str(data['encoder']) != bundle['encoder']:
            raise ValueError('Feature encoder does not match the trained model.')
        x = fusion(data['text'], data['image'], bundle['fusion'])
        groups, ids = data['group'], data['ids']
    if bundle['kind'] == 'merf':
        scores = bundle['model'].predict(x, np.ones((len(x), 1)), pd.Series(groups))
    else:
        scores = bundle['model'].predict_proba(x)[:, 1]
    uncertain = np.abs(scores-bundle['threshold']) < bundle['margin']
    output = pd.DataFrame({'id': ids, 'score': scores,
        'prediction': np.where(uncertain, 'uncertain', np.where(scores >= bundle['threshold'], 'fake', 'real'))})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f'Saved {len(output)} predictions to {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('extract'); p.add_argument('manifest'); p.add_argument('--output', default='artifacts/features.npz')
    p.add_argument('--model', default='openai/clip-vit-base-patch32'); p.add_argument('--batch-size', type=int, default=8); p.set_defaults(func=extract)
    p = sub.add_parser('train'); p.add_argument('features'); p.add_argument('--output', default='artifacts/models')
    p.add_argument('--merf-iterations', type=int, default=10); p.add_argument('--margin', type=float, default=.1); p.set_defaults(func=train)
    p = sub.add_parser('demo'); p.add_argument('--output', default='artifacts/synthetic.npz'); p.set_defaults(func=demo)
    p = sub.add_parser('predict'); p.add_argument('model'); p.add_argument('features')
    p.add_argument('--output', default='artifacts/predictions.csv'); p.set_defaults(func=predict)
    args = parser.parse_args()
    if hasattr(args, 'margin') and not 0 <= args.margin <= .5: parser.error('margin must be in [0, .5]')
    if hasattr(args, 'batch_size') and args.batch_size < 1: parser.error('batch size must be positive')
    if hasattr(args, 'merf_iterations') and args.merf_iterations < 1: parser.error('MERF iterations must be positive')
    args.func(args)


if __name__ == '__main__': main()
