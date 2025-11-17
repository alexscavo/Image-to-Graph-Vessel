# fitsne.py — local shim so "from fitsne import FItSNE" works
# Uses openTSNE if available; otherwise falls back to scikit-learn TSNE.

import numpy as np

try:
    from openTSNE import TSNE as _TSNE

    def FItSNE(X, no_dims=2, perplexity=30, seed=None, nthreads=0, **kwargs):
        X = np.asarray(X)
        tsne = _TSNE(
            n_components=no_dims,
            perplexity=perplexity,
            initialization="pca",
            n_jobs=None if not nthreads else nthreads,
            random_state=seed,
        )
        emb = tsne.fit(X)        # openTSNE returns an Embedding; convert to ndarray
        return np.asarray(emb)

except Exception:
    # Fallback to sklearn
    from sklearn.manifold import TSNE as _TSNE

    def FItSNE(X, no_dims=2, perplexity=30, seed=None, nthreads=0, **kwargs):
        X = np.asarray(X)
        tsne = _TSNE(
            n_components=no_dims,
            perplexity=perplexity,
            init="pca",
            random_state=seed,
        )
        return tsne.fit_transform(X)