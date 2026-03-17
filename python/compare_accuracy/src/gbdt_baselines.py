"""LightGBM baseline wrappers with monotonic constraints.

These baselines take the same ``cvec_sgn`` sign-constraint vector that
the kernel PFGD method uses and inject it as ``monotone_constraints``
into the GBDT learner.

Data convention
---------------
* ``X`` arrays are ``(nfeas, npts)`` – *features × samples* –
  matching the rest of the codebase.  They are transposed internally
  before being passed to the sklearn-compatible API.
* ``y`` is ``{-1, +1}``; converted to ``{0, 1}`` internally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_feature_frame(X: np.ndarray) -> pd.DataFrame:
    """Convert ``(nfeas, npts)`` arrays to a DataFrame with stable names."""
    X_t = np.asarray(X, dtype=float).T
    columns = [f"f{i}" for i in range(X_t.shape[1])]
    return pd.DataFrame(X_t, columns=columns)


def _use_monotone_constraints(mode_sgncon: str) -> bool:
    """Return whether monotone constraints should be enabled for GBDT baselines."""
    if mode_sgncon in {"sc", "mc"}:
        return True
    if mode_sgncon in {"sf", "mf"}:
        return False
    raise ValueError(f"Unsupported GBDT constraint mode: {mode_sgncon}")


def train_eval_lightgbm(
    X_tra: np.ndarray,
    y_tra: np.ndarray,
    X_tst: np.ndarray,
    cvec_sgn: np.ndarray,
    mode_sgncon: str,
) -> np.ndarray:
    """Train LightGBM classifier and return continuous test scores."""
    import lightgbm as lgb

    X_tra_t = _to_feature_frame(X_tra)
    X_tst_t = _to_feature_frame(X_tst)
    y_tra_01 = ((y_tra + 1) / 2).astype(int)

    unique_classes = np.unique(y_tra_01)
    if len(unique_classes) < 2:
        sco = 2.0 * unique_classes[0] - 1.0
        return np.full(X_tst_t.shape[0], sco)

    mc = [int(s) for s in cvec_sgn] if _use_monotone_constraints(mode_sgncon) else None

    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_child_samples=1,
        monotone_constraints=mc,
        random_state=0,
        verbose=-1,
    )
    model.fit(X_tra_t, y_tra_01)

    proba = model.predict_proba(X_tst_t)[:, 1]
    return 2.0 * proba - 1.0
