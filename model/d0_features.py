"""Feature extractors for pocker_d0's ensemble — shared by trainer + infer so train == serve.

The benchmark-supervised stack + monotone-XGB run on the phasberg features; the neural MLP runs
on the v2+phasberg UNION; the DRSE runs on the v2 features alone. Three feature views, four models.
"""
from features_v2 import extract_features_v2
from poker44_ml.features import chunk_features


def phasberg_dict(chunk):
    d = chunk_features(chunk or [])
    d["hand_count"] = float(len(chunk or []))
    return d


def v2_dict(chunk):
    return extract_features_v2(chunk or [])
