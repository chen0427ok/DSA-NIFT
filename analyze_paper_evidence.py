#!/usr/bin/env python3
"""Produce label-free and graph diagnostics used by the paper analysis.

This script deliberately does not infer per-document correctness on official
validation/test data: their gold labels are unavailable.  It quantifies what can
still be checked locally—graph structure, lexicon coverage, synthetic-text
distribution, and how interventions change predictions.
"""
from __future__ import annotations

import csv
import json
import pickle
import random
from pathlib import Path

import jieba
import networkx as nx
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from affective_graph import (
    seed_coherence,
    seeds_by_expansion,
    seeds_for_target,
    seeds_random,
)
from augment_generate import TARGET_BINS
from lexicon import FEATURE_NAMES, LexiconFeaturizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
PREDS = OUT / "preds"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def qstats(values) -> dict[str, float]:
    a = np.asarray(list(values), dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "p75": float(np.quantile(a, 0.75)),
        "max": float(a.max()),
    }


def graph_diagnostics() -> dict:
    with (OUT / "l3_graph.pkl").open("rb") as f:
        graph = pickle.load(f)
    nodes = list(graph)
    edges = list(graph.edges(data=True))
    rng = random.Random(2026)
    random_pairs = [rng.sample(nodes, 2) for _ in range(len(edges))]

    def attr_gap(pairs, attr):
        return [
            abs(float(graph.nodes[u][attr]) - float(graph.nodes[v][attr]))
            for u, v in pairs
        ]

    seed_modes = {}
    for mode in ("random", "lookup", "graph"):
        coherences, sizes, signatures, examples = [], [], set(), []
        rng = random.Random(42)
        for trial in range(10):
            for bin_i, (_, _, vr, ar, _) in enumerate(TARGET_BINS):
                if mode == "random":
                    seeds = seeds_random(graph, n=20, rng=rng)
                elif mode == "lookup":
                    seeds = seeds_for_target(graph, v_range=vr, a_range=ar, n=20)
                else:
                    seeds = seeds_by_expansion(
                        graph, v_range=vr, a_range=ar, n=20, rng=rng,
                        n_anchors=4, hops=2,
                    )
                words = [w for w, _, _ in seeds]
                coherences.append(seed_coherence(graph, words))
                sizes.append(len(words))
                signatures.add(tuple(words))
                if trial == 0:
                    examples.append({"bin": bin_i + 1, "words": words})
        seed_modes[mode] = {
            "coherence": qstats(coherences),
            "seed_set_size": qstats(sizes),
            "distinct_sets_of_50": len(signatures),
            "first_trial_examples": examples,
        }

    comps = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    result = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "components": len(comps),
        "largest_component_ratio": comps[0] / graph.number_of_nodes(),
        "average_degree": 2 * graph.number_of_edges() / graph.number_of_nodes(),
        "degree": qstats(dict(graph.degree()).values()),
        "clustering": nx.average_clustering(graph),
        "edge_weight": qstats(d["weight"] for _, _, d in edges),
        "valence": qstats(graph.nodes[w]["valence"] for w in nodes),
        "arousal": qstats(graph.nodes[w]["arousal"] for w in nodes),
        "valence_assortativity": nx.numeric_assortativity_coefficient(graph, "valence"),
        "arousal_assortativity": nx.numeric_assortativity_coefficient(graph, "arousal"),
        "edge_valence_gap": qstats(attr_gap([(u, v) for u, v, _ in edges], "valence")),
        "random_valence_gap": qstats(attr_gap(random_pairs, "valence")),
        "edge_arousal_gap": qstats(attr_gap([(u, v) for u, v, _ in edges], "arousal")),
        "random_arousal_gap": qstats(attr_gap(random_pairs, "arousal")),
        "top_degree": sorted(graph.degree(), key=lambda x: (-x[1], x[0]))[:15],
        "seed_modes": seed_modes,
    }
    for word in ("焦慮", "開心", "平靜", "疲憊", "憤怒"):
        if word in graph:
            nbrs = sorted(
                graph[word].items(), key=lambda x: -x[1]["weight"]
            )[:8]
            result.setdefault("neighborhoods", {})[word] = [
                {
                    "word": w,
                    "weight": float(edge["weight"]),
                    "valence": float(graph.nodes[w]["valence"]),
                    "arousal": float(graph.nodes[w]["arousal"]),
                }
                for w, edge in nbrs
            ]
    return result


def lexicon_diagnostics() -> dict:
    fz = LexiconFeaturizer()
    datasets = {
        "validation": ROOT / "DSANIDF_ValidationSet.csv",
        "test": ROOT / "DSANIDF_TestSet.csv",
        "train": HERE / "data" / "train.csv",
    }
    result = {"lexicon_entries": len(fz), "feature_names": FEATURE_NAMES}
    for name, path in datasets.items():
        rows = read_rows(path)
        text_key = "Text" if name != "train" else "text"
        feats = np.asarray([fz.featurize(r[text_key]) for r in rows])
        result[name] = {
            "documents": len(rows),
            "document_hit_rate": float(np.mean(feats[:, 1] > 0)),
            "coverage": qstats(feats[:, 0]),
            "hit_count_capped_normalized": qstats(feats[:, 1]),
            "arousal_mean_feature": qstats(feats[:, 6]),
            "arousal_std_feature": qstats(feats[:, 9]),
        }
    return result


def corpus_surface(path: Path) -> dict:
    rows = read_rows(path)
    texts = [r["text"] for r in rows]
    lengths = [len(t.replace(" ", "")) for t in texts]
    tokens = [
        tok for text in texts for tok in jieba.cut(text)
        if tok.strip() and any(ch.isalnum() for ch in tok)
    ]
    return {
        "documents": len(texts),
        "length": qstats(lengths),
        "under_50_ratio": float(np.mean(np.asarray(lengths) < 50)),
        "over_120_ratio": float(np.mean(np.asarray(lengths) > 120)),
        "type_token_ratio": len(set(tokens)) / len(tokens),
    }


def official_surface(path: Path) -> tuple[dict, list[int]]:
    rows = read_rows(path)
    texts = [r["Text"] for r in rows]
    lengths = [len(t.replace(" ", "")) for t in texts]
    tokens = [
        tok for text in texts for tok in jieba.cut(text)
        if tok.strip() and any(ch.isalnum() for ch in tok)
    ]
    return {
        "documents": len(texts),
        "length": qstats(lengths),
        "under_50_ratio": float(np.mean(np.asarray(lengths) < 50)),
        "over_120_ratio": float(np.mean(np.asarray(lengths) > 120)),
        "type_token_ratio": len(set(tokens)) / len(tokens),
    }, lengths


def pred_map(run: str, split: str) -> dict[str, np.ndarray]:
    rows = read_rows(PREDS / f"{run}_{split}.csv")
    return {
        r["ID"]: np.asarray(
            [float(r["valence_pred"]), float(r["arousal_pred"])], dtype=float
        )
        for r in rows
    }


def mean_prediction(runs: list[str], split: str) -> dict[str, np.ndarray]:
    maps = [pred_map(run, split) for run in runs]
    ids = maps[0].keys()
    return {i: np.mean([m[i] for m in maps], axis=0) for i in ids}


def prediction_summary(preds: dict[str, np.ndarray]) -> dict:
    a = np.vstack(list(preds.values()))
    return {
        "valence": qstats(a[:, 0]),
        "arousal": qstats(a[:, 1]),
        "va_correlation": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]),
    }


def match_hits(text: str, fz: LexiconFeaturizer) -> list[dict]:
    hits = []
    for tok in jieba.cut(text):
        if tok in fz.lex:
            v, a = fz.lex[tok]
            hits.append({"term": tok, "valence": 1 + 8 * v, "arousal": 1 + 8 * a})
    return hits


def compare_predictions(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    texts: dict[str, str],
    example_n: int = 6,
) -> dict:
    ids = sorted(set(left) & set(right))
    l = np.vstack([left[i] for i in ids])
    r = np.vstack([right[i] for i in ids])
    delta = l - r
    fz = LexiconFeaturizer()
    order = np.argsort(-np.abs(delta[:, 1]))[:example_n]
    return {
        "documents": len(ids),
        "mean_delta_left_minus_right": delta.mean(0).tolist(),
        "mean_absolute_delta": np.abs(delta).mean(0).tolist(),
        "prediction_correlation": [
            float(np.corrcoef(l[:, j], r[:, j])[0, 1]) for j in range(2)
        ],
        "left_summary": prediction_summary({i: left[i] for i in ids}),
        "right_summary": prediction_summary({i: right[i] for i in ids}),
        "largest_arousal_disagreements": [
            {
                "id": ids[k],
                "text": texts.get(ids[k], ""),
                "left": l[k].tolist(),
                "right": r[k].tolist(),
                "delta": delta[k].tolist(),
                "lexicon_hits": match_hits(texts.get(ids[k], ""), fz),
            }
            for k in order
        ],
    }


def prediction_diagnostics() -> dict:
    test_texts = {
        r["ID"]: r["Text"] for r in read_rows(ROOT / "DSANIDF_TestSet.csv")
    }
    val_texts = {
        r["ID"]: r["Text"] for r in read_rows(ROOT / "DSANIDF_ValidationSet.csv")
    }
    l1 = mean_prediction(["macbert_s1", "macbert_s2", "macbert_s42"], "test")
    nolex = mean_prediction(["nolex_s1", "nolex_s2", "nolex_s42"], "test")
    result = {
        "l1_vs_no_l1_test_3seed_ensemble": compare_predictions(
            l1, nolex, test_texts
        ),
        "e19_vs_e18_test": compare_predictions(
            pred_map("e19_source_aware", "test"),
            pred_map("e18_l1_intensity", "test"),
            test_texts,
        ),
        "e20_vs_e19_test": compare_predictions(
            pred_map("e20_rank_aug", "test"),
            pred_map("e19_source_aware", "test"),
            test_texts,
        ),
        "e21_vs_e19_validation": compare_predictions(
            pred_map("e21_dim_attention", "val"),
            pred_map("e19_source_aware", "val"),
            val_texts,
        ),
    }
    condition_runs = {
        "baseline": ["macbert_s1", "macbert_s2", "macbert_s42"],
        "N": ["aug_N_s1", "aug_N_s2", "aug_N_s42"],
        "A": ["aug_A_s1", "aug_A_s2", "aug_A_s42"],
        "C": ["aug_C_s1", "aug_C_s2", "aug_C_s42"],
        "E": ["aug_E_s1", "aug_E_s2", "aug_E_s42"],
        "F": ["aug_F_s1", "aug_F_s2", "aug_F_s42"],
        "F2": ["aug_F2_s1", "aug_F2_s2", "aug_F2_s42"],
    }
    result["augmentation_test_prediction_spread"] = {
        name: prediction_summary(mean_prediction(runs, "test"))
        for name, runs in condition_runs.items()
    }
    return result


def main() -> None:
    validation_surface, validation_lengths = official_surface(
        ROOT / "DSANIDF_ValidationSet.csv"
    )
    synthetic_surface = {}
    for name in ("N", "A", "C", "E", "F", "F2"):
        path = HERE / "data" / f"train_aug_{name}.csv"
        surface = corpus_surface(path)
        synth_lengths = [
            len(r["text"].replace(" ", "")) for r in read_rows(path)
        ]
        surface["length_vs_validation"] = {
            "wasserstein": float(
                wasserstein_distance(validation_lengths, synth_lengths)
            ),
            "ks": float(ks_2samp(validation_lengths, synth_lengths).statistic),
        }
        synthetic_surface[name] = surface
    report = {
        "graph": graph_diagnostics(),
        "lexicon": lexicon_diagnostics(),
        "validation_surface": validation_surface,
        "synthetic_surface": synthetic_surface,
        "predictions": prediction_diagnostics(),
    }
    path = OUT / "analysis" / "paper_evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(path)


if __name__ == "__main__":
    main()
