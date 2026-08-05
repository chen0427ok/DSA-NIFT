#!/usr/bin/env python3
"""Generate reproducible evidence, plots, tables, and paper-analysis docs.

Run from the project root:

    python baseline/analyze_paper_evidence.py

Official validation/test gold labels are not available locally.  Official metrics
are parsed from the repository's Markdown result tables; local prediction files
are used only for label-free distribution/correlation diagnostics.
"""
from __future__ import annotations

import csv
import json
import math
import pickle
import random
import re
import sys
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

import jieba

from affective_graph import (
    seed_coherence,
    seeds_by_expansion,
    seeds_for_target,
    seeds_random,
)
from augment_generate import TARGET_BINS
from lexicon import FEATURE_NAMES, LexiconFeaturizer


BASELINE = Path(__file__).resolve().parent
PROJECT = BASELINE.parent
OUT = BASELINE / "outputs"
ANALYSIS = OUT / "analysis"
PREDS = OUT / "preds"
DOCS = BASELINE / "docs"
PAPER = PROJECT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"


@dataclass
class MissingReport:
    entries: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        if message not in self.entries:
            self.entries.append(message)

    def require(self, path: Path, purpose: str) -> bool:
        if path.exists():
            return True
        self.add(f"MISSING: {path.relative_to(PROJECT)} — {purpose}")
        return False

    def write(self) -> None:
        text = "\n".join(self.entries) if self.entries else "No required files were missing."
        (ANALYSIS / "missing_files_report.txt").write_text(text + "\n", encoding="utf-8")


MISSING = MissingReport()


def ensure_dirs() -> None:
    for path in (ANALYSIS, FIGURES, TABLES):
        path.mkdir(parents=True, exist_ok=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name: str, rows: Iterable[dict] | pd.DataFrame) -> Path:
    path = ANALYSIS / name
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def clean_md(value: str) -> str:
    value = re.sub(r"\*+|`|⭐|🟢|🔴|←|→", "", value)
    value = re.sub(r"<[^>]+>", "", value)
    return value.strip()


def number(value: str) -> float:
    """Extract the first ordinary decimal number from a Markdown cell."""
    value = clean_md(value).replace(",", "")
    match = re.search(r"(?<![\w.])-?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else math.nan


def markdown_tables(path: Path) -> list[dict]:
    """Parse simple pipe tables and retain the nearest Markdown heading."""
    if not MISSING.require(path, "official-score/table source"):
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    heading = ""
    tables = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            heading = clean_md(line.lstrip("#").strip())
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?[\s:|-]+\|?$", lines[i + 1].strip()):
            raw = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw.append(lines[i].strip())
                i += 1
            cells = [[clean_md(x) for x in row.strip("|").split("|")] for row in raw]
            header = cells[0]
            body = cells[2:]
            rows = [dict(zip(header, r)) for r in body if len(r) == len(header)]
            tables.append({"heading": heading, "header": header, "rows": rows})
            continue
        i += 1
    return tables


def find_table(tables: list[dict], heading_contains: str, required_columns: Iterable[str] = ()) -> dict | None:
    for table in tables:
        if heading_contains in table["heading"] and all(c in table["header"] for c in required_columns):
            return table
    MISSING.add(f"TODO: could not parse Markdown table under heading containing '{heading_contains}'")
    return None


VAL_ID_MAP = {
    "1": "E1", "2": "E2", "3": "E3", "4": "E4", "5": "E5", "5b": "E5b",
    "10": "E10", "11b": "robertaL_s42", "12": "E12", "13": "E13",
    "18": "E18", "19": "E19", "20": "E20", "21": "E21b",
}


def parse_official_validation() -> dict[str, dict]:
    tables = markdown_tables(DOCS / "experiments.md")
    table = find_table(tables, "官方 validation 結果總表", ["#", "實驗"])
    out: dict[str, dict] = {}
    if not table:
        return out
    for row in table["rows"]:
        exp_id = clean_md(row.get("#", "")).lower()
        system = VAL_ID_MAP.get(exp_id)
        if not system:
            continue
        out[system] = {
            "system": system,
            "method": clean_md(row.get("實驗", "")),
            "V_MAE": number(row.get("V_MAE ↓", "")),
            "V_PCC": number(row.get("V_PCC ↑", "")),
            "A_MAE": number(row.get("A_MAE ↓", "")),
            "A_PCC": number(row.get("A_PCC ↑", "")),
        }
    return out


def parse_main_test_runs() -> dict[str, dict]:
    tables = markdown_tables(DOCS / "test_results.md")
    table = find_table(tables, "已取得的 test 分數", ["run", "架構"])
    out: dict[str, dict] = {}
    if not table:
        return out
    for row in table["rows"]:
        run = clean_md(row.get("run", ""))
        if run == "e19（predict.py 重跑）":
            continue
        out[run] = {
            "run": run,
            "method": clean_md(row.get("架構", "")),
            "V_MAE": number(row.get("V_MAE ↓", "")),
            "V_PCC": number(row.get("V_PCC ↑", "")),
            "A_MAE": number(row.get("A_MAE ↓", "")),
            "A_PCC": number(row.get("A_PCC ↑", "")),
        }
    nolex = find_table(tables, "no-L1 消融", ["seed", "V_MAE", "A_PCC"])
    if nolex:
        for row in nolex["rows"]:
            seed = clean_md(row.get("seed", ""))
            if seed not in {"s1", "s2", "s42"}:
                continue
            run = f"nolex_{seed}"
            out[run] = {
                "run": run,
                "method": "MacBERT without L1",
                "V_MAE": number(row.get("V_MAE", "")),
                "V_PCC": number(row.get("V_PCC", "")),
                "A_MAE": number(row.get("A_MAE", "")),
                "A_PCC": number(row.get("A_PCC", "")),
            }
    return out


def aggregate_scores(rows: list[dict], name: str, method: str) -> dict:
    result = {"system": name, "method": method, "n_seeds": len(rows)}
    for metric in ("V_MAE", "V_PCC", "A_MAE", "A_PCC"):
        vals = np.asarray([r[metric] for r in rows], dtype=float)
        result[metric] = float(vals.mean())
        result[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return result


def build_test_system_scores(test_runs: dict[str, dict]) -> dict[str, dict]:
    groups = {
        "E4": [f"macbert_s{s}" for s in (1, 2, 3, 4, 42)],
        "E13": [f"macbert_pseudo_s{s}" for s in (1, 2, 42)],
    }
    out: dict[str, dict] = {}
    for name, runs in groups.items():
        present = [test_runs[r] for r in runs if r in test_runs]
        if len(present) == len(runs):
            method = "MacBERT+L1 single-model seed mean" if name == "E4" else "teacher pseudo-label seed mean"
            out[name] = aggregate_scores(present, name, method)
        else:
            MISSING.add(f"MISSING official rows: {name} expected {runs}, found {[r for r in runs if r in test_runs]}")
    direct = {
        "E18": "e18_l1_intensity",
        "E19": "e19_source_aware",
        "E20": "e20_rank_aug",
        "roberta_s42": "roberta_s42",
        "robertaL_s42": "robertaL_s42",
    }
    for system, run in direct.items():
        if run in test_runs:
            out[system] = {"system": system, **test_runs[run], "n_seeds": 1}
        else:
            MISSING.add(f"MISSING official test score row: {run}")
    return out


def official_val_test(val: dict[str, dict], test: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for system in sorted(set(val) & set(test)):
        rows.append({
            "system": system,
            "method": val[system]["method"],
            **{f"val_{m}": val[system][m] for m in ("V_MAE", "V_PCC", "A_MAE", "A_PCC")},
            **{f"test_{m}": test[system][m] for m in ("V_MAE", "V_PCC", "A_MAE", "A_PCC")},
        })
    df = pd.DataFrame(rows)
    # Leaderboard-derived means contain binary artifacts such as
    # 0.3699999999999999 versus 0.37.  Preserve nominal ties at the precision
    # reported in the paper before computing ranks.
    ranked = df.copy()
    metric_columns = [f"{split}_{metric}" for split in ("val", "test")
                      for metric in ("V_MAE", "V_PCC", "A_MAE", "A_PCC")]
    ranked[metric_columns] = ranked[metric_columns].round(3)
    correlations = {}
    for metric in ("V_MAE", "V_PCC", "A_MAE", "A_PCC"):
        if len(df) >= 3:
            x = ranked[f"val_{metric}"].to_numpy()
            y = ranked[f"test_{metric}"].to_numpy()
            rho = float(spearmanr(x, y).statistic)
            null = np.asarray([spearmanr(x, perm).statistic for perm in permutations(y)])
            p_exact = float(np.mean(np.abs(null) >= abs(rho) - 1e-12))
            correlations[metric] = {
                "rho": rho, "p": p_exact, "n": len(df),
                "p_method": "two-sided exact permutation after 3-decimal tie handling",
            }
    write_csv("official_val_test_summary.csv", df)
    return df, correlations


def seed_and_l1_tables(test_runs: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    for group, runs in {
        "E4_L1": [f"macbert_s{s}" for s in (1, 2, 3, 4, 42)],
        "E13_pseudo": [f"macbert_pseudo_s{s}" for s in (1, 2, 42)],
        "no_L1": [f"nolex_s{s}" for s in (1, 2, 42)],
    }.items():
        present = [test_runs[r] for r in runs if r in test_runs]
        for row in present:
            seed_rows.append({"group": group, **row})
        if present:
            agg = aggregate_scores(present, group, "aggregate")
            seed_rows.append({"group": group, "run": "mean±std", **agg})
    seed_df = pd.DataFrame(seed_rows)
    write_csv("seed_variance.csv", seed_df)

    l1_rows = []
    for group in ("E4_L1", "no_L1"):
        rows = [r for r in seed_rows if r["group"] == group and r.get("run") != "mean±std"]
        if rows:
            agg = aggregate_scores(rows, group, "10-d L1" if group == "E4_L1" else "no lexicon")
            l1_rows.append(agg)
    if len(l1_rows) == 2:
        left, right = l1_rows
        delta = {"system": "L1_minus_noL1", "method": "difference of seed means", "n_seeds": "5 vs 3"}
        for metric in ("V_MAE", "V_PCC", "A_MAE", "A_PCC"):
            delta[metric] = left[metric] - right[metric]
            delta[f"{metric}_std"] = math.nan
        l1_rows.append(delta)
    l1_df = pd.DataFrame(l1_rows)
    write_csv("l1_ablation.csv", l1_df)
    return seed_df, l1_df


def parse_augmentation_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = markdown_tables(DOCS / "test_results.md")
    table = find_table(tables, "官方 test 分數（陸續回填）", ["條件", "seed"])
    detail = []
    if table:
        for row in table["rows"]:
            condition = clean_md(row.get("條件", ""))
            if condition not in {"N", "A", "C", "E", "F", "F2"}:
                continue
            detail.append({
                "condition": condition,
                "seed": clean_md(row.get("seed", "")),
                "V_MAE": number(row.get("V_MAE", "")),
                "V_PCC": number(row.get("V_PCC", "")),
                "A_MAE": number(row.get("A_MAE", "")),
                "A_PCC": number(row.get("A_PCC", "")),
            })
    detail_df = pd.DataFrame(detail)
    metadata = {
        "N": ("none", False, "fixed"), "A": ("random", False, "fixed"),
        "C": ("lookup", False, "fixed"), "E": ("graph", False, "fixed"),
        "F": ("graph", True, "fixed"), "F2": ("graph", True, "match_real"),
    }
    base_A_PCC, base_A_MAE = 0.372, 0.928
    rows = []
    for condition, (seed_mode, anchor, length) in metadata.items():
        sub = detail_df[detail_df["condition"] == condition] if not detail_df.empty else pd.DataFrame()
        if condition == "F" and len(sub) < 3:
            MISSING.add(f"TODO official scores: augmentation F has {len(sub)}/3 test seeds; F→F2 is not a controlled length-only estimate yet")
        data_file = BASELINE / "data" / f"train_aug_{condition}.csv"
        n_examples = len(read_rows(data_file)) if data_file.exists() else math.nan
        if not data_file.exists():
            MISSING.add(f"MISSING: {data_file.relative_to(PROJECT)} — augmentation corpus")
        row = {
            "condition": condition, "seed_mode": seed_mode, "style_anchor": anchor,
            "length_instruction": length, "num_examples": n_examples, "n_official_seeds": len(sub),
        }
        for metric in ("V_MAE", "V_PCC", "A_MAE", "A_PCC"):
            row[f"test_{metric}_mean"] = float(sub[metric].mean()) if len(sub) else math.nan
            row[f"test_{metric}_std"] = float(sub[metric].std(ddof=1)) if len(sub) > 1 else (0.0 if len(sub) else math.nan)
        row["delta_A_PCC_from_baseline"] = row["test_A_PCC_mean"] - base_A_PCC
        row["delta_A_MAE_from_baseline"] = row["test_A_MAE_mean"] - base_A_MAE
        rows.append(row)
    agg_df = pd.DataFrame(rows)
    write_csv("augmentation_ablation.csv", agg_df)
    return detail_df, agg_df


def qstats(values: Iterable[float]) -> dict[str, float]:
    a = np.asarray(list(values), dtype=float)
    return {
        "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
        "q05": float(np.quantile(a, .05)), "p25": float(np.quantile(a, .25)),
        "median": float(np.median(a)), "p75": float(np.quantile(a, .75)),
        "q95": float(np.quantile(a, .95)), "max": float(a.max()),
    }


def graph_diagnostics() -> dict:
    graph_path = OUT / "l3_graph.pkl"
    if not MISSING.require(graph_path, "graph diagnostics and seed coherence"):
        return {}
    try:
        with graph_path.open("rb") as f:
            graph = pickle.load(f)
    except Exception as exc:
        MISSING.add(f"ERROR loading {graph_path.relative_to(PROJECT)}: {exc}")
        return {}
    nodes = list(graph)
    edges = list(graph.edges(data=True))
    rng = random.Random(2026)
    random_pairs = [rng.sample(nodes, 2) for _ in range(len(edges))]

    def gap(pairs, attr):
        return [abs(float(graph.nodes[u][attr]) - float(graph.nodes[v][attr])) for u, v in pairs]

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
                    seeds = seeds_by_expansion(graph, v_range=vr, a_range=ar, n=20, rng=rng, n_anchors=4, hops=2)
                words = [w for w, _, _ in seeds]
                coherences.append(seed_coherence(graph, words)); sizes.append(len(words)); signatures.add(tuple(words))
                if trial == 0:
                    examples.append({"bin": bin_i + 1, "words": words})
        seed_modes[mode] = {
            "coherence": qstats(coherences), "seed_set_size": qstats(sizes),
            "distinct_sets_of_50": len(signatures), "first_trial_examples": examples,
        }
    comps = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    return {
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "density": nx.density(graph),
        "components": len(comps), "largest_component_ratio": comps[0] / graph.number_of_nodes(),
        "average_degree": 2 * graph.number_of_edges() / graph.number_of_nodes(),
        "degree": qstats(dict(graph.degree()).values()), "clustering": nx.average_clustering(graph),
        "edge_weight": qstats(d["weight"] for _, _, d in edges),
        "valence_assortativity": nx.numeric_assortativity_coefficient(graph, "valence"),
        "arousal_assortativity": nx.numeric_assortativity_coefficient(graph, "arousal"),
        "edge_valence_gap": qstats(gap([(u, v) for u, v, _ in edges], "valence")),
        "random_valence_gap": qstats(gap(random_pairs, "valence")),
        "edge_arousal_gap": qstats(gap([(u, v) for u, v, _ in edges], "arousal")),
        "random_arousal_gap": qstats(gap(random_pairs, "arousal")),
        "seed_modes": seed_modes,
    }


def text_surface(texts: list[str]) -> tuple[dict, list[int]]:
    lengths = [len(t.replace(" ", "")) for t in texts]
    tokens = [tok for text in texts for tok in jieba.cut(text) if tok.strip() and any(ch.isalnum() for ch in tok)]
    return {
        "documents": len(texts), "length": qstats(lengths),
        "under_50_ratio": float(np.mean(np.asarray(lengths) < 50)),
        "over_120_ratio": float(np.mean(np.asarray(lengths) > 120)),
        "type_token_ratio": len(set(tokens)) / len(tokens) if tokens else math.nan,
    }, lengths


def length_diagnostics() -> tuple[dict, pd.DataFrame]:
    val_path = PROJECT / "DSANIDF_ValidationSet.csv"
    if not MISSING.require(val_path, "official validation text length reference"):
        return {}, pd.DataFrame()
    val_rows = read_rows(val_path)
    val_surface, val_lengths = text_surface([r["Text"] for r in val_rows])
    result = {"official_validation": val_surface}
    csv_rows = [{"corpus": "official_validation", **flatten_surface(val_surface), "ks_vs_validation": 0.0, "wasserstein_vs_validation": 0.0}]
    for condition in ("N", "A", "C", "E", "F", "F2"):
        path = BASELINE / "data" / f"train_aug_{condition}.csv"
        if not MISSING.require(path, f"synthetic length distribution {condition}"):
            continue
        surface, lengths = text_surface([r["text"] for r in read_rows(path)])
        surface["length_vs_validation"] = {
            "ks": float(ks_2samp(val_lengths, lengths).statistic),
            "wasserstein": float(wasserstein_distance(val_lengths, lengths)),
        }
        result[condition] = surface
        csv_rows.append({
            "corpus": condition, **flatten_surface(surface),
            "ks_vs_validation": surface["length_vs_validation"]["ks"],
            "wasserstein_vs_validation": surface["length_vs_validation"]["wasserstein"],
        })
    df = pd.DataFrame(csv_rows)
    write_csv("length_distribution.csv", df)
    return result, df


def flatten_surface(surface: dict) -> dict:
    length = surface["length"]
    return {
        "documents": surface["documents"], "length_mean": length["mean"], "length_std": length["std"],
        "length_min": length["min"], "length_median": length["median"], "length_max": length["max"],
        "short_lt50_ratio": surface["under_50_ratio"], "long_gt120_ratio": surface["over_120_ratio"],
        "type_token_ratio": surface["type_token_ratio"],
    }


def lexicon_diagnostics() -> tuple[dict, pd.DataFrame]:
    try:
        fz = LexiconFeaturizer()
    except Exception as exc:
        MISSING.add(f"ERROR loading L1 lexicon: {exc}")
        return {}, pd.DataFrame()
    datasets = {
        "train": (BASELINE / "data" / "train.csv", "text"),
        "official_validation": (PROJECT / "DSANIDF_ValidationSet.csv", "Text"),
        "official_test": (PROJECT / "DSANIDF_TestSet.csv", "Text"),
    }
    result = {"lexicon_entries": len(fz), "feature_names": FEATURE_NAMES}
    rows = []
    for name, (path, key) in datasets.items():
        if not MISSING.require(path, f"L1 coverage for {name}"):
            continue
        records = read_rows(path)
        feats = np.asarray([fz.featurize(r[key]) for r in records])
        stats = {
            "documents": len(records), "document_hit_rate": float(np.mean(feats[:, 1] > 0)),
            "coverage": qstats(feats[:, 0]), "hit_count_capped_normalized": qstats(feats[:, 1]),
            "arousal_mean_feature": qstats(feats[:, 6]), "arousal_std_feature": qstats(feats[:, 9]),
        }
        result[name] = stats
        rows.append({
            "split": name, "documents": len(records), "document_hit_rate": stats["document_hit_rate"],
            "mean_token_coverage": stats["coverage"]["mean"], "median_token_coverage": stats["coverage"]["median"],
            "mean_normalized_hit_count": stats["hit_count_capped_normalized"]["mean"],
        })
    df = pd.DataFrame(rows)
    write_csv("l1_coverage.csv", df)
    return result, df


def normalize_prediction_file(path: Path) -> dict[str, np.ndarray]:
    rows = read_rows(path)
    out = {}
    for row in rows:
        low = {str(k).lower(): v for k, v in row.items()}
        ID = row.get("ID") or row.get("id") or low.get("id")
        v = low.get("valence_pred", low.get("valence"))
        a = low.get("arousal_pred", low.get("arousal"))
        if ID is not None and v not in (None, "") and a not in (None, ""):
            out[str(ID)] = np.asarray([float(v), float(a)])
    return out


def pred_map(run: str, split: str) -> dict[str, np.ndarray] | None:
    path = PREDS / f"{run}_{split}.csv"
    if not path.exists():
        MISSING.add(f"MISSING: {path.relative_to(PROJECT)} — prediction diagnostics")
        return None
    try:
        result = normalize_prediction_file(path)
        if not result:
            MISSING.add(f"UNPARSEABLE: {path.relative_to(PROJECT)} — no prediction columns found")
            return None
        return result
    except Exception as exc:
        MISSING.add(f"ERROR reading {path.relative_to(PROJECT)}: {exc}")
        return None


def mean_prediction(runs: list[str], split: str) -> dict[str, np.ndarray] | None:
    maps = [pred_map(run, split) for run in runs]
    if any(m is None for m in maps):
        return None
    valid = [m for m in maps if m is not None]
    ids = set(valid[0])
    for m in valid[1:]:
        ids &= set(m)
    return {i: np.mean([m[i] for m in valid], axis=0) for i in sorted(ids)}


def prediction_summary(name: str, split: str, preds: dict[str, np.ndarray]) -> dict:
    arr = np.vstack(list(preds.values()))
    row = {"system": name, "split": split, "n": len(arr)}
    for j, prefix in enumerate(("V", "A")):
        q = qstats(arr[:, j])
        row.update({
            f"{prefix}_mean": q["mean"], f"{prefix}_std": q["std"], f"{prefix}_min": q["min"],
            f"{prefix}_max": q["max"], f"{prefix}_q05": q["q05"], f"{prefix}_q25": q["p25"],
            f"{prefix}_q50": q["median"], f"{prefix}_q75": q["p75"], f"{prefix}_q95": q["q95"],
        })
    return row


def core_prediction_maps() -> dict[str, dict[str, np.ndarray]]:
    specs = {
        "E4_s42": ("single", ["macbert_s42"]), "E18": ("single", ["e18_l1_intensity"]),
        "E19": ("single", ["e19_source_aware"]), "E20": ("single", ["e20_rank_aug"]),
        "E13_3seed_mean": ("mean", ["macbert_pseudo_s1", "macbert_pseudo_s2", "macbert_pseudo_s42"]),
        "RoBERTa_base": ("single", ["roberta_s42"]), "RoBERTa_large": ("single", ["robertaL_s42"]),
        "L1_3seed_mean": ("mean", ["macbert_s1", "macbert_s2", "macbert_s42"]),
        "noL1_3seed_mean": ("mean", ["nolex_s1", "nolex_s2", "nolex_s42"]),
    }
    for cond in ("N", "A", "C", "E", "F", "F2"):
        specs[f"aug_{cond}_3seed_mean"] = ("mean", [f"aug_{cond}_s1", f"aug_{cond}_s2", f"aug_{cond}_s42"])
    result = {}
    for name, (mode, runs) in specs.items():
        preds = pred_map(runs[0], "test") if mode == "single" else mean_prediction(runs, "test")
        if preds:
            result[name] = preds
    return result


def prediction_diagnostics() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maps = core_prediction_maps()
    dist_df = pd.DataFrame([prediction_summary(name, "test", preds) for name, preds in maps.items()])
    write_csv("prediction_distribution.csv", dist_df)
    corr_rows = []
    names = sorted(maps)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            ids = sorted(set(maps[left_name]) & set(maps[right_name]))
            if not ids:
                continue
            left = np.vstack([maps[left_name][x] for x in ids]); right = np.vstack([maps[right_name][x] for x in ids])
            corr_rows.append({
                "system_i": left_name, "system_j": right_name, "n": len(ids),
                "V_corr": float(np.corrcoef(left[:, 0], right[:, 0])[0, 1]),
                "A_corr": float(np.corrcoef(left[:, 1], right[:, 1])[0, 1]),
                "V_mean_abs_diff": float(np.abs(left[:, 0] - right[:, 0]).mean()),
                "A_mean_abs_diff": float(np.abs(left[:, 1] - right[:, 1]).mean()),
            })
    corr_df = pd.DataFrame(corr_rows)
    write_csv("prediction_correlation.csv", corr_df)

    shift_rows = []
    if "L1_3seed_mean" in maps and "noL1_3seed_mean" in maps:
        l1, no = maps["L1_3seed_mean"], maps["noL1_3seed_mean"]
        for ID in sorted(set(l1) & set(no)):
            shift_rows.append({
                "ID": ID, "L1_V": l1[ID][0], "noL1_V": no[ID][0], "delta_V": l1[ID][0] - no[ID][0],
                "L1_A": l1[ID][1], "noL1_A": no[ID][1], "delta_A": l1[ID][1] - no[ID][1],
            })
    shift_df = pd.DataFrame(shift_rows)
    write_csv("l1_prediction_shift.csv", shift_df)

    example_rows = []
    test_text_path = PROJECT / "DSANIDF_TestSet.csv"
    if not shift_df.empty and test_text_path.exists():
        texts = {r["ID"]: r["Text"] for r in read_rows(test_text_path)}
        try:
            fz = LexiconFeaturizer()
            top = shift_df.assign(abs_delta_A=shift_df["delta_A"].abs()).nlargest(10, "abs_delta_A")
            for _, row in top.iterrows():
                text = texts.get(row.ID, "")
                hits = []
                for token in jieba.cut(text):
                    if token in fz.lex:
                        v, a = fz.lex[token]
                        hits.append(f"{token}(V={1+8*v:.2f},A={1+8*a:.2f})")
                example_rows.append({
                    "ID": row.ID, "text": text, "L1_A": row.L1_A, "noL1_A": row.noL1_A,
                    "delta_A": row.delta_A, "lexicon_hits": "；".join(hits),
                    "caution": "Official gold is unavailable; semantic plausibility is not correctness.",
                })
        except Exception as exc:
            MISSING.add(f"ERROR generating L1 qualitative examples: {exc}")
    write_csv("l1_examples.csv", example_rows)

    comparisons = {}
    for left, right in (("L1_3seed_mean", "noL1_3seed_mean"), ("E19", "E18"), ("E20", "E19"), ("RoBERTa_large", "E4_s42")):
        row = corr_df[((corr_df.system_i == min(left, right)) & (corr_df.system_j == max(left, right)))] if not corr_df.empty else pd.DataFrame()
        if len(row):
            comparisons[f"{left}_vs_{right}"] = row.iloc[0].to_dict()
    return {"comparisons": comparisons, "l1_examples": example_rows}, dist_df, corr_df, shift_df


def experiment_taxonomy(val: dict[str, dict], test: dict[str, dict]) -> pd.DataFrame:
    known = {
        1: ("Baseline", "baseline/data-source", "EmoBank only", "How far does adjacent general affect data transfer?"),
        2: ("DAPT", "baseline/data-source", "MLM on 200 target-domain texts", "Does small target-domain MLM adaptation help?"),
        3: ("Reflective corpora", "baseline/data-source", "Add DSA-MST and education reflection", "Does style-adjacent labelled data help?"),
        4: ("L1 lexicon fusion", "lexicon fusion", "Add 10 CVAW/CVAP aggregates", "Does explicit VA knowledge complement MacBERT?"),
        5: ("Raw synthetic augmentation", "synthetic augmentation", "400 bin-centred LLM texts", "Can controlled generation expand arousal ranking?"),
        10: ("5-seed ensemble", "ensemble", "Mean of five MacBERT seeds", "Does ensembling reduce prediction variance?"),
        11: ("Encoder variants", "encoder variant", "RoBERTa base/large", "Does encoder capacity change transfer?"),
        12: ("Multi-encoder ensemble", "ensemble", "Dimension-wise model fusion", "Does heterogeneous ensembling help?"),
        13: ("Teacher pseudo-label", "teacher pseudo-label", "Teacher relabel 318 synthetic texts", "Can teachers repair synthetic calibration?"),
        14: ("Frozen embedding + SVR", "encoder variant", "Frozen embedding regression", "Does a non-finetuned regressor complement the ensemble?"),
        15: ("Arousal/PCC loss weighting", "ranking loss", "Arousal and batch-PCC weights", "Does direct arousal emphasis help?"),
        16: ("L1+L2", "lexicon fusion", "OOV VA prediction features", "Does lexicon expansion overcome OOV?"),
        17: ("Arousal calibration", "ablation", "Post-hoc calibration", "Can MAE improve without changing ranking?"),
        18: ("L1++ intensity", "lexicon fusion", "31 handcrafted features", "Do arousal intensity cues add information?"),
        19: ("Source-aware loss", "source-aware training", "Down-weight general-corpus arousal", "Does source reliability weighting transfer?"),
        20: ("Ranking-only augmentation", "ranking loss", "Synthetic pairwise hinge", "Can ordinal signal avoid synthetic-label calibration?"),
        21: ("Dimension attention", "attention/pooling", "V/A-specific attention and multi-pooling", "Does mean pooling dilute arousal tokens?"),
        22: ("L1 + source-aware", "ablation", "Missing 2x2 cell; later planning reused ID", "Can source weighting work without intensity features?"),
        23: ("Target-style RAG", "synthetic augmentation", "Unlabelled validation style anchors", "Does target style reduce generation shift?"),
        24: ("LLM enrichment features", "lexicon fusion", "Structured arousal indicators", "Can LLM cues enrich arousal evidence?"),
        25: ("Frozen embedding stacking", "encoder variant", "Frozen embedding Ridge/SVR", "Do stronger frozen representations help?"),
        26: ("Arousal calibration", "ablation", "Post-processing calibration", "Can calibration reduce MAE safely?"),
    }
    rows = []
    for i in range(1, 27):
        name, category, change, question = known.get(i, (f"E{i} undocumented", "undocumented", "TODO", "TODO"))
        key = f"E{i}"
        v = val.get(key)
        t = test.get(key)
        if i == 11:
            v, t = val.get("robertaL_s42"), test.get("robertaL_s42")
        elif i == 21:
            v, t = val.get("E21b"), test.get("E21b")
        supported = "not run / no official evidence"
        if v and t:
            supported = "official validation and test available; compare transfer rather than val-only gain"
        elif v:
            supported = "official validation only"
        rows.append({
            "experiment_id": key, "system_name": name, "category": category, "method_change": change,
            "research_question": question, "uses_lexicon": i in {4, 16, 18, 19, 20, 21, 22, 24},
            "uses_intensity_features": i in {18, 19, 20, 21}, "uses_source_aware_loss": i in {19, 20, 21, 22},
            "uses_augmentation": i in {5, 13, 20, 23}, "uses_pseudo_label": i == 13,
            "uses_ranking_loss": i in {15, 20, 21}, "uses_ensemble": i in {10, 12},
            "encoder": "RoBERTa variants" if i == 11 else "MacBERT (default)",
            "official_val_available": bool(v), "official_test_available": bool(t),
            "main_supported_finding": supported,
            "caution": "Do not infer test utility from validation-only evidence." if not t else "Differences may remain below leaderboard resolution/uncertainty.",
        })
    df = pd.DataFrame(rows)
    write_csv("experiment_taxonomy.csv", df)
    return df


def savefig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def make_plots(paired: pd.DataFrame, correlations: dict, aug: pd.DataFrame, lengths: pd.DataFrame,
               dist: pd.DataFrame, corr: pd.DataFrame, shift: pd.DataFrame,
               prediction_maps: dict[str, dict[str, np.ndarray]]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .22, "figure.dpi": 140})
    if not paired.empty:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
        label_offsets = {
            "V_PCC": {
                "E13": (-22, 6), "E18": (-24, -12), "E19": (4, -12),
                "E20": (5, 6), "E4": (5, -12), "robertaL_s42": (4, -12),
            },
            "A_PCC": {
                "E13": (-29, 10), "E18": (-25, -13), "E19": (5, 4),
                "E20": (-27, 7), "E4": (7, -11), "robertaL_s42": (5, 4),
            },
        }
        for ax, dim, color in zip(axes, ("V_PCC", "A_PCC"), ("#2c7fb8", "#d95f0e")):
            x, y = paired[f"val_{dim}"], paired[f"test_{dim}"]
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max()); pad = max((hi - lo) * .14, .008)
            ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", color="0.55", lw=1)
            ax.scatter(x, y, s=48, color=color, edgecolor="white", zorder=3)
            for _, r in paired.iterrows():
                label = r.system.replace("robertaL_s42", "RoBERTa-L")
                offset = label_offsets[dim].get(r.system, (4, 4))
                ax.annotate(label, (r[f"val_{dim}"], r[f"test_{dim}"]), xytext=offset,
                            textcoords="offset points", fontsize=8)
            rho = correlations[dim]["rho"]
            ax.set_title(f"{'Valence' if dim[0]=='V' else 'Arousal'} (Spearman $\\rho$={rho:.2f}, n={len(paired)})")
            ax.set_xlabel("Official validation PCC (n=200)"); ax.set_ylabel("Official test PCC (n=1,100)")
        savefig(fig, "val_vs_test_pcc.pdf")

    score_rows = []
    if not paired.empty:
        score_rows += [{"system": r.system, "A_MAE": r.test_A_MAE, "A_PCC": r.test_A_PCC, "kind": "model"} for _, r in paired.iterrows()]
    if not aug.empty:
        score_rows += [{"system": f"Aug-{r.condition}", "A_MAE": r.test_A_MAE_mean, "A_PCC": r.test_A_PCC_mean, "kind": "augmentation"} for _, r in aug.iterrows() if not math.isnan(r.test_A_PCC_mean)]
    if score_rows:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for kind, color, marker in (("model", "#2c7fb8", "o"), ("augmentation", "#d95f0e", "s")):
            sub = [r for r in score_rows if r["kind"] == kind]
            ax.scatter([r["A_MAE"] for r in sub], [r["A_PCC"] for r in sub], color=color, marker=marker, label=kind, s=44)
            for r in sub: ax.annotate(r["system"], (r["A_MAE"], r["A_PCC"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Official test Arousal MAE (lower is better)"); ax.set_ylabel("Official test Arousal PCC (higher is better)"); ax.legend()
        savefig(fig, "arousal_mae_vs_pcc.pdf")

    if not aug.empty:
        ordered = aug.set_index("condition").reindex(["N", "A", "C", "E", "F", "F2"]).reset_index()
        fig, ax = plt.subplots(figsize=(6.2, 3.5))
        ax.errorbar(ordered.condition, ordered.test_A_PCC_mean, yerr=ordered.test_A_PCC_std, marker="o", capsize=3, color="#d95f0e")
        ax.axhline(.372, ls="--", color="#2c7fb8", label="No augmentation (0.372)")
        ax.set_ylabel("Official test Arousal PCC"); ax.set_xlabel("Component ladder"); ax.legend(fontsize=8)
        savefig(fig, "augmentation_component_decomposition.pdf")
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.scatter(ordered.test_A_MAE_mean, ordered.test_A_PCC_mean, color="#d95f0e", s=48)
        for _, r in ordered.iterrows(): ax.annotate(r.condition, (r.test_A_MAE_mean, r.test_A_PCC_mean), xytext=(4, 3), textcoords="offset points")
        ax.scatter([.928], [.372], color="#2c7fb8", marker="*", s=100, label="No augmentation")
        ax.set_xlabel("Test Arousal MAE"); ax.set_ylabel("Test Arousal PCC"); ax.legend(fontsize=8)
        savefig(fig, "augmentation_mae_pcc_tradeoff.pdf")

    if not lengths.empty:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        for corpus in ("official_validation", "C", "E", "F", "F2"):
            path = PROJECT / "DSANIDF_ValidationSet.csv" if corpus == "official_validation" else BASELINE / "data" / f"train_aug_{corpus}.csv"
            if not path.exists(): continue
            rows = read_rows(path); key = "Text" if corpus == "official_validation" else "text"
            vals = np.sort([len(r[key].replace(" ", "")) for r in rows]); ax.step(vals, np.arange(1, len(vals)+1)/len(vals), where="post", label=corpus)
        ax.set_xlabel("Text length (characters)"); ax.set_ylabel("Empirical CDF"); ax.set_xlim(left=0); ax.legend(fontsize=8)
        savefig(fig, "augmentation_length_distribution.pdf")

    selected = [x for x in ("E4_s42", "E19", "E13_3seed_mean", "RoBERTa_large", "aug_F2_3seed_mean") if x in prediction_maps]
    if selected:
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        ax.boxplot([np.vstack(list(prediction_maps[x].values()))[:, 1] for x in selected], labels=selected, showfliers=False)
        ax.set_ylabel("Predicted Arousal (official test, no gold)"); ax.tick_params(axis="x", rotation=18)
        savefig(fig, "prediction_distribution_arousal.pdf")

    if not corr.empty and selected:
        matrix = np.eye(len(selected))
        lookup = {(r.system_i, r.system_j): r.A_corr for _, r in corr.iterrows()}
        for i, a in enumerate(selected):
            for j, b in enumerate(selected):
                if i != j: matrix[i, j] = lookup.get(tuple(sorted((a, b))), math.nan)
        fig, ax = plt.subplots(figsize=(5.6, 4.6)); im = ax.imshow(matrix, vmin=.5, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(selected)), selected, rotation=35, ha="right"); ax.set_yticks(range(len(selected)), selected)
        for i in range(len(selected)):
            for j in range(len(selected)): ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="Arousal prediction correlation")
        savefig(fig, "prediction_correlation_heatmap.pdf")

    if not shift.empty:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
        for ax, dim in zip(axes, ("V", "A")):
            ax.scatter(shift[f"noL1_{dim}"], shift[f"L1_{dim}"], s=8, alpha=.35)
            lo = min(shift[f"noL1_{dim}"].min(), shift[f"L1_{dim}"].min()); hi = max(shift[f"noL1_{dim}"].max(), shift[f"L1_{dim}"].max())
            ax.plot([lo, hi], [lo, hi], "--", color="0.5"); ax.set_xlabel(f"no-L1 {dim}"); ax.set_ylabel(f"L1 {dim}"); ax.set_title("Valence" if dim == "V" else "Arousal")
        savefig(fig, "l1_vs_nol1_prediction_scatter.pdf")

    if "E4_s42" in prediction_maps and "E19" in prediction_maps:
        ids = sorted(set(prediction_maps["E4_s42"]) & set(prediction_maps["E19"]))
        e4 = np.vstack([prediction_maps["E4_s42"][x] for x in ids]); e19 = np.vstack([prediction_maps["E19"][x] for x in ids])
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
        for ax, j, title in zip(axes, (0, 1), ("Valence", "Arousal")):
            ax.scatter(e4[:, j], e19[:, j], s=8, alpha=.35); lo=min(e4[:,j].min(),e19[:,j].min()); hi=max(e4[:,j].max(),e19[:,j].max()); ax.plot([lo,hi],[lo,hi],"--",color="0.5"); ax.set_xlabel("E4"); ax.set_ylabel("E19"); ax.set_title(title)
        savefig(fig, "system_pairwise_arousal_scatter.pdf")


def latex_escape(s: str) -> str:
    return str(s).replace("_", "\\_").replace("±", "$\\pm$")


def write_latex_tables(paired: pd.DataFrame, seed: pd.DataFrame, l1: pd.DataFrame,
                       aug: pd.DataFrame, dist: pd.DataFrame) -> None:
    br = r"\\"
    lines = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"System & \multicolumn{4}{c}{Official validation} & \multicolumn{4}{c}{Official test} " + br,
             r" & V-MAE$\downarrow$ & V-PCC$\uparrow$ & A-MAE$\downarrow$ & A-PCC$\uparrow$ & V-MAE$\downarrow$ & V-PCC$\uparrow$ & A-MAE$\downarrow$ & A-PCC$\uparrow$ " + br,
             r"\midrule"]
    for _, r in paired.iterrows():
        lines.append(f"{latex_escape(r.system)} & {r.val_V_MAE:.3f} & {r.val_V_PCC:.3f} & {r.val_A_MAE:.3f} & {r.val_A_PCC:.3f} & {r.test_V_MAE:.3f} & {r.test_V_PCC:.3f} & {r.test_A_MAE:.3f} & {r.test_A_PCC:.3f} " + br)
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Official validation and test results for systems available on both splits.}", r"\label{tab:main-val-test}", r"\end{table*}"]
    (TABLES / "main_val_test_results.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")

    seed_summary = seed[seed["run"] == "mean±std"] if not seed.empty and "run" in seed.columns else seed
    lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lrrrr}", r"\toprule", r"System & V-MAE$\downarrow$ & V-PCC$\uparrow$ & A-MAE$\downarrow$ & A-PCC$\uparrow$ " + br, r"\midrule"]
    for _, r in seed_summary.iterrows(): lines.append(f"{latex_escape(r.group)} & {r.V_MAE:.3f}$\\pm${r.V_MAE_std:.3f} & {r.V_PCC:.3f}$\\pm${r.V_PCC_std:.3f} & {r.A_MAE:.3f}$\\pm${r.A_MAE_std:.3f} & {r.A_PCC:.3f}$\\pm${r.A_PCC_std:.3f} " + br)
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Seed-level variation on the official test leaderboard.}", r"\label{tab:seed-variance}", r"\end{table}"]
    (TABLES / "seed_variance.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")

    real_l1 = l1[l1["system"] != "L1_minus_noL1"] if not l1.empty and "system" in l1.columns else l1
    lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lrrrr}", r"\toprule", r"System & V-MAE$\downarrow$ & V-PCC$\uparrow$ & A-MAE$\downarrow$ & A-PCC$\uparrow$ " + br, r"\midrule"]
    for _, r in real_l1.iterrows(): lines.append(f"{latex_escape(r.system)} & {r.V_MAE:.3f} & {r.V_PCC:.3f} & {r.A_MAE:.3f} & {r.A_PCC:.3f} " + br)
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{L1 lexicon ablation on official test scores.}", r"\label{tab:l1-ablation}", r"\end{table}"]
    (TABLES / "l1_ablation.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")

    lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lrrr}", r"\toprule", r"Condition & Seeds & A-MAE$\downarrow$ & A-PCC$\uparrow$ " + br, r"\midrule"]
    for _, r in aug.iterrows(): lines.append(f"{r.condition} & {int(r.n_official_seeds)} & {r.test_A_MAE_mean:.3f} & {r.test_A_PCC_mean:.3f}$\\pm${r.test_A_PCC_std:.3f} " + br)
    lines += [r"No augmentation & 5 & 0.928 & 0.372$\pm$0.004 " + br, r"\bottomrule", r"\end{tabular}", r"\caption{Component ablation of synthetic augmentation on official test scores.}", r"\label{tab:augmentation-ablation}", r"\end{table}"]
    (TABLES / "augmentation_ablation.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")

    chosen = dist[dist["system"].isin(["E4_s42", "E19", "E13_3seed_mean", "RoBERTa_large", "aug_F2_3seed_mean"])] if not dist.empty and "system" in dist.columns else dist
    lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lrrrr}", r"\toprule", "System & V mean & V std & A mean & A std " + br, r"\midrule"]
    for _, r in chosen.iterrows(): lines.append(f"{latex_escape(r.system)} & {r.V_mean:.3f} & {r.V_std:.3f} & {r.A_mean:.3f} & {r.A_std:.3f} " + br)
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Label-free prediction distributions on the official test texts.}", r"\label{tab:prediction-distribution}", r"\end{table}"]
    (TABLES / "prediction_distribution.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


def fmt(x: float, digits: int = 3) -> str:
    return "TODO" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def write_docs(paired: pd.DataFrame, correlations: dict, l1_cov: pd.DataFrame, l1: pd.DataFrame,
               aug: pd.DataFrame, lengths: pd.DataFrame, dist: pd.DataFrame, corr: pd.DataFrame,
               graph: dict, taxonomy: pd.DataFrame) -> None:
    rho_v = correlations.get("V_PCC", {}).get("rho", math.nan); rho_a = correlations.get("A_PCC", {}).get("rho", math.nan)
    p_a = correlations.get("A_PCC", {}).get("p", math.nan); n = correlations.get("A_PCC", {}).get("n", 0)
    paired_systems = set(paired["system"]) if not paired.empty and "system" in paired.columns else set()
    coverage_splits = set(l1_cov["split"]) if not l1_cov.empty and "split" in l1_cov.columns else set()
    l1_systems = set(l1["system"]) if not l1.empty and "system" in l1.columns else set()
    aug_conditions = set(aug["condition"]) if not aug.empty and "condition" in aug.columns else set()
    e19 = paired[paired["system"] == "E19"].iloc[0] if "E19" in paired_systems else None
    rl = paired[paired["system"] == "robertaL_s42"].iloc[0] if "robertaL_s42" in paired_systems else None
    cov_test = l1_cov[l1_cov["split"] == "official_test"].iloc[0] if "official_test" in coverage_splits else None
    l1row = l1[l1["system"] == "E4_L1"].iloc[0] if "E4_L1" in l1_systems else None
    norow = l1[l1["system"] == "no_L1"].iloc[0] if "no_L1" in l1_systems else None
    graph_ratio = math.nan
    if graph:
        graph_ratio = graph["seed_modes"]["graph"]["coherence"]["mean"] / graph["seed_modes"]["lookup"]["coherence"]["mean"]
    f2 = aug[aug["condition"] == "F2"].iloc[0] if "F2" in aug_conditions else None
    def pair_metric(a: str, b: str, metric: str) -> float:
        if corr.empty or not {"system_i", "system_j", metric}.issubset(corr.columns):
            return math.nan
        lo, hi = sorted((a, b))
        row = corr[(corr["system_i"] == lo) & (corr["system_j"] == hi)]
        return float(row.iloc[0][metric]) if len(row) else math.nan

    l1_a_corr = pair_metric("L1_3seed_mean", "noL1_3seed_mean", "A_corr")
    e18_e19_corr = pair_metric("E18", "E19", "A_corr")
    e4_rl_corr = pair_metric("E4_s42", "RoBERTa_large", "A_corr")
    dist_systems = set(dist["system"]) if not dist.empty and "system" in dist.columns else set()
    e4_dist = dist[dist["system"] == "E4_s42"].iloc[0] if "E4_s42" in dist_systems else None
    f2_dist = dist[dist["system"] == "aug_F2_3seed_mean"].iloc[0] if "aug_F2_3seed_mean" in dist_systems else None
    codegen = f"""# Reproducible Paper Findings (Code-Generated)

Generated by `python baseline/analyze_paper_evidence.py`. Official validation/test scores are parsed from `baseline/docs/experiments.md` and `baseline/docs/test_results.md`; no official gold labels are assumed.

## Official validation is not a reliable selector for Arousal

Across the {n} primary systems in our main comparison with both official validation and test results, Spearman rank correlation is **{fmt(rho_v)}** for V-PCC and **{fmt(rho_a)}** for A-PCC (A-PCC tie-aware exact permutation p={fmt(p_a)}). Controlled no-L1 and augmentation ladders are analyzed separately because they use grouped multi-seed conditions rather than the primary-system comparison. The p-value does not establish statistical significance; the defensible observation is that validation Arousal ordering did not transfer reliably among our primary submissions.

- E19: validation A-PCC {fmt(e19.val_A_PCC if e19 is not None else math.nan)} → test {fmt(e19.test_A_PCC if e19 is not None else math.nan)}.
- RoBERTa-large: validation A-PCC {fmt(rl.val_A_PCC if rl is not None else math.nan)} → test {fmt(rl.test_A_PCC if rl is not None else math.nan)}.
- Safe wording: *Official validation was not a reliable Arousal model selector for our submitted systems.*
- Avoid: *Validation is useless* or *the ranking reversal is statistically significant*.

## Seed variance

Five identically configured E4-family single models have official test A-PCC 0.372±0.004. Leaderboard rounding means this is a resolution-limited estimate, but it is much smaller than the observed validation-to-test ordering changes.

## L1 lexicon fusion: high coverage but limited measurable test utility

Official-test document hit rate is **{fmt(100*(cov_test.document_hit_rate if cov_test is not None else math.nan), 2)}%** and mean token coverage is **{fmt(100*(cov_test.mean_token_coverage if cov_test is not None else math.nan), 2)}%**. L1 A-PCC is {fmt(l1row.A_PCC if l1row is not None else math.nan)} versus {fmt(norow.A_PCC if norow is not None else math.nan)} without L1. The evidence supports high coverage, but the test contribution is not clearly distinguishable from seed-level variation.

## Source-aware, intensity, ranking, encoder, and pseudo-label experiments

E19 was selected by public validation but did not transfer on A-PCC. E18 and E20 both return about 0.37 A-PCC on test. E13 teacher pseudo-labels average 0.370 A-PCC and 0.927 A-MAE, approximately matching the no-augmentation baseline. RoBERTa-large changes the observed ordering but does not justify a superiority claim without uncertainty-aware testing.

## Synthetic augmentation: intrinsic control vs downstream utility

Graph expansion improves seed coherence by **{fmt(graph_ratio, 1)}×** over lookup when the graph artifact is available. Official component means are generated in `outputs/analysis/augmentation_ablation.csv`. N→A captures arbitrary seed words; A→C VA filtering; C→E graph topology; E→F/F2 style and length alignment. F2 reaches A-PCC {fmt(f2.test_A_PCC_mean if f2 is not None else math.nan)} but A-MAE {fmt(f2.test_A_MAE_mean if f2 is not None else math.nan)}, compared with no-augmentation 0.372/0.928. Intrinsic control improved, but it did not clearly transfer to a better calibration–ranking combination.

## Prediction distribution analysis

`prediction_distribution.csv` and `prediction_correlation.csv` are label-free. L1/no-L1 Arousal predictions correlate at {fmt(l1_a_corr)}, E18/E19 at {fmt(e18_e19_corr)}, and E4/RoBERTa-large at {fmt(e4_rl_corr)}. F2 raises Arousal prediction std from {fmt(e4_dist.A_std if e4_dist is not None else math.nan)} (E4) to {fmt(f2_dist.A_std if f2_dist is not None else math.nan)}. High correlation indicates similar ordering, while mean/std changes describe calibration/spread; neither establishes correctness without gold labels.

## Length/style alignment

F2 has the smallest generated-vs-validation length KS/Wasserstein distances among the analyzed synthetic corpora, but it overproduces short examples. Exact values are in `length_distribution.csv`.

## Recommended paper artifacts

- `paper/figures/val_vs_test_pcc.pdf`
- `paper/figures/arousal_mae_vs_pcc.pdf`
- `paper/figures/augmentation_component_decomposition.pdf`
- `paper/figures/prediction_distribution_arousal.pdf`
- `paper/tables/main_val_test_results.tex`
- `paper/tables/augmentation_ablation.tex`

## Reproduction

```bash
python baseline/analyze_paper_evidence.py
```

Any unavailable artifact is recorded in `baseline/outputs/analysis/missing_files_report.txt`.
"""
    (DOCS / "PAPER_FINDINGS_ANALYSIS_CODEGEN.md").write_text(codegen, encoding="utf-8")

    exp_lines = ["# Experiment Map (Code-Generated)", "", "Generated from repository metadata and documented plans. `official_test_available=false` means no test claim is supported.", "", "| ID | System | Category | Change | Val | Test | Supported finding |", "|---|---|---|---|---:|---:|---|"]
    for _, r in taxonomy.iterrows(): exp_lines.append(f"| {r.experiment_id} | {r.system_name} | {r.category} | {r.method_change} | {'✓' if r.official_val_available else '—'} | {'✓' if r.official_test_available else '—'} | {r.main_supported_finding} |")
    exp_lines += ["", "## Interpretation discipline", "", "Validation-only experiments answer what looked promising on the public validation set; they do not establish official-test utility. Planned or failed runs remain TODO rather than negative evidence."]
    (DOCS / "EXPERIMENT_MAP_CODEGEN.md").write_text("\n".join(exp_lines)+"\n", encoding="utf-8")

    final = f"""# Final Evidence-Based Paper Analysis

## 1. Executive summary

The strongest contribution is diagnostic: Valence is comparatively stable, while public-validation Arousal ordering did not transfer reliably to the larger official test set. Graph, ranking, and length controls changed their intended intermediate quantities, but these changes did not yield a clearly better Arousal calibration–ranking combination.

## 2. Task setting and evaluation caveat

Official validation contains 200 texts and official test 1,100 texts. Their gold labels are unavailable locally; all official scores are leaderboard returns. Internal DSA-MST development scores are not official validation scores.

## 3. Dimension-dependent reliability

For {n} paired primary systems, V-PCC validation/test Spearman rho is {fmt(rho_v)}, while A-PCC rho is {fmt(rho_a)}. Controlled ablation groups are reported in separate tables. With only {n} systems and p={fmt(p_a)}, this is descriptive evidence, not a significance claim.

## 4. Public-validation Arousal ranking did not transfer

E19 moved from {fmt(e19.val_A_PCC if e19 is not None else math.nan)} on public validation to {fmt(e19.test_A_PCC if e19 is not None else math.nan)} on test. RoBERTa-large moved from {fmt(rl.val_A_PCC if rl is not None else math.nan)} to {fmt(rl.test_A_PCC if rl is not None else math.nan)}. We therefore frame public validation as an unreliable selector for Arousal among our submissions, not as universally invalid.

## 5. Valence is stable across systems

Valence PCC remains near 0.86–0.88 across paired systems and splits. This contrast makes the evaluation issue dimension-dependent rather than a blanket failure of the task.

## 6. L1 lexicon features

L1 hits {fmt(100*(cov_test.document_hit_rate if cov_test is not None else math.nan),2)}% of official-test documents, yet its mean A-PCC differs from no-L1 by only {fmt((l1row.A_PCC-norow.A_PCC) if l1row is not None and norow is not None else math.nan)}. Safe claim: high coverage with limited measurable test utility under available seed evidence.

## 7. Synthetic augmentation and calibration–ranking trade-off

F2 has A-PCC {fmt(f2.test_A_PCC_mean if f2 is not None else math.nan)} but A-MAE {fmt(f2.test_A_MAE_mean if f2 is not None else math.nan)}, versus 0.372 and 0.928 without augmentation. A small ranking increase co-occurs with worse absolute calibration.

## 8. Graph control and downstream performance

Graph expansion increases intrinsic seed coherence by {fmt(graph_ratio,1)}× over VA lookup, but C→E changes mean test A-PCC by only about 0.003. The graph demonstrably changes the control signal; that intrinsic gain did not clearly transfer downstream.

## 9. Style and length alignment

F2 most strongly changes length alignment, with exact KS/Wasserstein values in `length_distribution.csv`. It still overproduces short texts and does not recover baseline A-MAE.

## 10. Recommended tables and figures

Use the generated val/test table, augmentation table, val-vs-test scatter, augmentation decomposition, and Arousal distribution plot. Treat prediction-distribution figures as label-free diagnostics.

## 11. Claims supported by evidence

- Validation Arousal ranking did not transfer reliably among paired submissions.
- Valence was more stable than Arousal.
- Graph expansion improved intrinsic seed coherence/diversity.
- F2 improved length-distribution proximity relative to fixed-length generation.
- L1 coverage is high, while test utility is not clearly measurable.

## 12. Claims not supported

- Source-aware loss, L1, graph augmentation, or pseudo-labels significantly improve official test performance.
- The graph is useless: intrinsic graph effects are measurable.
- Any per-document official prediction is correct; official gold labels are unavailable.

## 13. Suggested wording

**Abstract:** In a zero in-domain-label setting, we find stable Valence but non-transferring public-validation Arousal rankings. Controlled graph/LLM augmentation improves intrinsic control and prediction spread, yet does not yield a clearly better test calibration–ranking trade-off.

**Conclusion:** Our results motivate uncertainty-aware model selection and explicit separation of intrinsic controllability from downstream utility in dimensional sentiment shared tasks.
"""
    (DOCS / "PAPER_FINDINGS_FINAL.md").write_text(final, encoding="utf-8")

    captions = """# Paper Table Captions

## `main_val_test_results.tex`
- Caption: Official validation and test results for systems evaluated on both splits.
- Interpretation: Valence is stable; Arousal ordering does not reliably transfer.
- Safe claim: Public validation was not a reliable Arousal selector among our paired submissions.
- Avoid: Validation is universally useless.

## `seed_variance.tex`
- Caption: Seed-level variation on official test leaderboard returns.
- Interpretation: Observed E4 seed variation is small at leaderboard precision.
- Safe claim: Training-seed variation does not explain the full val/test reversal.
- Avoid: True seed variance is exactly zero.

## `l1_ablation.tex`
- Caption: L1/no-L1 official test ablation.
- Interpretation: The observed mean difference is within available seed variation.
- Safe claim: L1 test utility is not clearly measurable here.
- Avoid: L1 is proven ineffective.

## `augmentation_ablation.tex`
- Caption: Controlled synthetic-augmentation component ladder.
- Interpretation: Arbitrary seed words and style/length alignment show larger observed changes than VA filtering or graph topology, while MAE remains worse than baseline.
- Safe claim: Intrinsic graph gains did not clearly transfer downstream.
- Avoid: The graph does nothing.

## `prediction_distribution.tex`
- Caption: Label-free prediction distribution on official test texts.
- Interpretation: Methods differ in mean/spread even without clearly different official scores.
- Safe claim: Distribution geometry changed.
- Avoid: Wider predictions are more correct.
"""
    (DOCS / "PAPER_TABLE_CAPTIONS.md").write_text(captions, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    val = parse_official_validation()
    test_runs = parse_main_test_runs()
    test_systems = build_test_system_scores(test_runs)
    paired, correlations = official_val_test(val, test_systems)
    seed_df, l1_df = seed_and_l1_tables(test_runs)
    aug_detail, aug_df = parse_augmentation_scores()
    graph = graph_diagnostics()
    lengths, length_df = length_diagnostics()
    lexicon, coverage_df = lexicon_diagnostics()
    pred_report, dist_df, corr_df, shift_df = prediction_diagnostics()
    taxonomy_df = experiment_taxonomy(val, test_systems)
    prediction_maps = core_prediction_maps()
    make_plots(paired, correlations, aug_df, length_df, dist_df, corr_df, shift_df, prediction_maps)
    write_latex_tables(paired, seed_df, l1_df, aug_df, dist_df)
    write_docs(paired, correlations, coverage_df, l1_df, aug_df, length_df, dist_df, corr_df, graph, taxonomy_df)
    report = {
        "provenance": {
            "official_validation": "baseline/docs/experiments.md",
            "official_test": "baseline/docs/test_results.md",
            "official_gold_available_locally": False,
            "command": "python baseline/analyze_paper_evidence.py",
        },
        "official_val_test": paired.to_dict(orient="records"),
        "spearman": correlations,
        "graph": graph,
        "lexicon": lexicon,
        "length": lengths,
        "predictions": pred_report,
        "augmentation": aug_df.to_dict(orient="records"),
    }
    (ANALYSIS / "paper_evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    MISSING.write()
    print("Analysis complete")
    print(f"  paired official systems: {len(paired)}")
    print(f"  analysis CSV/JSON: {ANALYSIS}")
    print(f"  paper figures: {FIGURES}")
    print(f"  LaTeX tables: {TABLES}")
    print(f"  missing/TODO entries: {len(MISSING.entries)}")
    if correlations:
        print(f"  A-PCC val/test Spearman: rho={correlations['A_PCC']['rho']:.3f}, p={correlations['A_PCC']['p']:.3f}")
    print("  strongly supported: dimension-dependent reliability; intrinsic graph control; augmentation calibration trade-off")
    print("  not supported: statistically significant test superiority for L1/source-aware/graph/pseudo-label methods")


if __name__ == "__main__":
    main()
