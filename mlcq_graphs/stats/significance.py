from __future__ import annotations

from scipy.stats import ttest_rel, wilcoxon


def significance_marker(p: float) -> str:
    """Return a significance marker string for a given p-value.

    Thresholds follow the locked convention in CONTEXT.md:
      p < 0.001 -> ***
      p < 0.01  -> **
      p < 0.05  -> *
      otherwise -> ns
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def run_baseline_tests(
    combo_scores: dict[str, list[float]],
    baseline_key: str,
    metric_name: str = "f1_macro_tuned",
) -> dict:
    """Compare each combo against a fixed baseline using paired t-test and Wilcoxon.

    Parameters
    ----------
    combo_scores:
        Mapping from combo key (JSON string) to list of per-seed metric values.
    baseline_key:
        The combo key that serves as the reference. Self-comparison is skipped.
    metric_name:
        Name of the metric being tested (stored in result metadata only).

    Returns
    -------
    dict
        Keyed by combo key (excluding baseline_key). Each entry contains:
        ttest_stat, ttest_p, ttest_sig, wilcoxon_stat, wilcoxon_p, wilcoxon_sig, n.
    """
    baseline_scores = combo_scores.get(baseline_key, [])
    results: dict[str, dict] = {}

    for key, scores in combo_scores.items():
        if key == baseline_key:
            continue

        entry: dict = {
            "vs_baseline": baseline_key,
            "metric": metric_name,
            "n": len(scores),
        }

        if len(scores) < 2 or len(baseline_scores) < 2:
            entry["skipped"] = "n < 2"
            results[key] = entry
            continue

        # Paired t-test
        tstat, tp = ttest_rel(scores, baseline_scores)
        entry["ttest_stat"] = float(tstat)
        entry["ttest_p"] = float(tp)
        entry["ttest_sig"] = significance_marker(float(tp))

        # Wilcoxon signed-rank — skip when all differences are zero
        diffs = [a - b for a, b in zip(scores, baseline_scores)]
        if all(d == 0.0 for d in diffs):
            entry["wilcoxon_stat"] = None
            entry["wilcoxon_p"] = None
            entry["wilcoxon_sig"] = "ns"
            entry["wilcoxon_note"] = "all differences zero; wilcoxon skipped"
        else:
            wstat, wp = wilcoxon(scores, baseline_scores, method="auto")
            entry["wilcoxon_stat"] = float(wstat)
            entry["wilcoxon_p"] = float(wp)
            entry["wilcoxon_sig"] = significance_marker(float(wp))

        results[key] = entry

    return results


def run_pairwise_tests(
    combo_scores: dict[str, list[float]],
    metric_name: str = "f1_macro_tuned",
) -> dict:
    """Build a full NxN pairwise comparison matrix.

    For each ordered pair (combo_a, combo_b) where combo_a != combo_b, runs
    a paired t-test and Wilcoxon signed-rank test.

    Parameters
    ----------
    combo_scores:
        Mapping from combo key (JSON string) to list of per-seed metric values.
    metric_name:
        Name of the metric being tested (stored in result metadata only).

    Returns
    -------
    dict
        Nested dict: {combo_a: {combo_b: {ttest_stat, ttest_p, ttest_sig,
        wilcoxon_stat, wilcoxon_p, wilcoxon_sig, n}}}.
    """
    keys = list(combo_scores.keys())
    matrix: dict[str, dict] = {}

    for i, key_a in enumerate(keys):
        matrix[key_a] = {}
        scores_a = combo_scores[key_a]
        for key_b in keys:
            if key_a == key_b:
                continue
            scores_b = combo_scores[key_b]

            entry: dict = {
                "metric": metric_name,
                "n": min(len(scores_a), len(scores_b)),
            }

            if len(scores_a) < 2 or len(scores_b) < 2:
                entry["skipped"] = "n < 2"
                matrix[key_a][key_b] = entry
                continue

            tstat, tp = ttest_rel(scores_a, scores_b)
            entry["ttest_stat"] = float(tstat)
            entry["ttest_p"] = float(tp)
            entry["ttest_sig"] = significance_marker(float(tp))

            diffs = [a - b for a, b in zip(scores_a, scores_b)]
            if all(d == 0.0 for d in diffs):
                entry["wilcoxon_stat"] = None
                entry["wilcoxon_p"] = None
                entry["wilcoxon_sig"] = "ns"
                entry["wilcoxon_note"] = "all differences zero; wilcoxon skipped"
            else:
                wstat, wp = wilcoxon(scores_a, scores_b, method="auto")
                entry["wilcoxon_stat"] = float(wstat)
                entry["wilcoxon_p"] = float(wp)
                entry["wilcoxon_sig"] = significance_marker(float(wp))

            matrix[key_a][key_b] = entry

    return matrix
