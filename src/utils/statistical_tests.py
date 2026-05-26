"""
Statistical tests for quantum simulation validation.

Provides hypothesis tests and confidence intervals for comparing
simulation results with experimental benchmarks and hardware data.
"""

import numpy as np
from scipy import stats


def z_test_proportion(p_sim, p_exp, n_shots, alpha=0.05):
    """
    Z-test for proportion (simulation vs. experiment).

    Tests H0: p_sim = p_exp against H1: p_sim != p_exp.

    Parameters
    ----------
    p_sim : float
        Simulated success probability.
    p_exp : float
        Experimental success probability.
    n_shots : int
        Number of measurement shots.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Test results.
    """
    se = np.sqrt(p_exp * (1 - p_exp) / n_shots)
    if se < 1e-10:
        return {"z_score": 0, "p_value": 1.0, "reject_H0": False}

    z = (p_sim - p_exp) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "z_score": z,
        "p_value": p_value,
        "reject_H0": p_value < alpha,
        "se": se,
        "alpha": alpha,
    }


def bootstrap_confidence_interval(data, n_bootstrap=10000,
                                   confidence=0.95, statistic=np.mean):
    """
    Compute bootstrap confidence interval.

    Parameters
    ----------
    data : array-like
        Sample data.
    n_bootstrap : int
        Number of bootstrap samples.
    confidence : float
        Confidence level.
    statistic : callable
        Statistic to compute.

    Returns
    -------
    dict
        Bootstrap CI results.
    """
    data = np.array(data)
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(sample)

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return {
        "mean": statistic(data),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "confidence": confidence,
        "se_bootstrap": np.std(bootstrap_stats),
    }


def chi_squared_goodness_of_fit(observed_counts, expected_probs,
                                 total_shots=None):
    """
    Chi-squared goodness-of-fit test.

    Parameters
    ----------
    observed_counts : dict
        Measurement outcome counts.
    expected_probs : dict
        Expected probabilities for each outcome.
    total_shots : int, optional
        Total shots.

    Returns
    -------
    dict
        Chi-squared test results.
    """
    if total_shots is None:
        total_shots = sum(observed_counts.values())

    all_outcomes = set(list(observed_counts.keys()) +
                       list(expected_probs.keys()))

    observed = []
    expected = []
    for outcome in all_outcomes:
        observed.append(observed_counts.get(outcome, 0))
        expected.append(expected_probs.get(outcome, 0) * total_shots)

    observed = np.array(observed)
    expected = np.array(expected)

    # Remove zero-expected bins
    mask = expected > 0
    observed = observed[mask]
    expected = expected[mask]

    if len(observed) < 2:
        return {"chi2": 0, "p_value": 1.0, "dof": 0, "reject_H0": False}

    chi2, p_value = stats.chisquare(observed, f_exp=expected)
    dof = len(observed) - 1

    return {
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "reject_H0": p_value < 0.05,
    }


def cohens_d(group1, group2):
    """
    Compute Cohen's d effect size.

    Parameters
    ----------
    group1, group2 : array-like
        Two samples to compare.

    Returns
    -------
    dict
        Effect size results.
    """
    g1 = np.array(group1)
    g2 = np.array(group2)

    n1, n2 = len(g1), len(g2)
    mean_diff = np.mean(g1) - np.mean(g2)

    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1))
        / (n1 + n2 - 2)
    )

    if pooled_std < 1e-10:
        d = 0
    else:
        d = mean_diff / pooled_std

    # Effect size interpretation
    if abs(d) < 0.2:
        interpretation = "negligible"
    elif abs(d) < 0.5:
        interpretation = "small"
    elif abs(d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "d": d,
        "interpretation": interpretation,
        "mean_diff": mean_diff,
        "pooled_std": pooled_std,
    }


def ks_test(sample1, sample2):
    """
    Kolmogorov-Smirnov two-sample test.

    Parameters
    ----------
    sample1, sample2 : array-like
        Two samples to compare.

    Returns
    -------
    dict
        KS test results.
    """
    stat, p_value = stats.ks_2samp(sample1, sample2)

    return {
        "ks_statistic": stat,
        "p_value": p_value,
        "reject_H0": p_value < 0.05,
    }


def monte_carlo_convergence(fidelity_function, n_samples_list=None,
                             n_repeats=10, **kwargs):
    """
    Test Monte Carlo convergence of a fidelity estimator.

    Parameters
    ----------
    fidelity_function : callable
        Function that returns a fidelity value.
    n_samples_list : list[int], optional
        Sample sizes to test.
    n_repeats : int
        Repeats per sample size.

    Returns
    -------
    dict
        Convergence data.
    """
    if n_samples_list is None:
        n_samples_list = [100, 500, 1000, 2000, 4096, 8192, 16384]

    results = {
        "n_samples": [],
        "mean_fidelity": [],
        "std_fidelity": [],
        "se_fidelity": [],
    }

    for n in n_samples_list:
        fidelities = []
        for _ in range(n_repeats):
            fid = fidelity_function(shots=n, **kwargs)
            fidelities.append(fid)

        results["n_samples"].append(n)
        results["mean_fidelity"].append(np.mean(fidelities))
        results["std_fidelity"].append(np.std(fidelities))
        results["se_fidelity"].append(np.std(fidelities) / np.sqrt(n_repeats))

    return results
