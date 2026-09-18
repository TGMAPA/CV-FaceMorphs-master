# cluster_quality_metrics.py

import numpy as np
import pandas as pd
from scipy import stats
from ControlledMorphGeneration.MorphGeneration_Pipeline_DistributionalSimilarity.TrustRegions.ControlledMorphGeneration import fit_candidate_distributions, compute_trust_region

# How precisely can we estimate the mean of the neighbor distances for each cluster?
def compute_t_student_metrics(distances, confidence=0.95):
    # Convert distances to a numeric NumPy array
    distances = np.asarray(distances, dtype=float)

    # Remove invalid or non-finite distance values
    distances = distances[np.isfinite(distances)]
    n = len(distances)

    # Return empty metrics if there are insufficient samples
    if n < 2:
        return {
            "n": n,
            "mean": np.nan,
            "std": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "ci_width": np.nan
        }

    # Compute sample mean and sample standard deviation
    mean = np.mean(distances)
    std = np.std(distances, ddof=1)

    # Compute significance level from the selected confidence level
    alpha = 1 - confidence

    # Compute critical t-value for the confidence interval
    t_critical = stats.t.ppf(
        1 - alpha / 2,
        df=n - 1
    )

    # Compute standard error of the sample mean
    standard_error = std / np.sqrt(n)

    # Compute confidence interval around the sample mean
    ci_lower = mean - t_critical * standard_error
    ci_upper = mean + t_critical * standard_error

    # Compute total confidence interval width
    ci_width = ci_upper - ci_lower

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_width
    }

# How geometrically extended is the cluster?
def compute_compression_metrics(embeddings):
    # Convert embeddings to a numeric NumPy array
    X = np.asarray(embeddings, dtype=float)

    # Return empty metrics if there are insufficient samples
    if len(X) < 2:
        return {
            "centroid_radius_mean": np.nan,
            "centroid_radius_std": np.nan,
            "centroid_radius_median": np.nan,
            "centroid_radius_q90": np.nan,
            "compression_cv": np.nan
        }

    # Compute the geometric centroid of the cluster
    centroid = np.mean(X, axis=0)

    # Compute the Euclidean distance from each embedding to the centroid
    radii = np.linalg.norm(X - centroid, axis=1)

    # Compute descriptive statistics of the centroid distances
    mean_radius = np.mean(radii)
    std_radius = np.std(radii, ddof=1)

    # Compute the coefficient of variation of the centroid distances
    cv = (
        std_radius / mean_radius
        if mean_radius > 0
        else np.nan
    )

    return {
        "centroid_radius_mean": mean_radius,
        "centroid_radius_std": std_radius,
        "centroid_radius_median": np.median(radii),
        "centroid_radius_q90": np.percentile(radii, 90),
        "compression_cv": cv
    }

# How close are the samples to each other inside a cluster?
def compute_density_metrics(neighbor_pairs, k=2):
    # Extract nearest-neighbor distances from the pairwise results
    distances = np.asarray(
        neighbor_pairs["distance"],
        dtype=float
    )

    # Remove invalid or non-finite distance values
    distances = distances[np.isfinite(distances)]

    # Return empty metrics if no valid distances are available
    if len(distances) == 0:
        return {
            "mean_knn_distance": np.nan,
            "median_knn_distance": np.nan,
            "knn_density": np.nan
        }

    # Compute mean and median nearest-neighbor distances
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)

    # Estimate local density as the inverse of the mean neighbor distance
    density = (
        1.0 / mean_distance
        if mean_distance > 0
        else np.nan
    )

    return {
        "mean_knn_distance": mean_distance,
        "median_knn_distance": median_distance,
        "knn_density": density
    }


def extract_distribution_quality(fitted_results):
    # Select the best-fitting distribution
    best = fitted_results.iloc[0]

    # Extract goodness-of-fit and model selection metrics
    return {
        "best_distribution": best["name"],
        "ks_statistic": best["ks_statistic"],
        "ks_p_value": best["p_value"],
        "aic": best["AIC"]
    }

# Do the obtained model and parameters hold up when we perturb or resample the data?
def bootstrap_distribution_stability(
        distances,
        candidate_distributions,
        n_iterations=100,
        confidence=0.90,
        random_state=42
    ):

    # Initialize the random number generator for reproducible sampling
    rng = np.random.default_rng(random_state)

    # Convert distances to a numeric NumPy array
    distances = np.asarray(distances, dtype=float)

    # Remove invalid or non-finite distance values
    distances = distances[np.isfinite(distances)]

    results = []

    n = len(distances)

    # Generate bootstrap samples and evaluate distribution fitting stability
    for i in range(n_iterations):

        # Generate a bootstrap sample by sampling with replacement
        bootstrap_sample = rng.choice(
            distances,
            size=n,
            replace=True
        )

        # Fit the candidate distributions to the bootstrap sample
        fitted = fit_candidate_distributions(
            bootstrap_sample,
            candidate_distributions
        )

        # Rank fitted distributions using KS statistic and AIC
        fitted = fitted.sort_values(
            by=["ks_statistic", "AIC"],
            ascending=[True, True]
        )

        # Remove models with non-finite quality metrics
        fitted = fitted[
            np.isfinite(fitted["ks_statistic"]) &
            np.isfinite(fitted["AIC"])
        ]

        if fitted.empty:
            continue

        # Select the best-fitting distribution
        best = fitted.iloc[0]

        # Compute the trust region for the selected distribution
        trust_region = compute_trust_region(
            best,
            confidence=confidence
        )

        # Store the results from the current bootstrap iteration
        results.append({
            "iteration": i,
            "distribution": best["name"],
            "ks": best["ks_statistic"],
            "p_value": best["p_value"],
            "AIC": best["AIC"],
            "trust_lower": trust_region["lower"],
            "trust_upper": trust_region["upper"]
        })

    return pd.DataFrame(results)


def summarize_bootstrap_stability(bootstrap_results):
    n = len(bootstrap_results)

    # Return an empty result if no bootstrap iterations were completed
    if n == 0:
        return {}

    # Compute the proportion of iterations selecting each distribution
    distribution_stability = (
        bootstrap_results["distribution"]
        .value_counts(normalize=True)
    )

    # Identify the distribution selected most frequently
    dominant_distribution = (
        distribution_stability.index[0]
    )

    # Compute the stability of the dominant distribution
    model_stability = (
        distribution_stability.iloc[0]
    )

    # Summarize the stability of the fitted model and trust region
    return {
        "bootstrap_iterations": n,
        "dominant_distribution": dominant_distribution,
        "model_stability": model_stability,
        "mean_ks": bootstrap_results["ks"].mean(),
        "std_ks": bootstrap_results["ks"].std(),
        "mean_aic": bootstrap_results["AIC"].mean(),
        "std_aic": bootstrap_results["AIC"].std(),
        "mean_trust_lower": bootstrap_results["trust_lower"].mean(),
        "mean_trust_upper": bootstrap_results["trust_upper"].mean(),
        "std_trust_lower": bootstrap_results["trust_lower"].std(),
        "std_trust_upper": bootstrap_results["trust_upper"].std()
    }