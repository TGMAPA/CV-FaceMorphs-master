# Pipeline for Demographic Morphability Analysis in Facial Embedding Manifold

## Demographic Annotation of the Manifold - Wed 13 May 22:26:55 GMT by MAPA

Each facial embedding is associated with demographic metadata such as ethnicity, age, and gender, allowing the reduced t-SNE space to be interpreted as a demographic-topological manifold instead of only a geometric representation.

## Density and Cluster Structure Analysis - Wed 13 May 22:26:55 GMT by MAPA

The manifold is analyzed using neighborhood and density-based methods to identify dense demographic regions, sparse areas, cluster boundaries, transition zones, and outlier identities that may exhibit different morphing behaviors and biometric vulnerabilities. ( **HDBSCAN** or Kernel Density Estimation)

## Distributional Similarity Analysis - Wed 13 May 22:26:55 GMT by MAPA

Demographic regions within the manifold are compared using a distribution-based metric in order to measure compatibility, separation, and transitions between demographic populations at a distributional level rather than through simple pairwise distances. (**Wasserstein Distance**)

### 1. Highly Compatible Regions

Clusters:

- Low Wasserstein,
- Similar demographics,
- Close boundary.

- Objective

Morphs:
- Natural,
- Smooth,
- High plausibility.

### 2. Transitional/Boundary Regions

Clusters:

- Medium Wasserstein,
- Partially mixed,
- Topologically neighboring.

- Objective

Evaluate:

- Biometric ambiguity,
- Demographic transition.

### 3. Cross-Demographic Compatible Regions

Clusters:

- Distinct demographics,
- Relatively low Wasserstein.

Objective

- Naturally compatible cross-demographic regions.

### 4. Extreme Regions

Clusters:
- High Wasserstein,
- Widely separated demographics.

- Objective: Adversarial/extreme morphs.

## Morph Pair Selection Strategy

Morph pairs are selected according to their topological and demographic relationships within the t-SNE manifold, including intra-cluster pairs, boundary-region pairs, bridge pairs between neighboring clusters, sparse-region pairs, and cross-demographic pairs with local geometric proximity.

## Morph Generation

Synthetic face morphs are generated from the selected identity pairs using a face morphing pipeline, producing samples with varying levels of demographic compatibility and manifold continuity for subsequent biometric vulnerability analysis.