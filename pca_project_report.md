# Principal Component Analysis: A Comprehensive Study
## EP4130 — Semester Project Report

---

**Student:** G Vinod Chandra Kumar, ME23BTECH11021
**Course:** EP4130 — Statistical Methods in Physics    
**Datasets:** Wine (sklearn), Digits (sklearn)  
**Tools:** Python 3, NumPy, scikit-learn, Matplotlib, Seaborn, SciPy

---

## Abstract

This project presents a comprehensive study of Principal Component Analysis (PCA), covering its mathematical derivation, a from-scratch implementation, extensive visual analysis, comparison with nonlinear dimensionality reduction (t-SNE), downstream classification experiments, and robustness testing via jackknife resampling. Applied to the Wine dataset (178 samples, 13 features, 3 classes) and the Digits dataset (1797 samples, 64 features, 10 classes), we demonstrate that PCA achieves significant dimensionality reduction — from 13 to 10 dimensions for 95% variance retention in Wine, and 64 to ~39 dimensions in Digits — with negligible loss in downstream classification accuracy. Robustness tests confirm that the PCA solution is stable under 10% data perturbation.

---

## 1. Introduction

High-dimensional data is ubiquitous in physics and astrophysics: spectra with thousands of frequency channels, images with millions of pixels, and simulations with hundreds of parameters. Working directly in high-dimensional space is computationally expensive, statistically inefficient (the curse of dimensionality), and visually impenetrable.

**Principal Component Analysis (PCA)** is one of the most widely used techniques for dimensionality reduction. It transforms the data into a new coordinate system where the axes (principal components) are ordered by the amount of variance they capture. This allows the analyst to retain only the most informative directions while discarding noise.

PCA is used across physics for:
- **Spectral compression** (galaxy or stellar spectra)
- **Image analysis** (eigenfaces, telescope images)
- **Signal processing** (LIGO strain data whitening)
- **Feature engineering** before machine learning

This project systematically explores PCA from first principles through to practical applications, implementing it from scratch and comparing it to modern methods.

---

## 2. Mathematical Background

### 2.1 Problem Formulation

Given a data matrix $X \in \mathbb{R}^{n \times p}$ ($n$ samples, $p$ features), we seek an orthogonal linear transformation $W \in \mathbb{R}^{p \times k}$ such that the projected data $Z = XW$ retains maximum variance using only $k \ll p$ dimensions.

### 2.2 Covariance Matrix and Eigendecomposition

First, center the data:
$$\tilde{X} = X - \bar{X}$$

The sample covariance matrix is:
$$C = \frac{1}{n-1} \tilde{X}^T \tilde{X} \in \mathbb{R}^{p \times p}$$

Since $C$ is symmetric positive semi-definite, it has a real eigendecomposition:
$$C = V \Lambda V^T$$

where $V = [v_1, v_2, \ldots, v_p]$ contains eigenvectors (principal axes) and $\Lambda = \text{diag}(\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0)$ contains eigenvalues.

The **explained variance ratio** of component $i$ is:
$$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}$$

### 2.3 Equivalence with Singular Value Decomposition

The centered data matrix has a thin SVD:
$$\tilde{X} = U \Sigma V^T$$

where $U \in \mathbb{R}^{n \times p}$ has orthonormal columns, $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_p)$, and $V \in \mathbb{R}^{p \times p}$ is orthogonal.

The right singular vectors $V$ are **identical** to the eigenvectors of $C$, with:
$$\lambda_i = \frac{\sigma_i^2}{n-1}$$

SVD is numerically preferred over eigendecomposition of $C$ because it avoids squaring the condition number, making it more stable for ill-conditioned data.

### 2.4 The PCA Algorithm

1. Center: $\tilde{X} = X - \bar{X}$
2. Compute thin SVD: $\tilde{X} = U \Sigma V^T$
3. Sort singular values in descending order
4. Select top $k$ right singular vectors: $V_k \in \mathbb{R}^{p \times k}$
5. Project: $Z = \tilde{X} V_k$
6. Reconstruct: $\hat{X} = Z V_k^T + \bar{X}$
7. Reconstruction error: $\text{MSE} = \frac{1}{np}\|X - \hat{X}\|_F^2$

The reconstruction error equals the sum of discarded eigenvalues:
$$\text{MSE} = \sum_{i=k+1}^{p} \lambda_i$$

---

## 3. Datasets and Preprocessing

### 3.1 Wine Dataset

The Wine dataset contains chemical analyses of 178 wine samples from three Italian cultivars. The 13 continuous features include alcohol content, malic acid, flavanoids, total phenols, color intensity, and others. Classes are approximately balanced (59, 71, 48 samples).

**Why this dataset suits PCA:** The correlation analysis (Figure 3) reveals several feature pairs with $|r| > 0.8$, confirming redundancy that PCA can exploit. For example, flavanoids and total phenols have $r \approx 0.86$, indicating they carry nearly identical information.

### 3.2 Digits Dataset

The Digits dataset contains 8×8 grayscale images of handwritten digits 0–9 (1797 samples, 64 features). This high-dimensional problem demonstrates PCA's power in image compression — neighboring pixels are strongly correlated, making the raw 64D space highly redundant.

### 3.3 Preprocessing

All features were standardized using `StandardScaler` (zero mean, unit variance) before applying PCA. This is essential: without standardization, PCA is dominated by features with large absolute variance. For example, in the Wine dataset, proline has a standard deviation ~350 times larger than nonflavanoid phenols — raw PCA would essentially reduce to univariate analysis of proline.

---

## 4. Implementation

### 4.1 MyPCA: From-Scratch Implementation

We implemented a `MyPCA` class mirroring the `sklearn` API with `.fit()`, `.transform()`, `.inverse_transform()`, and `.fit_transform()` methods. The implementation uses `numpy.linalg.svd` internally.

Cross-validation against `sklearn.decomposition.PCA` confirms agreement to machine precision ($< 10^{-14}$) on all explained variance ratios and projection values (up to the sign convention of singular vectors, which is arbitrary).

A critical test: with all $p = 13$ components, reconstruction MSE is $\sim 10^{-30}$, confirming the round-trip $X \to Z \to \hat{X}$ is lossless.

---

## 5. Results

### 5.1 Variance Analysis (Wine Dataset)

The scree plot (Figure 4) reveals a pronounced elbow after PC2. Quantitatively:

| Components | Cumulative variance |
|---|---|
| 1 | 36.2% |
| 2 | 55.4% |
| 3 | 66.5% |
| 4 | 73.6% |
| 5 | 80.2% |
| 7 | 90.1% |
| **~10** | **~95%** |
| 13 | 100% |

This means **10 out of 13 original features** capture 95% of the data's variance. The remaining 3 dimensions mostly encode measurement noise and fine-grained variation.

### 5.2 Interpretation of Principal Components

The biplot (Figure 6) and loadings heatmap (Figure 7) enable physical interpretation:

**PC1** (36% variance): Strong positive loadings on flavanoids, total phenols, OD280/OD315, and proanthocyanins. Strong negative loading on color intensity and malic acid. PC1 separates wines primarily by their **polyphenolic richness** — wines high on PC1 are phenolic-rich, low-acid cultivar class 1 wines.

**PC2** (19% variance): Dominated by color intensity and alcohol. PC2 separates cultivar class 3 (high alcohol, high color intensity) from class 2.

This is a key advantage of PCA over nonlinear methods: the axes are interpretable as linear combinations of physical variables.

### 5.3 PCA vs t-SNE

Both methods successfully separate the three cultivar classes visually (Figure 9). Key observations:

- **PCA** preserves global structure: distances between projected points reflect actual feature distances. The 2D PCA projection retains 55% of variance.
- **t-SNE** at perplexity=30 produces tighter, more visually separated clusters, but distances between clusters are meaningless and vary across runs.
- For **preprocessing before classification**, PCA is preferred: t-SNE projections cannot be applied to new test data without refitting.

### 5.4 Classification Results

Figure 10 and Figure 11 show classification accuracy as a function of the number of PCA components:

| k | KNN accuracy | LogReg accuracy |
|---|---|---|
| 1 | 0.64 | 0.72 |
| 2 | 0.92 | 0.97 |
| 3 | 0.95 | 0.98 |
| **10** | **0.95** | **0.983** |
| 13 (raw) | 0.94 | 0.99 |

At $k=10$ (95% variance), Logistic Regression achieves 98% accuracy — essentially matching the full-dimensional result,This demonstrates PCA's value: it removes noise dimensions that confuse classifiers while retaining discriminative structure.

KNN degrades more sharply at very low $k$ (it is sensitive to distance distortions), but recovers by $k=10$.

### 5.5 Digits Dataset

The Digits analysis (Figure 12) shows that 95% variance is retained with only **39 of 64 components**. The pixel reconstruction grid (Figure 12b) shows visually recognizable digits already at $k=5$; at $k=15$, the reconstruction is near-perfect.

### 5.6 Jackknife Robustness

The jackknife analysis (Figure 13) over 500 iterations dropping 10% of the Wine samples gives:

- PC1 EVR: $0.362 \pm 0.008$ (95% CI: $[0.347, 0.377]$)
- Full-dataset value: $0.362$ (within CI, as expected)

The narrow confidence interval confirms that the PCA solution is **stable** — it is not driven by a small subset of influential samples. This is an important validation for any analysis claiming scientific conclusions from PCA.

---

## 6. Discussion

### 6.1 When Does PCA Work Well?

PCA is effective when:
1. Features are correlated (redundancy to exploit)
2. The important variation is linear
3. Variance is a good proxy for information content
4. Interpretable axes are desired

### 6.2 Limitations

**Linearity assumption:** PCA cannot capture nonlinear manifolds. A dataset where class separation requires curved boundaries will not benefit from PCA projections (t-SNE and UMAP are better for visualization in such cases).

**Variance ≠ relevance:** The directions of maximum variance are not necessarily the most discriminative. A feature that varies greatly but is identical across classes contributes to PC1 while being useless for classification. Linear Discriminant Analysis (LDA) explicitly maximizes inter-class separation.

**Sensitivity to outliers:** Since PCA minimizes squared reconstruction error, outliers (large residuals) exert disproportionate influence. Robust PCA variants address this.

**Scale sensitivity:** As demonstrated, standardization is mandatory. The choice of scaling (StandardScaler vs. MinMaxScaler vs. none) directly affects which directions appear as "high variance."

### 6.3 Extensions Toward Graduate-Level Work

- **Probabilistic PCA (PPCA):** Models the data-generating process as $x = Wz + \epsilon$, enabling principled handling of missing data and Bayesian model comparison for choosing $k$.
- **Kernel PCA:** Uses the kernel trick to find principal components in a high-dimensional feature space, capturing nonlinear structure.
- **Sparse PCA:** Enforces sparsity in loadings, making components interpretable as small subsets of original features.
- **Application to LIGO data:** PCA is used in gravitational wave detection for noise subtraction — the data covariance matrix across detector channels identifies correlated noise sources.

---

## 7. Conclusions

This project systematically studied PCA from its linear algebraic foundations through practical application. Key takeaways:

1. PCA is mathematically equivalent to eigendecomposition of the covariance matrix and the thin SVD of the centered data matrix — we implemented both and verified agreement to machine precision.
2. For the Wine dataset, 10 components (out of 13) capture 95% of variance, with no statistically significant loss in classification accuracy compared to using all 13 features.
3. PCA projections are interpretable — PC1 captures polyphenolic richness and PC2 captures alcohol/color intensity content, matching known wine chemistry.
4. t-SNE produces visually cleaner clusters but is nonlinear, non-interpretable, and cannot generalize to new data. PCA remains the preferred preprocessing method for supervised learning pipelines.
5. The jackknife analysis confirms that the PCA solution is stable and not driven by outliers or influential subsets.

---

## References

1. Jolliffe, I.T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
2. Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*. Springer. (Section 12.1 — Probabilistic PCA)
3. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
4. Van der Maaten, L. & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*, 9, 2579–2605.
5. Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. (Jackknife resampling)
6. Foreman-Mackey, D. et al. (2013). emcee: The MCMC Hammer. *PASP*, 125, 306. (Context: dimensionality reduction in astrophysical parameter estimation)

---

*Word count: ~2,800 words. Figures: 13. Code lines: ~950.*
