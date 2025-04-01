

# Proposal plan for project (first draft)

__Authors:__ Victor Pekkari and Armon and Haghighi


__`Terms:`__ we have total of `9 weeks` for this project. During these 9 weeks we have to `learn` about RMT, some application/applications to ML, `program` something related and then `present` everything we have done in a final report.

We should just keep going if we feel like we are done with a phase before the set deadline. There is no need to wait for the next phase to begin.

## Phase 0: Come up with plan
- week 1

__`Goal:`__ Draft the initial project proposal and outline to share with our supervisor

Come up with what is important to learn in `random matrix theory` and what applications in `machine learning` we should study. here we should come up with several applications for the professor to choose which ones he thinks is the best one for us to pursue.

## Phase 1: Background and Theoretical Review
- week 2-5

__`Goal:`__ Build a solid foundation in `random matrix theory` and the chosen applications in `ML`.

We can do this by reading papers on current research as well as reading informations we find online about the field. the papers/pdfs we read can be found in the `papers-folder` of this project.


## Phase 2: Programming...
- week 6-7

__`Goal:`__ Program experiments or a project that centers around `RMT` in `ML`.

This part can either be a project where we try to build something, where we try to apply RMT to some machine learning concept. It can also be a set of experiments where we test our own hypothesis or theory that we have studied. The programming language should probably be `python` since it has a lot of helpful libraries

## Phase 3: Reporting
- week 7-8

__`Goal:`__ Get a written overleaf report, and a nice GitHub repo to present what we have done/learnt during this project

The report shuold be in the style of a scientific paper, reporting what we e.g have `learnt, programmed, concluded, experiments, done for project`. `Phase 3` is what we will ultimately be "graded" on.

## Phase 4
- week 9

__`Goal:`__ Correct report after review from the professor

This is the `final` phase


# Possible applications in ML  

### 1. Understanding the neural network training and generalization  
- The Hessian (second derivative of the loss function) captures the curvature of the loss landscape, and its eigenvalues provide insights into optimization and generalization.  
- RMT helps analyze the spectral density of Hessians, showing that neural network loss landscapes often have a bulk of near-zero eigenvalues with some outliers, which affects convergence behavior.  
- Fisher information matrices (which quantify the sensitivity of predictions to weight changes) also follow RMT distributions and can be used to study generalization capacity.  

**Applications:**  
✔ Diagnosing sharp vs. flat minima in loss landscapes (flat minima generalize better).  
✔ Understanding why neural networks avoid poor local minima despite non-convexity.  
✔ Designing better optimization algorithms (e.g., adapting learning rates based on spectral properties).  

---

### 2. Spectral pruning & model compression  
- Eigenvalue analysis of weight matrices identifies directions in the data space that contribute least to variance (small eigenvalues).  
- Spectral pruning removes these insignificant directions, leading to a smaller, more efficient network with minimal loss in performance.  

**Applications:**  
✔ Reducing model size while maintaining accuracy (important for deploying ML models on edge devices).  
✔ Improving efficiency in training and inference.  

---

### 3. Double descent & generalization in overparameterized models  
- The **double descent phenomenon** refers to how test error initially decreases with more model parameters, then increases, and then decreases again in overparameterized models.  
- RMT provides a theoretical framework to explain this behavior by analyzing covariance matrices of random feature models.  

**Applications:**  
✔ Helps justify why modern deep learning models can generalize well despite being heavily overparameterized.  
✔ Guides optimal model selection to avoid overfitting or underfitting.  

---

### 4. Understanding and optimizing kernel methods  
- Kernel methods, such as Support Vector Machines (SVMs) and Gaussian Processes (GPs), rely on Gram (kernel) matrices.  
- RMT helps analyze the eigenvalue distribution of these matrices, improving understanding of how kernels behave in high dimensions.  

**Applications:**  
✔ Selecting better kernel functions in high-dimensional learning tasks.  
✔ Improving the efficiency of kernel-based models by approximating kernel matrices with low-rank approximations.  

---

### 5. Feature selection & dimensionality reduction  
- Principal Component Analysis (PCA) and other dimensionality reduction techniques use covariance matrices, which can be analyzed using RMT.  
- Spiked covariance models help identify signal vs. noise in high-dimensional data.  

**Applications:**  
✔ Improving feature selection by separating meaningful features from noise.  
✔ Reducing overfitting by selecting robust feature representations.  

---

### 6. Stability & robustness in machine learning models  
- RMT can be used to study the stability of gradient-based optimization methods by analyzing the spectral properties of weight updates.  
- Helps identify when models are at risk of instability due to large eigenvalues in the Hessian.  

**Applications:**  
✔ Designing more robust optimization methods that avoid exploding or vanishing gradients.  
✔ Ensuring stable training in deep networks by controlling spectral properties.  


### 7. Distributed Machine Learning & Communication-Efficient Training  
- Large datasets require distributed learning, where models are trained across multiple machines with minimal communication overhead.  
- RMT helps analyze performance loss due to splitting data across multiple machines.  
- Weighted parameter averaging and iterative methods can be studied using high-dimensional statistics.  

**Applications:**  
✔ Optimizing distributed training strategies for large-scale machine learning models.  
✔ Reducing communication costs in federated learning settings.  
✔ Ensuring minimal performance loss when using distributed ordinary least squares (OLS) regression.  

---

### 8. Ridge Regression in High Dimensions  
- Ridge regression is widely used for regularization in high-dimensional data settings.  
- RMT helps analyze and optimize **one-shot distributed ridge regression** methods (e.g., the WONDER algorithm).  
- Mean squared error (MSE) can be studied using random-effects models, providing new insights into bias-variance tradeoffs in distributed regression.  

**Applications:**  
✔ Improving efficiency of ridge regression for large-scale machine learning tasks.  
✔ Designing new algorithms (like WONDER) that achieve high accuracy with reduced computation time.  
✔ Applying distributed ridge regression to large-scale datasets (e.g., the Million Song Dataset).  


### 7. Robustness in High-Dimensional Statistical Learning  
- When data is **high-dimensional**, classical methods can fail due to overfitting or noise sensitivity.  
- RMT provides theoretical **bounds on confidence intervals**, test errors, and robustness metrics.  

**Applications:**  
✔ Developing more **robust** machine learning models in **noisy or adversarial** environments.  
✔ Enhancing **outlier detection** in high-dimensional datasets.  
✔ Improving **feature selection** when dealing with correlated features.  

### 8. High-Dimensional Covariance Estimation & Inference  
- Many ML models rely on estimating covariance matrices, but **standard estimators fail** in high dimensions.  
- RMT provides **better-conditioned covariance estimators**, reducing instability in inverse calculations.  

**Applications:**  
✔ Enhancing **Gaussian Process regression** and kernel-based methods.  
✔ Improving **portfolio optimization** in financial ML by using **better-conditioned covariance matrices**.  
✔ Designing **better uncertainty quantification techniques** in probabilistic machine learning.  

---

### 9. Eigenvalue-Based Anomaly Detection  
- The eigenvalue spectrum of large datasets can reveal hidden **anomalous patterns** (e.g., fraud detection, cybersecurity threats).  
- RMT helps distinguish **true signals from random noise** in high-dimensional settings.  

**Applications:**  
✔ Developing **unsupervised anomaly detection** techniques based on spectral methods.  
✔ Improving **fraud detection in finance** using spectral signatures of transaction data.  
✔ Enhancing **cybersecurity** by detecting unusual activity in high-dimensional logs.  
