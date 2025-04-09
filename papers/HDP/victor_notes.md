# Notes on HDP

## Chapter 1

__Moment__ is the exptected value for a random variable raised to the power of k.

__Second Moment:__  $$\text{E}[X^2]$$

__Moment generating function:__ $$M_{X}(t) = E e^{tX}$$

---

__L-p norm:__ 

$$|| X||_{L^{p}} = (E |X|^p)^{\frac{1}{p}}$$

---

__Covariance:__

$$cov(X,Y) = E(X - E (x)) \cdot E(Y - E(Y)) = <X-E(X), Y-E(Y)>_{L^2}$$

---

__Jensen's inequality:__

$$\rho (E(X)) \leq E(\rho(X))$$

## Chapter 2

__Hoeffding’s inequality (Application in ML):__ 

In learning theory (e.g., VC dimension, PAC learning), Hoeffding shows that:

“The training error ≈ test error with high probability, if you have enough data.”

It provides the mathematical foundation for generalization.
