# SmallestZeroNorm

### Proposing a new sparse weight extraction method in a linear model:

 Let $y \in \mathbb{R}^n$ be the output vector, $F \in \mathbb{R}^{n \times d}$ be the factor matrix, $w \in \mathbb{R}^d$ be the weight vector, and $\varepsilon \in \mathbb{R}^n$ be the noise vector. 

Suppose many of the values in the factor loadings $w$ are zero (or negligibly small). The true underlying linear mode is 
$$
y = F w + \varepsilon\,.
$$

Given $y$ and $F$ (not knowing the noise vector $\varepsilon$), we want to find the sparsest $\hat w$ (in the sense of ***$\mathbf{\ell_0}$ norm***) that approximately solves the above linear model. Specifically, we want to solve the following optimization:

$$
\min \| w \|_0 \, \, \, \, \text{subject to }\, \frac{1}{n} \|y - F w \|_2^2 \leq b^2\,.
$$

$b$ is a parameter of the problem, that is adaptively trained in the optimization algorithm.

```SmallestZeroNorm.ipynb``` explains and implements the optimization method. In ```Test.ipynb``` we compare the performance of this method with Lasso. 