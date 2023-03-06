# Adaptive Cross Aproximation (ACA)
The ACA is a rank revealing matrix decomposition method which is a popular method used in BEM to compress low-rank blocks in the moment matrix. The idea is, given a $n \times m$ matrix $A$ with $\rank (A) < \min(m, n)$, to decompose the matrix $A$ to 

$$A^{n\times m} = U^{n\times r} V^{r\times m}\,.$$

The algorithm computes this approximation by iteratively selecting rows and columns by a pivoting criteria from the matrix $A$ and subtracting the multiplication of the row and colum of the matrix. This is repeated until the norm of the selected row and column reach a given convergence criterion.

## Pivoting Strategies 
Different strategies for choosing the next row or column can be used. the classical approach is the so called maximum value pivoting.

### Maximum Value Pivoting
This strategy selects the next row or column by the index of the maximum value of the previous chosen column or row. 

### Minimal Fill-Distance Pivoting
The fill distance strategy is a geometrical approach. In the BEM the low-rank blocks of the moment matrix combine different test and basis functions. Each basis function is placed on a node edge or in the barycentric mesh. This pivoting strategy selects the next row or column by minimizing the fill-distance form step to step in the block of basis functions. The fill-distance is defined for a set of basis function $X$ as 

$$sup\, min (dist(x, X_k)), \quad \forall x\in X\,,$$

where $X_k$ is the set of the already used basis functions of the set $X$. 

This strategy is used either for the columns or the rows, the corresponding rows or columns should be selected by the maximum value pivoting. It is not recommended to use it for both. 

Selecting the next basis function minimizing the fill distance is computationally expensive. We present a fast approach which gives often in the long run a smaller fill-distance, but not necessarily minimizes it in each step. 

The first basis function has to be selected by computing the fill distance for all functions and selecting the basis function resulting in the minimal fill-distance. With the first basis functions computed we get a distance for each basis function in the set. In the following steps we select the next basis function by the maximum distance of the previous step and update the distance for each basis function, where it gets smaller with the selected basis-function. This gives an algorithm with linear complexity. 


