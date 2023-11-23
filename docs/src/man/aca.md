# Adaptive Cross Aproximation (ACA)
The ACA is a rank revealing matrix decomposition method which is a popular method used in BEM to compress low-rank blocks in the moment matrix. The idea is, given a $n \times m$ matrix $A$ with $\text{rank} (A) < \min(m, n)$, to decompose the matrix $A$ to 

$$A^{n\times m} = U^{n\times r} V^{r\times m}\,.$$

The algorithm computes this approximation by iteratively selecting rows and columns by a pivoting criteria from the matrix $A$ and subtracting the multiplication of the row and colum of the matrix. This is repeated until the norm of the selected row and column reach a given convergence criterion.

## Pivoting Strategies 
Different strategies for choosing the next row or column can be used. the classical approach is the so called maximum value pivoting.

### Maximum Value Pivoting
This strategy selects the next row or column by the index of the maximum value of the previous chosen column or row. 

 


