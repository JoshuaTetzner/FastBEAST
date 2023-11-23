var documenterSearchIndex = {"docs":
[{"location":"man/hmatrix/#H-Matrix","page":"H-Matrix","title":"H-Matrix","text":"","category":"section"},{"location":"man/hmatrix/","page":"H-Matrix","title":"H-Matrix","text":"Here the Documentation of H-Matrix is done.","category":"page"},{"location":"functions/#Types","page":"Types and Functions","title":"Types","text":"","category":"section"},{"location":"functions/","page":"Types and Functions","title":"Types and Functions","text":"KMeansTreeOptions\nKMeansTreeNode","category":"page"},{"location":"functions/#FastBEAST.KMeansTreeOptions","page":"Types and Functions","title":"FastBEAST.KMeansTreeOptions","text":"KMeansTreeOptions <: TreeOptions\n\nIs the datatype that describes which tree the create_tree function creates.\n\nFields\n\niterations: number of iterations on each level, default is one iterations\nnchildren: defines the number of children of each node, default is two\nnmin: defines the minimum amount of datapoints which are needed in a    cluster so that it gets split up in subclusters, default is 1\nmaxlevel: defines the maximum amount of levels, default is 100.\nalgorithm: defines which algorithm is used. The :naive approach is not recommended.   Default is the wrapped ParallelKMeans algorithm.\n\n\n\n\n\n","category":"type"},{"location":"functions/#FastBEAST.KMeansTreeNode","page":"Types and Functions","title":"FastBEAST.KMeansTreeNode","text":"KMeansTreeNode{T} <: AbstractNode\n\nIs the datatype of each node in the K-Means Clustering Tree.\n\nFields\n\nparent::Union{KMeansTreeNode{T},Nothing}: is the superordinate cluster of   of the represented cluster\nchildren::Union{Vector{KMeansTreeNode{T}}, Nothing}: all directly    subordinated clusters of the represented cluster\nlevel::Integer: the level of the represented cluster\ncenter: the center by which the cluster is defined\nradius: the euclidian distance between the center and the farthest away   point\ndata::T: array containig the indices of the points in this cluster\n\n\n\n\n\n","category":"type"},{"location":"functions/#Functions","page":"Types and Functions","title":"Functions","text":"","category":"section"},{"location":"functions/","page":"Types and Functions","title":"Types and Functions","text":"FastBEAST.create_tree\nFastBEAST.iscompressable","category":"page"},{"location":"functions/#FastBEAST.create_tree","page":"Types and Functions","title":"FastBEAST.create_tree","text":"create_tree(points::Array{SVector{D, T}, 1}; treeoptions)\n\nCreates an algebraic tree for an givn set of datapoints. The returned  datastructure is the foundation for the algorithms in this package. \n\nArguments\n\npoints::Array{SVector{D, T}, 1}: is an array of    SVector.    Each    SVector   contains in general two or three float values, which discribe the position    in the space.\n\nKeywords\n\ntreeoptions::TreeOptions: this keyword defines by which tree is build.    TreeOptions is an abstract type which either can be BoxTreeOptions or   KMeansTreeOptions. Default type is BoxTreeOptions.\n\nReturns\n\nAbstractNode: the root of the created tree. AbstractNode is an abstract type    which either can be BoxTreeNode or KMeansTreeNode, depending on the keyword.\n\n\n\n\n\n","category":"function"},{"location":"functions/#FastBEAST.iscompressable","page":"Types and Functions","title":"FastBEAST.iscompressable","text":"iscompressable(sourcenode::AbstractNode, testnode::AbstractNode)\n\nDetermins whether two nodes of a tree are comressable. The criteria differs  between the Box Tree and the K-Means Clustering Tree. For the K-Means Clustering Tree two nodes can be compressed, if the distance  between the centers of two clusters is greater than the sum of their radius  multiplied by a factor of 1.5. For the Box Tree two nodes can be compressed, if the distance between the centers  of the two boxes is greater than the sum of the distances of each box's center to  one of its corners multiplied by a factor of 1.1.\n\nArguments\n\nsourcenode::AbstractNode: the node which is observed\ntestnode::AbstractNode: the node which is tested for compression\n\nReturns\n\ntrue: if the input nodes are compressable\nfalse: if the input nodes are not compressable\n\n\n\n\n\n","category":"function"},{"location":"gstarted/#Getting-Started","page":"Getting Started","title":"Getting Started","text":"","category":"section"},{"location":"gstarted/#Installation","page":"Getting Started","title":"Installation","text":"","category":"section"},{"location":"man/clustering/#Clustering-Strategies","page":"Clustering Strategies","title":"Clustering Strategies","text":"","category":"section"},{"location":"man/clustering/","page":"Clustering Strategies","title":"Clustering Strategies","text":"For the algorithms defined in this package the given set of datapoints have to be sorted in an algebraic tree.  Therefor the following strategies can be used. ","category":"page"},{"location":"man/clustering/#Box-Tree","page":"Clustering Strategies","title":"Box Tree","text":"","category":"section"},{"location":"man/clustering/#K-Means-Clustering-Tree","page":"Clustering Strategies","title":"K-Means Clustering Tree","text":"","category":"section"},{"location":"man/clustering/#Definition","page":"Clustering Strategies","title":"Definition","text":"","category":"section"},{"location":"man/clustering/","page":"Clustering Strategies","title":"Clustering Strategies","text":"The K-Means algorithm is a clustering strategy for n-dimensional spaces, in which each cluster is represented by its center. In this case the algorithm is implemented for a two or three dimensional euclidean space.","category":"page"},{"location":"man/clustering/#Idea","page":"Clustering Strategies","title":"Idea","text":"","category":"section"},{"location":"man/clustering/","page":"Clustering Strategies","title":"Clustering Strategies","text":"The goal of the algorithm is to divide the dataset in k clusters, which are then alternately divided in k clusters.  The procedure is as follows: At first there are k points chosen from the dataset as centers. Then each point is sorted to its closest center. For each resulting cluster the center is recalculated by its points and all points are resorted to the new centers. This step is repeated for a given number of iterations. The whole process is alternately repeated for each cluster.  ","category":"page"},{"location":"man/clustering/#Algorithm","page":"Clustering Strategies","title":"Algorithm","text":"","category":"section"},{"location":"man/clustering/","page":"Clustering Strategies","title":"Clustering Strategies","text":"Given is a random distribution of datapoints in 2D, which should be sorted in a binary tree.  Two points out of the dataset are chosen as the first centers. For each of the points the euclidean distance to both centers is calculated by:","category":"page"},{"location":"man/clustering/","page":"Clustering Strategies","title":"Clustering Strategies","text":"textdist = textnorm(x - c_i)quad textfor  i = 12k","category":"page"},{"location":"man/clustering/","page":"Clustering Strategies","title":"Clustering Strategies","text":"With x the location of the datapoint and c_i the centers. The point is then added to the closer center.","category":"page"},{"location":"man/clustering/","page":"Clustering Strategies","title":"Clustering Strategies","text":"(Image: ) (Image: )","category":"page"},{"location":"man/clustering/","page":"Clustering Strategies","title":"Clustering Strategies","text":"The centers are shown in the first picture in orange and each cluster is represented in the second picture by one color. To achieve a more even distribution between both clusters, the centers are recalculated. Therefor the distance between the center and each point of the cluster has to be minimized. This can be done by taking the mean over all points.","category":"page"},{"location":"man/clustering/","page":"Clustering Strategies","title":"Clustering Strategies","text":"    c_i = N_k^-1 * sum_n=1^N_k x_n","category":"page"},{"location":"man/clustering/","page":"Clustering Strategies","title":"Clustering Strategies","text":"The dataset is resorted to the updated centers, resulting in the following distribution.","category":"page"},{"location":"man/clustering/","page":"Clustering Strategies","title":"Clustering Strategies","text":"(Image: )","category":"page"},{"location":"man/clustering/","page":"Clustering Strategies","title":"Clustering Strategies","text":"As it can be seen the updated clusters can result in a more even distribution. These steps can then be repeated for a given amount of iterations.  The final clusters are afterwards taken as new distributions of datapoints and the algorithm can alternately be repeated until clusters with a minimum amount of points or a given number of level is reached. ","category":"page"},{"location":"man/clustering/#Comments","page":"Clustering Strategies","title":"Comments","text":"","category":"section"},{"location":"man/clustering/","page":"Clustering Strategies","title":"Clustering Strategies","text":"As it can be seen in the example more iterations will in general lead to more equal clusters. By default the wrapped ParallelKMeans.jl algorithm is used, which stops iterating when the new center of a cluster is close enough to the last iteration. This prevents unnecessary iterations and thus reduces the required computing time. For non homogenous distributions equal sized clusters can not always be reached, but more iterations will generate in general a better tree structure for the algorithms in this package. For the the algorithms in this package 100 iterations and two children for each level are recommended.","category":"page"},{"location":"man/fmm/#Fast-Multipole-Method-(FMM)","page":"Fast Multipole Method (FMM)","title":"Fast Multipole Method (FMM)","text":"","category":"section"},{"location":"man/fmm/","page":"Fast Multipole Method (FMM)","title":"Fast Multipole Method (FMM)","text":"The FMM improves the complexity of the matrix-vector product","category":"page"},{"location":"man/fmm/","page":"Fast Multipole Method (FMM)","title":"Fast Multipole Method (FMM)","text":"mathbfA mathbfx  = mathbfy","category":"page"},{"location":"man/fmm/","page":"Fast Multipole Method (FMM)","title":"Fast Multipole Method (FMM)","text":"form mathcalO(N²) to mathcalO(N), where A is the interaction matrix of monopole or dipole sources that evaluates the Green's function for a Laplace, Helmholtz or modified Helmholtz kernel. The BEM on th other hand is an integral method. To apply the FMM on the integral expressions in the BEM, the internal structure of the FMM must be changed. A fast implementation of a FMM is quiet difficult to achieve, changing the internal structure of the FMM is therefore often cumbersome and results in bad performance. ","category":"page"},{"location":"man/fmm/","page":"Fast Multipole Method (FMM)","title":"Fast Multipole Method (FMM)","text":"There a some highly optimized FMM codes such as the ExaFMM package for monopole and dipole interactions. To use such optimized algorithms with the BEM, Adelmann et al. present in [1] a black-box FMM approach. This method, called the error correction factor matrix method, approximates the BEM integrals by a quadrature. The  quadrature points are then treated as monopole or dipole source, which can directly plugged in to the FMM. The error generated by the quadrature is afterwards corrected. ","category":"page"},{"location":"man/fmm/#Error-Correction-Factor-Matrix-Method","page":"Fast Multipole Method (FMM)","title":"Error Correction Factor Matrix Method","text":"","category":"section"},{"location":"man/fmm/","page":"Fast Multipole Method (FMM)","title":"Fast Multipole Method (FMM)","text":"The method starts with discretizing the geometry. On each triangle the algorithm samples quadrature points which approximate the defined integral. This approximation is sufficient for well separated triangles. For triangles which are to close to each other the quadrature is not accurate and has to be corrected. To determine which triangles are well separated the geometry is clustered in a tree structure by any preferred clustering strategy. ","category":"page"},{"location":"man/fmm/","page":"Fast Multipole Method (FMM)","title":"Fast Multipole Method (FMM)","text":"Given the close interactions and the far interactions, we can setup the black-box FMM. For the close interactions we have to solve the BEM integrals directly giving the matrix mathbfS. Additionally we have to evaluate the Greens function for these points which is computed in the FMM and has to be corrected. This matrix is called the correction matrix mathbfC.  The FMM evaluates the interaction of points which are the quadrature points given a charge on each point. The charge is composed out of the vector x, the quadrature weight and the action of the test and trial functions on the weighted sums over the quadrature points. This is influence is described by the matrix mathbfP_1 and mathbfP_2. The  construction of these interaction matrices is discussed in detail later on.  With these matrices the product mathbfAx writes","category":"page"},{"location":"man/fmm/","page":"Fast Multipole Method (FMM)","title":"Fast Multipole Method (FMM)","text":"mathbfA mathbfx = mathbfP_2^T(mathbfG-mathbfC)mathbfP_1mathbfx + mathbfSmathbfx","category":"page"},{"location":"man/fmm/#References","page":"Fast Multipole Method (FMM)","title":"References","text":"","category":"section"},{"location":"man/fmm/","page":"Fast Multipole Method (FMM)","title":"Fast Multipole Method (FMM)","text":"[1] Adelman, Ross, Nail A. Gumerov, and Ramani Duraiswami. “FMM/GPU-Accelerated Boundary Element Method for Computational Magnetics and Electrostatics.” IEEE Transactions on Magnetics 53, no. 12 (December 2017): 1–11. https://doi.org/10.1109/TMAG.2017.2725951.","category":"page"},{"location":"man/aca/#Adaptive-Cross-Aproximation-(ACA)","page":"Adaptive Cross Aproximation (ACA)","title":"Adaptive Cross Aproximation (ACA)","text":"","category":"section"},{"location":"man/aca/","page":"Adaptive Cross Aproximation (ACA)","title":"Adaptive Cross Aproximation (ACA)","text":"The ACA is a rank revealing matrix decomposition method which is a popular method used in BEM to compress low-rank blocks in the moment matrix. The idea is, given a n times m matrix A with textrank (A)  min(m n), to decompose the matrix A to ","category":"page"},{"location":"man/aca/","page":"Adaptive Cross Aproximation (ACA)","title":"Adaptive Cross Aproximation (ACA)","text":"A^ntimes m = U^ntimes r V^rtimes m","category":"page"},{"location":"man/aca/","page":"Adaptive Cross Aproximation (ACA)","title":"Adaptive Cross Aproximation (ACA)","text":"The algorithm computes this approximation by iteratively selecting rows and columns by a pivoting criteria from the matrix A and subtracting the multiplication of the row and colum of the matrix. This is repeated until the norm of the selected row and column reach a given convergence criterion.","category":"page"},{"location":"man/aca/#Pivoting-Strategies","page":"Adaptive Cross Aproximation (ACA)","title":"Pivoting Strategies","text":"","category":"section"},{"location":"man/aca/","page":"Adaptive Cross Aproximation (ACA)","title":"Adaptive Cross Aproximation (ACA)","text":"Different strategies for choosing the next row or column can be used. the classical approach is the so called maximum value pivoting.","category":"page"},{"location":"man/aca/#Maximum-Value-Pivoting","page":"Adaptive Cross Aproximation (ACA)","title":"Maximum Value Pivoting","text":"","category":"section"},{"location":"man/aca/","page":"Adaptive Cross Aproximation (ACA)","title":"Adaptive Cross Aproximation (ACA)","text":"This strategy selects the next row or column by the index of the maximum value of the previous chosen column or row. ","category":"page"},{"location":"#FastBEAST.jl","page":"Home","title":"FastBEAST.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Some ideas for implementing an ACA in Julia targeting BEAST.","category":"page"},{"location":"#Package-Features","page":"Home","title":"Package Features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Fast linear Algebra","category":"page"},{"location":"#Example","page":"Home","title":"Example","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Example of the main feature of the package.","category":"page"}]
}
