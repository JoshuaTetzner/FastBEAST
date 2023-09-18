using Test
using ClusterTrees
using FastBEAST
using StaticArrays
using LinearAlgebra
using SparseArrays

# Do a 3D test with Laplace kernel

function OneoverRkernel(testpoint::SVector{3,T}, sourcepoint::SVector{3,T}) where T
    if isapprox(testpoint, sourcepoint, rtol=eps()*1e-4)
        return T(0.0)
    else
        return T(1.0) / (norm(testpoint - sourcepoint))
    end
end

function assembler(kernel, testpoints, sourcepoints)
    kernelmatrix = zeros(promote_type(eltype(testpoints[1]),eltype(sourcepoints[1])), 
                length(testpoints), length(sourcepoints))

    for j = 1:length(sourcepoints)
        for i = 1:length(testpoints)
            kernelmatrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
    return kernelmatrix
end


function assembler(kernel, matrix, testpoints, sourcepoints)
    for j = 1:length(sourcepoints)
        for i = 1:length(testpoints)
            matrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end

##

N =  1000


points = [@SVector rand(3) for i = 1:N]
 
@time tree = create_CT_tree(points, maxlevel=7)
block_tree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
@time nears, fars = FastBEAST.computeinteractions(block_tree)
kmat = assembler(OneoverRkernel, points, points);

rowblock, colblock = fars[4][1]

rowidcs = FastBEAST.value(block_tree.test_cluster, rowblock)
colidcs = FastBEAST.value(block_tree.test_cluster, colblock)

@views function fct(B, x, y)
    B[:,:] = kmat[x, y]
end

lm = LazyMatrix(fct, rowidcs, colidcs, Float64)

U, V, retU, retV, r, c = nca(lm, maxrank=100, tol=1e-4, svdrecompress=false)
