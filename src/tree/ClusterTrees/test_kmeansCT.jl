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
 
@time tree = create_CT_tree(points, nmin=1, maxlevel=5);

tree.levels

block_tree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
@time nears, fars = FastBEAST.computeinteractions(block_tree)
x=1

fars