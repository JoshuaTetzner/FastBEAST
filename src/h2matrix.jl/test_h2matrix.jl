using Test
using FastBEAST
using StaticArrays
using LinearAlgebra
using ClusterTrees

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

    for j in eachindex(sourcepoints)
        for i in eachindex(testpoints)
            kernelmatrix[i, j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
    return kernelmatrix
end


function assembler(kernel, matrix, testpoints, sourcepoints)
    for j in eachindex(sourcepoints)
        for i in eachindex(testpoints)
            matrix[i, j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end

##

N =  10000
points = [@SVector rand(3) for i = 1:N]

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, points[tdata], points[sdata])

fmat = assembler(OneoverRkernel, points, points)

@time tree = create_CT_tree(points, maxlevel=7)
block_tree = ClusterTrees.BlockTrees.BlockTree(tree, tree)

@time retvals = FastBEAST.H2Matrix(OneoverRkernelassembler, block_tree);

##

stree = create_tree(points, BoxTreeOptions(nmin=50))
ttree = create_tree(points, BoxTreeOptions(nmin=50))
@time hmat = HMatrix(
    OneoverRkernelassembler,
    ttree,
    stree,
    Int64,
    Float64,
    compressor=FastBEAST.ACAOptions(tol=1e-4)
)

##
