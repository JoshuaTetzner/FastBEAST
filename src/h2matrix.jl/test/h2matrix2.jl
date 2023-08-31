using Test
using FastBEAST
using StaticArrays
using LinearAlgebra

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

N =  20000

spoints = [@SVector rand(3) for i = 1:N]
tpoints = spoints

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, tpoints[tdata], spoints[sdata])

stree = create_tree(spoints, KMeansTreeOptions(maxlevel=4))
ttree = create_tree(tpoints, KMeansTreeOptions(maxlevel=4))
kmat = assembler(OneoverRkernel, tpoints, spoints)

@time nears, fars, tree = h2matrix(
    OneoverRkernelassembler,
    tpoints,
    spoints,
    Int64,
    Float64,
    compressor=FastBEAST.ACAOptions(tol=1e-4)
)


indices = []
for f in sorted_fars
    if f[1] == 4
        push!(indices, f[2])
    end
end

pindices = []


for index in indices
    s, t = FastBEAST.ClusterTree.indices(index, 4, tree)
    pindices = vcat(pindices, s)
    println(t)
end
pindices