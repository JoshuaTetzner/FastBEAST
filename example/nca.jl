using FastBEAST
using StaticArrays

## NCA
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
            kernelmatrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
    return kernelmatrix
end


function assembler(kernel, matrix, testpoints, sourcepoints)
    for j in eachindex(sourcepoints)
        for i in eachindex(testpoints)
            matrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end

##
using FastBEAST
using LinearAlgebra

N = 100
A = rand(N,N)

@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

U,S,V = svd(A)

S = [ i < 15 ? 10.0^(-i) : 0.0 for i = 1:N ]

A = U*diagm(S)*V'

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

U, V, Ut, Vt, rows, cols = nca(lm, svdrecompress=false, tol=10^-4)
##
norm(U*V-A)/norm(A)

##
norm(Ut * A[rows, cols]^-1 * Vt - A)/norm(A)

At = Ut * Ut[rows, :]^-1 * A[rows, cols] * Vt[:, cols]^-1 * Vt

norm(At-A)/norm(A)

##
using StaticArrays
using FastBEAST

N =  100

spoints = [@SVector rand(2) for i = 1:N]
tpoints = spoints

stree = create_tree(spoints, KMeansTreeOptions(maxlevel=4))
ttree = create_tree(tpoints, KMeansTreeOptions(maxlevel=4))

fullinteractions = SVector{2}[]
compressableinteractions = SVector{2}[]
    
FastBEAST.computerinteractions!(
    ttree,
    stree,
    fullinteractions,
    compressableinteractions
)

compressableinteractions[1]

for c in compressableinteractions
    println(c[1].data.indices)
end