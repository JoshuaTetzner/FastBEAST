using StaticArrays
using LinearAlgebra
using FastBEAST

##kernel
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

##
#sp, tp = testgeo_2d()
sp = [@SVector rand(3) for i = 1:1000]
tp = [@SVector rand(3) for i = 1:1000] + 3 .* [@SVector ones(3) for i = 1:1000]
A = assembler(OneoverRkernel, tp, sp)
sp
tp

_,sv,_=svd(A)
sv


@views function fct(B, x, y)
    B[:,:] = A[x, y]
end


##
piv = FastBEAST.FillDistance(tp)
rows, cols, V = pcur(
    fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), rowpivstrat=piv)
V

##
    V[1:10, :]
V[7, :]
V[8, :]

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)
U, V2 = aca(lm, rowpivstrat=piv, svdrecompress=false, tol=1e-4)
norm(V2[5,:])
norm(U[:,5])
V[3, :]
##

rank(A)
norm(A.-U*V2)/norm(A)

V = A[1:5, :]
U = A[:, 1:5]

U2 = V[1:5, 1:5]
V2 = copy(V)

## first iter
U2[:, 1] = U2[:, 1] .* (1/U2[1,1])
norm(V2[1, :]) 
norm(U2[:, 1]) 
## second
U2[:, 2] -= U2[:, 1] .* V2[1, 2]
U2[:, 2] = U2[:, 2] .* (1/U2[2,2])
U2[2,1]
V2[1, :] 
V2[2, :] -= U2[2, 1] .* V2[1, :] 
println((U2[2, 1] .* V2[1, :])[1:5]) 
norm(V2[2, :])
## third
U2[:, 3] -= U2[:, 1] .* V2[1, 3]
U2[:, 3] -= U2[:, 2] .* V2[2, 3]
U2[:, 3] = U2[:, 3] .* (1/U2[3,3])
V2[3, :] -= U2[3, 1] .* V2[1, :] 
V2[3, :] -= U2[3, 2] .* V2[2, :]
norm(V2[3, :]) 
##
V2
U2[:, 1:3] * V2[1:3, :] - V
U2[:, 2:2] * V2[2:2, :]

##
A = rand(10)
norm(A)
