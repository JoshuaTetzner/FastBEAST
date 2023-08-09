using FastBEAST
using StaticArrays


## Börm
using Test

N = 100
Gₜ = rand(N,N)

Q, Σ, P = svd(Gₜ)

Σ = [ i < 15 ? 10.0^(-i) : 0.0 for i = 1:N ]

# Matrices
Gₜ = Q * diagm(Σ) * P'
Gₜ₁ = Gₜ[1:50, :]
Gₜ₂ = Gₜ[51:100, :]
Q, Σ, P = svd(Gₜ)
Q₁, Σ₁, P₁ = svd(Gₜ₁)
Q₂, Σ₂, P₂ = svd(Gₜ₂)

# V leafs
Vₜ₁ = Q₁[:, 1:10]
Vₜ₂ = Q₂[:, 1:10]

# VV*Q = (q_1,...,q_k,0,...,0)
isapprox(norm((Vₜ₁*Vₜ₁'*Q₁)[:, 11:end])/norm(Vₜ₁*Vₜ₁'*Q₁), 0.0, atol=10^-15)
isapprox(norm((Vₜ₂*Vₜ₂'*Q₂)[:, 11:end])/norm(Vₜ₂*Vₜ₂'*Q₂), 0.0, atol=10^-15)

# X leafs
X₁ = Vₜ₁' * Gₜ₁
X₂ = Vₜ₂' * Gₜ₂

##
Gₜ_hat = zeros(Float64, size(X₁)[1] + size(X₂)[1], size(X₁)[2])
Gₜ_hat[1:size(X₁)[1], :] .= X₁
Gₜ_hat[size(X₁)[1]+1:end, :] .= X₂
Q_hat, Σ_hat, P_hat = svd(Gₜ_hat)
Vₜ_hat = Q_hat[:, 1:10]
Eₜ₁ = Vₜ_hat[1:size(Vₜ₁)[2], :]
Eₜ₂ = Vₜ_hat[size(Vₜ₁)[2]+1:end , :]
Xₜ = Vₜ_hat'*Gₜ_hat

helperV = zeros(Float64, size(Vₜ₁)[1]+size(Vₜ₂)[1], size(Vₜ₁)[2]+size(Vₜ₂)[2])
helperV[1:size(Vₜ₁)[1], 1:size(Vₜ₁)[2]] .= Vₜ₁
helperV[size(Vₜ₁)[1]+1:end, size(Vₜ₁)[2]+1:end] .= Vₜ₂

helperE = zeros(Float64, size(Eₜ₁)[1] + size(Eₜ₂)[1], size(Eₜ₁)[2]) 
helperE[1:size(Eₜ₁)[1], :] .= Eₜ₁
helperE[size(Eₜ₁)[1]+1:end, :] .= Eₜ₂

Vₜ_cluster = helperV*helperE

Vₜ_cluster + Q[:, 1:10]

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
##
@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

U,S,V = svd(A)

S = [ i < 15 ? 10.0^(-i) : 0.0 for i = 1:N ]

A = U*diagm(S)*V'

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

U, V = aca(lm)


##
norm(U*V-A)/norm(A)

##
rowindices = rand(1:100, 10)
colindices = rand(1:100, 10)

V = A[rowindices, :]
S = A[rowindices, colindices]
U = A[:, colindices]

Vinv = V[:, colindices]^-1
Uinv = U[rowindices, :]^-1

norm(U*Uinv*S*Vinv*V - A)/norm(A)
##
Ahelp = copy(A)
notconv = true

U = zeros(N, 10)
V = zeros(10, N)

rows = zeros(10)
cols = zeros(10)
for i = 1:10
    if i == 1
        nextrow = 1
    else
        nextrow = argmax(V[i-1, :])
    end
    rows[i] = nextrow
    U[:, i] = Ahelp[:, nextrow]

    nextcolumn = argmax(U[:, i])
    V[i, :] = Ahelp[nextcolumn, :]

    Ahelp = Ahelp -  U[1:end, i:i] * Ahelp[nextcolumn, nextrow]^-1 *V[i:i, 1:end]
    println(nextcolumn)
end 

##
A = rand(10, 10)

norm(A - A * A^-1*A)