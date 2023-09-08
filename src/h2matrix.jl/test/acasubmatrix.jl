using FastBEAST
using LinearAlgebra

A = rand(1000,1000)

U,S,V = svd(A)
S .= 0
S[1:30] = [10.0^(-i) for i = 1:30 ]
A = U*diagm(S)*V'

rc = Array(250:750)

Asub = A[rc, rc] 

@views function fct(B, x, y)
    B[:,:] = Asub[x, y]
end

lm = LazyMatrix(fct, Vector(1:size(Asub, 1)), Vector(1:size(Asub, 2)), Float64)

retU, retV, U, V, rowindices, colindices = nca(lm, maxrank=100, tol=1e-4, svdrecompress=false)


@views function fct2(B, x, y)
    B[:,:] = A[x, y]
end

lm2 = LazyMatrix(fct2, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)
Utrue, Vtrue = aca(lm2, maxrank=100, tol=1e-4)

V = A[rowindices, :]
U = A[:, colindices]
I = A[rowindices, colindices]^-1
norm(U*I*V-A)/norm(A)

norm(Utrue*Vtrue-A)/norm(A)


