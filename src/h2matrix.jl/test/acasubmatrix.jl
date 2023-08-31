using FastBEAST
using LinearAlgebra

A = rand(100000,100)

U,S,V = svd(A)
S .= 0
S[1:30] = [10.0^(-i) for i = 1:30 ]
A = U*diagm(S)*V'


@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

U, V = aca(lm, maxrank=100, tol=1e-4)

norm(U*V-A)/norm(A)