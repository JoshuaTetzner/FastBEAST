using Test
using FastBEAST
using StaticArrays
using LinearAlgebra
using ClusterTrees
 
##
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


Nt=200
points = [@SVector rand(3) for i = 1:Nt]




##
points2 = [((@SVector rand(3)))*2 + SVector(2, 0, 0) for i = 1:Nt]
for x = [2, 3] 
    for y = [0, 1, 2, 3]
        append!(points2, [(@SVector rand(3)) + SVector(x, y, 0) for i = 1:Nt])
    end
end

for x = [0, 1, 2, 3]
    for y = [2, 3]
        append!(points2, [(@SVector rand(3)) + SVector(x, y, 0) for i = 1:Nt])
    end
end

##far field
Nfar=10
for y = [0, 2, 4] 
    append!(points2, [(@SVector rand(3)) + SVector(4, y, 0) for i = 1:Nfar])
end
for x = [0, 2] 
    append!(points2, [(@SVector rand(3)) + SVector(x, 4, 0) for i = 1:Nfar])
end
##
@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, points[tdata], points[sdata])

A = assembler(OneoverRkernel, points2, points)

@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

am = allocate_aca_memory(Float64, size(lm, 1), size(lm, 2); maxrank=30)

filld = FastBEAST.FillDistance(points2)
tfd = FastBEAST.TrueFillDistance(points2)

# classic aca filldistance
@time U, V = aca(lm, rowpivstrat=filld, tol=1e-4);
norm(U*V-A)/norm(A)

##
@time U, V = aca(lm, tol=1e-4);
norm(U*V-A)/norm(A)


##


## rows
piv = FastBEAST.FillDistance(points2)
used = zeros(Bool, length(points2))
piv, first = FastBEAST.pivoting(piv, [], used, Vector(1:length(used)))
used[first] = true
rows = [first]
for i = 1:15
    push!(rows, FastBEAST.pivoting(piv, [], used, Vector(1:length(used))))
    used[rows[end]] = true
end

##cols
f = 1-15/length(points2)

_,truesv,_ = svd(A)
(truesv./truesv[1])[1:20]

_,sv,_ = svd(A[rows, 1:16])
(sv./sv[1])[1:15] #+ (sv./sv[1])[1:15]*f 

##
using Plots
tsv=truesv[1:16]
plot(Vector(1:16), sv./sv[1], yaxis=:log)
plot!(Vector(1:16), tsv./tsv[1], yaxis=:log)

sv
tsv
##tsv

function partialACA( 
    M::FastBEAST.LazyMatrix{I, K},
    fullmat,
    am;
    rowpivstrat=FastBEAST.MaxPivoting(1),
    columnpivstrat=FastBEAST.MaxPivoting(1),
    tol=1e-14
) where {I, K} 

    (maxrows, maxcolumns) = size(M)
    
    nrows=1
    ncols=1
    rows = []
    cols = []

    rowpivstrat, nextrow = FastBEAST.firstindex(rowpivstrat, abs.(am.U[1:maxrows, am.Jc]), M.τ)
    am.used_I[nextrow] = true
    push!(rows, nextrow)

    columnpivstrat, nextcolumn = FastBEAST.firstindex(
        columnpivstrat, abs.(am.V[am.Ic, 1:maxcolumns]), M.σ
    )
    push!(cols, nextcolumn)
    am.used_J[nextcolumn] = true
    
    M.μ(
        fullmat[1:nrows, 1:ncols], 
        rows,
        cols
    )

    for i = 1:20
        nextrow = FastBEAST.pivoting(
            rowpivstrat,
            abs.(am.U[1:maxrows, am.Jc]),
            am.used_I[1:maxrows],
            M.τ
        )
        nrows += 1
        am.used_I[nextrow] = true
        push!(rows, nextrow)

        @views M.μ(
            fullmat[nrows, 1:ncols], 
            [nextrow],
            cols
        )

        nextcolumn = FastBEAST.pivoting(
            columnpivstrat,
            abs.(am.V[am.Ic, 1:maxcolumns]),
            am.used_J[1:maxcolumns],
            M.σ
        )
        ncols += 1
        am.used_J[nextcolumn] = true
        push!(cols, nextcolumn)

        @views M.μ(
            fullmat[1:nrows, ncols], 
            rows,
            [nextcolumn]
        )

        _, sv, _ = svd(fullmat[1:nrows, 1:ncols])

        println(sv[end]/sv[1])
    end
    return rows, cols, fullmat
end

fullmat = zeros(Float64, 21, 21)
a, b, M = partialACA(
    lm, fullmat, am; rowpivstrat=filld, columnpivstrat = FastBEAST.FillDistance(points)
)
    
a
b

fullmat
_,sv,_ = svd(A[:, b])



sv ./ sv[1]