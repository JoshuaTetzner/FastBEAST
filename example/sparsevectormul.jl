using BEAST
using FastBEAST
using Printf
using LinearAlgebra
using StaticArrays
using SparseArrays
using BenchmarkTools

##
function logkernel(testpoint::SVector{2,T}, sourcepoint::SVector{2,T}) where T
    if isapprox(testpoint, sourcepoint, rtol=eps()*1e1)
        return 0.0
    else
        return - 2*π*log(norm(testpoint - sourcepoint))
    end
end

function assembler(kernel, testpoints, sourcepoints)
    kernelmatrix = zeros(
        promote_type(eltype(testpoints[1]),eltype(sourcepoints[1])), 
        length(testpoints), 
        length(sourcepoints)
    )

    for i = 1:length(testpoints)
        for j = 1:length(sourcepoints)
            kernelmatrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end

    return kernelmatrix
end

function assembler(kernel, matrix, testpoints, sourcepoints)
    for i = 1:length(testpoints)
        for j = 1:length(sourcepoints)
            matrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end
##
N = 2000
spoints = [@SVector rand(2) for i = 1:N]
sparsevector = sprand(Float64,N,0.001)
smat = sprand(Float64,N, N,0.01);

##
logkernelassembler(matrix, tdata, sdata) = assembler(
    logkernel, 
    matrix, 
    spoints[tdata], 
    spoints[sdata]
)

stree = create_tree(spoints, BoxTreeOptions(nmin=20))
@time kmat = assembler(logkernel, spoints, spoints)
@time hmat = HMatrix(logkernelassembler, stree, stree, Int64, Float64)

fullhmat = getfullmatrix(hmat, N, N)

@time sol = hmatmatrixmul(hmat, smat);
@time truesol = fullhmat*smat;


##
N =  10000
NT = 5000
sparsevector = Array(sprand(Float64,N,0.01))

spoints = [@SVector rand(2) for i = 1:N]
tpoints = 0.1*[@SVector rand(2) for i = 1:NT] + [SVector(1.0, 1.0) for i = 1:NT]

logkernelassembler(matrix, tdata, sdata) = assembler(
    logkernel,
    matrix,
    tpoints[tdata],
    spoints[sdata]
)

stree = create_tree(spoints, KMeansTreeOptions(iterations = 100, nmin=50000))
ttree = create_tree(tpoints, KMeansTreeOptions(iterations = 100, nmin=50000))

sparsevector = sprand(Float64,N,0.01)
@time solution = sparsevectormul2(
    colstomv(HMatrix(logkernelassembler, ttree, stree, T=Float64)),
    sparsevector
)
@time truesol = assembler(logkernel, tpoints, spoints)*sparsevector
println(norm(solution-truesol)/norm(truesol))

##
N = 1000;
spoints = [@SVector rand(2) for i = 1:N];
sparsevector = sprand(Float64,N,0.001);
kmat = assembler(logkernel, spoints, spoints);
##
logkernelassembler(matrix, tdata, sdata) = assembler(
    logkernel, 
    matrix, 
    spoints[tdata], 
    spoints[sdata]
)

stree = create_tree(spoints, KMeansTreeOptions(iterations=100, nmin=10))
hmat = HMatrix(logkernelassembler, stree, stree, T=Float64)

shmat = colstomv(hmat)
@btime solution = sparsevectormul(shmat, sparsevector)
@btime solution = sparsevectormul2(shmat, s[1,:])
truesol = kmat*sparsevector

##
function sp(
    shmat,
    smat
)
    sol = zeros(size(shmat.hmat)[1], size(shmat.hmat)[1])
    for i = 1:length(smat[:,1])
        sol[i,:] = sparsevectormul(shmat, smat[i,:])
    end
end

function sp2(
    shmat,
    smat
)
    sol = zeros(size(shmat.hmat)[1], size(shmat.hmat)[1])
    for i = 1:length(smat[:,1])
        sol[i,:] = sparsevectormul2(shmat, smat[i,:])
    end
end

##
N = 1000;
spoints = [@SVector rand(2) for i = 1:N];
smat = sprand(Float64,N, N,0.1);

logkernelassembler(matrix, tdata, sdata) = assembler(
    logkernel, 
    matrix, 
    spoints[tdata], 
    spoints[sdata]
)

stree = create_tree(spoints, KMeansTreeOptions(iterations=100, nmin=10))
hmat = HMatrix(logkernelassembler, stree, stree, Int64, Float64)
shmat = colstomv(hmat)
##
@time sp(shmat, smat)
@time sp2(shmat, smat)
@time sol = hmatmatrixmul(hmat, smat);
