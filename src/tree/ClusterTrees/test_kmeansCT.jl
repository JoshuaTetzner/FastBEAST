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

N =  10000


points = [@SVector rand(3) for i = 1:N]
 
@time tree = create_CT_tree(points, maxlevel=7)
block_tree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
@time nears, fars = FastBEAST.computeinteractions(block_tree)
x=1
##

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, points[tdata], points[sdata])

near_blocks = getfullrankblocks(block_tree, nears, OneoverRkernelassembler)
@time far_row_blocks = row_top_down(block_tree, fars, OneoverRkernelassembler);
@time far_column_blocks = column_top_down(block_tree, fars, OneoverRkernelassembler);
test_basis, trial_basis = setup_nested_basis(far_row_blocks, far_column_blocks, block_tree)
interactions = setup_interactions(far_row_blocks, far_column_blocks, OneoverRkernelassembler, fars, block_tree)
##
solutions = Iterators.filter(y->interactions[y].M.B_U==Int[], interactions[1])
sol = collect(solutions)
##


@time kmat = assembler(OneoverRkernel, points, points);
function h2mat(points, OneoverRkernelassembler)
    @time tree = create_CT_tree(points, maxlevel=7)
    block_tree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
    @time nears, fars = FastBEAST.computeinteractions(block_tree)
    near_blocks = getfullrankblocks(block_tree, nears, OneoverRkernelassembler)
    @time far_row_blocks = row_top_down(block_tree, fars, OneoverRkernelassembler)
    @time far_column_blocks = column_top_down(block_tree, fars, OneoverRkernelassembler)
    test_basis, trial_basis = setup_nested_basis(far_row_blocks, far_column_blocks, block_tree)
    interactions = setup_interactions(far_row_blocks, far_column_blocks, OneoverRkernelassembler, fars, block_tree)

    return near_blocks, test_basis, trial_basis, interactions
end

function hmat(points, OneoverRkernelassembler)
    tree = create_tree(points, KMeansTreeOptions(maxlevel=6))
    M = HMatrix(
        OneoverRkernelassembler,
        tree,
        tree,
        Int64,
        Float64,
        compressor=FastBEAST.ACAOptions(tol=1e-4)
        )
end

@time h2mat(points, OneoverRkernelassembler);

@time hmat(points, OneoverRkernelassembler);
##

for i in interactions 
    if length(i.M.B_U) == 1
        teb = test_basis[i.M.B_U[1]].M.UV
        trb = trial_basis[i.M.B_V[1]].M.UV
        println(norm(kmat[i.τ, i.σ] - teb * i.M.Z * trb)/norm(kmat[i.τ, i.σ]))

    end
end

interactions[30].τ
##

blk = kmat[interactions[4].τ, interactions[4].σ]

i = interactions[4]
teb = [test_basis[i.M.B_U[1]].M.UV, test_basis[i.M.B_U[2]].M.UV]
trb = [trial_basis[i.M.B_V[1]].M.UV, trial_basis[i.M.B_V[2]].M.UV]

M = vcat(teb[1]*i.M.T_U[1],teb[2]*i.M.T_U[2]) * i.M.Z * hcat(i.M.T_V[1]*trb[1], i.M.T_V[2]*trb[2])
norm(M-blk)/norm(blk)
##
#





##
N =  10000

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, points[tdata], points[sdata])
points = [@SVector rand(3) for i = 1:N]
 
@time tree = create_CT_tree(points, maxlevel=7)
block_tree = ClusterTrees.BlockTrees.BlockTree(tree, tree)

typeof(block_tree)

@time H2Matrix(
    OneoverRkernelassembler,
    block_tree,
    Int,
    Float64
);