using BEAST
using FastBEAST
using ClusterTrees
using LinearMaps

#storage for pivoting
struct ClusterMatrix{F} <: LinearMaps.LinearMap{F}
    U::Matrix{F}
    V::Matrix{F}
    τ::Vector{Int}
    σ::Vector{Int}
end

Base.size(lrm::ClusterMatrix) = (size(lrm.U, 1), size(lrm.V, 2))

#storage for pivoting
struct Pivotlrb{F}
    M::ClusterMatrix{F}
    τ::Vector{Int}
    σ::Vector{Int}
    roworcolidcs::Vector{Int}
    roworcolmap::Vector{UnitRange{Int}}
    children::Vector{Tuple{Int,UnitRange{Int}}}
end

struct H2MatrixBlock{I,K}
    Z::MatrixBlock{I,K,Matrix{K}}
    τ::Vector{I}
    σ::Vector{I}
    row_basis::I#Vector{I}
    col_basis::I#Vector{I}
end

struct H2BasisBlock{I,K}
    T::Union{Vector{Matrix{K}},Matrix{K}}
    children::Vector{I}
end

struct H2Matrix{I,K} <: LinearMaps.LinearMap{K}
    fullrankblocks::Vector{MatrixBlock{I,K,Matrix{K}}}
    lowrankblocks::Vector{H2MatrixBlock{I,K}}
    nestedtestbases::Vector{H2BasisBlock{I,K}}
    nestedtrialbases::Vector{H2BasisBlock{I,K}}
    fars
    rowdim::I
    columndim::I
end

function H2Matrix(
    matrixassembler::Function,
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    ::Type{K};
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
) where {T,K}
    nears, fars = computeinteractions(tree)
    println("Computing near interactions")
    fullrankblocks = getfullrankblocks(tree, nears, matrixassembler, K)

    println("Pivot selection")
    test_fars = test_top_down_pivots(
        tree, fars, matrixassembler, K; compressor=compressor
    )
    trial_fars = trial_top_down_pivots(
        tree, fars, matrixassembler, K; compressor=compressor
    )

    lowrankblocks = Vector{H2MatrixBlock{Int,K}}(undef, sum([length(far) for far in fars]))

    println("Compute blocks")
    ind = 0
    for levelfars in fars
        for far in levelfars
            ind += 1

            blk = zeros(
                K,
                length(test_fars[far[1]].τ[test_fars[far[1]].M.τ]),
                length(trial_fars[far[2]].σ[trial_fars[far[2]].M.σ]),
            )
            matrixassembler(
                blk,
                test_fars[far[1]].τ[test_fars[far[1]].M.τ],
                trial_fars[far[2]].σ[trial_fars[far[2]].M.σ],
            )
            lowrankblocks[ind] = H2MatrixBlock(
                MatrixBlock(
                    blk,
                    test_fars[far[1]].τ[test_fars[far[1]].M.τ],
                    trial_fars[far[2]].σ[trial_fars[far[2]].M.σ],
                ),
                test_fars[far[1]].τ,
                trial_fars[far[2]].σ,
                far[1],
                far[2],
            )
        end
    end

    println("Build Basis")
    test_basis = build_test_bases(tree.test_cluster, test_fars, K)
    trial_basis = build_trial_bases(tree.trial_cluster, trial_fars, K)
    rowdim = length(value(tree.test_cluster, 1))
    coldim = length(value(tree.trial_cluster, 1))

    return H2Matrix(
        fullrankblocks, lowrankblocks, test_basis, trial_basis, fars, rowdim, coldim
    )
end

function build_test_bases(tree::KMeansTree{T}, test_fars, ::Type{K}) where {T,K}
    test_basis = Vector{H2BasisBlock{Int,K}}(undef, length(tree.nodes))

    for idx in eachindex(test_fars)
        if isassigned(test_fars, idx)
            testfar = test_fars[idx]
            if testfar.children == []
                test_basis[idx] = H2BasisBlock(
                    test_fars[idx].M.U * test_fars[idx].M.U[test_fars[idx].M.τ, :]^-1, Int[]
                )
            else
                transfer = Vector{Matrix{K}}(undef, length(testfar.children))
                children = zeros(Int, length(testfar.children))

                for (ind, (child, range)) in enumerate(testfar.children)
                    transfer[ind] =
                        testfar.M.U[range, :][test_fars[child].M.τ, :] *
                        testfar.M.U[testfar.M.τ, :]^-1
                    children[ind] = child
                end

                test_basis[idx] = H2BasisBlock(transfer, children)
            end
        end
    end

    return test_basis
end

function build_trial_bases(tree::KMeansTree{T}, trial_fars, ::Type{K}) where {T,K}
    trial_basis = Vector{H2BasisBlock{Int,K}}(undef, length(tree.nodes))

    for idx in eachindex(trial_fars)
        if isassigned(trial_fars, idx)
            trialfar = trial_fars[idx]
            if trialfar.children == []
                trial_basis[idx] = H2BasisBlock(
                    trial_fars[idx].M.V[:, trial_fars[idx].M.σ]^-1 * trial_fars[idx].M.V,
                    Int[],
                )
            else
                transfer = Vector{Matrix{K}}(undef, length(trialfar.children))
                children = zeros(Int, length(trialfar.children))

                for (ind, (child, range)) in enumerate(trialfar.children)
                    transfer[ind] =
                        trialfar.M.V[:, trialfar.M.σ]^-1 *
                        trialfar.M.V[:, range][:, trial_fars[child].M.σ]

                    children[ind] = child
                end
                trial_basis[idx] = H2BasisBlock(transfer, children)
            end
        end
    end

    return trial_basis
end

function test_top_down_pivots(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    fars,
    matrixassembler,
    ::Type{K};
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
) where {T,K}
    test_lowrankblocks = Vector{Pivotlrb{K}}(undef, length(block_tree.test_cluster.nodes))

    interactionlist = sort_interactions_tree(
        length(block_tree.test_cluster.nodes), fars; testortrial=1
    )
    for nodeidx in eachindex(interactionlist)
        childidcs = FastBEAST.childidcs(block_tree.test_cluster, nodeidx)
        childrange = Tuple{Int,UnitRange{Int}}[]
        if ClusterTrees.haschildren(block_tree.test_cluster, nodeidx)
            childidcs = FastBEAST.childidcs(block_tree.test_cluster, nodeidx)

            idxcounter = 1
            for childidx in childidcs
                push!(
                    childrange,
                    (
                        childidx,
                        idxcounter:(
                            idxcounter + length(value(block_tree.test_cluster, childidx)) - 1
                        ),
                    ),
                )
                idxcounter += length(value(block_tree.test_cluster, childidx))
            end
        end

        col_set = Int[]
        colmaps = UnitRange{Int}[]
        idx = 1
        if interactionlist[nodeidx] != []
            for adm_blk in interactionlist[nodeidx]
                append!(col_set, value(block_tree.trial_cluster, adm_blk))
                push!(
                    colmaps,
                    idx:(idx + length(value(block_tree.trial_cluster, adm_blk)) - 1),
                )
                idx += length(value(block_tree.trial_cluster, adm_blk))
            end
        end
        if ClusterTrees.parent(block_tree.test_cluster, nodeidx) != 0 && isassigned(
            test_lowrankblocks, ClusterTrees.parent(block_tree.test_cluster, nodeidx)
        )
            parent = ClusterTrees.parent(block_tree.test_cluster, nodeidx)
            append!(col_set, test_lowrankblocks[parent].σ[test_lowrankblocks[parent].M.σ])
        end
        if col_set != []
            row_set = value(block_tree.test_cluster, nodeidx)

            test_lowrankblocks[nodeidx] = getcompressedmatrix(
                row_set,
                col_set,
                [nodeidx],
                colmaps,
                childrange,
                matrixassembler,
                K;
                compressor=compressor
            )
        end
    end

    return test_lowrankblocks
end

function trial_top_down_pivots(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    fars,
    matrixassembler,
    ::Type{K};
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
) where {T,K}
    trial_lowrankblocks = Vector{Pivotlrb{K}}(undef, length(block_tree.trial_cluster.nodes))

    interactionlist = sort_interactions_tree(
        length(block_tree.trial_cluster.nodes), fars; testortrial=2
    )

    for nodeidx in eachindex(interactionlist)
        childidcs = FastBEAST.childidcs(block_tree.trial_cluster, nodeidx)
        childrange = Tuple{Int,UnitRange{Int}}[]
        if ClusterTrees.haschildren(block_tree.trial_cluster, nodeidx)
            childidcs = FastBEAST.childidcs(block_tree.trial_cluster, nodeidx)

            idxcounter = 1
            for childidx in childidcs
                push!(
                    childrange,
                    (
                        childidx,
                        idxcounter:(
                            idxcounter + length(value(block_tree.trial_cluster, childidx)) - 1
                        ),
                    ),
                )
                idxcounter += length(value(block_tree.trial_cluster, childidx))
            end
        end

        row_set = Int[]
        rowmaps = UnitRange{Int}[]
        idx = 1
        if interactionlist[nodeidx] != []
            for adm_blk in interactionlist[nodeidx]
                append!(row_set, value(block_tree.test_cluster, adm_blk))
                push!(
                    rowmaps, idx:(idx + length(value(block_tree.test_cluster, adm_blk)) - 1)
                )
                idx += length(value(block_tree.test_cluster, adm_blk))
            end
        end
        if ClusterTrees.parent(block_tree.trial_cluster, nodeidx) != 0 && isassigned(
            trial_lowrankblocks, ClusterTrees.parent(block_tree.trial_cluster, nodeidx)
        )
            parent = ClusterTrees.parent(block_tree.trial_cluster, nodeidx)
            append!(row_set, trial_lowrankblocks[parent].τ[trial_lowrankblocks[parent].M.τ])
        end

        if row_set != []
            col_set = value(block_tree.trial_cluster, nodeidx)
            trial_lowrankblocks[nodeidx] = getcompressedmatrix(
                row_set,
                col_set,
                [nodeidx],
                rowmaps,
                childrange,
                matrixassembler,
                K;
                compressor=compressor
            )
        end
    end

    return trial_lowrankblocks
end
