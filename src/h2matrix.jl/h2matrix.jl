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

Base.size(lrm::ClusterMatrix) = (size(lrm.U,1), size(lrm.V,2))

#storage for pivoting
struct Pivotlrb{F}
    M::ClusterMatrix{F} 
    τ::Vector{Int}
    σ::Vector{Int}
    roworcolidcs::Vector{Int}
    roworcolmap::Vector{UnitRange{Int}}
    children::Vector{Tuple{Int, UnitRange{Int}}}
end

struct H2MatrixBlock{I, K}
    Z::MatrixBlock{I, K, Matrix{K}}
    τ::Vector{I}
    σ::Vector{I}
    row_basis::I#Vector{I}
    col_basis::I#Vector{I}
end

struct H2BasisBlock{I, K}
    T::Union{Vector{Matrix{K}}, Matrix{K}}
    children::Vector{I}
end

struct H2Matrix{I, K} <: LinearMaps.LinearMap{K}
    fullrankblocks::Vector{MatrixBlock{I, K, Matrix{K}}}
    lowrankblocks::Vector{H2MatrixBlock{I, K}}
    nestedtestbases::Vector{H2BasisBlock{I, K}}
    nestedtrialbases::Vector{H2BasisBlock{I, K}}
    fars
    rowdim::I
    columndim::I
end
function H2Matrix(
    matrixassembler::Function,
    tree::ClusterTrees.BlockTrees.BlockTree{T},
) where {T}

    nears, fars = computeinteractions(tree)

    fullrankblocks = getfullrankblocks(
        tree, 
        nears,
        matrixassembler
    )

    @time test_fars = test_top_down_pivots(tree, fars, matrixassembler)
    @time trial_fars = trial_top_down_pivots(tree, fars, matrixassembler)

    lowrankblocks = Vector{H2MatrixBlock{Int, Float64}}(undef, sum([length(far) for far in fars]))
    
    ind = 0
    for levelfars in fars
        for far in levelfars
            ind += 1
            #row_children = [child[1] for child in test_fars[far[1]].children]
            #col_children = [child[1] for child in trial_fars[far[2]].children]
            idx = findfirst(x->x==far[2], test_fars[far[1]].roworcolidcs)
            range = test_fars[far[1]].roworcolmap[idx]
            
            lowrankblocks[ind] = H2MatrixBlock(
                MatrixBlock(
                    test_fars[far[1]].M.V[:, range][:, trial_fars[far[2]].M.σ],
                    test_fars[far[1]].τ[test_fars[far[1]].M.τ],
                    trial_fars[far[2]].σ[trial_fars[far[2]].M.σ]
                ),
                test_fars[far[1]].τ,
                trial_fars[far[2]].σ,
                far[1],#row_children,
                far[2]#col_children
            )
        end
    end

    level = []
    for (i, far) in enumerate(fars[1:end-1])
        if far != []
            push!(level, i)
        end
    end

    test_basis = build_test_bases(tree.test_cluster, fars, level, test_fars)
    trial_basis = build_trial_bases(tree.trial_cluster, fars, level, trial_fars)
    rowdim = length(value(tree.test_cluster, 1))
    coldim = length(value(tree.trial_cluster, 1))
    return H2Matrix(
        fullrankblocks, lowrankblocks, test_basis, trial_basis, fars, rowdim, coldim
    )
end


function build_test_bases(
    tree::KMeansTree{T},
    fars,
    levels,
    test_fars
) where T

    maxlevel = length(tree.levels)
    test_basis = Vector{H2BasisBlock{Int, Float64}}(undef, length(tree.nodes))

    ## nested test basis
    for nodeidx in FastBEAST.clusterlink(tree, node=root(tree), target=maxlevel)
        test_basis[nodeidx] = H2BasisBlock(
            test_fars[nodeidx].M.U * test_fars[nodeidx].M.U[test_fars[nodeidx].M.τ, :]^-1,
            Int[]
        )
    end
    
    #transfer matrices
    for level in Iterators.reverse(levels)
        sfars = sort_interactions(fars[level], testortrial=1)
        for (nodeidx, _) in sfars
            transfer = Vector{Matrix{Float64}}(undef, length(test_fars[nodeidx[1]].children))
            children = zeros(Int, length(test_fars[nodeidx[1]].children))

            for (ind, (child, range)) in enumerate(test_fars[nodeidx[1]].children)
                transfer[ind] = test_fars[nodeidx[1]].M.U[range, :][test_fars[child].M.τ, :] * 
                    test_fars[nodeidx[1]].M.U[test_fars[nodeidx[1]].M.τ, :]^-1
                children[ind] = child
            end
            test_basis[nodeidx[1]] = H2BasisBlock(
                transfer,
                children
            )
        end

    end

    return test_basis
end


function build_trial_bases(
    tree::KMeansTree{T},
    fars,
    levels,
    trial_fars
) where T

    maxlevel = length(tree.levels)
    trial_basis = Vector{H2BasisBlock{Int, Float64}}(undef, length(tree.nodes))

    ## nested test basis
    for nodeidx in FastBEAST.clusterlink(tree, node=root(tree), target=maxlevel)
        trial_basis[nodeidx] = H2BasisBlock(
            trial_fars[nodeidx].M.V[:, trial_fars[nodeidx].M.σ]^-1 * trial_fars[nodeidx].M.V,
            Int[]
        )
    end
    
    #transfer matrices
    for level in Iterators.reverse(levels)
        sfars = sort_interactions(fars[level], testortrial=2)
        for (_, nodeidx) in sfars
            transfer = Vector{Matrix{Float64}}(undef, length(trial_fars[nodeidx[1]].children))
            children = zeros(Int, length(trial_fars[nodeidx[1]].children))

            for (ind, (child, range)) in enumerate(trial_fars[nodeidx[1]].children)
                transfer[ind] = trial_fars[nodeidx[1]].M.V[:, trial_fars[nodeidx[1]].M.σ]^-1 * 
                    trial_fars[nodeidx[1]].M.V[:, range][:, trial_fars[child].M.σ] 
                    
                children[ind] = child
            end
            trial_basis[nodeidx[1]] = H2BasisBlock(
                transfer,
                children
            )
        end

    end

    return trial_basis
end


function test_top_down_pivots(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    fars,
    matrixassembler;
    compressor=FastBEAST.ACAOptions(tol=1e-4)
) where T

    test_lowrankblocks = Vector{Pivotlrb{Float64}}(undef, length(block_tree.test_cluster.nodes))

    for level_fars in fars
        if level_fars != []

            sorted_fars = sort_interactions(level_fars, testortrial=1)
            
            for sfar in sorted_fars
                c = sfar[1][1]
                childidcs = FastBEAST.childidcs(block_tree.test_cluster, c)
                childrange = Tuple{Int, UnitRange{Int}}[]
                if ClusterTrees.haschildren(block_tree.test_cluster, c)
                    childidcs = FastBEAST.childidcs(block_tree.test_cluster, c)
                    
                    idxcounter = 1
                    for childidx in childidcs
                        push!(childrange, (childidx, idxcounter:(idxcounter+length(value(block_tree.test_cluster, childidx))-1)))
                        idxcounter += length(value(block_tree.test_cluster, childidx))
                    end
                end

                row_set = value(block_tree.test_cluster, c)
                col_set = Int[]
                colmaps = UnitRange{Int}[]
                idx = 1

                for adm_blk in sfar[2]
                    append!(col_set, value(block_tree.trial_cluster, adm_blk))
                    push!(colmaps, idx:(idx+length(value(block_tree.trial_cluster, adm_blk))-1))
                    idx += length(value(block_tree.trial_cluster, adm_blk))
                end
                
                if ClusterTrees.parent(block_tree.test_cluster, c) != 0 &&
                    isassigned(
                        test_lowrankblocks, ClusterTrees.parent(block_tree.test_cluster, c)
                    )
                    parent = ClusterTrees.parent(block_tree.test_cluster, c)
                    append!(col_set, test_lowrankblocks[parent].σ[test_lowrankblocks[parent].M.σ])
                end

                test_lowrankblocks[c] = getcompressedmatrix(
                    row_set, col_set, sfar[2],colmaps, childrange, matrixassembler, compressor=compressor
                )
            end
        end
    end

    return test_lowrankblocks
end


function trial_top_down_pivots(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    fars,
    matrixassembler;
    compressor=FastBEAST.ACAOptions(tol=1e-4)
) where T

    trial_lowrankblocks = Vector{Pivotlrb{Float64}}(undef, length(block_tree.trial_cluster.nodes))

    for level_fars in fars
        if level_fars != []

            sorted_fars = sort_interactions(level_fars, testortrial=2)
            
            for sfar in sorted_fars
                c = sfar[2][1]
                childidcs = FastBEAST.childidcs(block_tree.trial_cluster, c)
                childrange = Tuple{Int, UnitRange{Int}}[]
                if ClusterTrees.haschildren(block_tree.trial_cluster, c)
                    childidcs = FastBEAST.childidcs(block_tree.trial_cluster, c)
                    
                    idxcounter = 1
                    for childidx in childidcs
                        push!(childrange, (childidx, idxcounter:(idxcounter+length(value(block_tree.trial_cluster, childidx))-1)))
                        idxcounter += length(value(block_tree.trial_cluster, childidx))
                    end
                end

                col_set = value(block_tree.trial_cluster, c)
                row_set = Int[]
                rowmaps = UnitRange{Int}[]
                idx = 1

                for adm_blk in sfar[1]
                    append!(row_set, value(block_tree.test_cluster, adm_blk))
                    push!(rowmaps, idx:(idx+length(value(block_tree.test_cluster, adm_blk))-1))
                    idx += length(value(block_tree.test_cluster, adm_blk))
                end
                
                if ClusterTrees.parent(block_tree.trial_cluster, c) != 0 &&
                    isassigned(
                        trial_lowrankblocks, ClusterTrees.parent(block_tree.trial_cluster, c)
                    )
                    parent = ClusterTrees.parent(block_tree.trial_cluster, c)
                    append!(row_set, trial_lowrankblocks[parent].τ[trial_lowrankblocks[parent].M.τ])
                end

                trial_lowrankblocks[c] = getcompressedmatrix(
                    row_set, col_set, sfar[1], rowmaps, childrange, matrixassembler, compressor=compressor
                )
            end
        end
    end

    return trial_lowrankblocks
end
