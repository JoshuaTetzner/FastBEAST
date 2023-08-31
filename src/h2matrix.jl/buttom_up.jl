using LinearAlgebra
using LinearMaps
using ClusterTrees
using FastBEAST


struct ClusterMatrix{F} <: LinearMaps.LinearMap{F}
    U::Matrix{F}
    V::Matrix{F}
    τ::Vector{Int}
    σ::Vector{Int}
end


Base.size(lrm::ClusterMatrix) = (size(lrm.U,1), size(lrm.V,2))


function buttom_up(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    fars::Vector{Vector{Tuple{Int, Int}}},
    farmatrixassembler;
    compressor=FastBEAST.ACAOptions()
) where T

    MBL = MatrixBlock{Int, Float64, ClusterMatrix{Float64}}
    test_lowrankblocks = Vector{MBL}(undef, length(block_tree.test_cluster.nodes))
    trial_lowrankblocks = Vector{MBL}(undef, length(block_tree.test_cluster.nodes))

    leaf_blocks!(
        block_tree,
        sort_interactions(fars[end], testortrial=1),
        test_lowrankblocks,
        1,
        farmatrixassembler,
        compressor=compressor
    )
    leaf_blocks!(
        block_tree,
        sort_interactions(fars[end], testortrial=2),
        trial_lowrankblocks,
        2,
        farmatrixassembler,
        compressor=compressor
    )

    for far in reverse(fars[1:end-1])
        far == [] && break;
        testnode_blocks!(
            block_tree,
            sort_interactions(far, testortrial=1),
            test_lowrankblocks,
            1,
            farmatrixassembler,
            compressor=compressor
        )
        testnode_blocks!(
            block_tree,
            sort_interactions(far, testortrial=2),
            trial_lowrankblocks,
            1,
            farmatrixassembler,
            compressor=compressor
        )
    end

    return test_lowrankblocks, trial_lowrankblocks
end


function testnode_blocks!(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    sorted_nodes::Vector{Tuple{Vector{Int}, Vector{Int}}},
    lowrankblocks::Vector{MatrixBlock{Int, F, ClusterMatrix{Float64}}},
    testortrial::Int,
    matrixassembler;
    compressor=FastBEAST.ACAOptions()
) where {F, T}

    for node in sorted_nodes
        test_idcs = Int[]
        trial_idcs = Int[]

        for testnode_idx in node[1]
            for chd in ClusterTrees.children(block_tree.test_cluster, testnode_idx)
                append!(
                    test_idcs,
                    lowrankblocks[chd].M.τ
                )
            end
        end

        for trialnode_idx in node[2]
            for leave in ClusterTrees.leaves(block_tree.trial_cluster, trialnode_idx)
                append!(
                    trial_idcs,
                    vals(block_tree.trial_cluster, leave)
                )
            end
        end
        
        lowrankblocks[node[testortrial][1]] = (getcompressedmatrix(
            test_idcs, trial_idcs, matrixassembler, compressor=compressor
        ))
    end
end


function trialnode_blocks!(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    sorted_nodes::Vector{Tuple{Vector{Int}, Vector{Int}}},
    lowrankblocks::Vector{MatrixBlock{Int, F, ClusterMatrix{Float64}}},
    testortrial::Int,
    matrixassembler;
    compressor=FastBEAST.ACAOptions()
) where {F, T}

    testortrial == 1 ? trialortest = 2 : trialortest = 1
    for node in sorted_nodes
        test_idcs = Int[]
        trial_idcs = Int[]

        for testnode_idx in node[1]
            for leave in ClusterTrees.leaves(block_tree.test_cluster, testnode_idx)
                append!(
                    test_idcs,
                    vals(block_tree.test_cluster, leave)
                )
            end
        end

        for trialnode_idx in node[2]
            for chd in ClusterTrees.children(block_tree.trial_cluster, trialnode_idx)
                append!(
                    trial_idcs,
                    lowrankblocks[chd].M.τ
                )
            end
        end

        lowrankblocks[node[testortrial][1]] = (getcompressedmatrix(
            test_idcs, trial_idcs, matrixassembler, compressor=compressor
        ))
    end
end


function leaf_blocks!(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    sorted_nodes::Vector{Tuple{Vector{Int}, Vector{Int}}},
    lowrankblocks::Vector{MatrixBlock{Int, F, ClusterMatrix{Float64}}},
    testortrial::Int,
    matrixassembler;
    compressor=FastBEAST.ACAOptions()
) where {F, T}
    
    for node in sorted_nodes
        test_idcs = Int[]
        trial_idcs = Int[]
        
        for testnode_idx in node[1]
            append!(
                test_idcs,
                vals(ClusterTrees.BlockTrees.testcluster(block_tree), testnode_idx)
            )
        end

        for trialnode_idx in node[2]
            append!(
                trial_idcs,
                vals(ClusterTrees.BlockTrees.trialcluster(block_tree), trialnode_idx)
            )
        end

        lowrankblocks[node[testortrial][1]] = (getcompressedmatrix(
            test_idcs, trial_idcs, matrixassembler, compressor=compressor
        ))
    end
end


function getcompressedmatrix(
    test_idcs::Vector{Int},
    trial_idcs::Vector{Int},
    matrixassembler;
    compressor=FastBEAST.ACAOptions(tol=1e-4)
)
    lm = FastBEAST.LazyMatrix(
        matrixassembler,
        test_idcs,
        trial_idcs,
        Float64
    )

    am = allocate_aca_memory(
        Float64,
        length(test_idcs),
        length(trial_idcs),
        maxrank=Int(round(minimum([length(test_idcs), length(trial_idcs)])/2))
    )

    retU, retV, U, V, rowindices, colindices= nca(
        lm,
        am;
        rowpivstrat=compressor.rowpivstrat,
        columnpivstrat=compressor.columnpivstrat,
        tol=compressor.tol,
        svdrecompress=compressor.svdrecompress
    )

    return FastBEAST.MatrixBlock{Int, Float64, ClusterMatrix{Float64}}(
        ClusterMatrix(retU, retV, rowindices, colindices),
        test_idcs,
        trial_idcs
    )
end


function sort_interactions(
    fars::Vector{Tuple{Int, Int}};
    testortrial = 1
)
    
    testortrial == 1 ? trialortest = 2 : trialortest = 1
    
    sortedfars = Tuple{Vector{Int}, Vector{Int}}[]
    sfars = sort(fars, by = x -> x[testortrial])

    push!(sortedfars, ([sfars[1][1]],[sfars[1][2]]))
    for n ∈ 2:length(fars)
        if sfars[n][testortrial] == sfars[n-1][testortrial]
            push!(sortedfars[end][trialortest], sfars[n][trialortest])
        else
            push!(sortedfars, ([sfars[n][1]], [sfars[n][2]]))
        end
    end

    return sortedfars
end