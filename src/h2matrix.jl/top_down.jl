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


struct ClusterBasis{F} <: LinearMaps.LinearMap{F}
    UV::Matrix{F}
    i::Vector{Int}
end

Base.size(cb::ClusterBasis) = size(cb.UV)

struct NestedNode{F} <: LinearMaps.LinearMap{F}
    B_U::Vector{Int}
    B_V::Vector{Int}
    T_U::Vector{Matrix{F}}
    T_V::Vector{Matrix{F}}
    Z::Matrix{F}
end

Base.size(cn::NestedNode) = (size(cn.T_U, 1), size(cn.T_V, 2))


function row_top_down(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    fars,
    matrixassembler;
    compressor=FastBEAST.ACAOptions(tol=1e-4)
) where T

    MBL = MatrixBlock{Int, Float64, ClusterMatrix{Float64}}
    test_lowrankblocks = Vector{MBL}(undef, length(block_tree.test_cluster.nodes))

    for level_fars in fars
        if level_fars != []

            sorted_fars = sort_interactions(level_fars, testortrial=1)
            for sfar in sorted_fars
                c = sfar[1][1]
                row_set = value(block_tree.test_cluster, c)
                col_set = Int[]

                for adm_blk in sfar[2]
                    append!(col_set, value(block_tree.trial_cluster, adm_blk))
                end
                
                if ClusterTrees.parent(block_tree.test_cluster, c) != 0 &&
                    isassigned(
                        test_lowrankblocks, ClusterTrees.parent(block_tree.test_cluster, c)
                    )
                    append!(col_set, test_lowrankblocks[
                        ClusterTrees.parent(block_tree.test_cluster, c)
                    ].M.σ)
                end
                test_lowrankblocks[c] = getcompressedmatrix(
                    row_set, col_set, matrixassembler, compressor=compressor
                ) 
            end
        end
    end

    return test_lowrankblocks
end


function column_top_down(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    fars,
    matrixassembler;
    compressor=FastBEAST.ACAOptions(tol=1e-4)
) where T

    MBL = MatrixBlock{Int, Float64, ClusterMatrix{Float64}}
    trial_lowrankblocks = Vector{MBL}(undef, length(block_tree.trial_cluster.nodes))

    for level_fars in fars
        if level_fars != []

            sorted_fars = sort_interactions(level_fars, testortrial=2)
            for sfar in sorted_fars
                c = sfar[2][1]
                col_set = value(block_tree.trial_cluster, c)
                row_set = Int[]
                

                for adm_blk in sfar[1]
                    append!(row_set, value(block_tree.test_cluster, adm_blk))
                end
                
                if ClusterTrees.parent(block_tree.trial_cluster, c) != 0 &&
                    isassigned(
                        trial_lowrankblocks, ClusterTrees.parent(block_tree.trial_cluster, c)
                    )
                    
                    append!(row_set, trial_lowrankblocks[
                        ClusterTrees.parent(block_tree.trial_cluster, c)
                    ].M.τ)
                end
                
                trial_lowrankblocks[c] = getcompressedmatrix(
                    row_set, col_set , matrixassembler, compressor=compressor
                ) 
            end
        end
    end
    return trial_lowrankblocks

end


function setup_nested_basis(
    far_row_blocks,
    far_column_blocks,
    block_tree
)   
    test_clusterbasis = Vector{MatrixBlock{Int, Float64, ClusterBasis{Float64}}}(
        undef, length(block_tree.test_cluster.nodes)
    )
    trial_clusterbasis = Vector{MatrixBlock{Int, Float64, ClusterBasis{Float64}}}(
        undef, length(block_tree.trial_cluster.nodes)
    )
    
    for leave ∈ ClusterTrees.leaves(block_tree.test_cluster)
            test_clusterbasis[leave] = FastBEAST.MatrixBlock{Int, Float64, ClusterBasis{Float64}}(
                ClusterBasis{Float64}(
                    far_row_blocks[leave].M.U * 
                        far_row_blocks[leave].M.U[far_row_blocks[leave].M.τ, :]^-1,
                    far_row_blocks[leave].M.τ
                ),
                far_row_blocks[leave].τ,
                Int[]
            )
    end

    for leave ∈ ClusterTrees.leaves(block_tree.trial_cluster)
            trial_clusterbasis[leave] = FastBEAST.MatrixBlock{Int, Float64, ClusterBasis{Float64}}(
                ClusterBasis{Float64}(
                    far_column_blocks[leave].M.V[:, far_column_blocks[leave].M.σ]^-1 * 
                        far_column_blocks[leave].M.V,

                    far_column_blocks[leave].M.σ
                ),
                Int[],
                far_column_blocks[leave].σ
            )
    end

    return test_clusterbasis, trial_clusterbasis
end


function setup_interactions(
    far_row_blocks,
    far_column_blocks,
    assembler,
    fars,
    block_tree
)
    MBL_CN = MatrixBlock{Int, Float64, NestedNode{Float64}}
    lowrankblocks = MBL_CN[]

    for levelfars in fars
        if levelfars != []
            for far in levelfars
                testbasis = Int[]
                trialbasis = Int[]

                rows = far_row_blocks[far[1]].τ[far_row_blocks[far[1]].M.τ]
                cols = far_column_blocks[far[2]].σ[far_column_blocks[far[2]].M.σ]
                transfer_U = Matrix{Float64}[]
                transfer_V = Matrix{Float64}[]
                
                if ClusterTrees.haschildren(block_tree.test_cluster, far[1])                    
                    for child in ClusterTrees.children(block_tree.test_cluster, far[1])
                        tn = zeros(Int, length(far_row_blocks[child].M.τ))
                        for (i, τ) in enumerate(far_row_blocks[child].τ[far_row_blocks[child].M.τ])
                            tn[i] = findfirst(x->x==τ, far_row_blocks[far[1]].τ)
                        end
                        push!(transfer_U, 
                            far_row_blocks[far[1]].M.U[tn, :] *
                            far_row_blocks[far[1]].M.U[far_row_blocks[far[1]].M.τ, :]^-1
                        )
                        
                        push!(testbasis, child)
                    end

                    for child in ClusterTrees.children(block_tree.trial_cluster, far[2])
                        tn = zeros(Int, length(far_column_blocks[child].M.σ))
                        for (i, σ) in enumerate(far_column_blocks[child].σ[far_column_blocks[child].M.σ])
                            tn[i] = findfirst(x->x==σ, far_column_blocks[far[2]].σ)
                        end
                        push!(transfer_V, 
                            far_column_blocks[far[2]].M.V[:, far_column_blocks[far[2]].M.σ]^-1 *
                            far_column_blocks[far[2]].M.V[:, tn]
                        )

                        push!(trialbasis, child)
                    end
                else
                    testbasis=[far[1]]
                    trialbasis=[far[2]]
                end

                interaction = fullmatrix(assembler, rows, cols, Int, Float64).M

                push!(
                    lowrankblocks, 
                    FastBEAST.MatrixBlock{Int, Float64, NestedNode{Float64}}(
                        NestedNode(testbasis, trialbasis, transfer_U, transfer_V, interaction),
                        value(block_tree.test_cluster, far[1]),
                        value(block_tree.trial_cluster, far[2])
                    )
                )

            end
        end
    end

    return lowrankblocks
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
    maxrank = Int(round(
        length(test_idcs) * length(trial_idcs)/(length(test_idcs) + length(trial_idcs))))

    am = allocate_aca_memory(
        Float64,
        length(test_idcs),
        length(trial_idcs),
        maxrank=maxrank
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
