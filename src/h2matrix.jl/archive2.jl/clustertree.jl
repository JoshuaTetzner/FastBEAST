using FastBEAST
using StaticArrays
using LinearMaps

function cluster_tree(spoints, tpoints, maxlevel)
    
    stree = create_tree(spoints, KMeansTreeOptions(maxlevel=maxlevel))
    ttree = create_tree(tpoints, KMeansTreeOptions(maxlevel=maxlevel))

    fullinteractions = SVector{2}[]
    compressableinteractions = Vector{Vector{SVector{2}}}(undef, maxlevel)
    
    computerinteractions!(
        testtree,
        sourcetree,
        fullinteractions,
        compressableinteractions
    )

end

compressableinteractions = Vector{Vector{SVector{2}}}(undef, 4)

compressableinteractions[1] = [SVector(1, 2), SVector(1, 2),SVector(1, 2)]
compressableinteractions
##
mutable struct LeafInteractions{}
    snode::FastBEAST.AbstractNode
    tnodes::Vector{FastBEAST.AbstractNode} 
end

#struct MatrixBlock{I, F, T}
#    M::T
#    τ::Vector{I}
#    σ::Vector{I}
#end

struct ClusterMatrix{F} <: LinearMaps.LinearMap{F}
    U::Matrix{F}
    V::Matrix{F}
    τ::Vector{Int}
    σ::Vector{Int}
end

function h2matrix(
    matrixassembler::Function,
    testtree::T,
    sourcetree::T,
    ::Type{I},
    ::Type{K};
    farmatrixassembler=matrixassembler,
    compressor=ACAOptions()
) where {I, K, T <: FastBEAST.AbstractNode}
    

    cltesttree = clusterlink(testtree)
    clsourcetree = clusterlink(sourcetree)
    
    fullinteractions = SVector{2}[]
    compressableinteractions = SVector{2}[]
    
    FastBEAST.computerinteractions!(
        testtree,
        sourcetree,
        fullinteractions,
        compressableinteractions
    )

    println(length(compressableinteractions))

    MBF = FastBEAST.MatrixBlock{I, K, Matrix{K}}
    fullrankblocks = MBF[]

    for fullinteraction in fullinteractions
        push!(
            fullrankblocks,
            FastBEAST.getfullmatrixview(
                matrixassembler,
                fullinteraction[1],
                fullinteraction[2],
                I,
                K
            )
        )
    end

    
    MBL = FastBEAST.MatrixBlock{I, K, ClusterMatrix{K}}
    lowrankblocks = MBL[]

    for link in cltesttree[end]
        
        indices = Vector{I}(undef, 0)
        #println("link:", link.data.indices)
        for interaction in compressableinteractions
            if link == interaction[1] 
                indices = [indices; FastBEAST.indices(interaction[2])]
            end
        end
        if length(indices) > 0
        am = allocate_aca_memory(
            K,
            length(FastBEAST.indices(link)),
            length(indices),
            maxrank=compressor.maxrank
        )

        push!(
            lowrankblocks,
            getcompressedbasis(
                matrixassembler,
                FastBEAST.indices(link),
                indices,
                I,
                K,
                am;
                compressor
            )
        )

        end
    end
           
    return fullrankblocks, lowrankblocks
    
end

function getcompressedbasis(
    matrixassembler::Function,
    testindices,
    sourceindices,
    ::Type{I},
    ::Type{K},
    am;
    compressor=FastBEAST.ACAOptions()
) where {I, K}

        lm = FastBEAST.LazyMatrix(matrixassembler, testindices, sourceindices, K)

        retU, retV, U, V, rowindices, colindices= nca(
            lm,
            am;
            rowpivstrat=compressor.rowpivstrat,
            columnpivstrat=compressor.columnpivstrat,
            tol=compressor.tol,
            svdrecompress=compressor.svdrecompress
        )

        mbl = FastBEAST.MatrixBlock{I, K, ClusterMatrix{K}}(
            ClusterMatrix(retU, retV, rowindices, colindices),
            testindices,
            sourceindices
        )

    return mbl
end

function clusterlink(node::T) where T <: FastBEAST.AbstractNode
    cltree = Vector{T}[]

    push!(cltree, [node])

    if FastBEAST.haschildren(node)
        computelinks!(cltree, node.children)
    end

    return cltree
end

function computelinks!(cltree::Vector{Vector{T}}, nodes::Vector{T}) where T <: FastBEAST.AbstractNode
    
    push!(cltree, nodes)

    childnodes = T[]
    for node in nodes
        if FastBEAST.haschildren(node)
            for child in node.children
                push!(childnodes, child)
            end
        end
    end

    if length(childnodes) > 0
        computelinks!(cltree, childnodes)
    end
end

function leafinteractions(compressableinteractions::Vector{SVector{2}})

    maxlevel = maximum(
        [FastBEAST.level(compressableinteractions[i][1]) for i in eachindex(c1)]
    )

    leafs = SVector{2}[]
    nonleafs = SVector{2}[]

    for interaction in compressableinteractions
        if FastBEAST.level(interaction[1]) == maxlevel
            push!(leafs, interaction)
        else
            push!(nonleafs, interaction)
        end
    end

    return leaf, nonleafs
end