using LinearAlgebra
using LinearMaps
using ProgressMeter


struct NestedInteraction{K}
    M::Matrix{K}
    cluster::Tuple{Int, Int}
    τ::Vector{Int}
    σ::Vector{Int}
end


struct NestedMatrix{K}
    T::Matrix{K}
    children::Vector{Int}
end


isbasis(node::NestedMatrix) = (node.children == [])


struct H2Matrix{I, K} <: LinearMaps.LinearMap{K}
    fullrankblocks::Vector{MatrixBlock{I, K, Matrix{K}}}
    lowrankblocks::Vector{MatrixBlock{I, K, NestedInteraction{K}}}
    nestedrows::Vector{NestedMatrix{K}}
    nestedcolumns::Vector{NestedMatrix{K}}
    rowdim::I
    columndim::I
    nnz::I
end


function nnz(hmat::H2T) where H2T <: H2Matrix
    return hmat.nnz
end


function H2Matrix(
    matrixassembler::Function,
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    ::Type{I},
    ::Type{K};
    farmatrixassembler=matrixassembler,
    compressor=ACAOptions(),
    multithreading=false,
    verbose=false
) where {I, K, T <: AbstractNode}
    
    
end