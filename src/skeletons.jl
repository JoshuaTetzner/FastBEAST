abstract type MatrixView{T} end

import Base.:eltype

function eltype(mv::MT) where MT <:MatrixView
    return typeof(mv).parameters[1]
end

struct Adjoint{T} 
    mv::T
end

import Base.:adjoint
function adjoint(mv::T) where T
    return Adjoint(mv)
end

function adjoint(adjmv::Adjoint{T}) where T
    return adjmv.mv
end

function eltype(mv::Adjoint{MT}) where MT <:MatrixView
    return typeof(adjoint(mv)).parameters[1]
end

mutable struct FullMatrixView{T} <: MatrixView{T}
    matrix::Matrix{T}
    rightindices::Vector{Integer}
    leftindices::Vector{Integer}
    rowdim::Integer
    columndim::Integer
end

mutable struct FullMatrixView2
    matrix::Matrix{Float64}
    rightindices::Vector{Int64}
    leftindices::Vector{Int64}
    rowdim::Int64
    columndim::Int64
end

function FullMatrixView(matrix::Matrix{T}, 
                        rightindices::Vector{I}, 
                        leftindices::Vector{I},
                        rowdim::Integer,
                        columndim::Integer) where {T, I <: Integer}

    return FullMatrixView{T}(matrix, 
                             rightindices, 
                             leftindices, 
                             rowdim, 
                             columndim)
end

function FullMatrixView2(matrix::Matrix{Float64}, 
    rightindices::Vector{I}, 
    leftindices::Vector{I},
    rowdim::Integer,
    columndim::Integer) where {I <: Int64}

return FullMatrixView2(matrix, 
         rightindices, 
         leftindices, 
         rowdim, 
         columndim)
end

import Base.:*
function *(fmv::FMT, vecin::VT) where {FMT <:FullMatrixView, VT <: AbstractVector}
    T = promote_type(eltype(fmv), eltype(vecin))
    vecout = zeros(T, fmv.rowdim)

    vecout[fmv.leftindices] = fmv.matrix * vecin[fmv.rightindices]
    return vecout
end

function *(afmv::Adjoint{FMT}, vecin::VT) where {FMT <:FullMatrixView, VT <: AbstractVector}
    T = promote_type(eltype(afmv), eltype(vecin))
    vecout = zeros(T, afmv.mv.columndim)

    vecout[afmv.mv.rightindices] = adjoint(afmv.mv.matrix) * vecin[afmv.mv.leftindices]
    return vecout
end

function nnz(fmv::FullMatrixView)
    return size(fmv.matrix,1)*size(fmv.matrix,2)
end

mutable struct LowRankMatrixView{T} <: MatrixView{T}
    rightmatrix::Matrix{T}
    leftmatrix::Matrix{T}
    rightindices::Vector{Integer}
    leftindices::Vector{Integer}
    rowdim::Integer
    columndim::Integer
end

function LowRankMatrixView(rightmatrix::Matrix{T}, 
                            leftmatrix::Matrix{T}, 
                            rightindices::Vector{I},
                            leftindices::Vector{I},
                            rowdim::Integer,
                            columndim::Integer) where {T, I <: Integer}

    return LowRankMatrixView{T}(rightmatrix,
                                leftmatrix,
                                rightindices, 
                                leftindices, 
                                rowdim, 
                                columndim)
end

function *(lmv::LMT, vecin::VT) where {LMT <:LowRankMatrixView, VT <: AbstractVector}
    T = promote_type(eltype(lmv), eltype(vecin))
    vecout = zeros(T, lmv.rowdim)
    vecout[lmv.leftindices] = lmv.leftmatrix*(lmv.rightmatrix * vecin[lmv.rightindices])
    return vecout
end

function *(almv::Adjoint{LMT}, vecin::VT) where {LMT <: LowRankMatrixView, VT <: AbstractVector}
    T = promote_type(eltype(almv), eltype(vecin))
    vecout = zeros(T, almv.mv.columndim)

    vecout[almv.mv.rightindices] = almv.mv.rightmatrix'*(almv.mv.leftmatrix' * vecin[almv.mv.leftindices])
    return vecout
end

function nnz(lmv::LowRankMatrixView)
    return size(lmv.rightmatrix,1)*size(lmv.rightmatrix,2) + 
            size(lmv.leftmatrix,1)*size(lmv.leftmatrix,2)
end