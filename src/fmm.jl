using PyCall
helmholtz = pyimport("exafmm.helmholtz")
laplace = pyimport("exafmm.laplace")
using LinearAlgebra
using LinearMaps
using ProgressMeter
using SparseArrays

struct FMMMatrix{I, K} <: LinearMaps.LinearMap{K}
    fmm::Function
    B::SparseMatrixCSC{Float64, I}
    Bt::SparseMatrixCSC{Float64, I}
    BtCB::SparseMatrixCSC{K, I}
    fullmat::SparseMatrixCSC{K, I}
    rowdim::I
    columndim::I
end

abstract type FMMOptions end

struct HelmholtzFMMOptions{I, F} <: FMMOptions
    p::I
    ncrit::I
    wavek::F
end

struct LaplaceFMMOptions{I} <: FMMOptions
    p::I
    ncrit::I
end

function LaplaceFMMOptions()
    return LaplaceFMMOptions(8, 100)
end

function HelmholtzFMMOptions(wavek::F) where F 
    if isapprox(wavek, F(0))
        return LaplaceFMMOptions()
    else
        return HelmholtzFMMOptions(8, 100, wavek)
    end 
end

function Base.size(fmat::FMMMatrix, dim=nothing)
    if dim === nothing
        return (fmat.rowdim, fmat.columndim)
    elseif dim == 1
        return fmat.rowdim
    elseif dim == 2
        return fmat.columndim
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(fmat::Adjoint{T}, dim=nothing) where T <: FMMMatrix
    if dim === nothing
        return reverse(size(adjoint(fmat)))
    elseif dim == 1
        return size(adjoint(fmat),2)
    elseif dim == 2
        return size(adjoint(fmat),1)
    else
        error("dim must be either 1 or 2")
    end
end

@views function LinearAlgebra.mul!(y::AbstractVecOrMat, A::FMMMatrix, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))
    
    y = A.Bt * conj.(A.fmm(A.B*x)[1][:,1]) - A.BtCB*x + A.fullmat*x

    return y
end


@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    transA::LinearMaps.TransposeMap{<:Any,<:FMMMatrix},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, transA, x)

    fill!(y, zero(eltype(y)))

    y = A.Bt * conj.(A.fmm(A.B*x)[1][:,1]) - A.BtCB*x + A.fullmat*x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    transA::LinearMaps.AdjointMap{<:Any,<:FMMMatrix},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, transA, x)

    fill!(y, zero(eltype(y)))

    y = A.Bt * conj.(A.fmm(A.B*x)[1][:,1]) - A.BtCB*x + A.fullmat*x

    return y
end

function assemble_fmm(
    spoints::Matrix{F},
    tpoints::Matrix{F};
    options=LaplaceFMMOptions() 
) where F <: Real
    
    sources = laplace.init_sources(spoints, zeros(length(tpoints[:,1])))
    targets = laplace.init_targets(tpoints)

    fmm = laplace.LaplaceFmm(p=options.p, ncrit=options.ncrit, filename="test_file.dat")
    tree = laplace.setup(sources, targets, fmm)
    eval(charges) = eval_fmm(tree, fmm, charges, options)

    return eval

end

function assemble_fmm(
    spoints::Matrix{F},
    tpoints::Matrix{F},
    options::LaplaceFMMOptions{I} 
) where {I, F <: Real}

    println("HERE")

    sources = laplace.init_sources(spoints, zeros(length(tpoints[:,1])))
    targets = laplace.init_targets(tpoints)

    fmm = laplace.LaplaceFmm(p=options.p, ncrit=options.ncrit, filename="test_file.dat")
    tree = laplace.setup(sources, targets, fmm)
    eval(charges) = eval_fmm(tree, fmm, charges, options)

    return eval

end

function assemble_fmm(
    spoints::Matrix{F},
    tpoints::Matrix{F},
    options::HelmholtzFMMOptions{I, F} 
) where {I, F <: Real}

    sources = helmholtz.init_sources(spoints, zeros(length(tpoints[:,1])))
    targets = helmholtz.init_targets(tpoints)

    fmm = helmholtz.HelmholtzFmm(
        p=options.p,
        ncrit=options.ncrit,
        wavek=options.wavek,
        filename="test_file.dat"
    )
    tree = helmholtz.setup(sources, targets, fmm)
    eval(charges) = eval_fmm(tree, fmm, charges, options)
    
    return eval

end

function eval_fmm(
    tree,
    fmm,
    charges::Vector{ComplexF64},
    options::HelmholtzFMMOptions
)
    
    helmholtz.update_charges(tree, charges)
    helmholtz.clear_values(tree)               
    trg_values = helmholtz.evaluate(tree, fmm)

    return trg_values, fmm.verify(tree.leafs)
end

function eval_fmm(
    tree,
    fmm,
    charges::Vector{F},
    options::LaplaceFMMOptions{I}
) where {I, F <: Real} 

    laplace.update_charges(tree, charges)
    laplace.clear_values(tree)               
    trg_values = laplace.evaluate(tree, fmm)

    return trg_values, fmm.verify(tree.leafs)
end
