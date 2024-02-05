using BEAST
using ClusterTrees
using CompScienceMeshes
using FastBEAST
using LinearAlgebra

function interactionerror(h2mat, T)
    errs = []

    for lrb in h2mat.lowrankblocks
        push!(errs, norm(T[lrb.τ, lrb.σ] - fulltestblock(
            h2mat.nestedtestbases, lrb.row_basis
        ) * lrb.Z.M * fulltrialblock(h2mat.nestedtrialbases, lrb.col_basis))/norm(T[lrb.τ, lrb.σ]))
    end

    return errs
end


function truelrbmat(h2mat::FastBEAST.H2Matrix{I, K}, mat) where {I, K}

    fmat = zeros(K, h2mat.rowdim, h2mat.columndim)
    for lrb in h2mat.lowrankblocks
        fmat[lrb.τ, lrb.σ] = mat[lrb.τ, lrb.σ]
    end

    return fmat
end

function fulllrbmat(h2mat::FastBEAST.H2Matrix{I, K}) where {I, K}

    fmat = zeros(K, h2mat.rowdim, h2mat.columndim)

    idx = 0
    for lrb in h2mat.lowrankblocks
        idx+=1
        fmat[lrb.τ, lrb.σ] = fulltestblock(
            h2mat.nestedtestbases, lrb.row_basis
        ) * lrb.Z.M * fulltrialblock(h2mat.nestedtrialbases, lrb.col_basis)
    end

    return fmat
end

function fullmat(h2mat::FastBEAST.H2Matrix{I, K}) where {I, K}

    fmat = zeros(K, h2mat.rowdim, h2mat.columndim)

    for frb in h2mat.fullrankblocks
        fmat[frb.τ, frb.σ] = frb.M
    end
    idx = 0
    for lrb in h2mat.lowrankblocks
        idx+=1
        fmat[lrb.τ, lrb.σ] = fulltestblock(
            h2mat.nestedtestbases, lrb.row_basis
        ) * lrb.Z.M * fulltrialblock(h2mat.nestedtrialbases, lrb.col_basis)
    end

    return fmat
end

function fulltestblock(blks::Vector{FastBEAST.H2BasisBlock{I, K}}, idx::I) where {I, K}
    if blks[idx].children == []
        return blks[idx].T
    else
        fblk = fulltestblock(blks, blks[idx].children[1])*blks[idx].T[1]
        for childidx in 2:length(blks[idx].children)
            fblk = vcat(fblk, fulltestblock(blks, blks[idx].children[childidx]) * blks[idx].T[childidx])
        end

        return fblk
    end
end

function fulltrialblock(blks::Vector{FastBEAST.H2BasisBlock{I, K}}, idx::I) where {I, K}
    if blks[idx].children == []
        return blks[idx].T
    else
        fblk = blks[idx].T[1]*fulltrialblock(blks, blks[idx].children[1])
        for childidx in 2:length(blks[idx].children)
            fblk = hcat(fblk, blks[idx].T[childidx]*fulltrialblock(blks, blks[idx].children[childidx]))
        end

        return fblk
    end
end

##

c = 3e8
f = 1e8
λ = c/f
k = 2*π/λ

a = 1.0
Γ = CompScienceMeshes.meshsphere(1.0, 0.08)

SL = Maxwell3D.singlelayer(wavenumber=k)
X = raviartthomas(Γ)
@show length(X.pos)

##
T = assemble(SL, X, X)

# blockassembler
@views blkasm = BEAST.blockassembler(SL, X, X)
    
@views function assembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    blkasm(tdata,sdata,store)
end

##
@time tree = create_CT_tree(X.pos, nchildren=2, nmin=200, maxlevel=10);
block_tree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
tree.levels

@time h2mat = FastBEAST.H2Matrix(
    assembler, block_tree, ComplexF64, multithreading=false
);

@test norm(fullmat(h2mat)-T)/norm(T) ≈ 0.0 atol=1e-4
