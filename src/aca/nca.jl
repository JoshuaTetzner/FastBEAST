using FastBEAST 
using BEAST

mutable struct RandConvergence{I, K}
    indices::Matrix{I}
    rest::Matrix{K}
end

function pivoting(
    strat::FastBEAST.MaxPivoting{I},
    randconv::RandConvergence{I, K},
    roworcolumn,
    acausedindices,
    totalindices
) where {I, K}

    if maximum(roworcolumn) != 0 
        return argmax(roworcolumn .* (.!acausedindices))

    else 
        return argmin(acausedindices)
    end
end

function initconvergence(
    M::FastBEAST.LazyMatrix{I, K},
) where {I, K}
    mult = 1
    indices = zeros(I, mult*(size(M)[1] + size(M)[2]), 2)
    indices[:, 1] = rand(1:length(M.τ), mult*(size(M)[1] + size(M)[2]))
    indices[:, 2] = rand(1:length(M.σ), mult*(size(M)[1] + size(M)[2]))
    
    rest = zeros(K, mult*(size(M)[1] + size(M)[2]), 1)
    for ind in eachindex(rest)
        @views M.μ(
            rest[ind:ind, 1:1], 
            M.τ[indices[ind, 1]:indices[ind, 1]],
            M.σ[indices[ind, 2]:indices[ind, 2]]
        )
    end
    
    return RandConvergence(indices, rest)
end

function checkconvergence(
    normU::F,
    normV::F,
    maxrows::I,
    maxcolumns::I,
    randcon::RandConvergence{I, K},
    am::FastBEAST.ACAGlobalMemory{I, F, K},
    rowpivstrat::FastBEAST.MaxPivoting{I},
    columnpivstrat::FastBEAST.MaxPivoting{I},
    tol::F,
) where {I, F, K}

    if normU == 0 || normV == 0
        rowpivstrat = FastBEAST.MaxPivoting()
        
        return false, rowpivstrat, columnpivstrat
    else
        for i in eachindex(randcon.rest)
            @views randcon.rest[i] -= am.U[randcon.indices[i, 1], am.Jc] * am.V[am.Ic, randcon.indices[i, 2]] 
        end

        am.normUV += (normU * normV)^2

        for j = 1:(am.Jc-1)
            @views am.normUV += 2*real(dot(am.U[1:maxrows, am.Jc], am.U[1:maxrows, j]) * dot(
                am.V[am.Ic, 1:maxcolumns],
                am.V[j, 1:maxcolumns])
            )
        end
        
        meanrest = sum(abs.(randcon.rest).^2)/length(randcon.rest)

        conv = sqrt(meanrest*maxrows*maxcolumns) <= tol*sqrt(am.normUV) && normU*normV <= tol*sqrt(am.normUV)
        #conv = normU*normV <= tol*sqrt(am.normUV)
        return conv, rowpivstrat, columnpivstrat
    end
end

function nca(
    M::FastBEAST.LazyMatrix{I, K},
    am::FastBEAST.ACAGlobalMemory{I, F, K};
    rowpivstrat=MaxPivoting(1),
    columnpivstrat=MaxPivoting(1),
    tol=1e-14,
    svdrecompress=true
) where {I, F, K}

    

    U = zeros(K, size(am.U)[1], size(am.U)[2])
    V = zeros(K, size(am.V)[1], size(am.V)[2])
    rowindices = Int[]
    colindices = Int[]
    #println(size(U))
    #println(size(V))

    randconv = initconvergence(M)
    maxrowcols = 1
    if size(M)[1] > 0 && size(M)[2] > 0
        maxrowcols = Int(floor(size(M)[1]*size(M)[2]/(size(M)[1]+size(M)[2])))
    end

    #println(maxrowcols, " -U: ", size(U), " -V: ", size(V))


    isconverged = false    
    (maxrows, maxcolumns) = size(M)

    rowpivstrat, nextrow = FastBEAST.firstindex(rowpivstrat, M.τ)
    push!(rowindices, nextrow)
    am.used_I[nextrow] = true
    i = 1

    @views M.μ(
        am.V[am.Ic:am.Ic, 1:maxcolumns], 
        M.τ[nextrow:nextrow],
        M.σ[1:size(M,2)]
    )
    V[am.Ic:am.Ic, 1:maxcolumns] = am.V[am.Ic:am.Ic, 1:maxcolumns]

    @views nextcolumn = FastBEAST.pivoting(
        columnpivstrat,
        abs.(am.V[am.Ic, 1:maxcolumns]),
        am.used_J[1:maxcolumns],
        M.σ
    )
    push!(colindices, nextcolumn)

    am.used_J[nextcolumn] = true

    dividor = am.V[am.Ic, nextcolumn]
    if dividor != 0
        @views am.V[am.Ic:am.Ic, 1:maxcolumns] ./= dividor
    end

    @views M.μ(
        am.U[1:maxrows, am.Jc:am.Jc], 
        M.τ[1:size(M, 1)], 
        M.σ[nextcolumn:nextcolumn]
    )
    U[1:maxrows, am.Jc:am.Jc] = am.U[1:maxrows, am.Jc:am.Jc]


    @views normU = norm(am.U[1:maxrows, am.Jc])
    @views normV = norm(am.V[am.Ic, 1:maxcolumns])

    if isapprox(normU, 0.0) && isapprox(normV, 0.0)
        #println("Matrix seems to have exact rank: ", am.Ic)
        isconverged = true
    else
        isconverged, rowpivstrat, columnpivstrat = checkconvergence(
            normU,
            normV,
            maxrows,
            maxcolumns,
            randconv,
            am,
            rowpivstrat,
            columnpivstrat,
            tol
        )
    end
    
    while !isconverged && am.Jc < maxrowcols &&
        i <= length(M.τ)-1 &&
        i <= length(M.σ)-1 &&
        am.Jc < FastBEAST.maxrank(am)

        i += 1
        
        @views nextrow = pivoting(
            rowpivstrat,
            randconv,
            abs.(am.U[1:maxrows,am.Jc]),
            am.used_I[1:maxrows],
            M.τ
        )
        push!(rowindices, nextrow)

        am.used_I[nextrow] = true

        am.Ic += 1
        @views M.μ(
            am.V[am.Ic:am.Ic, 1:maxcolumns],
            M.τ[nextrow:nextrow],
            M.σ[1:size(M, 2)]
        )
        V[am.Ic:am.Ic, 1:maxcolumns] = am.V[am.Ic:am.Ic, 1:maxcolumns]

        
        @assert am.Jc == (am.Ic - 1)
        for k = 1:am.Jc
            for kk=1:maxcolumns
                am.V[am.Ic, kk] -= am.U[nextrow, k]*am.V[k, kk]
            end
        end

        @views nextcolumn = FastBEAST.pivoting(
            columnpivstrat,
            abs.(am.V[am.Ic, 1:maxcolumns]),
            am.used_J[1:maxcolumns],
            M.σ
        )
        push!(colindices, nextcolumn)

        dividor = am.V[am.Ic, nextcolumn]
        if dividor != 0
            @views am.V[am.Ic:am.Ic, 1:maxcolumns] ./= dividor
        end

        am.used_J[nextcolumn] = true
        am.Jc += 1
        
        @views M.μ(
            am.U[1:maxrows, am.Jc:am.Jc], 
            M.τ[1:size(M, 1)],
            M.σ[nextcolumn:nextcolumn]
        )
        U[1:maxrows, am.Jc:am.Jc] = am.U[1:maxrows, am.Jc:am.Jc]
      
        

        @assert am.Jc == am.Ic
        for k = 1:(am.Jc-1)
            for kk = 1:maxrows
                am.U[kk, am.Jc] -= am.U[kk, k]*am.V[k, nextcolumn]
            end
        end
        
        @views normU = norm(am.U[1:maxrows, am.Jc])
        @views normV = norm(am.V[am.Ic, 1:maxcolumns])

        if isapprox(normU, 0.0) && isapprox(normV, 0.0)
            #println("Matrix seems to have exact rank: ", am.Ic-1)
            am.Ic -= 1
            am.Jc -= 1
            isconverged = true
        else
            isconverged, rowpivstrat, columnpivstrat = checkconvergence(
                normU,
                normV,
                maxrows,
                maxcolumns,
                randconv,
                am,
                rowpivstrat,
                columnpivstrat,
                tol
            )
        end
    end

    if am.Jc == maxrowcols || am.Jc == FastBEAST.maxrank(am)
        println("WARNING: aborted ACA after maximum effective rank: ", FastBEAST.maxrank(am))
    end

    #if false rowpivstrat isa FastBEAST.MRFPivoting && rowpivstrat.uselessupdate
    #    #println("uselessupdate")
    #    am.Ic -= 1
    #    am.Jc -= 1
    #end

    if svdrecompress && am.Jc > 1
        @views Q,R = qr(am.U[1:maxrows,1:am.Jc])
        @views U,s,V = svd(R*am.V[1:am.Ic,1:maxcolumns])

        opt_r = length(s)
        for i in eachindex(s)
            if s[i] < tol*s[1]
                opt_r = i
                break
            end
        end

        A = (Q*U)[1:maxrows, 1:opt_r]
        B = (diagm(s)*V')[1:opt_r, 1:maxcolumns]

        am.U[1:maxrows, 1:am.Jc] .= 0.0
        am.V[1:am.Ic, 1:maxcolumns] .= 0.0
        am.used_J[1:maxcolumns] .= false
        am.used_I[1:maxrows] .= false
        am.Ic = 1
        am.Jc = 1
        am.normUV = 0.0

        return A, B
    else
        
        retU = am.U[1:maxrows, 1:am.Jc]
        retV = am.V[1:am.Ic, 1:maxcolumns]
        U = U[1:maxrows, 1:am.Jc]
        V = V[1:am.Ic, 1:maxcolumns]

        am.U[1:maxrows, 1:am.Jc] .= 0.0
        am.V[1:am.Ic, 1:maxcolumns] .= 0.0

        am.used_J[1:maxcolumns] .= false
        am.used_I[1:maxrows] .= false
        am.Ic = 1
        am.Jc = 1
        am.normUV = 0.0
        

        return retU, retV, U, V, rowindices, colindices
    end
end

function nca(
    M::FastBEAST.LazyMatrix{I, F};
    rowpivstrat=FastBEAST.MaxPivoting(1),
    columnpivstrat=FastBEAST.MaxPivoting(1),
    tol=1e-14,
    maxrank=40,
    svdrecompress=true
) where {I, F}

    return nca(
        M,
        allocate_aca_memory(F, size(M, 1), size(M, 2); maxrank=maxrank),
        rowpivstrat=rowpivstrat,
        columnpivstrat=columnpivstrat,
        tol=tol,
        svdrecompress=svdrecompress
    )

end
