using BEAST

abstract type PivStrat end;

struct FillDistance{B, I, F} <: PivStrat
    loc::Vector{SVector{3, F}}
    dist::Vector{F}
    firstindex::I
    parallel::B
end

struct LocalMax{I} <: PivStrat
    firstindex::I
end

function pivoting2(
    pivoting::FillDistance{B, I, F}, 
    roworcolumn, 
    acausedindices,
    totalindices
) where {B, I, F}
    filldis = zeros(Float64, length(totalindices))
    for i in eachindex(totalindices)
        dist = []
        for j in eachindex(acausedindices)
            if acausedindices[j]
                push!(dist, norm(
                    pivoting.loc[totalindices[i]] - 
                    pivoting.loc[totalindices[j]]
                ))
            end
        end
        filldis[i] = minimum(dist)
    end
    return argmax(filldis), maximum(filldis)
end

function pivoting(
    pivoting::FillDistance{B, I, F}, 
    roworcolumn, 
    acausedindices,
    totalindices
) where {B, I, F}

    ind = argmax(pivoting.dist)

    for j in 1:length(pivoting.loc)
        ndist = norm(pivoting.loc[j] - pivoting.loc[ind])
        if pivoting.dist[j] > ndist
            pivoting.dist[j] = ndist
        end
    end

    return ind, 1
end

function firstindex(pivoting::FillDistance{B, I, F}) where {B, I, F}
    mins = Vector(pivoting.loc[1])
    max = Vector(pivoting.loc[1])

    for loc in pivoting.loc
        for i in eachindex(loc)
            if mins[i] > loc[i]
                mins[i] = loc[i]
            elseif max[i] < loc[i]
                max[i] = loc[i]
            end
        end
    end

    midpoint = (max - mins) ./ 2

    for i in eachindex(pivoting.dist)
        pivoting.dist[i] = norm(pivoting.loc[i]-midpoint)
    end

    return argmin(pivoting.dist)
end

function pivoting(
    pivoting::LocalMax{I}, 
    roworcolumn, 
    acausedindices,
    totalindices
) where I

    if maximum(abs.(roworcolumn)) != 0 && !acausedindices[argmax(abs.(roworcolumn))]
        return argmax(abs.(roworcolumn)), maximum(abs.(roworcolumn))
    else
        maxval = 0
        index = 0
        for i in eachindex(acausedindices)
            if !acausedindices[i] && abs.(roworcolumn[i]) > maxval
                maxval = abs(roworcolumn[i])
                index = i
            end
        end
        return index, maxval
    end
end

function smartmaxlocal(roworcolumn, acausedindices)
    
    #println(acausedindices)
    #println((roworcolumn))
    if maximum(roworcolumn) != 0 
        if !acausedindices[argmax(roworcolumn)]
           return argmax(roworcolumn), maximum(roworcolumn)
        else
            throw("Something went wrong")
        end
    else
        return argmin(acausedindices), 0.0
    end
end


function modifiedaca(
    M::LazyMatrix{I, F},
    am::ACAGlobalMemory{F},
    pivstrat::PivStrat;
    tol=1e-14,
    svdrecompress=true
) where {I, F}

    Ic = 1
    Jc = 1

    (maxrows, maxcolumns) = size(M)
    if pivstrat isa FillDistance
        nextrow = firstindex(pivstrat)
    else
        nextrow = pivstrat.firstindex
    end
    am.used_I[nextrow] = true
    i = 1

    #if pivstrat isa FillDistance
    #    for j in 2:length(pivstrat.loc)
    #        pivstrat.dist[j] = norm(pivstrat.loc[j] - pivstrat.loc[pivstrat.firstindex])
    #    end
    #end
    #nextrow = pivstrat.firstindex
    #am.used_I[nextrow] = true
    #i = 1
    
    @views M.μ(
        am.V[Ic:Ic, 1:maxcolumns], 
        M.τ[nextrow:nextrow],
        M.σ[1:size(M,2)]
    )

    nextcolumn, maxval = smartmaxlocal(
        abs.(am.V[Ic, 1:maxcolumns]),
        am.used_J[1:maxcolumns]  
    )

    am.used_J[nextcolumn] = true

    dividor = am.V[Ic, nextcolumn]
    @views am.V[Ic:Ic, 1:maxcolumns] ./= dividor

    @views M.μ(
        am.U[1:maxrows, Jc:Jc], 
        M.τ[1:size(M, 1)], 
        M.σ[nextcolumn:nextcolumn]
    )
    
    @views normUVlastupdate = norm(am.U[1:maxrows, 1])*norm(am.V[1, 1:maxcolumns])
    normUVsqared = normUVlastupdate^2
    
    while normUVlastupdate > sqrt(normUVsqared)*tol &&
        i <= length(M.τ)-1 &&
        i <= length(M.σ)-1 &&
        Jc < maxrank(am) 

        i += 1
        @views nextrow, maxval = pivoting(
            pivstrat,
            am.U[1:maxrows, Jc],
            am.used_I[1:maxrows],
            M.τ
        )

        am.used_I[nextrow] = true

        Ic += 1
        @views M.μ(
            am.V[Ic:Ic, 1:maxcolumns],
            M.τ[nextrow:nextrow],
            M.σ[1:size(M, 2)]
        )

        @assert Jc == (Ic - 1)
        for k = 1:Jc
            for kk=1:maxcolumns
                am.V[Ic, kk] -= am.U[nextrow, k]*am.V[k, kk]
            end
        end

        nextcolumn, maxval = smartmaxlocal(
            abs.(am.V[Ic, 1:maxcolumns]),
            am.used_J[1:maxcolumns],
        )

        if (isapprox(am.V[Ic, nextcolumn], 0.0))
            normUVlastupdate = 0.0
            am.V[Ic:Ic, 1:maxcolumns] .= 0.0
            Ic -= 1
            println("Matrix seems to have exact rank: ", Ic)
        else
            dividor = am.V[Ic, nextcolumn]
            @views am.V[Ic:Ic, 1:maxcolumns] ./= dividor

            am.used_J[nextcolumn] = true

            Jc += 1
            @views M.μ(
                am.U[1:maxrows, Jc:Jc], 
                M.τ[1:size(M, 1)],
                M.σ[nextcolumn:nextcolumn]
            )

            @assert Jc == Ic
            for k = 1:(Jc-1)
                for kk = 1:maxrows
                    am.U[kk, Jc] -= am.U[kk, k]*am.V[k, nextcolumn]
                end
            end

            @views normUVlastupdate = big(norm(am.U[1:maxrows, Jc]))*big(norm(am.V[Ic, 1:maxcolumns]))

            normUVsqared += normUVlastupdate^2
            for j = 1:(Jc-1)
                @views normUVsqared += 2*real(dot(am.U[1:maxrows,Jc], am.U[1:maxrows,j])*dot(am.V[Ic,1:maxcolumns], am.V[j,1:maxcolumns]))
            end
        end
    end

    if Jc == maxrank(am)
        println("WARNING: aborted ACA after maximum allowed rank")
    end

    if svdrecompress && Jc > 1
        @views Q,R = qr(am.U[1:maxrows,1:Jc])
        @views U,s,V = svd(R*am.V[1:Ic,1:maxcolumns])

        opt_r = length(s)
        for i in eachindex(s)
            if s[i] < tol*s[1]
                opt_r = i
                break
            end
        end

        A = (Q*U)[1:maxrows,1:opt_r]
        B = (diagm(s)*V')[1:opt_r,1:maxcolumns]

        am.U[1:maxrows, 1:Jc] .= 0.0
        am.V[1:Ic, 1:maxcolumns] .= 0.0
        am.used_J[1:maxcolumns] .= false
        am.used_I[1:maxrows] .= false

        return A, B
    else
        retU = am.U[1:maxrows,1:Jc]
        retV = am.V[1:Ic,1:maxcolumns]
        am.U[1:maxrows, 1:Jc] .= 0.0
        am.V[1:Ic, 1:maxcolumns] .= 0.0
        am.used_J[1:maxcolumns] .= false
        am.used_I[1:maxrows] .= false

        return retU, retV
    end    
end

function modifiedaca(
    M::LazyMatrix{I, F},
    pivoting::PivStrat;
    tol=1e-14,
    maxrank=40,
    svdrecompress=false
) where {I, F}

    return modifiedaca(
        M,
        allocate_aca_memory(F, size(M, 1), size(M, 2); maxrank=maxrank),
        pivoting,
        tol=tol,
        svdrecompress=svdrecompress
    )
end


using Base.Threads

function modifiedaca2(
    M::LazyMatrix{I, F},
    am::ACAGlobalMemory{F},
    pivstrat::PivStrat;
    tol=1e-4,
    rank=5,
    svdrecompress=false
) where {I, F}

    Ic = 1
    Jc = 0

    (maxrows, maxcolumns) = size(M)
    nextrows = zeros(Int, rank)
    nextrows[1] = pivstrat.firstindex
    am.used_I[nextrows[1]] = true
    
    for j in 2:length(pivstrat.loc)
        pivstrat.dist[j] = norm(pivstrat.loc[j] - pivstrat.loc[pivstrat.firstindex])
    end

    for j = 2:rank
        nextrows[j], _ = pivoting(
            pivstrat,
            am.U[1:maxrows, 1],
            am.used_I[1:maxrows],
            M.τ
        )
        am.used_I[nextrows[j]] = true
    end

    @threads for j = 1:rank
        @views M.μ(
            am.V[j:j, 1:maxcolumns], 
            M.τ[nextrows[j]:nextrows[j]],
            M.σ[1:size(M,2)]
        ) 
    end
    normUVsqared = 0
    normUVlastupdate = 1
    r = 1
    while r <= rank && normUVlastupdate > sqrt(normUVsqared)*tol
        
        Jc += 1
        @views nextcolumn, _ = smartmaxlocal(
            abs.(am.V[r, 1:maxcolumns]),
            am.used_J[1:maxcolumns]
        )

        dividor = am.V[r:r, nextcolumn]
        @views am.V[r:r, 1:maxcolumns] ./= dividor
        am.used_J[nextcolumn] = true

        @views M.μ(
            am.U[1:maxrows, Jc:Jc], 
            M.τ[1:maxrows],
            M.σ[nextcolumn:nextcolumn]
        )

        
        for ind = 1:(r-1)
            for kk = 1:maxrows
                am.U[kk, Jc] -= am.U[kk, ind] .* am.V[ind, nextcolumn]
            end
        end
        
        for ind = (r+1):rank
            for kk = 1:maxcolumns
                am.V[ind, kk] -= am.U[nextrows[ind], r] .* am.V[r, kk]
            end
        end
        @views normUVlastupdate = norm(am.U[1:maxrows, Jc])*norm(am.V[r, 1:maxcolumns])

        normUVsqared += normUVlastupdate^2
        for j = 1:(Jc-1)
            @views normUVsqared += 2*real(dot(am.U[1:maxrows,Jc], am.U[1:maxrows,j])*dot(am.V[r,1:maxcolumns], am.V[j,1:maxcolumns]))
        end
        r += 1
       # println(normUVlastupdate)
    end

    if Jc == maxrank(am)
        println("WARNING: aborted ACA after maximum allowed rank")
    end

    retU = am.U[1:maxrows,1:Jc]
    retV = am.V[1:Jc,1:maxcolumns]
    am.U[1:maxrows, 1:Jc] .= 0.0
    am.V[1:Jc, 1:maxcolumns] .= 0.0
    am.used_J[1:maxcolumns] .= false
    am.used_I[1:maxrows] .= false

    return retU, retV    
end
