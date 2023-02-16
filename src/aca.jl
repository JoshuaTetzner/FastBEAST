struct LazyMatrix{I, F} <: AbstractMatrix{F}
    μ::Function
    τ::Vector{I}
    σ::Vector{I}
end

struct FillDistance{I, F}
    usedsrc::Vector{I}
    usedtrg::Vector{I}
    src::Matrix{F}
    trg::Matrix{F}
end

Base.size(A::LazyMatrix) = (length(A.τ), length(A.σ))

function Base.getindex(
    A::T,
    I,
    J
) where {K, F, T <: LazyMatrix{K, F}}

    Z = zeros(F, length(I), length(J))
    A.μ(Z, view(A.τ, I), view(A.σ, J))
    return Z
end

function LazyMatrix(μ::Function, τ::Vector{I}, σ::Vector{I}, ::Type{F}) where {I, F}
    return LazyMatrix{I, F}(μ, τ, σ)
end

@views function (A::LazyMatrix{K, F})(Z::S, I, J) where {K, F, S <: AbstractMatrix{F}}
    A.μ(view(Z, I, J), view(A.τ, I), view(A.σ, J))
end

struct ACAGlobalMemory{F}
    U::Matrix{F}
    V::Matrix{F}
    used_I::Vector{Bool}
    used_J::Vector{Bool}
end

function allocate_aca_memory(::Type{F}, maxrows, maxcolumns; maxrank = 40) where {F}
    U = zeros(F, maxrows, maxrank)
    V = zeros(F, maxrank, maxcolumns)
    used_I = zeros(Bool, maxrows)
    used_J = zeros(Bool, maxcolumns)
    return ACAGlobalMemory(U, V, used_I, used_J)
end

maxrank(acamemory::ACAGlobalMemory) = size(acamemory.U, 2)


function smartmaxlocal(roworcolumn, acausedindices)
    maxval = -1
    index = -1
    for i=1:length(roworcolumn)
        if !acausedindices[i]
            if abs(roworcolumn[i]) > maxval
                maxval = abs(roworcolumn[i]) 
                index = i
            end
        end
    end
    return index, maxval
end

function minfilldistance(used::Vector{T}, points::Matrix{F}) where {T, F}
    usedP = []
    for (i, val) in enumerate(used)
        if val
            #println("used: ", i)
            push!(usedP, i)
        end
    end
    vals = zeros(length(used))
    for i in eachindex(used)
        vals[i] = norm(points[i, :] - points[usedP[1], :])
        for j in eachindex(usedP)
            if vals[i] > norm(points[i, :] - points[usedP[j], :])
                vals[i] = norm(points[i, :] - points[usedP[j], :])
            end
        end
    end

    used[argmax(vals)] = true

    return argmax(vals) 
end

function aca(
    A, 
    M::LazyMatrix{I, F},
    am::ACAGlobalMemory{F},
    M2::FillDistance{T, F},
    pivstrat;
    tol=1e-14,
    svdrecompress=true
) where {T, I, F}

    Ic = 1
    Jc = 1

    (maxrows, maxcolumns) = size(M)

    nextrow = 1
    am.used_I[nextrow] = true
    M2.usedsrc[1] = true
    i = 1

    @views M.μ(
        am.V[Ic:Ic, 1:maxcolumns], 
        M.τ[nextrow:nextrow],
        M.σ[1:size(M,2)]
    )
    #if pivstrat == :max
        @views nextcolumn, maxval = smartmaxlocal(
            am.V[Ic:Ic, 1:maxcolumns],
            am.used_J[1:maxcolumns]
        )
    #else
    #    nextcolumn=1
    #end

    am.used_J[nextcolumn] = true
    M2.usedtrg[1] = true

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
        println("true", norm(am.U * am.V - A), ", ",normUVlastupdate)
        i += 1
        if pivstrat == :max
            @views nextrow, maxval = smartmaxlocal(
                am.U[1:maxrows,Jc],
                am.used_I[1:maxrows]
            )
        else
            nextrow = minfilldistance(M2.usedtrg, M2.trg)
        end

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
        #if pivstrat == :max
            @views nextcolumn, maxval = smartmaxlocal(
                am.V[Ic, 1:maxcolumns],
                am.used_J[1:maxcolumns]
            )
        #else
        #    nextcolumn = minfilldistance(M2.usedsrc, M2.src)
        #end

        if (isapprox(am.V[Ic, nextcolumn],0.0))
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

            @views normUVlastupdate = norm(am.U[1:maxrows,Jc])*norm(am.V[Ic,1:maxcolumns])

            normUVsqared += normUVlastupdate^2
            for j = 1:(Jc-1)
                @views normUVsqared += 2*abs(dot(am.U[1:maxrows,Jc], am.U[1:maxrows,j])*dot(am.V[Ic,1:maxcolumns], am.V[j,1:maxcolumns]))
            end
            normUVsqared += norm(am.U[1:maxrows,Jc])^2*nomr(am.V[Ic,1:maxcolumns])^2
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

function aca(
    A,
    M::LazyMatrix{I, F},
    M2::FillDistance{T, F},
    pivstrat;
    tol=1e-14,
    maxrank=40,
    svdrecompress=false
) where {T, I, F}

    return aca(
        A,
        M,
        allocate_aca_memory(F, size(M, 1), size(M, 2); maxrank=maxrank),
        M2,
        pivstrat,
        tol=tol,
        svdrecompress=svdrecompress
    )
end
