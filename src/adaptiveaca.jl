using BEAST

struct LazyMatrix{I, F} <: AbstractMatrix{F}
    μ::Function
    τ::Vector{I}
    σ::Vector{I}
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

function minfilldistance(used::Vector{T}, points::Matrix{F}) where {T, F}
    usedP = []
    for (i, val) in enumerate(used)
        if val
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

function mean(v::Array{F}) where F
    return sum(v) / length(v)
end

function variance(v::Array{F}; μ=sum(v)/length(v)) where F
    return sum((v .- μ).^2) / length(v)
end

function randomcheck(M::LazyMatrix{I, F}) where {I, F}
    rc = zeros(I, Int(round((length(M.τ) + length(M.σ)) / 2)), 2)
    vals = zeros(F, size(rc)[1], 1)

    rc[:, 1] = rand(1:length(M.τ), size(rc)[1])
    rc[:, 2] = rand(1:length(M.σ), size(rc)[1])

    for i in eachindex(vals[:,1])
        @views M.μ(vals[i:i, 1], M.τ[rc[i:i,1]], M.σ[rc[i:i,2]])
    end
    return rc, vals
end

function aca(
    M::LazyMatrix{I, F},
    am::ACAGlobalMemory{F},
    pivoting::Function;
    firstindex=1,
    tol=1e-14,
    svdrecompress=true
) where {I, F}

    rc, vals_check = randomcheck(M)
    Ic = 1
    Jc = 1

    (maxrows, maxcolumns) = size(M)

    nextrow = firstindex
    am.used_I[nextrow] = true
    i = 1

    @views M.μ(
        am.V[Ic:Ic, 1:maxcolumns], 
        M.τ[nextrow:nextrow],
        M.σ[1:size(M,2)]
    )
    
    @views nextcolumn, maxval = smartmaxlocal(
        am.V[Ic:Ic, 1:maxcolumns],
        am.used_J[1:maxcolumns],
        M.σ    
    )

    am.used_J[nextcolumn] = true

    dividor = am.V[Ic, nextcolumn]
    @views am.V[Ic:Ic, 1:maxcolumns] ./= dividor

    @views M.μ(
        am.U[1:maxrows, Jc:Jc], 
        M.τ[1:size(M, 1)], 
        M.σ[nextcolumn:nextcolumn]
    )

    for ci = 1:size(rc)[1]
        vals_check[ci] -= am.U[rc[ci, 1], Jc] .* am.V[Ic, rc[ci, 2]]
    end
    @views normUVlastupdate = norm(am.U[1:maxrows, 1])*norm(am.V[1, 1:maxcolumns])
    normUVsqared = normUVlastupdate^2
    CV_v2 = variance(abs.(am.V[Ic:Ic, :]).^2) / mean(abs.(am.V[Ic:Ic, :]).^2)^2
    CV_u2 = variance(abs.(am.U[:, Jc:Jc]).^2) / mean(abs.(am.U[:, Jc:Jc]).^2)^2
    CV = sqrt(CV_v2 + CV_u2 + CV_u2 * CV_v2)
    breaker = true
    #while breaker && # 
    while normUVlastupdate > sqrt(normUVsqared)*tol &&
        i <= length(M.τ)-1 &&
        i <= length(M.σ)-1 &&
        Jc < maxrank(am) 
        #println("CV: ", CV)
        #println(mean(abs.(vals_check.^2)), " < ", tol^2 / (length(M.τ) * length(M.σ)) * normUVsqared)
        #println(normUVlastupdate)
        i += 1
        @views nextrow, maxval = pivoting(
            am.U[1:maxrows,Jc],
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
        
        @views nextcolumn, maxval = smartmaxlocal(
            am.V[Ic, 1:maxcolumns],
            am.used_J[1:maxcolumns],
            M.σ
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
            @views normUVlastupdate = norm(am.U[1:maxrows, Jc])*norm(am.V[Ic, 1:maxcolumns])
            for ci = 1:size(rc)[1]
                vals_check[ci] -= am.U[rc[ci, 1], Jc] .* am.V[Ic, rc[ci, 2]]
            end

            normUVsqared += (norm(am.U[1:maxrows, Jc])*norm(am.V[Ic, 1:maxcolumns]))^2#normUVlastupdate^2
            for j = 1:(Jc-1)
                @views normUVsqared += 2*real(dot(am.U[1:maxrows,Jc], am.U[1:maxrows,j])*dot(am.V[Ic,1:maxcolumns], am.V[j,1:maxcolumns]))
            end

            
            CV_v2 = variance(abs.(am.V[Ic:Ic, :]).^2) / mean(abs.(am.V[Ic:Ic, :]).^2)^2
            CV_u2 = variance(abs.(am.U[:, Jc:Jc]).^2) / mean(abs.(am.U[:, Jc:Jc]).^2)^2
            CV = sqrt(CV_v2 + CV_u2 + CV_u2 * CV_v2)
            
            #if mean(abs.(vals_check.^2)) <= tol^2 / (length(M.τ) * length(M.σ)) * normUVsqared# && CV < 4
            #    breaker = false
            #end
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
    M::LazyMatrix{I, F};
    pivoting=smartmaxlocal,
    firstindex=1,
    tol=1e-14,
    maxrank=40,
    svdrecompress=false
) where {I, F}

    return aca(
        M,
        allocate_aca_memory(F, size(M, 1), size(M, 2); maxrank=maxrank),
        pivoting,
        firstindex=firstindex,
        tol=tol,
        svdrecompress=svdrecompress
    )
end

function minimalfilldistance(
    trial_functions::BEAST.Space,
    roworcolumn, 
    acausedindices,
    totalindices
)
    if maximum(acausedindices)
        filldis = zeros(Float64, length(totalindices))
        for i in eachindex(totalindices)
            dist = []
            for j in eachindex(acausedindices)
                if acausedindices[j]
                    push!(dist, norm(
                        trial_functions.pos[totalindices[i]] - 
                        trial_functions.pos[totalindices[j]]
                    ))
                end
            end
            filldis[i] = minimum(dist)
        end
        return argmax(filldis), maximum(filldis)
    else
        return 1, 1 
    end
end

function minimalfilldistance(
    trial_functions::Vector{SVector{3, Float64}},
    roworcolumn, 
    acausedindices,
    totalindices
)
    if maximum(acausedindices)
        filldis = zeros(Float64, length(totalindices))
        for i in eachindex(totalindices)
            dist = []
            for j in eachindex(acausedindices)
                if acausedindices[j]
                    push!(dist, norm(
                        trial_functions[totalindices[i]] - 
                        trial_functions[totalindices[j]]
                    ))
                end
            end
            filldis[i] = minimum(dist)
        end
        return argmax(filldis), maximum(filldis)
    else
        return 1, 1 
    end
end


function smartmaxlocal(roworcolumn, acausedindices, totalindices)
    maxval = -1
    index = -1
    for i = 1:length(roworcolumn)
        if !acausedindices[i]
            if abs(roworcolumn[i]) > maxval
                maxval = abs(roworcolumn[i]) 
                index = i
            end
        end
    end
    return index, maxval
end
##

a = rand(Bool, 10)
a[1] = true
argmin(a)