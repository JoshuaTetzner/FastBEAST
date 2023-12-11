using FastBEAST

function pivotingcur(
    strat::FastBEAST.FillDistance{I, F},
    acausedindices,
    totalindices
) where {I, F}

    if maximum(acausedindices) == 0
        firstind = firstindex(strat, totalindices)
        return firstind 
    else
        localind = argmax(strat.dist)

        for gind in eachindex(totalindices)
            if strat.dist[gind] > norm(strat.loc[gind] - strat.loc[localind])
                strat.dist[gind] = norm(strat.loc[gind] - strat.loc[localind])
            end
        end
        
        return localind
    end
end

function pivoting2(
    roworcolumn,
    acausedindices
)

    if maximum(roworcolumn) != 0 
        println(typeof(roworcolumn .* (.!acausedindices)))
        return argmax((roworcolumn .* (.!acausedindices)))
    else 
        println(typeof(acausedindices))
        return argmin(acausedindices)
    end
end

function pcur(
    fct::Function,
    rowidcs::Vector{Int},
    colidcs::Vector{Int};
    rowpivstrat=FastBEAST.MaxPivoting(),
    colpivstrat=FastBEAST.MaxPivoting(),
    maxrank=40,
    tol=1e-4
)
    rows = []#Vector(1:40)
    cols = []#Vector(1:40)

    ith = 1
    
    nrows = length(rowidcs)
    ncols = length(colidcs)
    
    fullV = zeros(Float64, maxrank, length(colidcs))
    reducedV = zeros(Float64, maxrank, length(colidcs))

    usedrows = zeros(Bool, length(rowidcs))
    usedcols = zeros(Bool, length(colidcs))

    rowpivstrat, nextrow = FastBEAST.firstindex(rowpivstrat, rowidcs)
    push!(rows, nextrow)
    
    usedrows[nextrow] = true
    @views fct(fullV[ith:ith, 1:ncols], nextrow:nextrow, 1:ncols)
    reducedV .= fullV
    
    colpivstrat, nextcol = FastBEAST.firstindex(colpivstrat, fullV[1, :], colidcs)
    push!(cols, nextcol)
    usedcols[nextcol] = true
    ith += 1
    
    conv = false
    normUV = norm(reducedV[1, 1:ncols])
    for i = 1:20 # iterations
        
        nextrow = pivotingcur(rowpivstrat, usedrows, rowidcs)
        #nextrow = rows[i+1]
        push!(rows, nextrow)
        usedrows[nextrow] = true
        @views fct(fullV[ith:ith, 1:ncols], nextrow:nextrow, 1:ncols)
        reducedV[ith:ith, 1:ncols] .= fullV[ith:ith, 1:ncols]

        for ind in 1:(ith-1)
            #multi = #reducedV[ith:ith, r] * (1/reducedV[r, r])
            #if r > 1
            #    multi = reducedV[ith:ith, r] - reducedV[ith:ith, r]*reducedV[r, ith]
            #end
            reducedV[ith:ith, :] -= reducedV[ind:ind, :] .* (
                reducedV[ith, ind] * 1/reducedV[ind, ind])#* 1/ fullV[ind, nextrow])
        end

        nextcol = pivoting2(reducedV[ith, :], usedcols)
        push!(cols, nextcol)
        usedcols[nextcol] = true
        println(norm(reducedV[ith:ith, :]))
        ith += 1
       
    end
    return rows, cols, reducedV
end

##
 a = [true, false, true, false]
 argmax(a)