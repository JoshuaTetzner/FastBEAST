using LinearAlgebra

function fdapprox(dist, pos)
    next = argmax(dist)

    for np in eachindex(dist)
        if dist[np] > norm(pos[np, :] - pos[next, :])
            dist[np] = norm(pos[np, :] - pos[next, :])
        end
    end

    return next, maximum(dist)

end

function fdtrue_stepwise(dist, pos)
    
    maxdist = zeros(length(dist))
    newdist = zeros(Float64, length(dist))
    for np in eachindex(dist)
        if dist[np] != 0
            for nxp in eachindex(dist)
                if dist[nxp] > norm(pos[np, :] - pos[nxp, :])
                    newdist[nxp] = norm(pos[np, :] - pos[nxp, :])
                else
                    newdist[nxp] = dist[nxp]
                end
            end

            maxdist[np] = maximum(newdist)
        else 
            maxdist[np] = 2maximum(dist)
        end
    end
    next = argmin(maxdist)
    
    for np in eachindex(dist)
        if dist[np] > norm(pos[np, :] - pos[next, :])
            dist[np] = norm(pos[np, :] - pos[next, :])
        end
    end

    return next, maximum(dist)
end

function distance_matrix(pos)
    DM = zeros(Float64, size(pos)[1], size(pos)[1])
    for i = 1:size(pos)[1]
        for j = 1:size(pos)[1]
            DM[i, j] = norm(pos[i, :] - pos[j, :]) 
        end
    end

    return DM
end

function combimatrix(len, dim)

    CM = zeros(Int, len^dim, dim)
    for j = 1:dim 
        index = 1
        num = len^(dim-(j-1))
        iter = Int(len^dim / num)
        for mult = 1:iter
            for i = 1:len
                for rep = 1:len^(dim-j)
                    CM[index, j] = i
                    index += 1
                end
            end
        end
    end

    return CM
end
##

pos = rand(Float64, 20, 5)


DM = distance_matrix(pos)
CM = combimatrix(size(pos)[1], 6)

##
distap2 = zeros(size(pos)[1])
distap = zeros(size(pos)[1])
disttr = ones(size(pos)[1]) .* 100



firstindex = fdtrue(disttr, pos)
distap .= disttr
distap2 .= disttr

##
pos = rand(Float64, 50, 2)
dist = ones(size(pos)[1]) .* 100

maxdist(dist, pos, 4)

##
err1 = []
push!(err1, maximum(distap2))
@time for i = 2:length(distap2)
    next, m = fdtrue_stepwise(distap2, pos)
    #println(next)
    push!(err1, m)
end

##
err2 = []
push!(err2, maximum(disttr))
@time for i = 2:length(distap)
    next, m = fdtrue_stepwise_fast(disttr, pos)
    push!(err2, m)
end

err3 = []
push!(err3, maximum(distap))
@time for i = 2:length(distap)
    next, m = fdapprox(distap, pos)
    #println(next)
    push!(err3, m)
end

disttr

using Plots
err1
plot(Array(1:length(distap)), err1, label="Approx Fill FillDistance2")
plot!(Array(1:length(distap)), err3, label="Approx Fill FillDistance")
plot!(Array(1:length(disttr)), err2, label="True Fill FillDistance")
