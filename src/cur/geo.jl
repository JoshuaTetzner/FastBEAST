using StaticArrays

function testgeo_1d(N=100)
    spoints = [@SVector rand(3) for i = 1:N]

    tpoints = [(@SVector rand(3)) + SVector(2, 0, 0) for i = 1:N]
    append!(tpoints, [(@SVector rand(3)) + SVector(3, 0, 0) for i = 1:N])

    Nff = 15
    append!(tpoints, [(@SVector rand(3))*2 + SVector(4, 0, 0) for i = 1:Nff])

    return spoints, tpoints
end


function testgeo_2d(N=100)
    spoints = [@SVector rand(3) for i = 1:N]

    tpoints = SVector{3, Float64}[]
    for x = 0:5
        for y = [-2, 2, 3]
            append!(tpoints, [(@SVector rand(3)) + SVector(x-2, y, 0) for i = 1:N])
        end
    end
    
    for x = [-2, 2, 3]
        for y = [-1, 0, 1]
            append!(tpoints, [(@SVector rand(3)) + SVector(x, y, 0) for i = 1:N])
        end
    end

    Nff = 15
    for x = [-4, 4]
        for y = [-4, -2, 0, 2, 4]
            append!(tpoints, [(@SVector rand(3))*2 + SVector(x, y, 0) for i = 1:Nff])
        end
    end

    for x = [-2, 0, 2]
        for y = [-4, 4]
            append!(tpoints, [(@SVector rand(3))*2 + SVector(x, y, 0) for i = 1:Nff])
        end
    end


    return spoints, tpoints
end
