using ParallelKMeans
using ClusterTrees
using StaticArrays
using LinearAlgebra


struct Data{F, T}
    center::SVector{3, F}
    radius::F
    values::Vector{T}
end


struct KMeansSettings
    max_iters::Int
    n_threads::Int
end


function KMeansSettings(;max_iters=20, n_threads=1)
    return KMeansSettings(max_iters, n_threads)
end


struct KMNode{D}
    node::ClusterTrees.PointerBasedTrees.Node{D}
    height::Int
end


struct KMeansTree{D} <: ClusterTrees.PointerBasedTrees.APBTree
    nodes::Vector{KMNode{D}}
    root::Int
    num_elements::Int
    levels::Vector{Int}
end


function KMeansTree(num_elements; center=SVector(0.0, 0.0, 0.0), radius=0.0, data=Int[])
    root = KMNode(ClusterTrees.PointerBasedTrees.Node(
        Data(center, radius, data),
        0,
        0,
        0, 
        0
    ), 0)

    return KMeansTree([root], 1, num_elements, Int[1])
end

ClusterTrees.root(tree::KMeansTree{D}) where D = tree.root
ClusterTrees.data(tree::KMeansTree{D}, node) where D = tree.nodes[node].node.data
ClusterTrees.parent(tree::KMeansTree, node_idx) = tree.nodes[node_idx].node.parent
ClusterTrees.PointerBasedTrees.nextsibling(tree::KMeansTree, node_idx) = 
    tree.nodes[node_idx].node.next_sibling
ClusterTrees.PointerBasedTrees.firstchild(tree::KMeansTree, node_idx) = 
    tree.nodes[node_idx].node.first_child


function rootstate(tree::KMeansTree, destination)
    node = root(tree)
    return node, 1, 0
end


function value(tree, node::Int)

    if !ClusterTrees.haschildren(tree, node)
        return tree.nodes[node].node.data.values
    else
        values = Int[]
        for leave ∈ ClusterTrees.leaves(tree, node)
            append!(
                values,
                tree.nodes[leave].node.data.values
            )
        end
        return values
    end
end


function indexedvalue(tree, node::Int)

    if !ClusterTrees.haschildren(tree, node)
        return tree.nodes[node].node.data.values
    else
        values = Int[]
        indices = Tuple{Int, UnitRange{Int}}[]
        startind = 1
        for leaf ∈ ClusterTrees.leaves(tree, node)
            append!(
                values,
                tree.nodes[leaf].node.data.values
            )

            push!(
                indices, 
                (leaf, startind:(startind + length(tree.nodes[leaf].node.data.values) - 1))
            )
            startind += (length(tree.nodes[leaf].node.data.values))
        end
        return values, indices
    end
end


function child!(
    tree::KMeansTree{D}, state, destination
) where D 
        
    maxlevel, num_children, points, kmeans_sttings, nmin = destination
    parent_node_idx, level, sibling_idx, point_idcs = state

    kmcluster = ParallelKMeans.kmeans(
        points[:, point_idcs],
        num_children;
        max_iters=kmeans_sttings.max_iters,
        n_threads=kmeans_sttings.n_threads
    )

    sorted_point_idcs = zeros(Int, length(point_idcs)+1, num_children)

    for (index, value) in enumerate(kmcluster.assignments)
        sorted_point_idcs[1, value] += 1
        sorted_point_idcs[sorted_point_idcs[1,value]+1, value] = point_idcs[index] 
    end

    center = SVector{3, Float64}([kmcluster.centers[j, 1] for j = 1:3])
    radius = maximum(norm.(eachcol(
        points[:, sorted_point_idcs[2:(sorted_point_idcs[1, 1] + 1), 1]] .- 
            kmcluster.centers[:, 1]
    )))

    isnmin = true
    for sidx in 1:num_children
        isnmin = isnmin && 
            length(sorted_point_idcs[2:(sorted_point_idcs[1, 1] + 1), sidx]) <= nmin
    end

    #check if maximum depth is reached
    if level < maxlevel && !isnmin
        push!(tree.nodes, KMNode(ClusterTrees.PointerBasedTrees.Node(
            Data(center, radius, Int[]),
            num_children,
            0,
            parent_node_idx,
            0
        ), level))

        node_idx = length(tree.nodes)
            
        level > length(tree.levels) && resize!(tree.levels, level)
        tree.levels[level] = node_idx

        state_sibling = (
            parent_node_idx,
            level,
            sibling_idx + 1,
            point_idcs,
            sorted_point_idcs,
            kmcluster
        )
        tree.nodes[node_idx].node.next_sibling = sibling!(tree, state_sibling, destination) 

        # Check if more than one node is left
        sorted_point_idcs[1,1] == 1 && return node_idx

        state_child = (
            node_idx,
            level+1,
            1,
            sorted_point_idcs[(2:sorted_point_idcs[1,1] + 1), 1]
        )
        tree.nodes[node_idx].node.first_child = child!(tree, state_child, destination)

        return node_idx
    else
        push!(tree.nodes, KMNode(ClusterTrees.PointerBasedTrees.Node(
            Data(center, radius, sorted_point_idcs[2:(sorted_point_idcs[1, 1] + 1), 1]),
            0,
            0,
            parent_node_idx,
            0
        ), level))

        node_idx = length(tree.nodes)

        level > length(tree.levels) && resize!(tree.levels, level)
        tree.levels[level] = node_idx

        state_sibling = (
            parent_node_idx, level, sibling_idx+1, point_idcs, sorted_point_idcs, kmcluster
        )
        tree.nodes[node_idx].node.next_sibling = sibling!(tree, state_sibling, destination) 

        return node_idx
    end
end


function sibling!(
    tree::KMeansTree{D}, state, destination
) where D
        
    maxlevel, num_children, points, kmeans_sttings, nmin = destination
    parent_node_idx, level, sibling_idx, point_idcs, sorted_point_idcs, kmcluster = state

    # Enough siblings?
    sibling_idx > num_children && return 0

    center = SVector{3, Float64}([kmcluster.centers[j, sibling_idx] for j = 1:3])
    radius = maximum(norm.(eachcol(points[
        :, sorted_point_idcs[2:(sorted_point_idcs[1, sibling_idx] + 1), sibling_idx]
        ] .- kmcluster.centers[:, sibling_idx]
    )))

    isnmin = true
    for sidx in 1:num_children
        isnmin = isnmin && 
            length(sorted_point_idcs[2:(sorted_point_idcs[1, 1] + 1), sidx]) <= nmin
    end

    #check if maximum depth is reached
    if level < maxlevel && !isnmin
        push!(tree.nodes, KMNode(ClusterTrees.PointerBasedTrees.Node(
            Data(center, radius, Int[]),
            num_children,
            0,
            parent_node_idx,
            0
        ), level))

        node_idx = length(tree.nodes)

        level > length(tree.levels) && resize!(tree.levels, level)
        tree.levels[level] = node_idx
            
        state_sibling = (
            parent_node_idx, level, sibling_idx+1, point_idcs, sorted_point_idcs, kmcluster
        )
        tree.nodes[node_idx].node.next_sibling = sibling!(tree, state_sibling, destination) 

        # Check if more than one node is left
        sorted_point_idcs[1, sibling_idx] == 1 && return node_idx

        state_child = (
            node_idx,
            level+1,
            1,
            sorted_point_idcs[(2:sorted_point_idcs[1,sibling_idx] + 1), sibling_idx]
        )
        tree.nodes[node_idx].node.first_child = child!(tree, state_child, destination)

        return node_idx
    else
        # Node is a leaf node -> the point_idcs are saved
        push!(tree.nodes, KMNode(ClusterTrees.PointerBasedTrees.Node(
            Data(
                center,
                radius,
                sorted_point_idcs[2:(sorted_point_idcs[1, sibling_idx] + 1), sibling_idx]
            ),
            0,
            0,
            parent_node_idx,
            0
        ), level))

        node_idx = length(tree.nodes)

        level > length(tree.levels) && resize!(tree.levels, level)
        tree.levels[level] = node_idx

        state_sibling = (
            parent_node_idx, level, sibling_idx+1, point_idcs, sorted_point_idcs, kmcluster
        )
        tree.nodes[node_idx].node.next_sibling = sibling!(tree, state_sibling, destination) 

        return node_idx
    end

    return node_idx
end 


function updatestate(block_tree::ClusterTrees.BlockTrees.BlockTree{T}, chd) where T
    d = ClusterTrees.data(block_tree, chd)
    test_center = d[1].center
    trial_center = d[2].center
    test_radius= d[1].radius
    trial_radius = d[2].radius
    return ((test_center, test_radius), (trial_center, trial_radius))
end


function listnearfarinteractions(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    block,
    state,
    nears::Vector{Tuple{Int, Int}},
    fars::Vector{Vector{Tuple{Int, Int}}},
    level::Int
) where T
    
    isfar(state) && (push!(fars[level], block); return)
    !ClusterTrees.haschildren(block_tree, block) && (push!(nears, block); return)
    for chd ∈ ClusterTrees.children(block_tree, block)
        chd_state = updatestate(block_tree, chd)
        listnearfarinteractions(block_tree, chd, chd_state, nears, fars, level+1)
    end
end


function isfar(state)

    η = 1.1
    test_state, trial_state = state
    test_center, test_radius = test_state
    trial_center, trial_radius = trial_state

    center_dist = norm(test_center - trial_center)

    if center_dist > η * (test_radius + trial_radius)
        return true
    else
        return false
    end
end


function create_CT_tree(
    points::Vector{SVector{3, F}};
    maxlevel=5,
    nmin=100,
    nchildren=2,
    kmeans_settings=KMeansSettings()
) where F <: Real

    pointsM = reshape(
        [point[i] for point in points for i = 1:3], 
        (3,length(points))
    )

    tree = KMeansTree(length(points))
    destination = (maxlevel, nchildren, pointsM, kmeans_settings, nmin)
    state = (1, 2, 1, Vector(1:length(points)))
    child!(tree, state, destination)
    tree.nodes[1].node.first_child = 2

    return tree
end


function computeinteractions(tree::ClusterTrees.BlockTrees.BlockTree{T}) where T

    nears = Tuple{Int, Int}[]
    # Need better way to check depth
    num_levels = length(tree.test_cluster.levels)
    fars = [Tuple{Int, Int}[] for l in 1:num_levels]
    
    root_state = (
        (
            tree.test_cluster.nodes[1].node.data.center,
            tree.test_cluster.nodes[1].node.data.radius
        ),
        (
            tree.trial_cluster.nodes[1].node.data.center,
            tree.trial_cluster.nodes[1].node.data.radius
        ),
    )
    root_level = 1

    listnearfarinteractions(tree, ClusterTrees.root(tree),
        root_state, nears, fars, root_level)

    return nears, fars
end


function clusterlink(tree::KMeansTree{D}; node=root(tree), target=1) where D
    Iterators.filter(
        n->tree.nodes[n].height==target, ClusterTrees.DepthFirstIterator(tree, node)
    )
end


function sort_interactions(
    fars::Vector{Tuple{Int, Int}};
    testortrial = 1
)
    
    testortrial == 1 ? trialortest = 2 : trialortest = 1
    
    sortedfars = Tuple{Vector{Int}, Vector{Int}}[]
    sfars = sort(fars, by = x -> x[testortrial])

    push!(sortedfars, ([sfars[1][1]],[sfars[1][2]]))
    for n ∈ 2:length(fars)
        if sfars[n][testortrial] == sfars[n-1][testortrial]
            push!(sortedfars[end][trialortest], sfars[n][trialortest])
        else
            push!(sortedfars, ([sfars[n][1]], [sfars[n][2]]))
        end
    end

    return sortedfars
end


function childidcs(
    tree::FastBEAST.KMeansTree{FastBEAST.Data{F, I}},
    nodeidx::I
) where {F, I}

    !ClusterTrees.haschildren(tree, nodeidx) && return 0

    
    nextchild = ClusterTrees.PointerBasedTrees.firstchild(tree, nodeidx)
    childidcs = Int[]
    while nextchild != 0 
        push!(childidcs, nextchild)
        nextchild = ClusterTrees.PointerBasedTrees.nextsibling(tree, nextchild)
    end

    return childidcs
end