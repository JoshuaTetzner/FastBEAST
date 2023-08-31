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
    levels::Vector{Int}
end


function KMeansTree(;center=SVector(0.0, 0.0, 0.0), radius=0.0, data=Int[])
    root = KMNode(ClusterTrees.PointerBasedTrees.Node(
        Data(center, radius, data),
        0,
        0,
        0, 
        0
    ), 0)

    return KMeansTree([root], 1, Int[1])
end


ClusterTrees.root(tree::KMeansTree) = tree.root
ClusterTrees.data(tree::KMeansTree, node) = tree.nodes[node].node.data
ClusterTrees.parent(tree::KMeansTree, node_idx) = tree.nodes[node_idx].node.parent
ClusterTrees.PointerBasedTrees.nextsibling(tree::KMeansTree, node_idx) = 
    tree.nodes[node_idx].node.next_sibling
ClusterTrees.PointerBasedTrees.firstchild(tree::KMeansTree, node_idx) = 
    tree.nodes[node_idx].node.first_child


function rootstate(tree::KMeansTree, destination)
    node = root(tree)
    return node, 1, 0
end


function value(tree::KMeansTree, node)

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


function child!(
    tree::KMeansTree{D}, state, destination
) where D 
        
    maxlevel, num_children, points, kmeans_sttings = destination
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

    if level < maxlevel
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
        
    maxlevel, num_children, points, kmeans_sttings = destination
    parent_node_idx, level, sibling_idx, point_idcs, sorted_point_idcs, kmcluster = state

    # Enough siblings?
    sibling_idx > num_children && return 0

    center = SVector{3, Float64}([kmcluster.centers[j, sibling_idx] for j = 1:3])
    radius = maximum(norm.(eachcol(points[
        :, sorted_point_idcs[2:(sorted_point_idcs[1, sibling_idx] + 1), sibling_idx]
        ] .- kmcluster.centers[:, sibling_idx]
    )))

    # Check if maximum tree depth is reached
    if level < maxlevel
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
    for chd ∈ children(block_tree, block)
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
    nchildren=2,
    kmeans_settings=KMeansSettings()
) where F <: Real

    pointsM = reshape(
        [point[i] for point in points for i = 1:3], 
        (3,length(points))
    )

    tree = KMeansTree()
    destination = (maxlevel, nchildren, pointsM, kmeans_settings)
    state = (1, 2, 1, Vector(1:length(points)))
    child!(tree, state, destination)
    tree.nodes[1].node.first_child = 2

    return tree
end


function computeinteractions(testtree::KMeansTree{D}, trialtree::KMeansTree{D}) where D

    block_tree = ClusterTrees.BlockTrees.BlockTree(testtree, trialtree)

    nears = Tuple{Int, Int}[]
    num_levels = length(testtree.levels)
    fars = [Tuple{Int, Int}[] for l in 1:num_levels]
    
    root_state = (
        (testtree.nodes[1].node.data.center, testtree.nodes[1].node.data.radius),
        (trialtree.nodes[1].node.data.center, trialtree.nodes[1].node.data.radius),
    )
    root_level = 1

    listnearfarinteractions(block_tree, ClusterTrees.root(block_tree),
        root_state, nears, fars, root_level)

    return nears, fars, block_tree
end


function clusterlink(tree::KMeansTree{D}, node=root(tree); target=1) where D
    Iterators.filter(
        n->tree.nodes[n].height==target, ClusterTrees.DepthFirstIterator(tree, node)
    )
end
