module ClusterTree

    using ClusterTrees
    using StaticArrays
    using FastBEAST
    using CompScienceMeshes
    using LinearAlgebra

    function updatestate(block_tree, chd, par_state)
        d = ClusterTrees.data(block_tree, chd)
        test_sector  = d[1].sector
        trial_sector = d[2].sector
        cs1 = ClusterTrees.LevelledTrees.center_size(test_sector, par_state[1][1], par_state[1][2])
        cs2 = ClusterTrees.LevelledTrees.center_size(trial_sector, par_state[2][1], par_state[2][2])
        return (cs1,cs2)
    end

    function listnearfarinteractions(block_tree, block, state, adm, nears, fars, level)
        adm(block_tree, block, state) && (push!(fars[level], block); return)
        !ClusterTrees.haschildren(block_tree, block) && (push!(nears,block); return)
        for chd ∈ children(block_tree, block)
            chd_state = updatestate(block_tree, chd, state)
            listnearfarinteractions(block_tree, chd, chd_state, adm, nears, fars, level+1)
        end
    end


    function isfar(block_tree, block, state)
        η = 1.1
        test_state, trial_state = state
        test_center, test_size = test_state
        trial_center, trial_size = trial_state
        center_dist = norm(test_center-trial_center)
        if (center_dist - sqrt(3)*(test_size+trial_size)) / (test_size+trial_size) > η
            return true
        else
            return false
        end
    end

    function create_tree(spoints, tpoints; minboxsize=1, ct= point(0.5,0.5,0.5), sz=0.5)

        stree = ClusterTrees.LevelledTrees.LevelledTree(ct, sz, Int[])
        ttree = ClusterTrees.LevelledTrees.LevelledTree(ct, sz, Int[])

         for (i,pt) in pairs(spoints)
            dest = (smallest_box_size=minboxsize, target_point=pt)
            state = ClusterTrees.LevelledTrees.rootstate(stree, dest)
            ClusterTrees.update!(stree, state, i, dest) do stree, node, i
                push!(data(stree, node).values, i)
            end
        end

         for (i,pt) in pairs(tpoints)
            dest = (smallest_box_size=minboxsize, target_point=pt)
            state = ClusterTrees.LevelledTrees.rootstate(ttree, dest)
            ClusterTrees.update!(ttree, state, i, dest) do ttree, node, i
                push!(data(ttree, node).values, i)
            end
        end

         block_tree = ClusterTrees.BlockTrees.BlockTree(stree,ttree)

        nears = []
        num_levels = length(stree.levels)
        fars = [[] for l in 1:num_levels]
        root_state = ((stree.center, stree.halfsize),(ttree.center,ttree.halfsize))
        root_level = 1
        listnearfarinteractions(
            block_tree,
            root(block_tree),
            root_state,
            isfar,
            nears,
            fars,
            root_level
        )

        return nears, fars, block_tree

    end
    
    function indices(srcbox::I, trgbox::I, blktree::T) where {I, T}

        sindices = ClusterTrees.data(blktree.trial_cluster, srcbox).values
        tindices = ClusterTrees.data(blktree.test_cluster, trgbox).values

        return sindices, tindices
    end
end
