using BEAST

function hassemble(
    operator::BEAST.AbstractOperator,
    test_functions,
    trial_functions;
    compressor=:aca,
    tol=1e-4,
    treeoptions=BoxTreeOptions(nmin=100),
    maxrank=200,
    threading=:single,
    quadstrat=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
    verbose=false,
    svdrecompress=false
)

    @views blkasm = BEAST.blockassembler(operator, test_functions, trial_functions)
    
    @views function assembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end

    @views farblkasm = BEAST.blockassembler(
        operator,
        test_functions,
        trial_functions,
        quadstrat=quadstrat
    )
    
    @views function farassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        farblkasm(tdata,sdata,store)
    end


    test_tree = create_tree(test_functions.pos, treeoptions)
    trial_tree = create_tree(trial_functions.pos, treeoptions)

    @time hmat = HMatrix(assembler, test_tree, trial_tree, 
                         compressor=compressor, T=scalartype(operator), tol=tol, maxrank=maxrank,
                         threading=threading, farmatrixassembler=farassembler, verbose=verbose,
                         svdrecompress=svdrecompress)
    return hmat
end

# The following to function ensure that no dynamic dispatching is
# performed since we know already that all triangles are well-separate

# Copied from BEAST/examples/quadstrat.jl
function BEAST.quaddata(op, tref, bref,
    tels, bels, qs::BEAST.DoubleNumQStrat)

    qs = BEAST.DoubleNumWiltonSauterQStrat(qs.outer_rule, qs.inner_rule, 1, 1, 1, 1, 1, 1)
    BEAST.quaddata(op, tref, bref, tels, bels, qs)
end

# Copied from BEAST/examples/quadstrat.jl
function BEAST.quadrule(op, tref, bref,
    i ,τ, j, σ, qd, qs::BEAST.DoubleNumQStrat)

    return BEAST.DoubleQuadRule(
        qd.test_qp[1,i],
        qd.bsis_qp[1,j])
end


# Safe evaluation of Greens function
struct SafeDoubleNumQStrat{R}
    outer_rule::R
    inner_rule::R
end

struct SafeDoubleQuadRule{P,Q}
    outer_quad_points::P
    inner_quad_points::Q
end

function BEAST.quadrule(op, tref, bref, i ,τ, j, σ, qd, qs::SafeDoubleNumQStrat)

    return SafeDoubleQuadRule(
        qd.test_qp[1,i],
        qd.bsis_qp[1,j])
end

function BEAST.quaddata(
    op::BEAST.Helmholtz3DOp,
    test_refspace::BEAST.LagrangeRefSpace,
    trial_refspace::BEAST.LagrangeRefSpace,
    test_elements,
    trial_elements,
    qs::SafeDoubleNumQStrat
)

    test_eval(x)  = test_refspace(x,  Val{:withcurl})
    trial_eval(x) = trial_refspace(x, Val{:withcurl})

    test_qp = BEAST.quadpoints(test_eval,  test_elements,  (qs.outer_rule,))
    bsis_qp = BEAST.quadpoints(trial_eval, trial_elements, (qs.inner_rule,))

    return (;test_qp, bsis_qp)
end
