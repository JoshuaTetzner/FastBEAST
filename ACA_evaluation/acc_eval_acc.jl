using Plots
using JLD2

truevals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]

##
using StatsPlots

err = load("ACA_evaluation/cc_slm_fullerr.jld2")["err"]
plot(truevals, truevals, xaxis=:log, yaxis=:log, label="tolerance")
errorline!(truevals, Float64.(err), errorstyle=:plume, label=false, 
    title="ACA accuracy max pivoting", legend=:topleft)

##
err = load("ACA_evaluation/cc_fd_fullerr.jld2")["err"]
plot(truevals, truevals, xaxis=:log, yaxis=:log, label="tolerance")
errorline!(truevals, Float64.(err), errorstyle=:plume, label=false, 
    title="ACA accuracy filldistance", legend=:topleft)

##
err = load("ACA_evaluation/cc_slm_fullerr_noVP.jld2")["err"]
plot(truevals, truevals, xaxis=:log, yaxis=:log, label="tolerance")
errorline!(truevals, Float64.(err), errorstyle=:plume, label=false, 
    title="ACA accuracy max pivoting, scalar potential", legend=:topleft)

##
err = load("ACA_evaluation/cc_fd_fullerr_noVP.jld2")["err"]
plot(truevals, truevals, xaxis=:log, yaxis=:log, label="tolerance")
errorline!(truevals, Float64.(err), errorstyle=:plume, label=false, 
    title="ACA accuracy filldistance, scalar potential", legend=:topleft)

##
err = load("ACA_evaluation/cc_slm_fullerr_noSP.jld2")["err"]
plot(truevals, truevals, xaxis=:log, yaxis=:log, label="tolerance")
errorline!(truevals, Float64.(err), errorstyle=:plume, label=false, 
    title="ACA accuracy max pivoting, vector potential", legend=:topleft)

##
err = load("ACA_evaluation/cc_fd_fullerr_noSP.jld2")["err"]
plot(truevals, truevals, xaxis=:log, yaxis=:log, label="tolerance")
errorline!(truevals, Float64.(err), errorstyle=:plume, label=false, 
   title="ACA accuracy filldistance, vector potential", legend=:topleft)
