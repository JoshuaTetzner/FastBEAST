using JLD2
using Plots

## 1e-3
hc_fd = load("ACA_evaluation/cc_fd3.jld2")["1e3"]
variance = sum((log.(hc_fd) .- log(1e-3)).^2)/length(hc_fd)
var = string(round(variance, digits=4))
histogram(hc_fd, xaxis=:log, xlim=[10^-4, 10^-2],
    ylabel="number of startvalues", legend=false, 
    xlabel="accuracy", 
    title="classic crit, filldistance, σ="*var)

##
hc_fd = load("ACA_evaluation/cc_slm3.jld2")["1e3"]
variance = sum((log.(hc_fd) .- log(1e-3)).^2)/length(hc_fd)
var = string(round(variance, digits=4))
histogram(hc_fd, xaxis=:log, xlim=[10^-4, 10^-2],
    ylabel="number of startvalues", legend=false, 
    xlabel="accuracy", 
    title="classic crit, slm, σ="*var)

## 1e-4
hc_fd = load("ACA_evaluation/cc_fd4.jld2")["1e4"]
variance = sum((log.(hc_fd) .- log(1e-4)).^2)/length(hc_fd)
var = string(round(variance, digits=4))
histogram(hc_fd, xaxis=:log, xlim=[10^-5, 10^-3],
    ylabel="number of startvalues", legend=false, 
    xlabel="accuracy", 
    title="classic crit, filldistance, σ="*var)

##
hc_fd = load("ACA_evaluation/cc_slm4.jld2")["1e4"]
variance = sum((log.(hc_fd) .- log(1e-4)).^2)/length(hc_fd)
var = string(round(variance, digits=4))
histogram(hc_fd, xaxis=:log, xlim=[10^-5, 10^-3],
    ylabel="number of startvalues", legend=false, 
    xlabel="accuracy", 
    title="classic crit, slm, σ="*var)    

## 1e-8
hc_fd = load("ACA_evaluation/cc_fd8.jld2")["1e8"]
variance = sum((log.(hc_fd) .- log(1e-8)).^2)/length(hc_fd)
var = string(round(variance, digits=4))
histogram(hc_fd, xaxis=:log, xlim=[10^-9, 10^-7],
    ylabel="number of startvalues", legend=false, 
    xlabel="accuracy", 
    title="classic crit, filldistance, σ="*var)

##
hc_fd = load("ACA_evaluation/cc_slm8.jld2")["1e8"]
variance = sum((log.(hc_fd) .- log(1e-8)).^2)/length(hc_fd)
var = string(round(variance, digits=4))
histogram(hc_fd, xaxis=:log, xlim=[10^-9, 10^-7],
    ylabel="number of startvalues", legend=false, 
    xlabel="accuracy", 
    title="classic crit, slm, σ="*var)

## 1e-12
hc_fd = load("ACA_evaluation/cc_fd12.jld2")["1e12"]
variance = sum((log.(hc_fd) .- log(1e-12)).^2)/length(hc_fd)
var = string(round(variance, digits=4))
histogram(hc_fd, xaxis=:log, xlim=[10^-13, 10^-11],
    ylabel="number of startvalues", legend=false, 
    xlabel="accuracy", 
    title="classic crit, filldistance, σ="*var)

##
hc_fd = load("ACA_evaluation/cc_slm12.jld2")["1e12"]
variance = sum((log.(hc_fd) .- log(1e-12)).^2)/length(hc_fd)
var = string(round(variance, digits=4))
histogram(hc_fd, xaxis=:log, xlim=[10^-13, 10^-11],
    ylabel="number of startvalues", legend=false, 
    xlabel="accuracy", 
    title="classic crit, slm, σ="*var)