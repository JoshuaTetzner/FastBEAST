using JLD2
using Plots

##
hc_fd = load("ACA_evaluation/hc_fd.jld2")["sc_fd"]
variance = sum((log.(hc_fd) .- log(1e-4)).^2)/length(hc_fd)
var = string(round(variance, digits=4))
histogram(hc_fd, xaxis=:log, xlim=[10^-5, 10^-3], ylim=[0,280],
    ylabel="number of startvalues", legend=false, 
    xlabel="accuracy", 
    title="heldrig crit, filldistance, σ="*var)

##
hc_slm = load("ACA_evaluation/hc_slm.jld2")["sc_slm"]
variance = sum((log.(hc_slm) .- log(1e-4)).^2)/length(hc_slm)
var = string(round(variance, digits=4))
histogram(hc_slm, xaxis=:log, xlim=[10^-5, 10^-3], ylim=[0,280],
    ylabel="number of startvalues", legend=false, 
    xlabel="accuracy", 
    title="heldrig crit, max. pivoting, σ="*var)

##
sc_fd = load("ACA_evaluation/sc_fd.jld2")["sc_fd"]
variance = sum((log.(sc_fd) .- log(1e-4)).^2)/length(sc_fd)
var = string(round(variance, digits=4))
histogram(sc_fd, xaxis=:log, xlim=[10^-5, 10^-3], ylim=[0,280],
    ylabel="number of startvalues", legend=false, 
    xlabel="accuracy", 
    title="classic crit, filldistance, σ="*var)
##
sc_slm = load("ACA_evaluation/sc_slm.jld2")["sc_slm"]
variance = sum((log.(sc_slm) .- log(1e-4)).^2)/length(sc_slm)
var = string(round(variance, digits=4))
histogram(sc_slm, xaxis=:log, xlim=[10^-5, 10^-3], ylim=[0,280],
    ylabel="number of startvalues", legend=false, 
    xlabel="accuracy", 
    title="classic crit, max. pivoting, σ="*var)



## 
random_fd = load("ACA_evaluation/random_fd.jld2")["random_fd"]
histogram(random_fd, xaxis=:log, xlim=[10^-5, 10^-3], ylim=[0,150],
    ylabel="number of startvalues", legend=false, 
    xlabel="accuracy", 
    title="heldrig convergence crit without CV, filldistance")
##
random_sml = load("ACA_evaluation/random_slm.jld2")["random_slm"]
histogram(random_sml, xaxis=:log, xlim=[10^-5, 10^-3], ylim=[0,150],
        ylabel="number of startvalues", legend=false, 
        xlabel="accuracy", 
        title="Heldrig convergence crit without CV, max. pivoting")
##
pt = XS.pos

mean = sum(pt)./length(pt)
diff = []
for el in pt
    push!(diff, norm(el - mean))
end
hc_fd
##
indices = []
for i = eachindex(hc_fd)
    if hc_fd[i] < 1e-4
        push!(indices, i)
    end
end
indices
##
histogram(hc_fd[indices], xaxis=:log, xlim=[10^-5, 10^-3],
        ylabel="number of startvalues", legend=false, 
        xlabel="accuracy", 
        title="Heldrig convergence crit without CV, max. pivoting, best")

##
random_sml[argmax(diff)]
##
scatter()
for ind in indices
    scatter!([XS.pos[ind][1]], [XS.pos[ind][2]],
    xlim=[0,1], ylim=[0,1], markercolor=:blue, legend=false)
end
scatter!()
##
XS.pos[indices][1]