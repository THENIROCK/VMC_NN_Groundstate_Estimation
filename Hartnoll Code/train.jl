module Train

include("sampler.jl")
include("wavefunc.jl")
include("obs.jl")

using Flux
using Zygote: gradient, withgradient
using BSON: @save, @load
using Statistics
using .Sampler: sample
using .Wavefunc: log_prob
using .Obs

export save_model, load_model!, minimize, evaluate, gauge_evaluate

# utility: log-mean-exp for numerical stability
function logmeanexp(x::AbstractVector)
    m = maximum(x)
    return log(mean(exp.(x .- m))) + m
end

# Save model parameters and metadata
function save_model(mdl, filename::AbstractString; metadata=Dict())
    ps = Flux.params(mdl)
    arr = [cpu(Array(p)) for p in ps]
    @save filename arr metadata
end

# Load parameters back into model
function load_model!(mdl, filename::AbstractString)
    data = BSON.load(filename)
    arr = data["arr"]
    ps = Flux.params(mdl)
    @assert length(arr) == length(ps)
    for (a, p) in zip(arr, ps)
        p .= a
    end
end

# Evaluate expectation value of an observable using Obs.evaluate
function evaluate(wf, sampler, op, num_samples::Int; normalized::Bool=true, filename::String="", batch_size::Int=1000)
    if filename != ""
        load_model!(wf, filename * "_wf.bson")
        load_model!(sampler, filename * "_sm.bson")
    end
    vals = Float64[]
    ones = Float64[]
    n_batches = fld(num_samples, batch_size)
    for i in 1:n_batches
        v = evaluate(op, wf, sampler, batch_size; normalized=normalized)
        if !isnan(v)
            push!(vals, v)
        end
        if !normalized
            push!(ones, 1.0)
        end
    end
    m = mean(vals)
    s = std(vals) / sqrt(length(vals))
    if normalized
        println("Value: ", round(m, sigdigits=8), " ± ", round(s, sigdigits=8))
        return m, s
    else
        m2 = mean(ones)
        s2 = std(ones) / sqrt(length(ones))
        println("Value: ", round(m, sigdigits=8), " ± ", round(s, sigdigits=8),
                " / ", round(m2, sigdigits=8), " ± ", round(s2, sigdigits=8),
                " = ", round(m/m2, sigdigits=8))
        return m, s
    end
end

# Evaluate observable averaged over gauge orbits using Obs.gauge_evaluate
function gauge_evaluate(wf, sampler, op, num_samples::Int, algebra, group; normalized::Bool=true, filename::String="")
    if filename != ""
        load_model!(wf, filename * "_wf.bson")
        load_model!(sampler, filename * "_sm.bson")
    end
    vals = Float64[]
    batch = 1000
    n_batches = fld(num_samples, batch)
    for i in 1:n_batches
        v = gauge_evaluate(op, wf, sampler, algebra, group, batch; normalized=normalized)
        if !isnan(v)
            push!(vals, v)
        end
    end
    m = mean(vals)
    s = std(vals) / sqrt(length(vals))
    println("Gauge Value: ", round(m, sigdigits=8), " ± ", round(s, sigdigits=8))
    return m, s
end

# Private gradient steps
function _minimize_step(wf, hamil, sampler, optimizer, batch_size::Int)
    ps = Flux.params(wf)
    # use Obs.evaluate instead of hamil.evaluate
    loss = real(Obs.evaluate(hamil, wf, sampler, batch_size))
    gs = gradient(() -> real(Obs.evaluate(hamil, wf, sampler, batch_size)), ps)
    Flux.Optimise.update!(optimizer, ps, gs)
    return loss
end

function _sampler_step(wf, hamil, sampler, optimizer, batch_size::Int)
    ps = Flux.params(sampler)
    loss, gs = withgradient(ps) do
        samps, logp = sample(sampler, batch_size; with_log_prob=true)
        target_log = log_prob(wf, samps)
        log_norm = logmeanexp(target_log .- logp)
        target_log .-= log_norm
        return mean(logp .- target_log)
    end
    Flux.Optimise.update!(optimizer, ps, gs)
    return loss
end

function _sampler_step_with_wavefunc(wf, hamil, sampler, optimizer, batch_size::Int)
    ps_s = Flux.params(sampler)
    ps_w = Flux.params(wf)
    ps  = Flux.Params(vcat(ps_s, ps_w))
    loss, gs = withgradient(ps) do
        samps, logp = sample(sampler, batch_size; with_log_prob=true)
        target_log = log_prob(wf, samps)
        log_norm = logmeanexp(target_log .- logp)
        target_log .-= log_norm
        return mean(logp .- target_log)
    end
    Flux.Optimise.update!(optimizer, ps, gs)
    return loss
end

# Main minimization loop
function minimize(wf, hamil, obs_dict, sampler, lr::Ref{<:Real}, optimizer;
                  batch_size::Int=100, max_epochs::Int=1000, num_iters::Int=1000,
                  thres::Real=4.0, filename::String="")
    lr_min = 1e-6
    last_loss = Inf
    tot_fail = 0
    for epoch in 1:max_epochs
        losses = Float64[]
        for iter in 1:num_iters
            # KL divergence estimation
            samps, logp = sample(sampler, batch_size; with_log_prob=true)
            target_log = log_prob(wf, samps)
            kl = mean(logp .- (target_log .- logmeanexp(target_log .- logp)))
            if kl > thres
                loss = _sampler_step_with_wavefunc(wf, hamil, sampler, optimizer, batch_size)
            else
                _minimize_step(wf, hamil, sampler, optimizer, batch_size)
                loss = _sampler_step(wf, hamil, sampler, optimizer, batch_size)
            end
            push!(losses, loss)
        end
        loss_mean = mean(losses)
        println("Epoch $epoch: loss = $loss_mean")
        for (k, op) in obs_dict
            v, s = evaluate(wf, sampler, op, batch_size; normalized=true)
            println("  $k: $v ± $s")
        end
        if filename != ""
            save_model(wf, filename * "_wf.bson"; metadata=Dict("epoch"=>epoch, "loss"=>loss_mean))
            save_model(sampler, filename * "_sm.bson"; metadata=Dict("epoch"=>epoch, "loss"=>loss_mean))
        end
        if loss_mean > last_loss
            tot_fail += 1
            if tot_fail == 10
                tot_fail = 0
                lr[] /= 2
                println("Learning rate reduced to ", lr[])
            end
        else
            tot_fail = 0
        end
        last_loss = loss_mean
        if lr[] < lr_min
            println("Stopping: lr < ", lr_min)
            break
        end
    end
    return last_loss
end

end # module Train
