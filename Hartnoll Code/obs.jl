module Obs

include("./algebra.jl")
include("./sampler.jl")
include("./wavefunc.jl")

using Functors
using Flux: gradient
using Statistics: mean

using .Algebra: vector_to_matrix, matrix_to_vector, random_algebra_element, action as group_apply, infinitesimal_action as group_inf
using .Sampler: sample
import .Wavefunc: log_psi

export Observable, group_action, casimir, minimal_BMN_energy, evaluate, gauge_evaluate

# A simple wrapper for an observable: f::Function # (wf, x) -> complex log amplitude
struct Observable
    f::Function
end

# -- Basic evaluations ------------------------------------------
# Raw evaluation on configurations
evaluate(obs, wf, x) = obs.f(wf, x)

# Expectation value evaluation (Monte Carlo)
function evaluate(obs, wf, sampler, batch_size::Int=100; normalized::Bool=true)
    vectors, log_prob = sample(sampler, batch_size; with_log_prob=true)
    density = obs.f(wf, vectors) .+ conj.(log_psi(wf, vectors))
    one     = log_psi(wf, vectors) .+ conj.(log_psi(wf, vectors))
    lp = log_prob
    if normalized
        num = mean(exp.(density .- lp))
        den = mean(exp.(one     .- lp))
        return num / den
    else
        return mean(exp.(density .- lp))
    end
end

# -- Gauge-invariant evaluation --------------------------------
function gauge_evaluate(obs, wf, sampler, mm, grp, batch_size::Int=100; normalized::Bool=true)
    vectors, log_prob = sample(sampler, batch_size; with_log_prob=true)
    mats = vector_to_matrix(mm, vectors)
    g = grp.random_element(batch_size)
    mats_g = group_apply(g, mats)
    vectors_g = real.(matrix_to_vector(mm, mats_g))
    density = obs.f(wf, vectors_g) .+ conj.(log_psi(wf, vectors))
    one     = log_psi(wf, vectors_g) .+ conj.(log_psi(wf, vectors))
    lp = log_prob
    if normalized
        num = mean(exp.(density .- lp))
        den = mean(exp.(one     .- lp))
        return num / den
    else
        return mean(exp.(density .- lp))
    end
end

# -- Arithmetic operators ---------------------------------------
import Base: *, +

function *(obs::Observable, c::Number)
    return Observable((wf, x) -> obs.f(wf, x) + log(c))
end

function *(c::Number, obs::Observable)
    return obs * c
end

function +(o1::Observable, o2::Observable)
    return Observable((wf, x) -> begin
        s = o1.f(wf, x)
        t = o2.f(wf, x)
        m = max(real(s), real(t))
        return log(exp(s - m) + exp(t - m)) + Complex(m, 0)
    end)
end

# -- Helpers for building observables ---------------------------
function make_action_func(func::Function)
    return Observable((wf, x) -> func(wf, x))
end

function log_psi(aw, x)
    return aw.f(x)
end

function matrix_quadratic_potential(mm)
    return make_action_func((wf, x) -> begin
        mats = vector_to_matrix(mm, x)
        b = mm.num_b
        vals = [sum(abs2, mats[j,1:b,:,:]) for j in 1:size(x,1)]
        return log.(vals) + log_psi(wf, x)
    end)
end

function matrix_commutator_potential(mm)
    return make_action_func((wf, x) -> begin
        mats = vector_to_matrix(mm, x)
        b = mm.num_b; batch = size(x,1)
        vals = zeros(batch)
        for j in 1:batch, i1 in 1:b-1, i2 in i1+1:b
            C = mats[j,i1,:,:]*mats[j,i2,:,:] - mats[j,i2,:,:]*mats[j,i1,:,:]
            vals[j] += sum(abs2, C)
        end
        return log.(vals) + log_psi(wf, x)
    end)
end

function fermion_number(mm)
    return make_action_func((wf, x) -> begin
        vals = sum(x[:, mm.dim_b+1:end], dims=2)
        return log.(vec(vals)) + log_psi(wf, x)
    end)
end

function create_fermion(x, mode)
    parity = sum((1 .- cumsum(mode, dims=2)).*x, dims=2) .% 2
    sign   = 1 .- 2 .* parity
    coef   = sign .* (1 .- sum(x .* mode, dims=2))
    new_x  = min.(x .+ mode, 1f0)
    return log.(coef .+ 1e-10), new_x
end

function annihilate_fermion(x, mode)
    parity = sum((1 .- cumsum(mode, dims=2)).*x, dims=2) .% 2
    sign   = 1 .- 2 .* parity
    coef   = sign .* sum(x .* mode, dims=2)
    new_x  = max.(x .- mode, 0f0)
    return log.(coef .+ 1e-10), new_x
end

function matrix_fermionic_bilinear(mm, bf; adjoint_a=true, adjoint_b=false)
    return make_action_func((wf, x) -> begin
        batch = size(x,1); f = mm.dim_f
        aux = rand(1:f, batch, 2)
        ferm = zeros(batch, f, 2)
        for j in 1:batch
            ferm[j,aux[j,1],1] = 1; ferm[j,aux[j,2],2] = 1
        end
        function tom(fm)
            pad = hcat(zeros(batch, mm.dim_b), fm[:,:,1])
            return vector_to_matrix(mm, pad)
        end
        bos = vector_to_matrix(mm, x)[:,1:mm.num_b,:,:]
        m1 = tom(ferm); m2 = tom(permuteddims(ferm, (1,3,2)))
        lc = log.(bf(m1, m2, bos) .+ 1e-10)
        x_b = x[:,1:mm.num_b]; x_f = x[:,mm.dim_b+1:end]
        c1, x_f = adjoint_a ? annihilate_fermion(x_f, ferm[:,:,1]) : create_fermion(x_f, ferm[:,:,1])
        c2, x_f = adjoint_b ? annihilate_fermion(x_f, ferm[:,:,2]) : create_fermion(x_f, ferm[:,:,2])
        return lc .+ c1 .+ c2 .+ log_psi(wf, x)
    end)
end

function group_action(mm, grp, dg; bosonic_only=false)
    obs = nothing
    if mm.num_b > 0
        function fb(wf, x)
            mats = vector_to_matrix(mm, x)
            inf  = group_inf(dg, mats)
            db   = inf[:,1:mm.num_b,:,:]
            dz   = zeros(size(inf,1), mm.dim_f, mm.alg.N, mm.alg.N)
            full = cat(db, dz; dims=2)
            dv   = matrix_to_vector(mm, full)
            dr   = gradient(xx->real(log_psi(wf,xx)), x)[1]
            di   = gradient(xx->imag(log_psi(wf,xx)), x)[1]
            gc   = Complex.(dr, di)
            d    = sum(gc .* dv, dims=2)
            return log(-1im .* vec(d)) .+ log_psi(wf, x)
        end
        obs = Observable((wf, x)->fb(wf, x))
    end
    if mm.dim_f > 0 && !bosonic_only
        of = matrix_fermionic_bilinear(mm, (m1,m2,bos)->1im*sum(conj.(m1).*group_inf(dg,m2)); adjoint_a=false)
        obs = obs === nothing ? of : Observable((wf, x)->obs.f(wf, x) + of.f(wf, x))
    end
    return obs
end

function casimir(mm, grp; bosonic_only=false)
    return make_action_func((wf, x) -> begin
        dg = random_algebra_element(mm, size(x,1))
        op = group_action(mm, grp, dg; bosonic_only=bosonic_only)
        return op.f(op.f(wf, x), x)
    end)
end

function minimal_BMN_energy(mm, g, mu; bosonic_only=false)
    N = mm.alg.N
    bf1 = (m1,m2,bos)->sum(tr(m1*(-bos[:,1,:,:].-1im*bos[:,2,:,:])*m2), dims=2)
    bf2 = (m1,m2,bos)->sum(tr(m1'*(-bos[:,1,:,:].+1im*bos[:,2,:,:])*m2'), dims=2)
    Hb = make_action_func((wf, x) ->
        0.5*casimir(mm, mm; bosonic_only=true).f(wf, x)
      + 0.5*mu^2*matrix_quadratic_potential(mm).f(wf, x)
      - 0.25*g^2*matrix_commutator_potential(mm).f(wf, x)
    )
    if bosonic_only
        return Hb
    end
    F1 = 0.5*g*matrix_fermionic_bilinear(mm, bf1; adjoint_a=false)
    F2 = 0.5*g*matrix_fermionic_bilinear(mm, bf2; adjoint_b=true)
    F3 = 1.5*mu*fermion_number(mm)
    return Observable((wf, x) ->
        Hb.f(wf, x) + F1.f(wf, x) + F2.f(wf, x) + F3.f(wf, x) - (N^2-1)*mu
    )
end

end # module Obs
