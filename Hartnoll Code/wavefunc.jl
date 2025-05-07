module Wavefunc

include("./naf.jl")
include("./dense.jl")

using Flux
using Functors
using .NAF: BNAFDensityEstimator
using .DenseNet: ConditionalDenseNetwork

export BlockAutoregressiveWavefunction, log_psi, log_prob

# A block autoregressive neural wavefunction
mutable struct BlockAutoregressiveWavefunction
    dim_b::Int           # bosonic dimension
    dim_f::Int           # fermionic dimension
    alpha::Int           # hidden size multiplier
    bosonic_network      # BNAFDensityEstimator or Nothing
    fermion_real         # ConditionalDenseNetwork or Nothing
    fermion_imag         # ConditionalDenseNetwork or Nothing
end

# Register for Flux parameter extraction
Functors.@functor BlockAutoregressiveWavefunction

# Constructor
function BlockAutoregressiveWavefunction(bosonic_dim::Int, fermionic_dim::Int; alpha::Int=20)
    return BlockAutoregressiveWavefunction(
        bosonic_dim,
        fermionic_dim,
        alpha,
        nothing,
        nothing,
        nothing
    )
end

# Lazy init bosonic autoregressive network
function _get_bosonic_autoregressive(wf)
    if wf.bosonic_network === nothing
        wf.bosonic_network = BNAFDensityEstimator([wf.dim_b, wf.alpha * wf.dim_b, wf.dim_b])
    end
    return wf.bosonic_network
end

# Lazy init fermionic amplitude networks
function _get_fermionic_amplitudes(
    wf::BlockAutoregressiveWavefunction,
    sample_b::AbstractMatrix,
    sample_f::AbstractMatrix
)
    if wf.fermion_real === nothing
        wf.fermion_real = ConditionalDenseNetwork([wf.dim_b, wf.dim_b, 1], wf.dim_f)
    end
    if wf.fermion_imag === nothing
        wf.fermion_imag = ConditionalDenseNetwork([wf.dim_b, wf.dim_b, 1], wf.dim_f)
    end
    real_part = wf.fermion_real(sample_b, sample_f)
    imag_part = wf.fermion_imag(sample_b, sample_f)
    return real_part .+ im .* imag_part
end

# Log-amplitude of the wavefunction
function log_psi(wf, x::AbstractMatrix)
    # x: (batch_size, dim_b + dim_f)
    sample_b = x[:, 1:wf.dim_b]
    sample_f = wf.dim_f > 0 ? x[:, wf.dim_b+1:end] : zeros(size(x,1), 0)

    # Bosonic part: autoregressive density
    dist_b = _get_bosonic_autoregressive(wf)
    log_prob_b = NAF.log_prob(dist_b, sample_b)  # use fully qualified call

    # Fermionic part: conditional amplitudes
    psi_f = wf.dim_f > 0 ? _get_fermionic_amplitudes(wf, sample_b, sample_f) : ones(ComplexF32, size(x,1))

    return log_prob_b ./ 2 .+ log.(psi_f)
end

# Real-valued log-probability: |ψ|² = exp(log_prob)
function log_prob(wf, x::AbstractMatrix)
    return 2 .* real.(log_psi(wf, x))
end

end # module Wavefunc
