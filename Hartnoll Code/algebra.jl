module Algebra

using LinearAlgebra, Random

export LieAlgebra, random_algebra_element, vector_to_matrix, matrix_to_vector,
       action, infinitesimal_action, SU, SO2, MatrixModel

# -------------------------------------------------------------------
# Base Lie algebra
# -------------------------------------------------------------------
struct LieAlgebra
    N::Int                     # matrix size
    dim::Int                   # number of basis elements
    basis::Vector{Matrix{ComplexF64}}  # basis[i] is N×N Hermitian
    metric::Matrix{ComplexF64}         # inner‐product matrix
end

function LieAlgebra(basis::Vector{<:AbstractMatrix{ComplexF64}})
    N = size(basis[1],1)
    dim = length(basis)
    @assert all(ishermitian, basis) "Basis must be Hermitian"
    M = zeros(ComplexF64, dim, dim)
    for i in 1:dim, j in 1:dim
        M[i,j] = tr(basis[i]*basis[j])
    end
    return LieAlgebra(N, dim, basis, M)
end

"Draw random algebra elements (batch,dim) with norm √dim." 
random_algebra_element(alg, batch_size::Int) = begin
    X = randn(Float64, batch_size, alg.dim)
    norms = sqrt.(sum(abs2, X, dims=2))
    return sqrt(alg.dim) .* X ./ norms
end

"Expand (batch,dim) → (batch,N,N) via basis." 
function vector_to_matrix(alg::LieAlgebra, vec::AbstractMatrix{<:Real})
    batch, d = size(vec)
    @assert d == alg.dim
    out = Array{ComplexF64,3}(undef, batch, alg.N, alg.N)
    for b in 1:batch
        M = zeros(ComplexF64, alg.N, alg.N)
        for i in 1:d
            M .+= vec[b,i] * alg.basis[i]
        end
        out[b,:,:] = M
    end
    return out
end

"Project (batch,N,N) → (batch,dim) via basis." 
function matrix_to_vector(alg::LieAlgebra, mats::Array{ComplexF64,3})
    batch, n1, n2 = size(mats)
    @assert n1==alg.N && n2==alg.N
    out = Array{Float64,2}(undef, batch, alg.dim)
    invM = inv(alg.metric)
    for b in 1:batch
        trvec = ComplexF64[ tr(mats[b,:,:]*alg.basis[i]) for i in 1:alg.dim ]
        coeff = invM * trvec
        out[b,:] = real.(coeff)
    end
    return out
end

"""
    action(alg::LieAlgebra, g, x)

Returns g * x * g† for a batch of unitary matrices g and input matrices x.
Supports x with shape (batch, ..., N, N).
"""
function action(alg, g::Array{ComplexF64,3}, x::AbstractArray{ComplexF64})
    nd = ndims(x)
    batch = size(g,1)
    @assert size(g,2)==alg.N && size(g,3)==alg.N
    @assert size(x,1)==batch && size(x,nd-1)==alg.N && size(x,nd)==alg.N
    dims = size(x)
    d_flat = prod(dims[2:nd-2])
    x_flat = reshape(x, (batch, d_flat, alg.N, alg.N))
    y_flat = similar(x_flat)
    for b in 1:batch
        Gb  = view(g, b, :, :)
        GbH = adjoint(Gb)
        for t in 1:d_flat
            y_flat[b, t, :, :] = Gb * x_flat[b, t, :, :] * GbH
        end
    end
    return reshape(y_flat, size(x))
end

"""
    infinitesimal_action(alg::LieAlgebra, dg, x)

Returns i [dg, x] for a batch of hermitian matrices dg (in vector form) and input matrices x.
Supports x with shape (batch, ..., N, N) and dg with shape (batch, dim).
"""
function infinitesimal_action(alg::LieAlgebra, dg::AbstractMatrix{<:Real}, x::AbstractArray{ComplexF64})
    nd = ndims(x)
    batch = size(dg,1)
    @assert size(dg,2)==alg.dim
    @assert size(x,1)==batch && size(x,nd-1)==alg.N && size(x,nd)==alg.N
    dims   = size(x)
    d_flat = prod(dims[2:nd-2])
    x_flat = reshape(x, (batch, d_flat, alg.N, alg.N))
    dg_mats = vector_to_matrix(alg, dg)
    y_flat  = similar(x_flat)
    for b in 1:batch
        M = view(dg_mats, b, :, :)
        for t in 1:d_flat
            y_flat[b, t, :, :] = M * x_flat[b, t, :, :] - x_flat[b, t, :, :] * M
        end
    end
    return reshape(1im .* y_flat, size(x))
end

# -------------------------------------------------------------------
# Special Unitary algebra SU(N)
# -------------------------------------------------------------------
function SU(N::Int)
    basis = Matrix{ComplexF64}[]
    for i in 1:N-1, j in i+1:N
        m1 = zeros(ComplexF64,N,N); m1[i,j]=1/sqrt(2); m1[j,i]=1/sqrt(2)
        push!(basis,m1)
        m2 = zeros(ComplexF64,N,N); m2[i,j]=-1im/sqrt(2); m2[j,i]=1im/sqrt(2)
        push!(basis,m2)
    end
    for k in 1:N-1
        m = zeros(ComplexF64,N,N)
        for j in 1:k
            m[j,j] = 1/sqrt(k*(k+1))
        end
        m[k+1,k+1] = -sqrt(k/(k+1))
        push!(basis,m)
    end
    return LieAlgebra(basis)
end

# -------------------------------------------------------------------
# Special Orthogonal algebra SO(2)
# -------------------------------------------------------------------
function SO2()
    S = ComplexF64[0 -1im; 1im 0]
    return LieAlgebra([S])
end

# -------------------------------------------------------------------
# MatrixModel wrapper (fixed)
# -------------------------------------------------------------------
struct MatrixModel
    alg::LieAlgebra
    num_b::Int
    num_f::Int
    dim_b::Int
    dim_f::Int
    dim::Int
    num::Int
end

# Construct a MatrixModel
function MatrixModel(alg::LieAlgebra, num_b::Int, num_f::Int=0)
    d_alg = alg.dim
    dm_b  = num_b * d_alg
    dm_f  = num_f * d_alg
    return MatrixModel(alg, num_b, num_f, dm_b, dm_f, dm_b + dm_f, num_b + num_f)
end

# Delegate MatrixModel fields and fallback to algebra fields
import Base: getproperty
function getproperty(mm, s::Symbol)
    # if property exists on MatrixModel, return it
    if hasfield(MatrixModel, s)
        return getfield(mm, s)
    else
        # otherwise delegate to the underlying LieAlgebra
        return getproperty(mm.alg, s)
    end
end

# Block vector→matrix: flat (batch, dim) → (batch, num, N, N)
# function vector_to_matrix(mm::MatrixModel, vec::AbstractMatrix{<:Real})
#     batch, D = size(vec)
#     @assert D == mm.dim
#     # reshape into (batch*num, alg.dim)
#     small = reshape(vec, batch * mm.num, mm.alg.dim)
#     # convert each small vector block via the base Algebra.vector_to_matrix
#     mats  = vector_to_matrix(mm.alg, small)    # (batch*num, N, N)
#     # reshape into (batch, num, N, N)
#     return reshape(mats, batch, mm.num, mm.N, mm.N)
# end

# # 1) “Real” implementation: only Array{Float64,2}
# function vector_to_matrix(mm, vec::Array{Float64,2})
#     batch, D = size(vec)
#     @assert D == mm.dim

#     # reshape into (batch*num, alg.dim)
#     d_alg = mm.dim_b ÷ mm.num_b
#     small = reshape(vec, batch*mm.num, d_alg)

#     # convert each small vector → N×N
#     mats = vector_to_matrix(mm.alg, small)   # (batch*num, N, N)

#     # reshape back to (batch, num, N, N)
#     return reshape(mats, batch, mm.num, mm.N, mm.N)
# end

# # 2) Float32 fallback: convert to Float64 and dispatch above
# function vector_to_matrix(mm, vec::Array{Float32,2})
#     return vector_to_matrix(mm, Float64.(vec))
# end

# # 3) Generic fallback: any other Number type (e.g. Int, UInt, etc.)
# function vector_to_matrix(mm, vec::AbstractMatrix{T}) where {T<:Real}
#     return vector_to_matrix(mm, Float64.(vec))
# end



# Block matrix→vector: (batch, num, N, N) → (batch, dim)
# function matrix_to_vector(mm::MatrixModel, mats::Array{ComplexF64,4})
#     batch, num, N1, N2 = size(mats)
#     @assert num == mm.num && N1 == mm.N && N2 == mm.N
#     # flatten to (batch*num, N, N)
#     flat  = reshape(mats, batch * num, N1, N2)
#     # project via Algebra.matrix_to_vector
#     small = matrix_to_vector(mm.alg, flat)     # (batch*num, alg.dim)
#     # reshape into (batch, dim)
#     return reshape(small, batch, mm.dim)
# end

# # 1) “Real” implementation: only Array{ComplexF64,4}
# function matrix_to_vector(mm, mats::Array{ComplexF64,4})
#     batch, num, N1, N2 = size(mats)
#     @assert num == mm.num && N1 == mm.N && N2 == mm.N

#     # flatten into (batch*num, N, N)
#     flat = reshape(mats, batch*num, N1, N2)

#     # delegate to the LieAlgebra
#     small = matrix_to_vector(mm.alg, flat)   # (batch*num, alg.dim)

#     # reshape back to (batch, dim)
#     return reshape(small, batch, mm.dim)
# end

# # 2) ComplexF32 fallback: convert to ComplexF64 and dispatch above
# function matrix_to_vector(mm::MatrixModel, mats::Array{ComplexF32,4})
#     return matrix_to_vector(mm, ComplexF64.(mats))
# end

# # 3) Generic numeric fallback: any 4-d Number array
# function matrix_to_vector(mm::MatrixModel, mats::AbstractArray{T,4}) where {T<:Number}
#     return matrix_to_vector(mm, ComplexF64.(mats))
# end


# Delegate action/infinitesimal_action to the underlying algebra
function action(mm, g::Array{ComplexF64,3}, x::AbstractArray{ComplexF64})
    return action(mm.alg, g, x)
end

function infinitesimal_action(mm, dg::AbstractMatrix{<:Real}, x::AbstractArray{ComplexF64})
    return infinitesimal_action(mm.alg, dg, x)
end

end # module Algebra