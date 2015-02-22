abstract Model

type ANN <: Model
    w::Array{Float64,1}
    dims::Array{Int64,1}
    L::Int64

    function ANN(dims::Vector, stdDev::Real)
        numWeights = 0
        L = size(dims, 1) - 1
        for l in 1:L
            numWeights += (dims[l]+1) * dims[l+1]
        end
        w = randn(numWeights) * stdDev
        new(w, dims, L)
    end
end

#Quasi_Neuton_Methods neural network

type QNN <: Model
    w::Array{Float64,1}
    wl::Array{Float64,1}
    dims::Array{Int64,1}
    L::Int64

    function QNN(dims::Vector, stdDev::Real)
        numWeights = 0
        L = size(dims, 1) - 1
        for l in 1:L
            numWeights += (dims[l]+1) * dims[l+1]
        end
        w = randn(numWeights) * stdDev
        wl= deepcopy(w)
        new(w, wl, dims, L)
    end
end

type LQNN <: Model
    w::Array{Float64,1}
    wl::Array{Float64,1}
    dims::Array{Int64,1}
    L::Int64

    function LQNN(dims::Vector, stdDev::Real)
        numWeights = 0
        L = size(dims, 1) - 1
        for l in 1:L
            numWeights += (dims[l]+1) * dims[l+1]
        end
        w = randn(numWeights) * stdDev
        wl= deepcopy(w)
        new(w, wl, dims, L)
    end
end
