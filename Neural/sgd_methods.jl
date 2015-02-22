function sgd(model::Model,
             X::Array{Float64,2},
             Y::Array{Float64,2},
             numEpochs::Integer,
             alpha::Real = 0.1,
             eta::Real = 0.5,
             batchSize::Integer = 32)
    N = size(X, 2)
    momentums = {zeros(d) for d in getDims(model)}
    for i in 1:numEpochs
        MSE = 0.0
        indices = shuffle([1:N])
        for n in 1:batchSize:N
            tmpMSE, grads = gradient(model,
                X[:, indices[n:min(n+batchSize-1, end)]],
                Y[:, indices[n:min(n+batchSize-1, end)]])
            MSE += tmpMSE
            momentums = {eta * m - alpha * g for (m, g) in zip(momentums, grads)}
            update(model, momentums)
        end
        MSE /= fld((N + batchSize - 1), batchSize)
        println("Epoch $i, MSE = $MSE")
    end
    model
end

