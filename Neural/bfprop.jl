#forward-propagete
function predict(ann::Model,
                 X::Matrix)
    outputs = {}
    Z = X
    for l in 1:ann.L
        Z = vcat(Z, ones(1, size(Z, 2))) # add bias
        push!(outputs, Z) #Datalayer?
        W = getWeights(ann, l) #??
        A = W * Z #InnerProducts
        if l < ann.L
            Z = tanh(A) #Neuron
        else
            Z = softmax(A, 1) #Softmaxloss
        end
    end
    push!(outputs, Z)
    Z, outputs
end
#back-propagate
function gradient(ann::Model,
                  X::Matrix,
                  T::Matrix)
    gradients = {}
    N = size(X, 2)
    Y, outputs = predict(ann, X)
    dEdX = Y - T
    MSE = sum(dEdX.^2) / (2 * N)
    for l in ann.L:-1:1
        W = getWeights(ann, l)
        Z = outputs[l+1]
        X = outputs[l]
        if l == ann.L
            Deltas = dEdX #SoftMaxlossLayer
        else
            Zd = 1 - Z .* Z
            Deltas = Zd .* dEdX #Inner-productLayer
            Deltas = Deltas[1:end-1, :] # Get rid of bias
        end
        unshift!(gradients, Deltas * X' / N)
        if l > 1
          dEdX = W' * Deltas#ErrorPropagate
        end
    end
    MSE, gradients
end
