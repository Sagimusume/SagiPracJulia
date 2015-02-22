function gradline(ann::Model,
                  X::Matrix,
                  T::Matrix,
                  new_W)
    gradients = {}
    N = size(X, 2)
    Y, outputs = predictline(ann, X, new_W)
    dEdX = Y - T

    for l in ann.L:-1:1
        W = new_W[l]
        Z = outputs[l+1]
        X = outputs[l]
        if l == ann.L
            Deltas = dEdX
        else
            Zd = 1 - Z .* Z
            Deltas = Zd .* dEdX
            Deltas = Deltas[1:end-1, :] # Get rid of bias
        end
        unshift!(gradients, Deltas * X' / N)
        if l > 1
          dEdX = W' * Deltas
        end
    end
    gradients
end

function predictline(ann::Model,
                     X::Matrix,
                     new_W)
    outputs = {}
    Z = X
    for l in 1:ann.L
        Z = vcat(Z, ones(1, size(Z, 2)))
        push!(outputs, Z)
        W = new_W[l]
        A = W * Z
        if l < ann.L
            Z = tanh(A)
        else
            Z = softmax(A, 1)
        end
    end
    push!(outputs, Z)
    Z, outputs
end



