const LS_C1 = 1.0e-4
const LS_C2 = 0.9
const LS_SCALE_UP = 2.0
const LS_SCALE_DOWN = 0.6
const LS_ITER_MAX = 100
const LS_STEP0 = 10

function crossentropy(model,X,Y)
    E = 0
    Z,outoputs = predict(model,X)

    for i = 1:size(Y,2)
        E -= sum(Y[:,i]'*log(Z[:,i]))
    end
    return E
end

function crossentropy(model,X,Y,W)
    E = 0
    Z,outoputs = predictline(model,X,W)

    for i = 1:size(Y,2)
        E -= sum(Y[:,i]'*log(Z[:,i]))
    end
    return E
end


function line_search(model,X,Y,D,grads)

    alpha = LS_STEP0/(sum(map(norm,D)))

    E = crossentropy(model,X,Y)
    gradEs = grads
    for c = 1:LS_ITER_MAX
        new_W = [getWeights(model,l) for l in 1:model.L]+alpha*D
        new_E = crossentropy(model,X,Y,new_W)
        new_grads = gradline(model,X,Y,new_W)

        armijoの条件
        if new_E > E + LS_C1*alpha*sum(map(sum,(map(.*,D,gradEs))))
            alpha *= LS_SCALE_DOWN
            continue
        end

        wolfeの条件
        if sum(map(sum,(map(.*,D,new_grads)))) <
                            LS_C2*sum(map(sum,(map(.*,D,gradEs))))
            alpha *= LS_SCALE_UP
            continue
        end

        break

    end
    alpha
end
