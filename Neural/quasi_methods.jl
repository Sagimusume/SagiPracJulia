const QN_ITER_EPS = 1.0e-1
const QN_HESSE0 = 1.0e-2


function find_step(model::QNN,
                   B::Array{Float64,2},
                   X::Array{Float64,2},
                   Y::Array{Float64,2},
                   grads::Array{Any,1})
    d = -B*cell2vec(grads)
    d = [vec2cell(model,d,l) for l = 1:model.L]
    alpha = line_search(model,X,Y,d,grads)
    return alpha*d
end


function quasi_newton_method(model::QNN,
                             X::Array{Float64,2},
                             Y::Array{Float64,2},
                             numEpochs::Integer)
    W = deepcopy(model.w)
    SME, grads = gradient(model,X,Y)
    #ヘッセ行列の逆行列
    B = eye(length(model.w))/QN_HESSE0
    αD = find_step(model,B,X,Y,grads)
    update(model,αD)

    for i in 1:numEpochs
        tmpSME, next_grads = gradient(model,X,Y)
        y = cell2vec(next_grads - grads)
        s = model.w - W
        #BFGS
        sᵀy = (s'y)[1]
        A = B*y*s'
        B += (sᵀy + y'B*y)[1]/(sᵀy)^2 * (s*s') - (A + A')/sᵀy

        αD = find_step(model,B,X,Y,next_grads)
        W = deepcopy(model.w)
        update(model, αD)

        grads = next_grads
        if reduce(&,map(norm,αD).< QN_ITER_EPS)
            break
        end

        println("SME = $(tmpSME/size(X,2)) Epoch =",i)
    end
end



