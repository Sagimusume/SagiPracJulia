function find_step(model::LQNN,
                   X::Array{Float64,2},
                   Y::Array{Float64,2},
                   p::Array{Float64,1},
                   grads::Array{Any,1})
    #cell array
    p = [vec2cell(model, p,l) for l = 1:model.L]
    #あるみほ・うるふ
    #alpha =  line_search(model,X,Y,p,grads)
    alpha = rand() #乱数振ったほうが収束早いし良い(謎) 0<alpha<1 この範囲以外だと収束悪い(謎)
    return alpha*p
end

function l_bfgsNN(model::LQNN,
                    X::Array{Float64,2},
                    Y::Array{Float64,2},
                    numEpochs::Integer)
    W = deepcopy(model.w)
    SME, grads = gradient(model,X,Y)
    #パラメータ
    k, m = 1, 10
    y = spzeros(length(model.w),numEpochs)
    s = spzeros(length(model.w),numEpochs)
    p = -cell2vec(grads)
    a = zeros(numEpochs)
    b = 0

    αP = find_step(model,X,Y,p,grads)
    update(model,αP)

    for i in 1:numEpochs
        #Caluculate gradient
        tmpSME, next_grads = gradient(model,X,Y)

        #si,yi
        y[:,k] = cell2vec(next_grads - grads)
        s[:,k] = model.w - W

        #two-loop recursion
        for i = k:-1:max(k-m,1)
            a[i] = (s[:,i]'p)[1]/(s[:,i]'y[:,i])[1]
            p = p - a[i]*y[:,i]
        end

        p = (s[:,k]'y[:,k])[1]/(y[:,k]'y[:,k])[1]*p

        for i = max(k-m,1):k
            b = (y[:,i]'p)[1]/(s[:,i]'y[:,i])[1]
            p = p + ((a[i]-b)*s[:,i])
        end

        if k > m
            #Discard vector pair sk−m, yk−m from memory storage;;
            y[:,k-m] = spzeros(length(model.w),1)
            s[:,k-m] = spzeros(length(model.w),1)
            a[k-m] = 0
        end

        αP = find_step(model,X,Y,vec(p),next_grads)

        W = deepcopy(model.w)
        p = -cell2vec(next_grads)
        grads = next_grads


        update(model,αP)


        k += 1
        println("SME = $(tmpSME/size(X,2)) Ephoc = $i")
        #=
        if tmpSME/size(X,2) < 0.0001
            break
        end
        =#
    end
end


