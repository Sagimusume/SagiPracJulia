#MNISTデータ編集
function preprocess{R}(data::(Array{R,2},Array{R,1}))
    trainX, trainLabels = data
    trainX /= maximum(trainX)
    N = size(trainX, 2)
    trainY = zeros(length(Set(trainLabels)), N)
    # 1-of-k 符号法
    for n in 1:N
        trainY[trainLabels[n]+1, n] = 1
    end
    trainX, trainY
end

#Cell2vec
function cell2vec(v::Array{Any,1})
    c = Float64[]
    for i = 1:length(v)
        append!(c,vec(v[i]))
    end
    return c
end

function vec2cell(ann::Model, p::Vector{Float64}, layer::Integer)
    offset = 1
    for l in 1:layer-1
        offset += (ann.dims[l]+1) * ann.dims[l+1]
    end
    reshape(view(p, offset:offset+(ann.dims[layer]+1)*ann.dims[layer+1]-1),
            ann.dims[layer+1], ann.dims[layer]+1)
end


#プロット用
function meshgrid{T<:Real}(X::Union(Vector{T},Range{T}),
                           Y::Union(Vector{T},Range{T}))
    gr = []
    for x = X
        for y = Y
            append!(gr,[x,y])
        end
    end
    reshape(gr,2,(length(X)*length(Y)))
end

#スコア計測
function score(model::Model,
                  X::Array{Float64,2},
                  Y::Array{Float64,2})
    prediction, outputs = predict(model, X)
    N = size(Y, 2)
    correct = 0
    for n in 1:N
        c1 = indmax(prediction[:, n])
        c2 = indmax(Y[:, n])
        if c1 == c2
            correct += 1
        end
    end
    accuracy = correct / N
    correct, accuracy
end
#正答率
function test(model,testX, testY)

    correct, accuracy = score(model, testX, testY)
    accuracy *= 100
    println("$correct correct predictions ($accuracy% accuracy) on test set")
end
