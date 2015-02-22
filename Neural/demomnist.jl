using MNIST


function score(model::Model,
                  X::Array{Float64,2},
                  Y::Array{Float64,2})
    prediction, outputs = Neural.predict(model, X)
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



function demo_qnewton(numEpochs::Integer = 10,
                      stdDev::Real = 0.05,
                      H::Vector = [100])
    trainX, trainY = preprocess(traindata())
    D = size(trainX, 1)
    F = size(trainY, 1)
    model = QNN([D, H, F], stdDev)
    model = quasi_newton_method(model,
                             trainX,
                             trainY,
                             numEpochs)
end

function demo_lqnewton(numEpochs::Integer = 100,
                      stdDev::Real = 0.05,
                      H::Vector = [100])
    trainX, trainY = Neural.preprocess(traindata())
    D = size(trainX, 1)
    K = size(trainY, 1)
    model = LQNN([D, H, K], stdDev)
    l_bfgsNN(model,
             trainX,
             trainY,
             numEpochs)
    return model
end


function demo_ann(numEpochs::Integer = 10,
                  alpha::Real = 0.1,
                  eta::Real = 0.5,
                  batchSize::Integer = 32,
                  stdDev::Real = 0.05,
                  H::Vector = [100])
    trainX, trainY = preprocess(traindata())
    D = size(trainX, 1)
    F = size(trainY, 1)
    model = ANN([D, H, F], stdDev)
    model = sgd(model, trainX, trainY, numEpochs, alpha, eta, batchSize)
end

function demo_lqnewton(numEpochs::Integer = 100,
                      stdDev::Real = 0.05,
                      H::Vector = [100])
    trainX, trainY = Neural.preprocess(traindata())
    D = size(trainX, 1)
    K = size(trainY, 1)
    model = LQNN([D, H, K], stdDev)
    l_bfgsNN(model,
             trainX,
             trainY,
             numEpochs)
    return model
end

function test(model)
    testX, testY = Neural.preprocess(testdata())
    correct, accuracy = score(model, testX, testY)
    accuracy *= 100
    println("$correct correct predictions ($accuracy% accuracy) on test set")
end
