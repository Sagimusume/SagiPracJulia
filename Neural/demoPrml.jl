using Distributions, Neural

numEpochs = 30
#データ数
N = 200
#学数データパラメータ
#入力層次元
D = 2
#隱れ層次元
M = 4
#出力層次元
K = 4

#学習データ(目的変数)作成
function make_object(x)
    t = zeros(K,N)
    t[1,:] = x[2,:] .> (x[1,:] + 0.7)
    t[2,:] = (x[2,:].>0.3*x[1,:]+0.3) & (~(bool(t[1,:])))
    t[3,:] = (x[2,:] .> 1.2*x[1,:]) & ~(bool(t[1,:]) | bool(t[2,:]))
    t[4,:] = ~(bool(t[1,:])|(bool(t[2,:]) | bool(t[3,:])) )
    return t
end
#学習データ
function train_data()
    x = rand(Uniform(-1,1),(2,N))
    t = make_object(x)
    return x,t
end
#準ニュートン法(BFGS)学習
function demo_qnewton(numEpochs::Integer = 100,
                      stdDev::Real = 0.05,
                      H::Vector = [4])
    trainX, trainY = train_data()
    D = size(trainX, 1)
    K = size(trainY, 1)
    model = QNN([D, H, K], stdDev)
    quasi_newton_method(model,
                             trainX,
                             trainY,
                             numEpochs)
    return model
end

#L-BFGS法
function demo_lqnewton(numEpochs::Integer = 100,
                      stdDev::Real = 0.05,
                      H::Vector = [4])
    trainX, trainY = train_data()
    D = size(trainX, 1)
    K = size(trainY, 1)
    model = LQNN([D, H, K], stdDev)
    l_bfgsNN(model,
             trainX,
             trainY,
             numEpochs)
    return model
end

#確率的勾配降下法(SGD)学習
function demo_ann(numEpochs::Integer = 10,
                  alpha::Real = 0.1,
                  eta::Real = 0.5,
                  batchSize::Integer = 32,
                  stdDev::Real = 0.05,
                  H::Vector = [4])
    trainX, trainY = train_data()
    D = size(trainX, 1)
    K = size(trainY, 1)
    model = ANN([D, H, K], stdDev)
    model = sgd(model, trainX, trainY, numEpochs, alpha, eta, batchSize)
end

#テストデータ
function test_data()
    x = rand(Uniform(-1,1),(2,N))
    t = make_object(x)
    return x,t
end

#プロット
using PyPlot

function plot_test(model,x,t)
    color = (t[1,:]+2*t[2,:] + 3*t[3,:]+4*t[4,:])
    Z = Neural.meshgrid(linspace(-1,1),linspace(-1,1));
    Z = Neural.predict(model,float64(Z));
    Z2 = [indmax(Z[1][:,i]) for i = 1:10000]
    scatter(x[1,:], x[2,:],c=color,s=50,)
    pcolor(linspace(-1,1,100),linspace(-1,1,100),reshape(float(Z2),100,100),alpha=0.15)
end

function initqnn()
    x, t = train_data()
    model = demo_qnewton(500)
    Neural.test(model,test_data()...)
    plot_test(model,x,t)
end

function initsgd()
    x,t = train_data()
    model = demo_ann(500)
    Neural.test(model,test_data()...)
    plot_test(model,x,t)
end

function initlqn()
    x,t = train_data()
    model = demo_lqnewton(100)
    Neural.test(model,test_data()...)
    plot_test(model,x,t)
end
