using Distributions

#データのパラメータ
N = 100

xmin = 0; xmax = 10
ymin = 0; ymax = 10

#基底関数パラメタ
BASIS_SIGMA = 3.0
BASIS_COUNT = 4
DIM = BASIS_COUNT^2
#繰り返し回数
ITERATION = 100
#training data
function data()
    x = rand(Uniform(xmin,xmax),N)
    y = rand(Uniform(ymin,ymax),N)
    t = (y .< 5 -(x-2).^2) | ((x-8).^2+(y-7).^2 .< 3)
    return x, y, t
end
#トレーニングデータ
train_x,train_y,train_t = data()

#基底関数座標点
basis_center = [(x,y) for x = linspace(xmin,xmax,BASIS_COUNT),
                          y = linspace(ymin,ymax,BASIS_COUNT)]
#基底定義
gaussian_basis(x::Real,y::Real,cx::Real,cy::Real) =
                          exp(-((x-cx)^2 + (y-cy)^2)/(2*BASIS_SIGMA^2))
#計画行列
X = Float64[gaussian_basis(x,y,cx,cy) for (cx,cy) = basis_center,
                                          (x,y)   = zip(train_x,train_y)]'

#基底関数(Prml. 3.4)
phi(x,y) = Float64[gaussian_basis(x, y, cx, cy) for (cx,cy) in basis_center]

#シグモイド関数
sigmoid(x) = 1./(1+exp(-x))

#===== IRLS ======#
#=
function yR(w)
    s = sigmoid(X*w)
    return (s,diagm(s.*(1-s)))
end

w = zeros(DIM)

for i = 1:ITERATION
    y, R = yR(w)
    w += inv(X'R*X)*X'*(train_t-y)
end
=#
#==== 最急降下法 ====#
#=
function sd()
    α   = 0.2
    Stop = 0.5

    w = zeros(DIM)
    for i = 1:ITERATION
        grad = X'*(train_t - sigmoid(X*w)) - w
        w += α*grad
        grad_norm = norm(grad)
        if grad_norm < Stop
            break
        end
    end
    return w
end
w = sd()
=#
#==== 確率的勾配降下法(逐次処理版)====##(Prml 3.52・5.41)

η = 10.0

function sgd()
    w = zeros(DIM)
    for i = 1:2
        for n = 1:N
            x = train_x[n]
            y = train_y[n]
            t = train_t[n]
            grad = (t - sigmoid(w'phi(x,y))) .* phi(x,y)
            w += η/(i+1) * grad
        end
    end
    return w
end
#パラメータ推定
w = sgd()

#ヘッセ行列::ベイズ予測分布パラメータ作成
H = X'*diagm(sigmoid(X*w).*(1-sigmoid(X*w)))*X + I #(Prml. 4.143)行列表現
Σ = inv(H)
u = w

#予測分布(事後分布をラプラス近似し、予測分布をsigmoid(...)で近似)
function predict(x)
    mu = u'x  #(Prml. 4.149)
    sigma = u'Σ*u #(Prml. 4.150)
    sigmoid(mu[1]/sqrt(sigma[1]*π/8))[1] #(Prml. 4.153)
end

#プロット
using PyPlot
N = 100
xmin = 0; xmax = 10
ymin = 0; ymax = 10
X = linspace(xmin,xmax,100)
Y = linspace(ymin,ymax,100)
Z = Float64[predict((w'phi(x,y))[1]) for x = X,y = Y]
pcolor(X,Y,Z,alpha=0.3)
#scatter(train_x,train_y, c=train_t)

#スコア計測
function score(w)
    X,Y,T = data()

    scatter(X,Y, c=T)

    prediction = Float64[predict((w'phi(X[i],Y[i]))[1]) for i = 1:N]
    c = countnz(int(prediction) .== T)
    accuracy = c
    println("$c correct predictions ($accuracy% accuracy) on test set")
end

score(w)
