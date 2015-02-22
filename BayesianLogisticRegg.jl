#入力次元
D = 2
#データ数
N = 100

x1 = randn(N,D)
x2 = randn(N,D) + 5
#入力データ作成
x = [x1,x2]

t1 = zeros(N)
t2 = ones(N)
#教師データ作成
t = [t1,t2]

sigmoid(a) = 1./(1+exp(-a))
p_y_given(x,w) = sigmoid(x*w.+b)

#パラメータ初期化
w = rand(D)
b = randn()

eta = 0.1
err = Any[]

ITERATION = 500
#勾配
function grad(x,t,w,b)
    err = (t-p_y_given(x,w))
    w = -sum(x'*err,2) + w
    b = -sum(err) + b
    return w,b
end

#勾配降下法
for i = 1:ITERATION
    wg,bg = grad(x,t,w,b)
    #H = x'*diagm((sigmoid(x*w)).*(1-sigmoid(x*w)))*x
    w -= eta * wg + w/100
    b -= eta * bg + b/100
end
#ラプラス近似パラメータ
H = I + sum([p_y_given(x[i,:],w).*(1-p_y_given(x[i,:],w)).*x[i,:]'*x[i,:] for i = 1:N])
Σ = inv(H)
u = w

#予測分布
function predict(x)
    mu = u'x + b
    sigma = u'Σ*u
    sigmoid(mu/sqrt(sigma*π/8))[1]
end

#プロット
using PyPlot

N = 100
xmin = -10; xmax = 10
ymin = -10; ymax = 10
X1 = linspace(xmin,xmax,100)
Y1 = linspace(ymin,ymax,100)
Z1 = Float64[predict([x,y]) for x = X1,y = Y1]
pcolor(X1,Y1,Z1,alpha=0.3)

scatter(x1[:,1],x1[:,2],c="r")
scatter(x2[:,1],x2[:,2],c="b")
