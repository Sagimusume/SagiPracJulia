using PyPlot, Distributions

function hmc_sampler(current_q,U::Function,grad_U::Function,ϵ::Float64, L::Int64)
    #q = 一般化座標   -> x ->Sample
    #p = 一般化運動量 -> r->Momentum
    #  = 初期値を乱数でとる->座標qにおける勾配を元に運動量(従って底にて運動量が最大化する)運動量の初期値は正規分布に従いランダムに選ばれる
    #    Loop-運動量に従い座標を微小変化させる->end-loop->現在のハミルトニアンと過去のハミルトニアンの差をexpに入れ再利用するか判断する。->end one step

    q = current_q
    p = randn(length(q))[1]
    current_p = p
    #リープフロッグ
    for i = 1:L
        p += ϵ * grad_U(q) / 2
        q += ϵ * p
        p += ϵ * grad_U(q) / 2
    end
    #リープフロッグおわり

    #ハミルトニアン計算
    current_U = U(current_q)
    current_K = sum(current_p.^2) / 2
    proposed_U = U(q)
    proposed_K = sum(p.^2) / 2

    current_H  = -current_U + current_K
    proposed_H = -proposed_U + proposed_K


    if rand() < exp(current_H - proposed_H)
        return q, proposed_U
    else
        return current_q, current_U
    end
end

function HMC(x,f::Function,gradient::Function,epsilon::Float64,L::Int64, Iter::Int64)
    P = length(x)
    xs = zeros((Iter,P))
    progress = zeros(Iter)
    for iter = 1:Iter
        x, fx = hmc_sampler(x, f, gradient, epsilon, L)
        xs[iter,:] = x
        progress[iter] = fx
    end
    return xs, progress
end
#################### Objective function #####################
u = 4

f(x) = log((2/3)*exp(-1/2*x.^2) + (1/3)*exp(-1/2*(x-u).^2))
g(x,n) = -(exp(n*x)*(x-n)+2*exp(n^2/2)*x)/(exp(n*x)+ 2*exp(n^2/2))
g(x) = g(x,u)

######################## Main #############################
x0 = 0.1
Iter = 10000
epsilon = 0.01
L = 50

Xs,progress = HMC(x0, f, g, epsilon ,L ,Iter)

####################### Plot #############################
n, bins, patches = PyPlot.hist(Xs,bins=100,normed=true)
a(x) = 2/3*pdf(Normal(),x) + 1/3*pdf(Normal(u),x)
line = a(bins)
PyPlot.plot(bins,line,"r--",linewidth=1)
