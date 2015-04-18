function hmc_sampler(current_q::Array{Float64},U::Function,grad_U::Function,ϵ::Float64, L::Int64)
    #q = 一般化座標   -> x ->Sample
    #p = 一般化運動量 -> r->Momentum
    #  = 初期座標x0を適当にとる->座標qにおける勾配を元に運動量(従って底にて運動量が最大化する)をえる。運動量の初期値は正規分布に従いランダムに選ばれる
    #    Loop-運動量に従い座標を微小変化させる->end-loop->現在のハミルトニアンと過去のハミルトニアンの差をexpに入れ。->end one step

    q = current_q
    p = randn(length(q))
    current_p = p
    #リープフロッグ
    for i = 1:L
        p += ϵ * grad_U(q) / 2.
        q += ϵ * p
        p += ϵ * grad_U(q) / 2.
    end
    #リープフロッグおわり

    #ハミルトニアン計算
    current_U = U(current_q)
    current_K = sum(current_p.^2) / 2
    proposed_U = U(q)
    proposed_K = sum(p.^2) / 2

    current_H  = -current_U + current_K
    proposed_H = -proposed_U + proposed_K

    #棄却判定
    if rand() < exp(current_H - proposed_H)
        return q, proposed_U
    else
        return current_q, current_U
    end
end

function HMC(x::Array{Float64},f::Function,gradient::Function,epsilon::Float64,L::Int64, Iter::Int64)
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

#目的関数
ρ = 0.98
μ = [0, 0]
S  = [1 ρ; ρ 1]
Sⁱ = inv(S)
detS = det(S)

#f(x) = log(exp(-(x-μ)'Sⁱ*(x-μ)/2)/(2π*sqrt(detS)))[1]
f(x) = (-(x-μ)'Sⁱ*(x-μ)/2)[1]

g(x) = -Sⁱ*(x-μ)/2
#

x0 = [11.1, 0.5]
Iter = 1000
epsilon = 0.01
L = 50

Xs,progress = HMC(x0, f, gradient, epsilon ,L ,Iter)

X = linspace(-3., 3.)
Y = linspace(-3., 3.)
Z = Float64[exp(f([x, y])) for y=Y, x=X]

xs = Xs[:,1]
ys = Xs[:,2]

using PyPlot

pcolor(X,Y,Z,alpha=0.5)
plot(xs,ys)
scatter(xs,ys)
