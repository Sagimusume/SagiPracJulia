function hmc_sampler(current_q::Array{Float64},U::Function,grad_U::Function,ϵ::Float64, L::Int64)
    #q = 一般化座標   -> x ->Sample
    #p = 一般化運動量 -> r ->Momentum

    q = current_q
    p = randn(length(q))
    current_p = p
    #リープフロッグ
    for i = 1:L
        p += ϵ * grad_U(q) / 2.
        q += ϵ * p
        p += ϵ * grad_U(q) / 2.
    end

    #ハミルトニアン計算
    current_U = U(current_q)
    current_K = sum(current_p.^2) / 2
    proposed_U = U(q)
    proposed_K = sum(p.^2) / 2

    current_H  = -current_U + current_K
    proposed_H = -proposed_U + proposed_K

    #棄却判定?
    if rand() < exp(current_H - proposed_H)
        return q
    else
        return current_q
    end
end

function HMC(x::Array{Float64},f::Function,gradient::Function,epsilon::Float64,L::Int64, Iter::Int64)
    P = length(x)
    xs = zeros((Iter,P))
    for i = 1:Iter
        x = hmc_sampler(x, f, gradient, epsilon, L)
        xs[i,:] = x
    end
    return xs
end

################# Objectiv function ####################
ρ = 0.98
μ = [0, 0]
S  = [1 ρ; ρ 1]
Sⁱ = inv(S)

ρ2 = 0.5
μ2 = [8., 3.]
S2  = [1 ρ2; ρ2 1]
Sⁱ2 = inv(S2)

f(x) = log((1/3*exp(-(x-μ)'Sⁱ*(x-μ)/2)) + 2/3*exp(-(x-μ2)'Sⁱ2*(x-μ2)/2))[1]

g(x) = (1/3*exp(-(x-μ)'Sⁱ*(x-μ)/2)[1]*(-Sⁱ*(x-μ)/2)) + (2/3*exp(-(x-μ2)'Sⁱ2*(x-μ2)/2)[1]*-Sⁱ2*(x-μ2)) /
                      (1/3*exp(-(x-μ)'Sⁱ*(x-μ)/2))[1] + (2/3*exp(-(x-μ2)'Sⁱ2*(x-μ2)/2))[1]

############### Main ######################
x0 = [0.0, 0.0]#[2.1, 4.5]
Iter = 10000
epsilon = 0.01
L = 50

Xs = HMC(x0, f, gradient, epsilon ,L ,Iter)

############### Plot ######################
X = linspace(-3., 10.)
Y = linspace(-3., 8.)
Z = Float64[exp(f([x, y])) for y=Y, x=X]

xs = Xs[:,1]
ys = Xs[:,2]

using PyPlot

pcolor(X,Y,Z,alpha=0.5)
plot(xs,ys)
scatter(xs,ys)
