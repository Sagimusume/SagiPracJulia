using Distributions
#平均[0, 0], 分散[1 ρ; ρ 1] の場合
ρ = 0.98
μ = [0, 0]
S  = [1 ρ; ρ 1]
Sⁱ = inv(S)
detS = det(S)

#目的関数
#f(x) = exp(-(x-μ)'Sⁱ*(x-μ)/2)/(2π*sqrt(detS))

function sample_one(x)
    new_x = rand(Normal(ρ*x[2], 1-ρ^2))
    new_y = rand(Normal(ρ*new_x, 1-ρ^2))
    return [new_x, new_y]
end

#イテレーション回数
N = 30

x = [-3, 2]

xs = zeros(N);ys = zeros(N)

for i = 1:N
    xs[i] = x[1]
    ys[i] = x[2]
    x = sample_one(x)
end

using PyPlot

l = linspace(-3.,3.)
Z = Float64[f([x,y])[1] for x = l,y=l]
pcolor(l,l,Z,alpha=0.5)

scatter(xs, ys)

