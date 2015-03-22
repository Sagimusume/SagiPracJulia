using Distributions

ρ = 0.98
μ = [0., 0]
S  = [1. ρ; ρ 1]
Sⁱ = inv(S)
detS = det(S)

α = -0.98

#Objective function
#f(x) = exp(-(x-μ)'Sⁱ*(x-μ)/2)/(2π*sqrt(detS))
f(x)  = pdf(MvNormal(μ,S),x)
#mixtur of gaussianぽい
#g(x) = 1/3*exp(-(x-μ)'Sⁱ*(x-μ)/2)/(2π*sqrt(detS)) + 2/3*exp(-(x-[1.5 ,0.5])'Sⁱ*(x-[1.5 ,0.5])/2)/(2π*sqrt(detS))

function sample_one(x)
    new_x = ρ*x[2] + α*(rand(Normal(ρ*x[2], 1-ρ^2)) - ρ*x[2]) + (1-ρ^2)*sqrt(1-α^2)*rand(Normal())
    new_y = ρ*new_x + α*(rand(Normal(ρ*new_x, 1-ρ^2)) - ρ*new_x) + (1-ρ^2)*sqrt(1-α^2)*rand(Normal())
    return [new_x, new_y]
end

BURNIN = 0
N = 30

x = [-3, 2]

burn_x = zeros(BURNIN);burn_y = zeros(BURNIN)
for i = 1:BURNIN
    burn_x = x[1]
    burn_y = x[2]
    x = sample_one(x)
end

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

