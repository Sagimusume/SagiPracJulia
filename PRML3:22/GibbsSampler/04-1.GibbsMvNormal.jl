using Distributions, PyPlot
##################### Objective function ####################
#####使いません#####
ρ = 0.98
μ = [0, 0]
S  = [1 ρ; ρ 1]
Sⁱ = inv(S)
detS = det(S)

f(x) = exp(-(x-μ)'Sⁱ*(x-μ)/2)/(2π*sqrt(detS))

#################### Gibbs Sampler ########################
function sample_one(x)
    new_x = rand(Normal(ρ*x[2], 1-ρ^2))  # Conditional distribution of f1
    new_y = rand(Normal(ρ*new_x, 1-ρ^2)) # conditional distributon of f2
    return [new_x, new_y]
end

######################## Main #############################
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


################## Plot ###########################
X = linspace(-3., 3.)
Y = linspace(-3., 3.)
Z = Float64[f([x,y])[1] for x = X,y=Y]

pcolor(X,Y,Z,alpha=0.5)
scatter(xs, ys)

