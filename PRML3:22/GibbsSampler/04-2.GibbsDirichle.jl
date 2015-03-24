using Distributions

a =[3, 2, 5]
z = sum([gamma(x) for x = a])
#Dirichlet:pdf
function p(x)
    if x[1] + x[2] > 1
        return -0.1
    end
    return x[1]^a[1] * x[2]^a[2] * (1-x[1]-x[2])^a[3] / z
end

#dirichlet GibbsSampling
function sample_one(x)
    #Gibbs Dirichlet Dist Sampling from Beta Conditional Dist
    new_x = rand(Beta(a[1]+1, a[3]+1))*(1-x[2])
    new_y = rand(Beta(a[1]+1, a[2]+1))*(1-new_x)
    return [new_x, new_y]
end

#SampleingMain
BURNIN = 100
N = 100
x = [0,0]
burn_x = zeros(BURNIN);burn_y=zeros(BURNIN)
for i = 1:BURNIN
    burn_x = x[1]
    burn_y = x[2]
    x = sample_one(x)
end

sample_x = zeros(N); sample_y = zeros(N)

for i = 1:N
    sample_x[i] = x[1]
    sample_y[i] = x[2]
    x = sample_one(x)
end

