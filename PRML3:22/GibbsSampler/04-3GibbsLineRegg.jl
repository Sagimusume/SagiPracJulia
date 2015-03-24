# MAP推定するよ
using Distributions

xmin = 0
xmax = 1
ymin = -1.5
ymax = 1.5

NGAUSS = 5
GAUSS_S = 0.1 # ガウス基底の分散パラメータ
GAUSS_MU = linspace(xmin, xmax, NGAUSS)

# aの事前分布
a0  = zeros(NGAUSS)
s02 = 1.0e2^2

#σ²の事前分布のパラメータ
const α0 = 1.0e-2
const β0 = 1.0e2

# 学習データ
n1 = 100
const x = linspace(0,1,n1)
const y = sin(3pi*x) .+ rand(n1)*0.3
n = length(x)

# 繰り返し使う行列・ベクトルを作っておく
psi(x1) = Float64[exp(-(x - m).^2/(2*GAUSS_S.^2)) for m in GAUSS_MU,x = x1]
X = psi(x)'
const XᵀX = X'X
const Xᵀy = X'y

# π(a|σ²,D)
function p_a(s2)
    Σ = inv(XᵀX/s2 + 1/s02)
    a_dash = Σ*(Xᵀy/s2 + a0/s02)
    return rand(MvNormal(vec(a_dash),Σ))
end

# π(σ²|a,D)
function p_s2(a)
    α = α0 + n
    β = β0 + norm(y-X*a)^2
    rand(InverseGamma(α/2,β/2))
end

# ギブスサンプラー
function sampler(BURNIN)
    a  = zeros(NGAUSS)
    s2 = 1.0
    for i = 1:BURNIN
        a  = p_a(s2)
        s2 = p_s2(a)
    end
    while true
        a  = p_a(s2)
        s2 = p_s2(a)
        produce(a,s2)
    end
end


NSAMPLES = 10000
a_samples = zeros(NSAMPLES,NGAUSS)
s2_samples = zeros(NSAMPLES)

sim = @task sampler(1000)
for i = 1:NSAMPLES
    a_samples[i;:], s2_samples[i] = consume(sim)
end


a_MAP  = mean(a_samples,1)
s2_MAP = mean(s2_samples)

xmin = 0;  xmax = 1
ymin = -1; ymax = 5
t = linspace(0,1,100)

function predictive(sim, x)
    a,s2 = consume(sim)
    rand(Normal((psi(x)'a)[1], sqrt(s2)))
end

means   = Float64[]
ys_high = Float64[]
ys_low  = Float64[]
averages = Float64[]
xs = linspace(xmin, xmax,100)

for t = xs
    NGAUSS = 100
    average = psi(t)'vec(a_MAP)
    s = sqrt(mean([(average[1] - predictive(sim, t))^2 for i = 1:NGAUSS]))
    append!(averages,average)
    append!(ys_high,average + s)
    append!(ys_low,average - s)
end

using PyPlot

