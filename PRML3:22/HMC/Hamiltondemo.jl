using Lora

function f(v::Array{Float64,1})
  x = v[1]
  y = v[2]
  if x < 0
    return -Inf
  end
  2*log(x) - x*y^2 - y^2 + 2*y - 4*x
end
function gradient(v::Array{Float64,1})
  x = v[1]
  y = v[2]
  # TODO: Deal with x < 0 boundary
  dx = 2/x - y^2 - 4
  dy = -2*x*y - 2*y + 2
  [dx, dy]
end


mcmodel = model(f,grad=gradient,init=rand(2))
#mcchain = run(mcmodel, HMC(0.01), SerialMC(101:1000))
mcchain= run(mcmodel, MH(0.5), SerialMC(101:1000))

X = linspace(0,2)
Y = linspace(-1,2)
Z = [exp(f([x, y])) for y=Y, x=X]

xs = mcchain.samples[:,1]
ys = mcchain.samples[:,2]

using PyPlot
pcolor(X,Y,Z)
scatter(xs,ys)
