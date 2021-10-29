# written by Emad Ghalenoei in 2021
# (c) Emad.Ghalenoei, 2021
# This is a Julia script for Inversion of gravity and magentic data
# This code does not contain data and kernel. Therfore, I simulated random numbers for them. You need to use your own data and kernels.


using LinearAlgebra
using NearestNeighbors

Ndatapoints = 30 # number of data points
dis_s = range(0, stop = 1, length = Ndatapoints)
dg_obs = rand(Float64, (Ndatapoints,1)) # gravity data
dT_obs = rand(Float64, (Ndatapoints,1)) # magnetic data
#println.(dis_s)
Nx = 100 # number of regular cells in x axis
Nz = 100 # number of regular cells in z axis
x = range(0, stop = 1, length = Nx) # x vector assuming each cell has 1 meter dimension
z = range(0, stop = 1, length = Nz) # z vector assuming each cell has 1 meter dimension
X = x' .* ones(Nz)                  #2D X model space
Z = ones(Nx)' .* z                  #2D Z model space
#display("text/plain", X)
#display("text/plain", Z)
Y = copy(X)
DISMODEL = copy(X)

X = vec(X)
Z = vec(Z)

Kernel_Grv = rand(Float64, (Ndatapoints,Nx*Nz)) #define the gravity kernel here
Kernel_Mag = rand(Float64, (Ndatapoints,Nx*Nz)) #define the magnetic kernel here
TrueDensityModel = zeros(Nx,Nz)                 #define True Density Model
TrueDensityModel[90:100,20:80] .= 0.35
TrueDensityModel[80:89,40:60] .= 0.25
TrueDensityModel[60:79,50:55] .= -0.25
TrueSUSModel = TrueDensityModel/50              #define True Sus Model


dg_true = Kernel_Grv * vec(TrueDensityModel) # Unit(mGal)
dT_true = Kernel_Mag * vec(TrueSUSModel)      # Unit(nT)

Nnode = 30
xc = rand(Float64, (Nnode,1))     #initial x parameters
zc = rand(Float64, (Nnode,1))     #initial z parameters
rhoc = rand(Float64, (Nnode,1))   #initial density contrast parameters

#@time begin
for imcmc in 1:200000
    dist = zeros(Nx * Nz, Nnode)
    for i in 1:Nnode
        dist[:, i] = (((X .- xc[i, 1]) .^ 2) + ((Z .- zc[i, 1]) .^ 2)) .^ 0.5   # compute distance
    end

    v, ind = findmin(dist, dims = 2)   # knn
    ind = map(t -> t[2], ind)          # knn
    density_model = rhoc[ind, 1]       # density model from knn partitioning
    rg = dg_obs - Kernel_Grv * density_model     #gravity residuals

    sus_model = density_model / 50
    sus_model[density_model.<0.2] .= 0
    rT = dT_obs - Kernel_Mag * sus_model         #magnetic residuals

    N = length(rg)
    sqN = sqrt(N)
    sigma_g = norm(rg) / sqN               # standard deviation of gravity residuals
    sigma_T = norm(rT) / sqN               # standard deviation of magnetic residuals
    LogL = -N * log(sigma_g * sigma_T)     # LogLikelihood function
end
#end

points = rand(2, Nx*Nz)
data = rand(2, Nnode)
rho = rand(Nnode,1)
kdtree = KDTree(data)
idxs, dists = knn(kdtree, points, 1, true)
idxs2 = copy(hcat(idxs))
density_model_2 = rho[idxs2,1]
