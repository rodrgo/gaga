% testGPUPartialSVD

m = 2^11;
n = 2^12;
r = 2^5;
ens = 1; % 1=Gaussian, 2 = Uniform as the distributions for the random data 
         % which create mxr matrix A, and rxn matrix B, to form Mat=AB.
Tol = 0.005;
maxIter = 50;

timeCPU = 0;
timeGPU = 0;
tocGPU = 0;

A=randn(m,n)/sqrt(m);
g_Mat=single(A);

ticGPU = tic;
[g_U,g_S,g_V,gTime] = gpuPartialSVD_v2(g_Mat, int32(m), int32(n), int32(r), single(Tol), int32(maxIter));
tocGPU=toc(ticGPU)

ticCPU = tic;
[cU,cS,cV]=svds(A,r);
tocCPU=toc(ticCPU)

acceleration=tocCPU/tocGPU
timings = [tocCPU tocGPU]

%break;

c_s=diag(cS);
gs=sort(g_S,'descend');
TopTensingvals_rel_diff=(gs(1:10)'-single(c_s(1:10))')./c_s(1:10)'

ReconstructMATrel_normdifference=[norm(g_U*diag(g_S)*g_V'-single(cU*cS*cV'))]/norm(cU*cS*cV')
