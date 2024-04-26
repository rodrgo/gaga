% testGPUPartialSVD_SPI

clear all

addpath /home/suic/Desktop/Dissertation/Test/Auxiliary/

m = 2^7
n = 2^7
r = 2^3
ens = 1; % 1=Gaussian, 2 = Uniform as the distributions for the random data 
         % which create mxr matrix A, and rxn matrix B, to form Mat=AB.
Tol = 1e-5;
maxIter = 500;

timeCPU = 0;
timeGPU = 0;
tocGPU = 0;

A=randn(m,n)/sqrt(m);
g_Mat=single(A);
U0 = randn(m,r);
g_U0 = single(U0);

ticGPU1 = tic;
[g1_U,g1_S,g1_V,g1Time] = gpuPartialSVD_v2(g_Mat, int32(m), int32(n), int32(r), single(Tol), int32(maxIter));
tocGPU1=toc(ticGPU1);

ticGPU2 = tic;
[g2_U,g2_S,g2_V,g2Time] = gpuPartialSVD_SPI(g_Mat, int32(m), int32(n), int32(r), g_U0, single(Tol), int32(maxIter));
tocGPU2=toc(ticGPU2);



ticmysvd = tic;
[myU, myS, myV] = mysvd(A, r, U0, Tol);
tocmysvd = toc(ticmysvd);

ticCPU = tic;
[cU,cS,cV]=svds(A,r);
tocCPU=toc(ticCPU);



accleration1=tocCPU/tocGPU1
acceleration2=tocCPU/tocGPU2
timings = [tocCPU tocGPU1 tocGPU2 tocmysvd]

%break;

c_s=diag(cS);
gs1=sort(g1_S,'descend');
gs2=sort(g2_S,'descend');
mys = diag(myS);

% TopTensingvals_rel_diff1=(gs1(1:10)'-single(c_s(1:10))')./c_s(1:10)'
% TopTensingvals_rel_diff2=(gs2(1:10)'-single(c_s(1:10))')./c_s(1:10)'
% TopTensingvals_rel_diffmy=(mys(1:10)'-c_s(1:10)')./c_s(1:10)'
% TopTensingvals_rel_diffmy_2=(single(mys(1:10))'-gs2(1:10)')./single(mys(1:10))'


ReconstructMATrel_normdifference1=[norm(g1_U*diag(g1_S)*g1_V'-single(cU*cS*cV'),'fro')]/norm(cU*cS*cV', 'fro')
ReconstructMATrel_normdifference2=[norm(g2_U*diag(g2_S)*g2_V'-single(cU*cS*cV'),'fro')]/norm(cU*cS*cV', 'fro')
ReconstructMATrel_normdifferencemy=[norm(myU*myS*myV'-cU*cS*cV','fro')]/norm(cU*cS*cV', 'fro')
ReconstructMATrel_normdifferencemy_2=[norm(single(myU*myS*myV')-g2_U*diag(g2_S)*g2_V','fro')]/norm(myU*myS*myV', 'fro')
