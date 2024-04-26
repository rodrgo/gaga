% 27. August 2014
% Test speed of mysvd (matlab). mysvd (CUDA) with svds


clear all

addpath /home/suic/Desktop/Dissertation/Test/Auxiliary/

n_array = [ 50,100,200,300,400,500,600, 700, 800, 900,1000,1500,2000,2500,3000,3500, 4000,4500, 5000];
numberTests = 5;
ens = 1; % 1=Gaussian, 2 = Uniform as the distributions for the random data 
         % which create mxr matrix A, and rxn matrix B, to form Mat=AB.
maxIter = 5000;

timesvds2 = zeros(numberTests, length(n_array));
timesvds4 = zeros(numberTests, length(n_array));
timesvds6 = zeros(numberTests, length(n_array));
timesvds8 = zeros(numberTests, length(n_array));
timemysvd2 = zeros(numberTests, length(n_array));
timemysvd4 = zeros(numberTests, length(n_array));
timemysvd6 = zeros(numberTests, length(n_array));
timemysvd8 = zeros(numberTests, length(n_array));
timemysvdGPU2 = zeros(numberTests, length(n_array));
timemysvdGPU4 = zeros(numberTests, length(n_array));
timemysvdGPU6 = zeros(numberTests, length(n_array));
timemysvdGPU8 = zeros(numberTests, length(n_array));

GPUtimemysvdGPU8 = zeros(numberTests, length(n_array));


% errorsvds = zeros(numberTests, length(n_array));
% errormysvd = zeros(numberTests, length(n_array));
% errormysvdGPU = zeros(numberTests, length(n_array));


for n_it = 1:length(n_array)
    
    for test = 1: numberTests

        n = n_array(n_it)
        m =n;
        r = round(n*0.1);

        A=randn(m,n);
        g_Mat=single(A);
        U0 = randn(m,r);
        g_U0 = single(U0);
        
        
   
        
        %%
%         Tol = 1e-2;
%         options.tol = Tol;
%         
%          
%         ticsvds = tic;
%         [U,S,V]=svds(A,r, 'L',options);
%         timesvds2(test, n_it) =toc(ticsvds);
%         
%         ticGPU = tic;
%         [gU,gS,gV,gTime] = gpuPartialSVD_SPI(g_Mat, int32(m), int32(n), int32(r), g_U0, single(Tol), int32(maxIter));
%         timemysvdGPU2(test, n_it)=toc(ticGPU);
% 
%         %errormysvdGPU(test, n_it) = norm(gU*diag(gS)*gV' - single(U*S*V'), 'fro')/norm(single(U*S*V'),'fro');
% 
%         ticmysvd = tic;
%         [myU, myS, myV] = mysvd(A, r, U0, Tol);
%         timemysvd2(test, n_it) = toc(ticmysvd);
% 
%         %errormysvd2(test, n_it) = norm(myU*myS*myV' -single(U*S*V'), 'fro')/norm(single(U*S*V'), 'fro');
% 
%          %%
%         Tol = 1e-4;
%         options.tol = Tol;
%         
%          
%         ticsvds = tic;
%         [U,S,V]=svds(A,r, 'L',options);
%         timesvds4(test, n_it) =toc(ticsvds);
%         
%         ticGPU = tic;
%         [gU,gS,gV,gTime] = gpuPartialSVD_SPI(g_Mat, int32(m), int32(n), int32(r), g_U0, single(Tol), int32(maxIter));
%         timemysvdGPU4(test, n_it)=toc(ticGPU);
% 
%         %errormysvdGPU2(test, n_it) = norm(gU*diag(gS)*gV' - single(U*S*V'), 'fro')/norm(single(U*S*V'),'fro');
% 
%         ticmysvd = tic;
%         [myU, myS, myV] = mysvd(A, r, U0, Tol);
%         timemysvd4(test, n_it) = toc(ticmysvd);
% 
%         %errormysvd(test, n_it) = norm(myU*myS*myV' -single(U*S*V'), 'fro')/norm(single(U*S*V'), 'fro');
% 
% 
%          %%
%         Tol = 1e-6;
%         options.tol = Tol;
%         
%          
%         ticsvds = tic;
%         [U,S,V]=svds(A,r, 'L',options);
%         timesvds6(test, n_it) =toc(ticsvds);
%         
%         ticGPU = tic;
%         [gU,gS,gV,gTime] = gpuPartialSVD_SPI(g_Mat, int32(m), int32(n), int32(r), g_U0, single(Tol), int32(maxIter));
%         timemysvdGPU6(test, n_it)=toc(ticGPU);
% 
%         %errormysvdGPU2(test, n_it) = norm(gU*diag(gS)*gV' - single(U*S*V'), 'fro')/norm(single(U*S*V'),'fro');
% 
%         ticmysvd = tic;
%         [myU, myS, myV] = mysvd(A, r, U0, Tol);
%         timemysvd6(test, n_it) = toc(ticmysvd);
% 
%         %errormysvd(test, n_it) = norm(myU*myS*myV' -single(U*S*V'), 'fro')/norm(single(U*S*V'), 'fro');

        
        %%

        Tol = 1e-8;
        options.tol = Tol;
        
         
        ticsvds = tic;
        [U,S,V]=svds(A,r, 'L',options);
        timesvds8(test, n_it) =toc(ticsvds);
        
        ticGPU = tic;
        [gU,gS,gV,gTime] = gpuPartialSVD_SPI(g_Mat, int32(m), int32(n), int32(r), g_U0, single(Tol), int32(maxIter));
        timemysvdGPU8(test, n_it)=toc(ticGPU);
        GPUtimemysvdGPU8(test, n_it) = gTime(2);

        %errormysvdGPU2(test, n_it) = norm(gU*diag(gS)*gV' - single(U*S*V'), 'fro')/norm(single(U*S*V'),'fro');

        ticmysvd = tic;
        [myU, myS, myV] = mysvd(A, r, U0, Tol);
        timemysvd8(test, n_it) = toc(ticmysvd);

        %errormysvd(test, n_it) = norm(myU*myS*myV' -single(U*S*V'), 'fro')/norm(single(U*S*V'), 'fro');


    end

end

load('SVDS_MYSVD_GPU.mat', 'timesvds2', 'timemysvd2', 'timemysvdGPU2', 'timesvds4', 'timemysvd4', 'timemysvdGPU4', 'timesvds6', 'timemysvd6', 'timemysvdGPU6')
figure
subplot(1,3,1)
semilogy(n_array, mean(timesvds2),  n_array, mean(timesvds4), n_array, mean(timesvds6),n_array, mean(timesvds8))
xlabel('matrix dimension n')
ylabel('computation time in sec')
axis([0,5000,1e-3, 1e3])
title('svds')

subplot(1,3,2)
semilogy(n_array, mean(timemysvd2), n_array, mean(timemysvd4),n_array, mean(timemysvd6),n_array, mean(timemysvd8))
xlabel('matrix dimension n')
ylabel('computation time in sec')
axis([0,5000,1e-3, 1e3])
title('mysvd MATLAB')

subplot(1,3,3)
semilogy(n_array, mean(timemysvdGPU2), n_array, mean(timemysvdGPU4),n_array, mean(timemysvdGPU6),n_array, mean(timemysvdGPU8))
xlabel('matrix dimension n')
ylabel('computation time in sec')
axis([0,5000,1e-3, 1e3])
legend('tol= 10^{-2}','tol= 10^{-4}','tol= 10^{-6}','tol= 10^{-8}', 'Location', 'NorthEastOutside')
title('mysvd CUDA')


%save('SVDS_MYSVD_GPU.mat', 'timesvds2', 'timemysvd2', 'timemysvdGPU2', 'timesvds4', 'timemysvd4', 'timemysvdGPU4', 'timesvds6', 'timemysvd6', 'timemysvdGPU6')

save('SVDS_MYSVD_GPU.mat', 'timesvds8', 'timemysvd8', 'timemysvdGPU8', '-append')