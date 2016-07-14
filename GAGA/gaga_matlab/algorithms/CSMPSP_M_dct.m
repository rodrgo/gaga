function [vec, residNorm_prev, iter, time_sum, ksum] = CSMPSP_M_dct(y, dct_rows, k, m, n, tol, maxiter)
% This function has the usage 
% [vec, residNorm_prev, iter, time_sum, ksum] = CSMPSP_M_dct(y, dct_rows, k, m, n, tol, maxiter);
% y is the input measurements, 
% dct_rows are the subsampled rows of the DCT matrix,
% k is the assumed sparsity, 
% m is the number of measurements, 
% n is the dimension of the signal one attempts to recover,
% tol is the error tolerance, 
% maxiter is a maximum number of iterations.
% The outputs are:
%  vec= the approximation to the sparse vector which produced y;
%  residNorm_prev is a vector of the previous 16 residual norms;
%  iter is the number of iterations from the algorithm;
%  time_sum is the cummulative sum of time taken in each iteration;
%  ksum is the number of elements in the bin containing the kth largest
%  element.
% The problem parameters m (number of measurements) and n (ambient
% dimension of the signal and vec) are passed globally.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% One difference in the matlab code is that we never employ the idea of
% only counting a fraction of the bins.  This is because the counting must
% be done in serial anyway, so there is no advantage in counting only a
% subset of the bins.  The advantage in GAGA comes from reducing the number
% of conflicts with atomicAdd by not couting the bins containing the
% smallest entries.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Jeffrey D. Blanchard and Jared Tanner, 2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%
% Initialization %
%%%%%%%%%%%%%%%%%%

%ensure initial approximation is the zero vector
vec = zeros(n,1);

num_bins = max(round(n/20),1000);

iter = 0;
time_sum = 0;
mu = 0;
kbin = 1;
grad_kbin = 1;

minValue = 0;
maxChange = 0;


resid = y;


err = norm(resid);

err_start = err;

resid_tol = tol;

residNorm_diff = 1;

residNorm_prev=zeros(1,16);         % Stores the previous 16 residual norms.
residNorm_evolution=ones(1,16);    % Stores the previous 16 changes in residual norms.

% Some variables for the CG projection step.
tolCG = resid_tol*resid_tol;
iterCG=0;
maxiterCG=15;
errCG=1;
tol2 = 10*tolCG;

%residNorm_prevCG=zeros(1,16);
%residNorm_evolutionCG=ones(1,16);

% Some stopping parameters
fail = 0;
convergenceRate = 0;
                                  
%%%%%%%%%%%%%%%%%%%%%%%%%
%  Set the Linear Bins  %
%%%%%%%%%%%%%%%%%%%%%%%%%

vec = AT_dct(y, m, n, dct_rows);

max_value = MaxMagnitude(vec);

slope = (num_bins/max_value);

bin = zeros(n,1);
bin_counters = zeros(num_bins,1);

[kbin, ksum, minValue, bin, bin_counters] = FindSupportSet(vec, bin, bin_counters, slope, max_value, max_value, minValue, kbin, k, num_bins);


vec = threshold(vec, bin, kbin);

% ******* SET ALL VARIABLES FOR THE CG PROJECTION ****************
iterCG=0;
residNorm_prevCG=zeros(1,16);
residNorm_evolutionCG=ones(1,16);
    
residNorm_diffCG = 1;
residNorm_prevCG(16) = residNorm_prev(16);

resid_update = A_dct(vec, m, n, dct_rows);
resid = y - resid_update;
    
grad = AT_dct(resid, m, n, dct_rows);
grad = threshold(grad, bin, kbin);
    
grad_previous = grad;
    
mu = dot(grad_previous, grad_previous);
errCG = dot(resid, resid);
residNorm_prevCG(16) = errCG;


while ( (errCG > tolCG) && (iterCG < maxiterCG) && (residNorm_diffCG > tol2) )
        
    [vec, grad, grad_previous, residNorm_prevCG, mu] = RestrictedCG_M_dct(vec, grad, grad_previous, y, dct_rows, bin, residNorm_prevCG, kbin, m, n, mu);
        
    errCG = residNorm_prevCG(16);
    iterCG = iterCG + 1;
    
    % check for convergence
    residNorm_evolutionCG(1:15) = residNorm_evolutionCG(2:16);
    residNorm_evolutionCG(16) = residNorm_prevCG(15)-residNorm_prevCG(16);
    residNorm_diffCG = max(residNorm_evolutionCG);
end
        
err = sqrt(errCG);
residNorm_prev(16) = err;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main CSMPSP Projection Loop %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while ( (err>resid_tol) && (iter < maxiter) && (err < 100 * err_start) && (residNorm_diff > .01*tol) && (fail == 0) ) 
    
    timeiter=tic;
    
    grad = AT_dct(resid, m, n, dct_rows);
    
    grad_max = MaxMagnitude(grad);

    grad_slope = (num_bins/grad_max);


    grad_bin = zeros(n,1);
    grad_bin_counters = zeros(num_bins,1);

    [grad_kbin, grad_ksum, grad_min, grad_bin, grad_bin_counters] = FindSupportSet(grad, grad_bin, grad_bin_counters, grad_slope, grad_max, grad_max, 0, grad_kbin, k, num_bins);
    
    bin = joinSupports(bin, grad_bin, kbin, grad_kbin);
    
    
    % ******* RESET ALL VARIABLES FOR THE CG PROJECTION ****************
    iterCG=0;
    residNorm_prevCG=zeros(1,16);
    residNorm_evolutionCG=ones(1,16);
    
    residNorm_diffCG = 1;
    residNorm_prevCG(16) = residNorm_prev(16);
    
    resid_update = A_dct(vec, m, n, dct_rows);
    resid = y - resid_update;
    
    grad = AT_dct(resid, m, n, dct_rows);
    grad = threshold(grad, bin, kbin);
    
    grad_previous = grad;
    
    mu = dot(grad_previous, grad_previous);
    errCG = dot(resid, resid);
    residNorm_prevCG(16) = errCG;
    
    while ( (errCG > tolCG) && (iterCG < maxiterCG) && (residNorm_diffCG > tol2) )
        
        [vec, grad, grad_previous, residNorm_prevCG, mu] = RestrictedCG_M_dct(vec, grad, grad_previous, y, dct_rows, bin, residNorm_prevCG, kbin, m, n, mu);
        
        errCG = residNorm_prevCG(16);
        iterCG = iterCG + 1;
        
        % check for convergence
        residNorm_evolutionCG(1:15) = residNorm_evolutionCG(2:16);
        residNorm_evolutionCG(16) = residNorm_prevCG(15)-residNorm_prevCG(16);
        residNorm_diffCG = max(residNorm_evolutionCG);
    end

    max_value = MaxMagnitude(vec);

    slope = (num_bins/max_value);

    bin = zeros(n,1);
    bin_counters = zeros(num_bins,1);

    [kbin, ksum, minValue, bin, bin_counters] = FindSupportSet(vec, bin, bin_counters, slope, max_value, max_value, 0, kbin, k, num_bins);
    
    vec = threshold(vec, bin, kbin);
    
    resid_update = A_dct(vec, m, n, dct_rows);
    resid = y - resid_update; 
    err = norm(resid);
    
    %recording the convergence of the residual
    residNorm_prev(1:15) = residNorm_prev(2:16);
    residNorm_prev(16)=err;

    residNorm_evolution(1:15)=residNorm_evolution(2:16);
    residNorm_evolution(16) = residNorm_prev(15)-residNorm_prev(16);
    residNorm_diff = max(abs(residNorm_evolution));
    
    if (iter>125)
        convergenceRate = (residNorm_prev(16)/residNorm_prev(1))^(1/16);
        if (convergenceRate > 0.999)
            fail = 1;
        end
    end
    
    iter = iter+1;
    time_sum = time_sum + toc(timeiter);
    
end

