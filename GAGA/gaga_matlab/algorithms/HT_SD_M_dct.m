function [vec, residNorm_prev, iter, time_sum, ksum] = HT_SD_M_dct(y, dct_rows, k, m, n, tol, maxiter)
% HT_SD_matlab_dct performs a single Hard Thresholding operation on the set
% of k largest values of AT(y), and then performs Steepest Descent
% restricted to this k-subspace.
% This function has the usage 
% [vec, residNorm_prev, iter, time_sum, ksum] = HT_SD_M_dct(y, dct_rows, k, tol, maxiter);
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

minValue = 0;
maxChange = 0;


resid = y;

err = norm(resid);

tol2 = tol;
err_start = err;

residNorm_diff = 1;
residNorm_evolution_max = 0;
residNorm_evolution_change = 1;

residNorm_prev=zeros(1,16);         % Stores the previous 16 residual norms.
residNorm_evolution=ones(1,16);    % Stores the previous 16 changes in residual norms.
                 
%%%%%%%%%%%%%%%%%%%%%%%%%
%  Set the Linear Bins  %
%%%%%%%%%%%%%%%%%%%%%%%%%

vec = AT_dct(y, m, n, dct_rows);

max_value = MaxMagnitude(vec);

slope = (num_bins/max_value);
intercept = max_value;

bin = zeros(n,1);
bin_counters = zeros(num_bins,1);

[kbin, ksum, minValue, bin, bin_counters] = FindSupportSet(vec, bin, bin_counters, slope, intercept, max_value, minValue, kbin, k, num_bins);

vec = threshold(vec, bin, kbin);
grad = zeros(size(vec));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main Steepest Descent Loop %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while ( (err>tol) && (iter < maxiter) && (residNorm_diff > tol2) )
    
    timeiter=tic;

    [vec, grad, resid, residNorm_prev, maxChange] = RestrictedSD_M_dct(vec, grad, y, resid, dct_rows, bin, bin_counters, residNorm_prev, num_bins, kbin, m, n);

    vec=threshold(vec, bin, kbin);

    err = residNorm_prev(16);

    residNorm_evolution(1:15)=residNorm_evolution(2:16);
    residNorm_evolution(16) = residNorm_prev(16)-residNorm_prev(15);
    residNorm_diff = max(residNorm_evolution);
        
    iter = iter+1;
    time_sum = time_sum + toc(timeiter);

   
end
