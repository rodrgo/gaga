function [vec, residNorm_prev, iter, time_sum, ksum] = HT_CG_M_dct(y, dct_rows, k, m, n, tol, maxiter)
% HT_SD_matlab_dct performs a single Hard Thresholding operation on the set
% of k largest values of AT(y), and then performs Conjugate Gradient
% restricted to this k-subspace.
% This function has the usage 
% [vec, residNorm_prev, iter, time_sum, ksum] = HT_CG_M_dct(y, dct_rows, k, m, n, tol, maxiter);
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

tolCG = tol*tol;
tol2 = 10*tolCG;

residNorm_diff = 1;

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

% Our main variables are three n-vectors
%   vec will store the thresholded vector x
%   grad will store the descent direction p
%   grad_previous will store the residual r
% We also use three working variables
%   vec_thresh is a working n-vector
%   resid and resid_update are working m-vectors

% We now need to compute the residual and initial descent direction.
% Using the algorithm listed on page 35 of Greenbaum.

% Initially compute y - A * vec
resid_update = A_dct(vec, m, n, dct_rows);
resid = y - resid_update;

% The residual for us is AT*( y - A*vec)
% and then thresholded to bin<=kbin.
% Put this result in grad and then grad_previous
grad = AT_dct(resid, m, n, dct_rows);
grad = threshold(grad, bin, kbin);
grad_previous = grad;

% At this point CG is ready, now setup the main CG loop.

% the input variable are ready for CG.
%   bin is our candidate support set.
%   vec is our dandidate kbin sparse guess for the solution.
%   grad is the kbin restricted search direction.
%   grad_previous is the kbin restricted residual.

mu = dot(grad_previous,grad_previous);
err = dot(resid,resid);
residNorm_prev(16) = err;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main Conjugate Gradient Loop %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while ( (err>tolCG) && (iter < maxiter) && (residNorm_diff > tol2) ) 
    
    timeiter=tic;

    [vec, grad, grad_previous, residNorm_prev, mu] = RestrictedCG_M_dct(vec, grad, grad_previous, y, dct_rows, bin, residNorm_prev, kbin, m, n, mu);

    %vec=threshold(vec, bin, kbin);

    err = residNorm_prev(16);

    residNorm_evolution(1:15)=residNorm_evolution(2:16);
    residNorm_evolution(16) = residNorm_prev(16)-residNorm_prev(15);
    residNorm_diff = max(abs(residNorm_evolution));
        
    iter = iter+1;
    time_sum = time_sum + toc(timeiter);
    
   
end

