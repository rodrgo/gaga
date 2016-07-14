function [vec, residNorm_prev, iter, time_sum, ksum] = IHT_M_gen(y, A_gen, k, m, n, tol, maxiter)
% This function has the usage 
% [vec, residNorm_prev, iter, time_sum, ksum] = IHT_M_gen(y, A_gen, k, m, n, tol, maxiter);
% y is the input measurements, 
% A_gen is the matrix,
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
kbin = 1;
mu = 0.65;

minValue = 0;
maxChange = 0;

resid = y;

err = norm(resid);

resid_tol = tol;

tol2 = tol;
err_start = err;

residNorm_diff = 1;

residNorm_prev=zeros(1,16);         % Stores the previous 16 residual norms.
residNorm_evolution=ones(1,16);    % Stores the previous 16 changes in residual norms.

% Some stopping parameters
fail = 0;
convergenceRate = 0;
                 
%%%%%%%%%%%%%%%%%%%%%%%%%
%  Set the Linear Bins  %
%%%%%%%%%%%%%%%%%%%%%%%%%

grad = A_gen'*y;

max_value = MaxMagnitude(grad);

slope = (num_bins/max_value);

bin = zeros(n,1);
bin_counters = zeros(num_bins,1);




%%%%%%%%%%%%%%%%%
% Main IHT Loop %
%%%%%%%%%%%%%%%%%

while ( (err>resid_tol) && (iter < maxiter) && (err < 100 * err_start) && (residNorm_diff > .01*tol) && (fail == 0) )
    
    timeiter=tic;
    
    % Determine the steepest descent direction.
    grad  = A_gen'*resid;
    
    maxChange = MaxMagnitude(grad);
    maxChange = 2*mu*maxChange;
    
    % Take a fixed size step in the steepest descent direction.
    vec = vec + mu*grad;

    if (minValue <= maxChange)
        max_value = MaxMagnitude(vec);
        slope = (num_bins/max_value);
    end
    
    % Identify the support of the k largest entries
    [kbin, ksum, minValue, bin, bin_counters] = FindSupportSet(vec, bin, bin_counters, slope, max_value, maxChange, minValue, kbin, k, num_bins);
    
    % threshold the vector to that support set
    vec=threshold(vec, bin, kbin);
   
    % update the residual
    resid_update = A_gen*vec;
    resid = y - resid_update;
    
    err = norm(resid);
    
    %recording the convergence of the residual
    residNorm_prev(1:15) = residNorm_prev(2:16);
    residNorm_prev(16)=err;
    residNorm_evolution(1:15)=residNorm_evolution(2:16);
    residNorm_evolution(16) = residNorm_prev(15)-residNorm_prev(16);
    residNorm_diff = max(residNorm_evolution);
    
    if (iter>749)
        convergenceRate = (residNorm_prev(16)/residNorm_prev(1))^(1/16);
        if (convergenceRate > 0.999)
            fail = 1;
        end
    end
    
    iter = iter+1;
    time_sum = time_sum + toc(timeiter);

   
end

