function [vec, residNorm_prev, iter, time_sum, ksum time_per_iteration, time_supp_set] = NIHT_M_timings_dct(y, dct_rows, k, m, n, tol, maxiter, supp_flag)
% This function has the usage 
% [vec, residNorm_prev, iter, time_sum, ksum time_per_iteration, time_supp_set] 
%       = NIHT_M_timings_dct(y, dct_rows, k, tol, maxiter, supp_flag);
% y is the input measurements, 
% dct_rows are the subsampled rows of the DCT matrix,
% k is the assumed sparsity, 
% m is the number of measurements, 
% n is the dimension of the signal one attempts to recover,
% tol is the error tolerance,
% maxiter is a maximum number of iterations.
% supp_flag determines the method for identifying the support set 
%     (0 = dynamic binning, 1 = always binning, 2 = dynamic sorting, 3 = always sorting).
% The outputs are:
%  vec is the approximation to the sparse vector which produced y;
%  residNorm_prev is a vector of the previous 16 residual norms;
%  iter is the number of iterations from the algorithm;
%  time_sum is the cummulative sum of time taken in each iteration;
%  ksum is the number of elements in the bin containing the kth largest
%    element;
%  time_per_iteration is a vector of times for each iteration;
%  time_supp_set is a vector of times for identifying the support set
%    in each iteration;
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Allocate Timings Storage %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

time_per_iteration = zeros(maxiter,1);
time_supp_set = zeros(maxiter,1);

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

resid_tol = tol;

residNorm_diff = 1;

residNorm_prev=zeros(1,16);         % Stores the previous 16 residual norms.
residNorm_evolution=ones(1,16);    % Stores the previous 16 changes in residual norms.

% Some stopping parameters
fail = 0;
convergenceRate = 0;

% Parameter for FindSupportSet_sort
T = 0;

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
grad = zeros(size(vec));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main Steepest Descent Loop %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while ( (err>resid_tol) && (iter < maxiter) && (err < 100 * err_start) && (residNorm_diff > .01*tol) && (fail == 0) )
    
    timeiter=tic;

    [vec, grad, resid, residNorm_prev, maxChange] = RestrictedSD_M_dct(vec, grad, y, resid, dct_rows, bin, bin_counters, residNorm_prev, num_bins, kbin, m, n);


    start_time_supp=tic;
    
    if (supp_flag < 2)    
        if (minValue <= maxChange)
            max_value = MaxMagnitude(vec);
            slope = (num_bins/max_value);
        end
    end
    
    if (supp_flag == 0)
        [kbin, ksum, minValue, bin, bin_counters] = FindSupportSet(vec, bin, bin_counters, slope, max_value, maxChange, minValue, kbin, k, num_bins);
    else if (supp_flag == 1)
            [kbin, ksum, minValue, bin, bin_counters] = FindSupportSet(vec, bin, bin_counters, slope, max_value, max_value, minValue, kbin, k, num_bins);
        else if (supp_flag == 2)
                kbin = 1;
                [vec, bin, T] = FindSupportSet_sort(vec, bin, T, maxChange, k);
            else if (supp_flag == 3)
                    kbin = 1;
                    [vec, bin, T] = FindSupportSet_sort(vec, bin, T, T, k);
                end
            end
        end
    end
    
    vec=threshold(vec, bin, kbin);
    
    time_supp=toc(start_time_supp);
   

    err = residNorm_prev(16);

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
    
    time = toc(timeiter);
    
    time_per_iteration(iter) = time;
    time_supp_set(iter) = time_supp;
    time_sum = time_sum + time;

   
end

