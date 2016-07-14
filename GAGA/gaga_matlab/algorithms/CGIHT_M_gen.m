function [vec, residNorm_prev, iter, time_sum, ksum] = CGIHT_M_gen(y, A_gen, k, m, n, tol, maxiter, restartFlag)
% This function has the usage 
% [vec, residNorm_prev, iter, time_sum, ksum] = NIHT_M_gen(y, A_gen, k, m, n, tol, maxiter);
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
mu = 0;
kbin = 1;

minValue = 0;
maxChange = 0;


resid = y;

err = norm(resid);


err_start = err;

resid_tol = tol;

residNorm_diff = 1;

residNorm_prev=zeros(1,16);         % Stores the previous 16 residual norms.
residNorm_evolution=ones(1,16);    % Stores the previous 16 changes in residual norms.

% Some stopping parameters
fail = 0;
convergenceRate = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%
%  Set the Linear Bins  %
%%%%%%%%%%%%%%%%%%%%%%%%%

vec = A_gen'*y;

max_value = MaxMagnitude(vec);

slope = (num_bins/max_value);

bin = zeros(n,1);
bin_counters = zeros(num_bins,1);

[kbin, ksum, minValue, bin, bin_counters] = FindSupportSet(vec, bin, bin_counters, slope, max_value, max_value, minValue, kbin, k, num_bins);

vec = threshold(vec, bin, kbin);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the residual AT*(y-A*vec) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% calculate r^0 and the thresholded version of r^0 in grad_previous and grad_prev_thresh respectively

resid_update = A_gen*vec;
resid = y-resid_update;

err = norm(resid);
residNorm_prev(16) = err;

% compute the gradient and its restriction to the current support
grad_previous = A_gen'*resid;
grad=grad_previous;
grad_prev_thresh = threshold_one(grad_previous, bin, kbin);

% check for convergence
residNorm_evolution(1:15) = residNorm_evolution(2:16);
residNorm_evolution(16) = residNorm_prev(15) - residNorm_prev(16);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main Steepest Descent Loop %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% a flag equal to 1 indicates beta in CG should be set to zero
beta_to_zero_flag = 1;

% a flag equal to 1 indicates the support has changed, 0 that it is the same
suppChange_flag = 1;  % this must start with 1
kbin_prev = 0;

while ( (err>resid_tol) && (iter < maxiter) && (err < 100 * err_start) && (abs(residNorm_diff) > .01*tol) && (fail == 0) )
    
    timeiter=tic;

    if (restartFlag)
      [vec, grad, grad_previous, grad_prev_thresh, residNorm_prev, mu, maxChange] = RestrictedCGwithSupportEvolution_M_gen(vec, grad, grad_previous, grad_prev_thresh, y, A_gen, bin, residNorm_prev, kbin, m, n, mu, beta_to_zero_flag);
    else
      if (suppChange_flag)
        [vec, grad, grad_previous, grad_prev_thresh, residNorm_prev, mu, maxChange] = UnrestrictedCG_M_gen(vec, grad, grad_previous, grad_prev_thresh, y, A_gen, bin, residNorm_prev, kbin, m, n, mu, beta_to_zero_flag);
      else
        [vec, grad, grad_previous, grad_prev_thresh, residNorm_prev, mu, maxChange] = RestrictedCGwithSupportEvolution_M_gen(vec, grad, grad_previous, grad_prev_thresh, y, A_gen, bin, residNorm_prev, kbin, m, n, mu, beta_to_zero_flag);    
      end % ends if (suppChange_flag)
    end   % ends if (restartFlag)

    % after the first iteration compute beta
    beta_to_zero_flag = 0;

    if (minValue <= maxChange)
        max_value = MaxMagnitude(vec);
        slope = (num_bins/max_value);

        % save the previous support information
        bin_prev = bin;
        kbin_prev = kbin;
	
	% find the new support
	[kbin, ksum, minValue, bin, bin_counters] = FindSupportSet(vec, bin, bin_counters, slope, max_value, maxChange, minValue, kbin, k, num_bins);

	% check if the support has changed
	suppChange_flag = checkSupportEvolution(bin, bin_prev, kbin, kbin_prev);

	if (suppChange_flag)
	  if (restartFlag)
	    beta_to_zero_flag = 1;
	  else
	    % Since the support changed and we are not restarting,
	    % the previous search direction in grad_prev_thresh is thresholded to the wrong support set.
            % Restrict the CG search direction stored in grad_previous to the new support and store in grad_prev_thresh.
	    grad_prev_thresh = threshold_one(grad_previous, bin, kbin);
	  end  % ends the if (restartFlag)
	end  % ends the if (suppChange_flag)

    else   % this is the else for if (minValue <= maxChange)
	suppChange_flag = 0;
	minValue = minValue - maxChange;
    end    % ends the if (minValue <= maxChange)
    

    vec=threshold(vec, bin, kbin);

    err = residNorm_prev(16);

    residNorm_evolution(1:15)=residNorm_evolution(2:16);
    residNorm_evolution(16) = residNorm_prev(15)-residNorm_prev(16);
    residNorm_diff = max(residNorm_evolution);
    
    if (iter>749)
        convergenceRate = (residNorm_prev(16)/residNorm_prev(1))^(1/15);
        if (convergenceRate > 0.999)
            fail = 1;
        end
    end
    
    iter = iter+1;
    time_sum = time_sum + toc(timeiter);

	%    beta_to_zero_flag = 1;
   
end

%[(err>resid_tol) (iter < maxiter)  (err < 100 * err_start)  (residNorm_diff > .01*tol)  (fail == 0)]

