function [vec, grad, resid, residNorm_prev, maxChange] = RestrictedSD_M_dct(vec, grad, y, resid, dct_rows, bin, bin_counters, residNorm_prev, num_bins, kbin, m, n)
% RestrictedSD_M_dct performs one step of the Steepest Descent solution of
% A*vec = y where vec is restricted to being supported on the entries in
% bin that values larger than kbin.  It has the usage:
% [vec, grad, resid, residNorm_prev, maxChange] 
%     = RestrictedSD_M_dct(vec, grad, y, resid, dct_rows, bin, bin_counters, residNorm_prev, num_bins, kbin, m, n)

resid_update = A_dct(vec, m, n, dct_rows);
resid = y - resid_update; 
err = norm(resid);
    
%recording the convergence of the residual
residNorm_prev(1:15) = residNorm_prev(2:16);
residNorm_prev(16)=err;

%compute the gradient and its restriction to the support of vec
grad = AT_dct(resid, m, n, dct_rows);
grad_thresh = threshold(grad, bin, kbin);  %in GAGA we use d_vec_thresh as it already exists outside this function

%Now we need to compute the dctA(restricted gradient) in order to compute
%the steepest descent step size mu.
resid_update = A_dct(grad_thresh, m, n, dct_rows);
err = norm(resid_update);
mu = norm(grad_thresh);
if ( mu < 400 * err)
    mu = mu/err;
    mu = mu*mu;
else
    mu = 0;
end

%Now ad mu times the gradient to vec
vec = vec + mu*grad;

%return twice the maximum possible change in this update
maxChange = 2*mu*MaxMagnitude(grad);
