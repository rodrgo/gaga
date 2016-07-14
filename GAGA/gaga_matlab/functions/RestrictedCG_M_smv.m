function [vec, grad, grad_previous, residNorm_prev mu] = RestrictedCG_M_smv(vec, grad, grad_previous, y, A_smv, bin, residNorm_prev, kbin, m, n, mu)
% RestrictedCG_M_smv performs one step of the Conjugate Gradient solution of
% A*vec = y where vec is restricted to being supported on the entries in
% bin that values larger than kbin.  It has the usage:
% [vec, grad, grad_previous, residNorm_prev] 
%     = RestrictedCG_M_smv(vec, grad, grad_previous, A_smv, bin, residNorm_prev, kbin, m, n)
% IMPORTANT: vec and grad are considered to be thresholded already.
%            vec_thresh is a working n-vector
%            resid and resid_update are working m-vectors

% compute A_dct * grad and store the value in resid
resid = A_smv*grad;

% update vec
alpha = dot(resid, resid);
alpha = mu /alpha;
vec = vec + alpha*grad;


% compute AT_dct * resid to update the residual
vec_thresh = A_smv'*resid;
vec_thresh = threshold(vec_thresh, bin, kbin);

% update the residual which is called grad_previous
grad_previous = grad_previous - alpha * vec_thresh;

% update the search direction
beta = mu;
mu = dot(grad_previous, grad_previous);
beta = mu / beta;

grad = beta*grad + grad_previous;

%recording the convergence of the residual
% in the weighted norm for CG
% < (vec_input - vec), AT*A(vec_input-vec)> = <y-Avec, y-Avec>.
residNorm_prev(1:15) = residNorm_prev(2:16);

resid_update = A_smv*vec;
resid = y - resid_update;
err = dot(resid, resid);
residNorm_prev(16)=err;

