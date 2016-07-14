function [vec, grad, grad_previous, grad_prev_thresh, residNorm_prev, mu, maxChange] = RestrictedCGwithSupportEvolution_M_gen(vec, grad, grad_previous, grad_prev_thresh, y, A_gen, bin, residNorm_prev, kbin, m, n, mu, beta_to_zero_flag)
% RestrictedCG_M_gen performs one step of the Conjugate Gradient solution of
% A*vec = y where vec is restricted to being supported on the entries in
% bin that values larger than kbin.  It has the usage:
% [vec, grad, grad_previous, residNorm_prev] 
%     = RestrictedCG_M_gen(vec, grad, grad_previous, A_gen, bin, residNorm_prev, kbin)
% IMPORTANT: vec and grad are considered to be thresholded already.
%            vec_thresh is a working n-vector
%            resid and resid_update are working m-vectors

% compute A_dct * grad and store the value in resid
resid_update = A_gen*vec;
resid = y - resid_update;

%recording the convergence of the residual
% in the weighted norm for CG
% < (vec_input - vec), AT*A(vec_input-vec)> = <y-Avec, y-Avec>.
residNorm_prev(1:15) = residNorm_prev(2:16);
err = norm(resid);
residNorm_prev(16)=err;

% compute the residual r=AT*(y-Ax) and store in grad
grad = A_gen'*resid;

% store a thresholded copy of grad in vec_thresh
vec_thresh = threshold_one(grad, bin, kbin);

% beta is a built-in function in Matlab so we use bbeta.
if (beta_to_zero_flag == 1)
  bbeta = 0;
  % We compute mu in case the support does not change in the next iteration.
  mu = dot(vec_thresh, vec_thresh);
else
  bbeta = mu;
  mu = dot(vec_thresh, vec_thresh);
  if (mu < (1000*bbeta))
    bbeta = mu / bbeta;
  else
    bbeta = 0;
  end
end

% compute the new search direction
grad_previous = grad + bbeta*grad_previous;

% store a thresholded version of grad_previous in grad_prev_thresh
grad_prev_thresh = threshold_one(grad_previous, bin, kbin);

% resid_update is used to store A*grad_prev_thresh
resid_update = A_gen*grad_prev_thresh;

% calculate the step-size alpha
alpha = dot(resid_update, resid_update);
if ( mu < (1000*alpha) )
  alpha = mu / alpha;
else
  alpha = 0;
end

% update the approximation by stepping length alpha in the search direction grad_previous
vec = vec + alpha*grad_previous;

%return twice the maximum possible change in this update
maxChange = 2*alpha*MaxMagnitude(grad_previous);


