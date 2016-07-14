function vec_thresh = threshold_one(vec, bin, kbin)
% vec_thresh = threshold_one(vec, bin, kbin) duplicates the kernel by the same name in
% GAGA.  threshold_one takes a vector, a vector of bin assignments, and the kbin
% value, and writes a thresholded version to vec_thresh.
% This differs from the function threshold in that it writes a new vector rather than overwriting the input vector.
% It does this via a vectorized conditional.

vec_thresh=vec.*(bin<=kbin);
