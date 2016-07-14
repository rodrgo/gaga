function vec = threshold(vec, bin, kbin)
% vec = threshold(vec, bin, kbin) duplicates the kernel by the same name in
% GAGA.  threshold takes a vector, a vector of bin assignments, and the kbin
% value, and then sets all entries of vec to 0 if the entry belongs in a
% bin larger than kbin.  It does this via a vectorized conditional.

vec=vec.*(bin<=kbin);