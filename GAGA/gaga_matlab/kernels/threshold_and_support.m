function [vec, support] = threshold_and_support(vec, support, T)
% [vec, support] = threshold_and_support(vec, support, T) duplicates the kernel by the same name in
% GAGA.  It takes every element in vec which is less than T and sets that
% element to zero while simultaneously setting a bin index to 2.

vec = vec.*(abs(vec)>=T);
support(vec==0) = 2;