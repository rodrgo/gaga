function y = soft(x, T)
% y=soft(x, T) is a duplicate of the kernel __soft which is in the kernels
% file of GAGA and taken from the SpaRSA code of Lee and Wright.  It soft
% thresholds x with the value T (zeroing all values of x with magnitude less
% than T) and writes this thresholded version of the vector to the y.

y = max(abs(x)-T,0);

y = (y./(y+T)).*x;
