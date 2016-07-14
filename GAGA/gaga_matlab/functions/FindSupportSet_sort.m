function  [vec, bin, T] = FindSupportSet_sort(vec, bin, T, maxChange, k)
% [vec, bin, T] = FindSupportSet_sort(vec, bin, T, maxChange, k) is a
% duplicate of the function by the same name in functions.cu in GAGA.  It
% uses hard thresholding based on the value of the kth largest magnitude in
% vec.  It returns a thresholded version of vec, bin which is a vector of
% 1s and 2s with 1 indicated in the support, and 2 indicated outside the
% support, and an updated version of the threshold level T.

if (T>maxChange)
    T = T - maxChange;
    [vec, bin] = threshold_and_support(vec, bin, T);
else
    mag_vec = magnitudeCopy(vec);
    mag_vec = sort(mag_vec,'descend');
    T = mag_vec(k);
    bin = ones(size(bin));
    [vec, bin] = threshold_and_support(vec, bin, T);
end
    
