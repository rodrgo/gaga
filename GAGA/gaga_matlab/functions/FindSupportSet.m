function [kbin, ksum, minValue, bin, bin_counters] = FindSupportSet(vec, bin, bin_counters, slope, intercept, maxChange, minValue, kbin, k, num_bins);
% [kbin, ksum, minValue, bin, bin_counters] = FindSupportSet(vec, bin, bin_counters, slope, intercept, maxChange, minValue, kbin, k, num_bins);
% This function duplicates the function by the same name in functions.cu in
% GAGA.  First, it checks if the support could possibly have changed.  If
% not, it (essentially) does nothing.  If so, it rebins the vector, counts
% the bins, and then identifies which bin contains the kth largest value.  


    if (minValue > maxChange)
        minValue = minValue - maxChange;
        ksum = sum(bin_counters(1:kbin));
    else
        [bin, bin_counters] = LinearBinning(vec, bin, bin_counters, num_bins, slope, intercept);
        kbin=1;
        ksum=0;
        while (ksum<k && kbin<=num_bins)
            ksum = ksum + bin_counters(kbin);
            kbin=kbin+1;
        end
        kbin=kbin-1;
        
        minValue = intercept-(kbin/slope);
    end