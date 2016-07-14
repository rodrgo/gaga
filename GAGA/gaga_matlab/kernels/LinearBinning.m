function [bin, bin_counters] = LinearBinning(vec, bin, bin_counters, num_bins, slope, intercept)
% [bin, bin_counters] = LinearBinning(vec, bin, bin_counters, num_bins,
% slope, intercept) duplicates the kernel of the same name in GAGA.  This
% kernel projects each element of the list into a bin of width 1/slope
% along the line segment from 0 to intercept.  It then counts how many
% entries were assigned to each bin.

bin_counters = zeros(num_bins, 1);

bin = max(1,min( floor(slope*(intercept - abs(vec)))+1, num_bins));
    
for j=1:length(bin)
    bin_counters(bin(j))=bin_counters(bin(j))+1;
end
