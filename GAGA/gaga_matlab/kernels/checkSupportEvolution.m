function suppChange_flag = checkSupportEvolution(bin, bin_prev, kbin, kbin_prev)
% suppChange_flag = checkSupportEvolution(bin, bin_prev, kbin, kbin_prev)
% duplicates the kernel by the same in GAGA.
% This function is used to determine if the support has changed from the previous
% iteration in CGIHT.

n = length(bin);
if (length(bin_prev) ~= n)
    error('The bin vectors must be the same length.')
end

suppChange_flag = 0;

for index=1:n
    if ( (bin_prev(index) <= kbin_prev) && (bin(index) > kbin) )
        suppChange_flag = 1;
        %break;
    end
end
