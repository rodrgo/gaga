function bin = joinSupports(bin, grad_bin, kbin, grad_kbin)
% bin = joinSupports(bin, grad_bin, kbin, grad_kbin) duplicates the kernel
% by the same in GAGA.
% This function is used in CSMPSP to take the union of the support of the
% previous iterations with the support of the k largest magnitudes in the
% gradient.

n = length(bin);
if (length(grad_bin) ~= n)
    error('The bin vectors must be the same length.')
end

for index=1:n
    if (grad_bin(index) <= grad_kbin)
        if (bin(index) > kbin)
            bin(index) = kbin;
        end
    end
end
