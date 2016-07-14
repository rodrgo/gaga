function checkSupport = check_support(vec_input,  vec)
% support_counter = check_support(vec_input,  vec) duplicates the kernel by the same name in
% GAGA.  This function returns a vector of 4 numbers listing the number of true
% positives, false positives, true negatives, false negatives where a
% positive is a nonzero value, vec_input is the original vector, and vec is
% the approximation output from a greedy algorithm.

n = length(vec_input);
checkSupport = zeros(4,1);

if (length(vec) ~= n) 
    error('Both input vectors must have the same length.')
else
    for jj=1:n
        if (vec_input(jj)~=0)
            if (vec(jj)~=0) checkSupport(1)=checkSupport(1)+1;
            else checkSupport(4)=checkSupport(4)+1;
            end
        else if (vec(jj)~=0) checkSupport(2)=checkSupport(2)+1;
            else checkSupport(3)=checkSupport(3)+1;
            end
        end
    end
end
