function [vec_input, y, A_smv, seed] = createProblem_smv(k, m, n, vecDistribution, p, ensemble, seed)
% [vec_input, y, dct_rows, seed] = createProblem_dct(k, m, n, vecDistribution, seed)
% This function is considerably less complicated than the version in GAGA
% due to the fact that the rand function uses the Mersenne Twister as
% default.  Therefore, there are no kernels associated with this matlabe
% version of the function to generate the Twister produced random numbers.

rand('state',seed);
randn('state',seed);

vec_input=zeros(n,1);
[junk,supp]=sort(rand(n,1));

if (vecDistribution == 0)
    vec_input(supp(1:k)) = rand(k,1);
else if (vecDistribution == 1)
        vec_input(supp(1:k))=sign(randn(k,1));
    else if (vecDistribution == 2)
            vec_input(supp(1:k)) = randn(k,1);
        else error('vecDistribution options: 0=uniform, 1=random{-1,1}, 2=Gaussian')
        end
    end
end

nz=n*p;
rows = zeros(nz,1); 
cols = zeros(nz,1);

nLeftNeighbors = zeros(m, 1);

for jj=1:n
    cols((jj-1)*p+1:jj*p)=jj;
    %need to make p independent integers from 1:m
    if (5*p>m) | (m<2000) %2000 is speed transition point
      temp = randperm(m);
    else
      made_row_vals=0;
      max_tries=10;
      count_tries=0;
      while (made_row_vals==0) & (count_tries<max_tries)
        tmp_rows=ceil(m*rand(max(p+5,ceil(1.3*p)),1));
        tmp_rows=intersect(tmp_rows,tmp_rows);
        if length(tmp_rows)>=p
          temp=tmp_rows(1:p);
          made_row_vals=1;
        end
        count_tries=count_tries+1;
      end
    end
    rows((jj-1)*p+1:jj*p)=temp(1:p);
    nLeftNeighbors(rows((jj-1)*p+1:jj*p)) = nLeftNeighbors(rows((jj-1)*p+1:jj*p)) + 1;
end

% correct for zero-rows
zeroLeftNeighbors = find(nLeftNeighbors == 0);
if any(zeroLeftNeighbors)
	populatedLeftNeighbors = find(nLeftNeighbors > 1);
	for l = 1:length(zeroLeftNeighbors)
		i = populatedLeftNeighbors(l);
		j = find(rows == i, 1);
		rows(j) = zeroLeftNeighbors(l);
	end
end


if (ensemble == 1)
    vals = ones(nz,1);
else if (ensemble ==2)
        vals = sign(randn(nz,1));
    end
end
%vals = vals/sqrt(p);

A_smv = sparse(rows, cols, vals, m, n);

y=A_smv*vec_input;
