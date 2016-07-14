function [vec_input, y, dct_rows, seed] = createProblem_dct_noise(k, m, n, vecDistribution, seed, noise_level)
% [vec_input, y, dct_rows, seed] = createProblem_dct_noise(k, m, n, vecDistribution, seed, noise_level)
% This function creates a random problem with m rows subsampled from the dct and returned as dct_rows.
% The initial vector, vec_input, has a randomly selected support set of size k.
% The entries of the vector are from vecDistribution: (0 = uniform(0,1), 1 = {-1,1}, 2 = Gaussian).
% Gaussian noise is added to the measurement vector y.  The noise is scaled to have norm equal to noise_level*norm(y).

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

[junk,rows]=sort(rand(n,1));
dct_rows=rows(1:m);

y=A_dct(vec_input, m, n, dct_rows);

noise = randn(m,1);
noise_scale=noise_level*norm(y,2)/norm(noise,2);
noise = noise_scale*noise;

y = y + noise;

