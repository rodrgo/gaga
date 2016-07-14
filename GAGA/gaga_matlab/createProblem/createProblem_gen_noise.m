function [vec_input, y, A_gen, seed] = createProblem_gen_noise(k, m, n, vecDistribution, ensemble, seed, noise_level)
% [vec_input, y, A_gen, seed] = createProblem_gen_noise(k, m, n, vecDistribution, ensemble, seed, noise_level)
% This function creates a random problem with a dense matrix from ensemble: (1 = Gaussian, 2 = {-1,1}).
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

A_gen = randn(m,n);
if (ensemble == 2)
    A_gen = sign(A_gen);
end
A_gen = A_gen/sqrt(m);

y=A_gen*vec_input;

noise = randn(m,1);
noise_scale=noise_level*norm(y,2)/norm(noise,2);
noise = noise_scale*noise;

y = y + noise;
