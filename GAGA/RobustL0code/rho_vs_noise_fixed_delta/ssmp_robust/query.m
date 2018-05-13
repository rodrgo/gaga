
run('../init.m');

algs = {'ssmp_robust'};
maxiters = {'4k'};
gpuNumber = 6;

ns = [2^18];
d = 7;

deltas = [1/100 1/50 1/10 1/5];
seed = 1;

rho_start = 0;
rho_step = 0.01;
tests_per_rho = 10;

noise_levels = 10.^linspace(-3,-1, 20);

RES_TOL = 0;
SOL_TOL = 1;

l0_thresh = 2;

rho_vs_noise_fixed_delta(algs, maxiters, gpuNumber, ns, d, deltas, RES_TOL, SOL_TOL, seed, rho_start, rho_step, tests_per_rho, noise_levels, l0_thresh);

