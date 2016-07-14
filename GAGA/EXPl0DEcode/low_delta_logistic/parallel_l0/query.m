
run('../init.m');

algs = {'parallel_l0'};
maxiters = {'1k'};
gpuNumber = 1;

ns = [2^22 2^24 2^26];
d = 7;

deltas = [0.001];
TOL = 1e-5;
seed = 1;

rho_start = 0;
rho_step = 0.01;
tests_per_rho = 30;

timing_fixed_delta(algs, maxiters, gpuNumber, ns, d, deltas, TOL, seed, rho_start, rho_step, tests_per_rho)

