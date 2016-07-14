
run('../init.m');

algs = {'smp'};
maxiters = {'1k'};
gpuNumber = 0;

ns = [2^20];
d = 7;

deltas = [1/100 1/10];
TOL = 1e-5;
seed = 1;

rho_start = 0;
rho_step = 0.01;
tests_per_rho = 10;

timing_fixed_delta(algs, maxiters, gpuNumber, ns, d, deltas, TOL, seed, rho_start, rho_step, tests_per_rho)

