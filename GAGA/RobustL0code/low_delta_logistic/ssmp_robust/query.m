
run('../init.m');

algs = {'ssmp_robust'};
maxiters = {'1k'};
gpuNumber = 0;

ns = [2^20 2^22 2^24];
d = 7;
deltas = [0.001 0.01 0.05];

%RES_TOL = 1;
RES_TOL = 0;
SOL_TOL = 1;
l0_thresh = 2;
noise_levels = [1e-3, 1e-2, 1e-1];

seed = 1;
rho_start = 0;
rho_step = 0.01;
tests_per_rho = 15;

timing_fixed_delta(algs, maxiters, gpuNumber, ns, d, deltas, RES_TOL, SOL_TOL, l0_thresh, noise_levels, seed, rho_start, rho_step, tests_per_rho)

