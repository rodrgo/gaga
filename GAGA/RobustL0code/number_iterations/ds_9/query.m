
clear all;

noise_levels = [1e-2];
RES_TOL = 1;
SOL_TOL = 2;

GPUnumber = 0;

addpath('..');

maxiter_str = '1k';
l0_thresh = 2
alg_name = 'deterministic_robust_l0';

number_of_iterations(alg_name, noise_levels, RES_TOL, SOL_TOL, GPUnumber, maxiter_str, l0_thresh);
plot_number_of_iterations(alg_name, RES_TOL, SOL_TOL);

