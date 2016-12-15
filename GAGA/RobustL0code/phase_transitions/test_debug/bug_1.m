% =================================
% Replicate segmentation fault error
% =================================
% @ 28 October 2016 = Error occured during phase transition 
% @ 28 October 2016 = couldn't replicate the error

% gpu_data_smv parameters (before error)
% seed increment = 111
% After five tests with current value of k

% ---------------------------------
% Add necessary paths
% ---------------------------------

HOME = '~/src/robust_l0/GAGA_1_2_0/';

GAGA_PATH = fullfile(HOME, 'GAGA/gaga/');
addpath(GAGA_PATH);

addpath(fullfile(HOME, 'GAGA/RobustL0code/phase_transitions/generate_data/'));

% ------------------------
% Specify problem
% ------------------------

k = 31561;
m = 210406;
n = 262144;
d = 7;
vecDistribution = 2;
seed = 1105007 + 111;
noise_level = 0.001;

alg = 'deterministic_robust_l0';

gpuNumber = 0;
MAXITER = int32(3*k);
RES_TOL = 2*1e-2; % How many SD's from E[\|noise\|_2^2] is the residual allowed to differ

vecDistribution = 'gaussian';
band_percentage = 0.0;

delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02];

options = gagaOptions('gpuNumber', int32(gpuNumber), ...
		'maxiter', int32(MAXITER), ...
		'tol', single(RES_TOL), ...
		'noise', single(noise_level), ...
		'vecDistribution', vecDistribution, ...
		'seed', seed, ...
		'kFixed','on', ...
		'band_percentage', band_percentage);

[errors times iters supp conv xhat] = gaga_cs(alg, 'smv', int32(k), int32(m), int32(n), int32(d), options);

