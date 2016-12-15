
%%%% Set the paths

HOME='~/src/robust_l0/GAGA_1_2_0/';
GAGA_PATH=fullfile(HOME, 'GAGA/gaga/');
addpath(GAGA_PATH);

%%%%

alg = 'deterministic_robust_l0';
delta = 0.001;
rho = 0.06; 
n = 2^23;

m = ceil(n*delta); 
k = ceil(rho*m);
d = 7;

gpuNumber = 0;
MAXITER = k;
RES_TOL = 1;
noise_level = 0.1;
vecDistribution = 'gaussian';
l0_thresh = 2;
seed = 1;
band_percentage = 0.0;

options = gagaOptions('gpuNumber', gpuNumber, 'maxiter', MAXITER,'tol', RES_TOL,'noise', single(noise_level),'vecDistribution', vecDistribution, 'l0_thresh', l0_thresh, 'seed', seed, 'kFixed','on', 'band_percentage', band_percentage, 'debug_mode', 1);

[errors, times, iters, supp, conv, xhat] = gaga_cs(alg, 'smv', int32(k), int32(m), int32(n), int32(d), options);

SOL_TOL = 1;
mean_err1 = m*noise_level*sqrt(2/pi);
sd_err1 = sqrt(m)*noise_level*sqrt(1 - 2/pi);
mean_signal_norm = k*sqrt(2/pi);
upper_bound = (mean_err1 + SOL_TOL*sd_err1)/mean_signal_norm;

fprintf('mean_err1 = %5.6f\nsd_err1 = %5.6f\nmean_nrm=%5.6f\n', ...
	mean_err1, sd_err1, mean_signal_norm);
fprintf('error_1 = %5.6f, upper_bound = %5.6f\n', errors(1), upper_bound);
fprintf('norm(xhat) = %7.10f\nnorm(xhat) == 0? %d\n', norm(xhat), norm(xhat) == 0);

