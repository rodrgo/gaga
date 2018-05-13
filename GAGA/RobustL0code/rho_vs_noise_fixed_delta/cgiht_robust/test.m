
run('../init.m');
l0_thresh = 2;

%k = 13632;
k = 9438;
m = 52429;
n = 262144;
p = 7;
noise_level = 0.0013;

seed = 123988 + 111;
RES_TOL = 0;

options = gagaOptions('maxiter',4*k, ...
	'tol', RES_TOL, ...
	'noise', noise_level, ...
	'seed', seed, ...
	'gpuNumber', 0, ...
	'vecDistribution','gaussian', ...
	'kFixed','on', ...
	'l0_thresh', int32(l0_thresh), ...
	'debug_mode', 0);

alg = 'cgiht_robust';

[norms times iters d e f resRecord timeRecord] = gaga_cs(alg, 'smv', k, m, n, p, options);
fprintf('\n%s: norm1=%g, time=%g, iters=%d\n\n', alg, norms(1), times(1), iters);

