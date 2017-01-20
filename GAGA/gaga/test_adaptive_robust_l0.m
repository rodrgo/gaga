
delta = 0.1;
rho = 0.26;
n = 2^18;
m = ceil(delta*n);
k = ceil(rho*m);
p = 7;

RES_TOL = 2;
l0_thresh = 2;

noise_levels = [1e-3, 1e-2, 0.05, 0.075, 0.1, 0.15, 0.2];
noise_levels = [1e-3];

seed = 7532;
for noise_level = noise_levels
	options = gagaOptions('maxiter',k, ...
		'tol', RES_TOL, ...
		'noise', noise_level, ...
		'seed', seed + 1, ...
		'gpuNumber', 0, ...
		'vecDistribution','gaussian', ...
		'kFixed','on', ...
		'l0_thresh', int32(l0_thresh), ...
		'debug_mode', 0);

	fprintf('****** Adaptive robust-l0\n');
	[norms times iters d e f resRecord timeRecord] = gaga_cs('adaptive_robust_l0', 'smv', k, m, n, p, options);
	fprintf('\n');
	fprintf('ada_rob: norm1=%g, time=%g, iters=%d\n', norms(1), times(1), iters);
	fprintf('\n');

	if (true)
	fprintf('****** Deterministic robust-l0\n');
	[norms times iters d e f resRecord timeRecord] = gaga_cs('deterministic_robust_l0', 'smv', k, m, n, p, options);
	fprintf('\n');
	fprintf('det_rob: norm1=%g, time=%g, iters=%d\n', norms(1), times(1), iters);
	fprintf('\n');
	end
end

