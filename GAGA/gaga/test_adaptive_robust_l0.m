
delta = 0.1;
rho = 0.26;
rho = 0.25;
rho = 0.2;
rho = 0.07;
n = 2^18;
n = 2^20;
m = ceil(delta*n);
k = ceil(rho*m);
p = 7;

l0_thresh = 2;

noise_levels = [1e-3, 1e-2, 0.05, 0.075, 0.1, 0.15, 0.2];
noise_levels = [1e-3];
noise_levels = [1e-1];

algorithms = {'robust_l0', 'robust_l0_adaptive', 'robust_l0_adaptive_trans', 'robust_l0_trans', 'ssmp_robust', 'smp_robust'};
algorithms = {'robust_l0', 'robust_l0_adaptive', 'robust_l0_adaptive_trans', 'robust_l0_trans', 'cgiht_robust', 'ssmp_robust'};

seed = 7532;
RES_TOL = 0;
for noise_level = noise_levels

	for i = 1:length(algorithms)
		options = gagaOptions('maxiter',k, ...
			'tol', RES_TOL, ...
			'noise', noise_level, ...
			'seed', seed, ...
			'gpuNumber', 0, ...
			'vecDistribution','gaussian', ...
			'kFixed','on', ...
			'l0_thresh', int32(l0_thresh), ...
			'debug_mode', 1);
	
		alg = algorithms{i};

		fprintf('****** %s\n', alg);
		[norms times iters d e f resRecord timeRecord] = gaga_cs(alg, 'smv', k, m, n, p, options);
		fprintf('\n%s: norm1=%g, time=%g, iters=%d\n\n', alg, norms(1), times(1), iters);

	end


end

