
delta = 0.2;
rho = 0.1;
n = 2^18;
m = ceil(delta*n);
k = ceil(rho*m);
p = 7;

RES_TOL = 0;
l0_thresh = 2;

noise_levels = 0.1;

alg_1 = 'ssmp_robust';
alg_2 = 'robust_l0';

for noise_level = noise_levels
	options = gagaOptions('maxiter',k, ...
		'tol', RES_TOL, ...
		'noise', noise_level, ...
		'seed', seed + 1, ...
		'gpuNumber', 0, ...
		'vecDistribution','gaussian', ...
		'kFixed','on', ...
		'l0_thresh', int32(l0_thresh), ...
		'debug_mode', 1);

	fprintf('****** %s\n', alg_1);
	[norms times iters check_support conv_rate xout resRecord timeRecord] = gaga_cs(alg_1, 'smv', k, m, n, p, options);
	fprintf('\n');
	fprintf('%s: norm1=%g, time=%g, iters=%d\n', alg_1, norms(1), times(1), iters);
	fprintf('%s: tp=%g, fp=%g, tn=%d, fn=%d\n', alg_1, check_support(1), check_support(2), check_support(3), check_support(4));
	fprintf('\n');
	% Check support is tp, fp, tn, fn

	fprintf('****** %s\n', alg_2);
	[norms times iters check_support conv_rate xout resRecord timeRecord] = gaga_cs(alg_2, 'smv', k, m, n, p, options);
	fprintf('\n');
	fprintf('%s: norm1=%g, time=%g, iters=%d\n', alg_2, norms(1), times(1), iters);
	fprintf('%s: tp=%g, fp=%g, tn=%d, fn=%d\n', alg_2, check_support(1), check_support(2), check_support(3), check_support(4));
	fprintf('\n');
	seed = seed + 1;

end

