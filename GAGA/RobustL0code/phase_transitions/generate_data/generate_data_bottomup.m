function generate_data_bottomup(alg_list, n_list, nonZero_list, delta_list, vecDistribution, RES_TOL, SOL_TOL, tests_per_k, gpuNumber, maxiter_str, band_percentage, noise_levels, l0_thresh)

	if nargin < 12
		noise_level = 0.0;
	end

	if nargin < 13
		l0_thresh = 2;
	end

	if nargin < 11
		band_percentage = 0;
	end
	delta_list = sort(delta_list);
	results_matrix = [];
	m_minimum = max(nonZero_list)^2 + 2;

	tic;
	 
	seed = 1;
	for nl = 1:length(noise_levels)

			noise_level = noise_levels(nl);

			for ii = 1:length(n_list)
				n = n_list(ii);

				for pp = 1:length(nonZero_list);
					d = nonZero_list(pp);
					for hh = 1:length(alg_list)
						alg = alg_list{hh};
						for jj = 1:length(delta_list)
							m = ceil(n*delta_list(jj)); 
							rho = 0;

							converged = ones(1, tests_per_k);
							while any(converged) 
							
								rho = rho + 0.01;
								k = ceil(rho*m);
								
								if strcmp(maxiter_str(2), 'k')
									MAXITER = str2num(maxiter_str(1))*k;
								else
									MAXITER = k;
								end

								if any(ismember({'robust_l0'}, {alg}))
									MAXITER = min(k, 1000);
								end
								
								for i = 1:tests_per_k

									seed = seed + 111;
									options = gagaOptions('gpuNumber', gpuNumber, 'maxiter', MAXITER,'tol', RES_TOL,'noise', single(noise_level),'vecDistribution', vecDistribution, 'l0_thresh', l0_thresh, 'seed', seed, 'kFixed','on', 'band_percentage', band_percentage);
									%dlmwrite('problems.txt', [n m k d seed], 'delimiter', '\t', '-append');

									[errors times iters supp conv xhat] = gaga_cs(alg, 'smv', int32(k), int32(m), int32(n), int32(d), options);

									% We check convergence for robust_l0 in a different way

									if any(ismember({'deterministic_robust_l0', 'robust_l0', 'ssmp_robust', 'ssmp'}, {alg}))
											% Inspired by SSMP theoretical guarantee
											% \|x - xhat\|_1 = O(E[\|eta\|_1])
											% \|eta\|_1 = \sum_i |eta_i| with each eta_i ~ N(0, sigma^2)
											% |eta_i| is distributed as a half normal distribution and has
											%		  E[|eta_i|] = sigma*sqrt(2/pi)
											%		  Var[|eta_i|] = sigma^2*(1-2/pi)
											% In this case, SOL_TOL measures how many standard deviations from the mean we are tolerating.
											%
											% Note that errors(1) = \|x - xhat\|_1/\|x\|_1
											% Gaga's interface for automated testing doesn't return "x".
											% However, E[\|x\|_1] = k*1*sqrt(2/pi)
											% We use this as an approximation
											% 
											mean_err1 = m*noise_level*sqrt(2/pi);
											sd_err1 = sqrt(m)*noise_level*sqrt(1 - 2/pi);
											mean_signal_norm = k*sqrt(2/pi);
											converged(i) = (errors(1) <= (mean_err1 + SOL_TOL*sd_err1)/mean_signal_norm);
									else
											converged(i) = (errors(2) <= SOL_TOL);
									end

								end
								%fprintf('mean_err1 = %7.10f, sd_err1 = %7.10f\n', mean_err1, sd_err1);

							end
							display(sprintf('%s: n=%d, m=%d, nzs=%d, k=%d completed after %f secs.', alg, n, m, d, k, toc));
							
							line = [hh n m d k noise_level toc];
							dlmwrite('results.txt', line, 'delimiter', '\t', '-append');

						end %ends m_loop
					end %ends alg_list loop
				end %ends nonZero_list loop
			end %ends n_list loop
			line = [0000 0000 0000 0000 0000 0000]
			dlmwrite('results.txt', line, 'delimiter', '\t', '-append');
	end
end
