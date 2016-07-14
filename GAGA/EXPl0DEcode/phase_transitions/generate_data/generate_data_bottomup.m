function generate_data_bottomup(alg_list, n_list, nonZero_list, delta_list, vecDistribution, TOL, tests_per_k, gpuNumber, maxiter_str, band_percentage)

	if nargin < 10
		band_percentage = 0;
	end
	delta_list = sort(delta_list);
	results_matrix = [];
	m_minimum = max(nonZero_list)^2 + 2;

	tic;
	 
	seed = 1;
	for ii = 1:length(n_list)
		n = n_list(ii);
		MAXITER = ceil(2*n);

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
						
						if nargin == 9
							if strcmp(maxiter_str(2), 'k')
								MAXITER = str2num(maxiter_str(1))*k;
							end
						end
						
						for i = 1:tests_per_k

							seed = seed + 111;
							options = gagaOptions('gpuNumber', gpuNumber, 'maxiter', MAXITER,'tol', TOL,'noise', single(0.0),'vecDistribution', vecDistribution, 'seed', seed, 'kFixed','on', 'band_percentage', band_percentage);
							%dlmwrite('problems.txt', [n m k d seed], 'delimiter', '\t', '-append');
							[errors times iters supp conv xhat] = gaga_cs(alg, 'smv', int32(k), int32(m), int32(n), int32(d), options);
							converged(i) = (errors(2) <= TOL);

						end

					end
					display(sprintf('%s: n=%d, m=%d, nzs=%d, k=%d completed after %f secs.', alg, n, m, d, k, toc));
					
					line = [hh n m d k toc];
					dlmwrite('results.txt', line, 'delimiter', '\t', '-append');

				end %ends m_loop
			end %ends alg_list loop
		end %ends nonZero_list loop
	end %ends n_list loop
	line = [0000 0000 0000 0000 0000 0000]
	dlmwrite('results.txt', line, 'delimiter', '\t', '-append');
end
