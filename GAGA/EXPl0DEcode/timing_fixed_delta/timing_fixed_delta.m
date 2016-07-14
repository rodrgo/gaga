function timing_fixed_delta(algs, maxiters, gpuNumber, ns, d, deltas, TOL, seed, rho_start, rho_step, tests_per_rho)

for ii = 1:length(ns)

	n = ns(ii); 
	samples = cell(size(algs));

	for l = 1:length(deltas)

		delta = deltas(l);
		m = ceil(delta*n);
		fprintf('Generating n = %d, delta = %g\n', n, delta);
		
		rho = rho_start;
		
		for j = 1:length(samples)
			samples{j} = ones(1, tests_per_rho);
		end

		runWhile = 1 > 0;
		
		while runWhile 
		
			rho = rho + rho_step;
			k = ceil(rho*m);
			
			for i = 1:tests_per_rho

				for j = 1:length(samples)
					if any(samples{j})
						if strcmp(maxiters{j}(2), 'k')
							MAXITER = str2num(maxiters{j}(1))*k;
						elseif strcmp(maxiters{j}, 'n')
							MAXITER = n;
						elseif strcmp(maxiters{j}(2), 'n')
							MAXITER = str2num(maxiters{j}(1))*n;
						else
							MAXITER = k;
						end

						% Solve problem
						options = gagaOptions('gpuNumber', int32(gpuNumber), 'maxiter',int32(MAXITER),'tol', single(TOL),'noise',single(0.0),'vecDistribution', 'gaussian', 'seed', int32(seed), 'kFixed','on');
						[errors times iters supp conv xhat] = gaga_cs(algs{j}, 'smv', int32(k), int32(m), int32(n), int32(d), options);
						seed = seed + 111;

						samples{j}(i) = (errors(2) <= 10*TOL);
					end
				end
			end

			runWhile = 1 < 0;
			for j = 1:length(samples)
				if any(samples{j})
					samples{j} = ones(1, tests_per_rho);
					runWhile = 1 > 0;
				end
			end

		end

	end

end

end
