function rho_vs_noise_fixed_delta(algs, maxiters, gpuNumber, ns, d, deltas, RES_TOL, SOL_TOL, seed, rho_start, rho_step, tests_per_rho, noise_levels, l0_thresh)

load_clips;

for nl = 1:length(noise_levels)

    noise_level = noise_levels(nl);
    fprintf('Noise level = %g\n', noise_level);

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

            runWhile = true;
            
            while runWhile 
            
                rho = rho + rho_step;
                k = ceil(rho*m);
                
                for i = 1:tests_per_rho

                    for j = 1:length(samples)
                        alg = algs{j};
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
                            options = gagaOptions('gpuNumber', int32(gpuNumber), 'maxiter',int32(MAXITER),'tol', single(RES_TOL),'noise',single(noise_level),'vecDistribution', 'gaussian', 'l0_thresh', l0_thresh, 'seed', int32(seed), 'kFixed','on');
                            [errors times iters supp conv xhat] = gaga_cs(alg, 'smv', int32(k), int32(m), int32(n), int32(d), options);
                            seed = seed + 111;

                            if any(ismember({'robust_l0', 'robust_l0_adaptive', 'robust_l0_trans', 'robust_l0_adaptive_trans', 'ssmp_robust', 'smp_robust', 'cgiht_robust'}, {alg}))
                              mean_err1 = m*noise_level*sqrt(2/pi);
                              sd_err1 = sqrt(m)*noise_level*sqrt(1 - 2/pi);
                              mean_signal_norm = k*sqrt(2/pi);
                              upper_bound = min((mean_err1 + SOL_TOL*sd_err1)/mean_signal_norm, UPPER_BOUND_CLIP);
                              samples{j}(i) = (errors(1) <= upper_bound);
                              %samples{j}(i) = (errors(1) <= (mean_err1 + SOL_TOL*sd_err1)/mean_signal_norm);
                            else
                              samples{j}(i) = (errors(2) <= 10*SOL_TOL);
                            end
                        end
                    end
                end

                runWhile = false;
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

end
