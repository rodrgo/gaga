function timing_fixed_delta(algs, maxiters, gpuNumber, ns, d, deltas, RES_TOL, SOL_TOL, l0_thresh, noise_levels, seed, rho_start, rho_step, tests_per_rho)

for nl = 1:length(noise_levels)

  noise_level = noise_levels(nl);

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
              options = gagaOptions('gpuNumber', int32(gpuNumber), 'maxiter',int32(MAXITER),'tol', single(RES_TOL),'noise',single(noise_level),'vecDistribution', 'gaussian', 'l0_thresh', l0_thresh, 'seed', int32(seed), 'kFixed','on');
              [errors times iters supp conv xhat] = gaga_cs(algs{j}, 'smv', int32(k), int32(m), int32(n), int32(d), options);
              seed = seed + 111;

              if any(ismember({'deterministic_robust_l0', 'robust_l0', 'ssmp_robust'}, {algs{j}}))
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
                samples{j}(i) = (errors(1) <= (mean_err1 + SOL_TOL*sd_err1)/mean_signal_norm);
              else
                samples{j}(i) = (errors(2) <= SOL_TOL);
              end

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

end
