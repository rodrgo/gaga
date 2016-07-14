function [x, resNormList, iter, time_sum, ksum] = SMP_M_smv(y, A, A_rows, d, k, m, n, TOL, MAXITER)
	
	ksum = 0;

	MAXITER = 2*k;
	timeRecord = zeros(MAXITER + 1, 1);
	timeRecord(1) = 0;

	% Create initial vector

	x = zeros(n, 1);
	res = y;
	resNorm = norm(res);

	resNormList = zeros(1, 16);
	resNormList(end) = resNorm;

	resNormInitial = resNorm;
	resNormEvolution = ones(1,16);
	resNormDiff = 1;

	resRecord = zeros(MAXITER + 1, 1);
	resRecord(1) = resNorm;

	% Initialise

	u = zeros(n, 1);
	
	iter = 0;

	while ( (resNorm > TOL) && (iter <= MAXITER) && (resNorm < 100 * resNormInitial) && (resNormDiff > .01*TOL) )
		
		timeIteration = tic;

		for i = 1:n
			u(i) = median( res(A_rows( (i - 1)*d + 1:i*d) ) ); 
		end
		
		[~, ind] = sort(abs(u), 'descend');
		u(ind(2*k+1:end)) = 0;
		
		x = x + u;

		[~, ind] = sort(abs(x), 'descend');
		x(ind(k+1:end)) = 0;

		% Update residual

		res = y - A*x;
		resNorm = norm(res);
		resNormList(1:(end - 1)) = resNormList(2:end);
		resNormList(end) = resNorm;

		resNormEvolution(1:(end - 1)) = resNormEvolution(2:end);
		resNormEvolution(end) = resNormList(end - 1) - resNormList(end);
		resNormDiff = max(resNormEvolution);

		iter = iter + 1;
		resRecord(iter + 1) = resNorm;
		timeRecord(iter + 1) = timeRecord(iter) + toc(timeIteration);
	
	end
	time_sum = sum(timeRecord(1:(iter+1)));
	
end
