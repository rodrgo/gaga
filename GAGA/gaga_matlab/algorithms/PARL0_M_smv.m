function [x, resNormList, iter, time_sum, ksum] = PARL0_M_smv(res, A_rows, d, k, m, n, TOL, MAXITER)

	ksum = 0;
	EPS = k*1e-15;

	timeRecord = zeros(MAXITER + 1, 1);
	timeRecord(1) = 0;

	% Create initial vector

	x = zeros(n, 1);

	% Set up stopping conditions 
	
	stopSize = max([2*d, 16]);

	resNorm = norm(res);

	resNormList = zeros(1, stopSize);
	resNormList(end) = resNorm;

	residNormEvolution = ones(1, stopSize);

	resCycling = zeros(1, stopSize);

	resNormInitial = resNorm;
	
	resRecord = zeros(MAXITER + 1, 1);
	resRecord(1) = resNorm;

	% Initialise

	l0ScoreList = zeros(1, n);

	nodes = zeros(n, 1);
	updates = zeros(n, 1);

	isCycling = 0;
	residNorm_diff = 100;
	iter = 0;

	while ((resNorm > TOL) && (iter <= MAXITER) && (resNorm < 100*resNormInitial) && (isCycling == 0) && (residNorm_diff > EPS))
		
		timeIteration = tic;

		offset = 1 + mod(iter, d);

		nUpdatedNodes = 0;

		for node = 1:n
			update = res(A_rows(offset + d*(node - 1)));
			if abs(update) > EPS
				rightValues = res(A_rows((node - 1)*d + 1:node*d ));
				nOverlaps = length(rightValues(abs(rightValues - update) <= EPS));
				nZeros = length(rightValues(abs(rightValues) <= EPS));
				l0ScoreList(node) = nOverlaps - nZeros;
				if l0ScoreList(node) > 1
					nUpdatedNodes = nUpdatedNodes + 1;
					nodes(nUpdatedNodes) = node;
					updates(nUpdatedNodes) = update;
					%fprintf('%d - %d = %d\n', nOverlaps, nZeros, l0ScoreList(node));
				end
			end
		end

		% Update x

		if nUpdatedNodes > 0
			x(nodes(1:nUpdatedNodes)) = x(nodes(1:nUpdatedNodes)) + updates(1:nUpdatedNodes);
			for i = 1:nUpdatedNodes 
				node = nodes(i);
				update = updates(i);
				nodeNeighbours = A_rows((node - 1)*d + 1:node*d);
				res(nodeNeighbours) = res(nodeNeighbours) - update;
				res(nodeNeighbours(abs(res(nodeNeighbours) - update) <= EPS)) = 0;
			end
		end
		
		l0ScoreList(:) = 0;

		% Stopping conditions

		resNorm = norm(res);
		resNormList(1:(end - 1)) = resNormList(2:end);
		resNormList(end) = resNorm;

		% residNorm_diff 

		residNormEvolution(1:end-1) = residNormEvolution(2:end);
		residNormEvolution(end) = resNormList(end - 1) - resNormList(end);

		residNorm_diff = residNormEvolution(1); 
		for i = 1:length(residNormEvolution)
			if residNorm_diff > residNormEvolution(i)
				residNorm_diff = residNormEvolution(i);
			end
		end
		
		% cycling

		resCycling(1:end-1) = resCycling(2:end);
		resCycling(end) = resNorm;
		if (iter > length(resCycling))
			isCycling = 1;
			for i = 3:2:length(resCyling)
				isCycling = isCycling*(resCycling(i) == resCycling(i - 2));
			end
		end

		% Increase iter
				
		iter = iter + 1;
		resRecord(iter + 1) = resNorm;
		timeRecord(iter + 1) = timeRecord(iter) + toc(timeIteration);
	
	end
	time_sum = sum(timeRecord(1:(iter+1)));
	resNormList = resNormList(end-15:end);
end
