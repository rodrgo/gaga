function [x, resNormList, iter, time_sum, ksum] = ER_M_smv(y, A_rows, d, k, m, n, TOL, MAXITER)
	
	ksum = 0;
	EPS = k*1e-16;
	ROUND = 10.^(-floor(log10(EPS)));

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

	% Index

	A_cols = zeros(size(A_rows));
	for j = 1:n
        	A_cols((j-1)*d + 1 : j*d) = j; 
	end
	[A_rows_ind, ind] = sort(A_rows, 'ascend');
	cols_in = A_cols(ind)';
	rows = [1; find(A_rows_ind(2:end) - A_rows_ind(1:end-1)) + 1; length(A_cols) + 1]';

	% Initialise

	overlaps = zeros(n, 1);
	updates = zeros(n, 1);
	for i = 1:n
		rightValues = res( A_rows( (i - 1)*d + 1: i*d ) );
		rightValues = rightValues( abs(rightValues) > EPS );
		rightValuesRound = floor( ROUND*rightValues )/ROUND;
		if numel(rightValuesRound) > 0
			[modeValue, overlaps(i)] = mode(rightValuesRound); 
			updates(i) = rightValues( find(rightValuesRound == modeValue, 1) );
		else
			updates(i) = 0;
			overlaps(i) = 0;
		end
	end

	cycling = zeros(1, 2*(d + 1));
	isCycling = 0;
	
	iter = 0;
	foundOverlaps = 1;

	while ( (resNorm > TOL) && (iter <= MAXITER) && (resNorm < 100 * resNormInitial) && foundOverlaps && (~isCycling) )
		
		timeIteration = tic;

		[maxOverlap, node] = max(overlaps);

		if maxOverlap > 2 
			optimalUpdate = updates(node);
			x(node) = x(node) + optimalUpdate;
			nodeNeighbors = A_rows( (node - 1)*d + 1:node*d );
			res(nodeNeighbors) = res(nodeNeighbors) - optimalUpdate;
			res(nodeNeighbors(abs(res(nodeNeighbors) - optimalUpdate) <= EPS)) = 0;
			for j = 1:length(nodeNeighbors)
				row = nodeNeighbors(j);
				leftNeighbors = cols_in(rows( row ):rows(row + 1) - 1);				
				for l = 1:length(leftNeighbors)
					col = leftNeighbors(l);
					rightValues = res( A_rows( (col - 1)*d + 1: col*d ) );
					rightValues = rightValues( abs(rightValues) > EPS );
					rightValuesRound = floor( ROUND*rightValues )/ROUND;
					if numel(rightValuesRound) > 0
						[modeValue, overlaps(col)] = mode(rightValuesRound); 
						updates(col) = rightValues( find(rightValuesRound == modeValue, 1) );
					else
						updates(col) = 0;
						overlaps(col) = 0;
					end
				end
			end
		else
			foundOverlaps = 0;
		end 
		%fprintf('iter = %d\terr = %g\th_node = %d\th_score = %d\th_mode = %g\n', iter, resNorm, node, maxOverlap, optimalUpdate);

		% cycling
		
		nodes = node;

		for i = 1:length(nodes)
			cycling(1:(end - 1)) = cycling(2:end);
			cycling(end) = nodes(i);
		end

		if (iter > length(cycling))
			if ~any(cycling ~= cycling(1))
				isCycling = 1;
			end
			if ~any(cycling(1:2:end) ~= cycling(1)) && ~any(cycling(2:2:end) ~= cycling(2))
				isCycling = 1;
			end
		end

		% Update residual

		resNorm = norm(res);
		resNormList(1:(end - 1)) = resNormList(2:end);
		resNormList(end) = resNorm;

		resNormEvolution(1:(end - 1)) = resNormEvolution(2:end);
		resNormEvolution(end) = resNormList(end - 1) - resNormList(end);

		iter = iter + 1;
		resRecord(iter + 1) = resNorm;
		timeRecord(iter + 1) = timeRecord(iter) + toc(timeIteration);
	
		%fprintf('maxOverlap = %d, node = %d, update = %5.20f, resNorm = %5.10f\n', maxOverlap, node, optimalUpdate, resNorm);
	end
	time_sum = sum(timeRecord(1:(iter + 1)));
end
