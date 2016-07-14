function [x, resNormList, iter, time_sum, ksum] = SSMP_M_smv(y, A_rows, d, k, m, n, TOL, MAXITER)

	ksum = 0;

	timeRecord = zeros(MAXITER + 1, 1);
	timeRecord(1) = 0;
	inner_iterations = k;

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

	updates = zeros(n, 1);
	for i = 1:n
		updates(i) = median( res(A_rows( (i - 1)*d + 1:i*d )) );
	end
	
	iter = 0;

	while ( (resNorm > TOL) && (iter <= MAXITER) && (resNorm < 100 * resNormInitial) && (resNormDiff > .01*TOL) )
		
		timeIteration = tic;

		%for i = 1:inner_iterations
			[~, node] = max(abs(updates));
			%fprintf('node = %d, update = %5.10f\n', node, updates(node));
			optimalUpdate = updates(node);
			x(node) = x(node) + optimalUpdate;
			nodeNeighbors = A_rows( (node - 1)*d + 1:node*d );
			res(nodeNeighbors) = res(nodeNeighbors) - optimalUpdate;
			for j = 1:length(nodeNeighbors)
				row = nodeNeighbors(j);
				leftNeighbors = cols_in(rows( row ):rows(row + 1) - 1);				
				for l = 1:length(leftNeighbors)
					col = leftNeighbors(l);
					updates(col) = median( res(A_rows( (col - 1)*d + 1:col*d )) );
				end
			end
		%end

		if mod(iter, inner_iterations) == 0
			[~, ind] = sort(abs(x), 'descend');
			x(ind(k+1:end)) = 0;
		end

		% Update residual

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
	[~, ind] = sort(abs(x), 'descend');
	x(ind(k+1:end)) = 0;

	time_sum = sum(timeRecord(1:(iter + 1)));
end
