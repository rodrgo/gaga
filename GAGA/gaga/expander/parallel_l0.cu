
/*
**************************************************
**		parallel-l0			**
** 	 Expander Recovery of the Future	**
**************************************************
*/

inline void parallel_l0(float *d_vec, float *d_y, float *d_res, int *d_rows, int *d_cols, float *d_vals, float tol, const int maxiter, const int k, const int m, const int n, const int d, const int alpha, const int nz, float *resRecord, float *timeRecord, int *p_iter, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm){

	int iter = *p_iter;
	int offset = 0;
	timeRecord[0] = 0.0;
	
	float * d_Ax;
	float * d_scores;
	float * d_updates;
	int * d_found_candidates;
	
	cudaMalloc((void**)&d_Ax, m * sizeof(float));
	cudaMalloc((void**)&d_scores, n * sizeof(float));
	cudaMalloc((void**)&d_updates, n * sizeof(float));
	cudaMalloc((void**)&d_found_candidates, 1 * sizeof(int));

	computeResidual(d_res, d_y, d_Ax, d_vec, d_rows, d_cols, d_vals, nz, m, n,
		 numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
	
	float norm_res;
	float norm_res_start;
	norm_res = cublasSnrm2(m, d_res, 1);
	norm_res_start = norm_res;
	resRecord[0] = norm_res;

	int isCycling = 0;
	//int ind_max = -1;
	int zero = 0;
	int h_found_candidates = 1;

	float *resCycling;
	int resCyclingLength = 2*d;
	resCycling = (float*) malloc(sizeof(float)*resCyclingLength);
	for (int i = 0; i < resCyclingLength; i++)
		resCycling[i] = 0.0;

	float *residNorm_prev;
	float *residNorm_evolution;
	float residNorm_diff = 1.0f;
	int residNorm_length = 2*d;
	residNorm_prev = (float*) malloc(sizeof(float)*residNorm_length);
	residNorm_evolution = (float*) malloc(sizeof(float)*residNorm_length);
	for (int i = 0; i < residNorm_length; i++){
		residNorm_prev[i] = 0.0;
		residNorm_evolution[i] = 1.0f;
	}

	while( (iter < maxiter) & (norm_res > tol) & (norm_res < (100*norm_res_start)) & (residNorm_diff > 0.000001) & (isCycling == 0) & (h_found_candidates == 1)){

		// time variables
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		// BEGIN STEP
		offset = iter % d;

		cudaMemcpy(d_found_candidates, &zero, sizeof(int), cudaMemcpyHostToDevice);
		
		// compute l0-scores in parallel

		cuda_compute_scores_l0<<<numBlocks, threadsPerBlock>>>(d_scores, d_res, d_vec, d_updates, d_rows, d_found_candidates, n, d, alpha, offset); 

		// check if we found good candidates

		cudaMemcpy(&h_found_candidates, d_found_candidates, sizeof(int), cudaMemcpyDeviceToHost);

		
		// If we didn't, get the one with maximum score and update d_vec
		/*
		if (h_found_candidates == 0){
			ind_max = cublasIsamax(n, d_scores, 1) - 1;
			cudaMemcpy(d_vec+ind_max, d_updates+ind_max, sizeof(float), cudaMemcpyDeviceToDevice);
		}
		*/

		// Regardless of the outcome, need to recompute the residual.
		// To avoid device-host communication, we do this on the device.

		cuda_update_residual<<<numBlocks, threadsPerBlock>>>(d_res, d_rows, d_updates, n, d);
		//cuda_update_residual_np<<<numBlocksnp, threadsPerBlocknp>>>(d_res, d_rows, d_updates, n, d);

		//computeResidual(d_res, d_y, d_Ax, d_vec, d_rows, d_cols, d_vals, nz, m, n,
		//	 numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
		
		norm_res = cublasSnrm2(m, d_res, 1);

		// END STEP

		// check for no change in residual 
		for (int i = 0; i < residNorm_length - 1; i++){
			residNorm_prev[i] = residNorm_prev[i + 1];
		}
		residNorm_prev[residNorm_length - 1] = norm_res;
		for (int i = 0; i < residNorm_length - 1; i++){
			residNorm_evolution[i] = residNorm_evolution[i + 1];
		}
		residNorm_evolution[residNorm_length - 1] = residNorm_prev[residNorm_length - 2] - residNorm_prev[residNorm_length - 1];
		residNorm_diff = max_list(residNorm_evolution, residNorm_length);


		// Check for cycling
		// Check works, but not in all cases. Need more precise check.
		for (int i = 0; i < resCyclingLength - 1; i++)
			resCycling[i] = resCycling[i + 1]; 
		resCycling[resCyclingLength - 1] = norm_res;
		if (iter > resCyclingLength){
			isCycling = 1;
			for (int i = 3; i < resCyclingLength; i = i + 2)
				isCycling = isCycling*(resCycling[i] == resCycling[i - 2]);
		}

		// end timing
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	
		iter = iter + 1;
		resRecord[iter] = norm_res;
		timeRecord[iter] = timeRecord[iter - 1] + time;

		//printf("iter = %d\toffset = %d\tnorm_Res = %g\tnorm_diff = %g\tisCycling = %d\th_found_candidates = %d\n", iter, offset, norm_res, residNorm_diff, isCycling, h_found_candidates);

	}

	*p_iter = iter;

	free(resCycling);
	free(residNorm_prev);
	free(residNorm_evolution);

	// clean GPU
	cudaFree(d_Ax);
	cudaFree(d_scores);
	cudaFree(d_updates);
	cudaFree(d_found_candidates);

}

