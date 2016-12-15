
/*
**********************************************************
**			SSMP_robust 			**
** 	Sequantial Sparse Matching Pursuit index	**
**********************************************************
*/

// Change stopping conditions to those of robust_l0 and deterministic_robust_l0
inline void ssmp_robust(float *d_vec, float *d_y, float *resid, float *resid_update, 
	int *d_rows, int *d_cols, float *d_vals, 
	int *d_rm_rows_index, int *d_rm_cols, int max_nonzero_rows_count,
	int *d_bin, int *d_bin_counters, int *h_bin_counters, 
	float *residNorm_prev, float tol, const int maxiter, const int num_bins, 
	const int k, const int m, const int n, const int p, const int nz, float noise_level,
	int *p_iter, float err, int *p_sum, float *p_time_sum, 
	dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, 
	dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin,
	dim3 numBlocksr, dim3 threadsPerBlockr,
	float *timeRecord, float *resRecord){

	timeRecord[0] = 0;

	float h_sigma_noise = noise_level;

	int iter = *p_iter;
	float time_sum = *p_time_sum;
	int k_bin = 0;

	float alpha = 0.25f;
	int MaxBin = (int)(num_bins * (1 - alpha));
	float minVal = 0.0f;
	float maxChange = 1.0f;
	float max_value = MaxMagnitude(d_vec, n);
	float slope = ((num_bins - 1)/(max_value));
	
	int h_node;
	
	cublasInit();

	int S = k;

	float * d_Ax;
	float * d_scores;
	float * d_medians;
	float * d_res;
	float * d_aux_nz;
	float * d_aux_rp;

	int r = max_nonzero_rows_count * p;
	
	cudaMalloc((void**)&d_Ax, m * sizeof(float));
	cudaMalloc((void**)&d_scores, n * sizeof(float));
	cudaMalloc((void**)&d_medians, n * sizeof(float));
	cudaMalloc((void**)&d_res, m * sizeof(float));
	cudaMalloc((void**)&d_aux_nz, nz * sizeof(float));
	cudaMalloc((void**)&d_aux_rp, p * r * sizeof(float));
	
	computeResidual(d_res, d_y, d_Ax, d_vec, d_rows, d_cols, d_vals, nz, m, n,
		 numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
	
	err = cublasSnrm2(m, d_res, 1);

	resRecord[0] = err;

	float err_start = err;
	float resid_tol = tol;

	float residNorm_diff = 1.0f;
	float residNorm_evolution[16];

	// Variables to record convergence
	for(int i = 0; i < 16; i++){
		residNorm_prev[i] = 0;
		residNorm_evolution[i] = 1.0f;
	}

	cuda_compute_scores_ssmp<<< numBlocks, threadsPerBlock >>>(d_scores, d_medians, d_res, d_rows, d_vals, d_aux_nz, n, p);

	/*
	float h_score;
	float h_median;
	*/

	float norm_res_mean = (h_sigma_noise*h_sigma_noise)*((float) m);
	float norm_res_sd = (h_sigma_noise*h_sigma_noise)*sqrtf(2 * ((float) m)); 

	while( (iter < maxiter) & (err*err - norm_res_mean > resid_tol*norm_res_sd) & (err < (100*err_start)) & (residNorm_diff > 0.01*tol) ){
		
		// time variables
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		h_node = cublasIsamax(n, d_scores, 1) - 1;

		/*
		cudaMemcpy(&h_score, d_scores+h_node, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_median, d_medians+h_node, sizeof(float), cudaMemcpyDeviceToHost);
		printf("iter = %d\terr = %g\th_node = %d\th_score = %g\th_median = %g\n", iter, err, h_node, h_score, h_median);
		*/

		cudaUpdateSignal<<<numBlocks, threadsPerBlock>>>(d_vec, h_node, d_medians);

		if ( (iter % S == 0) & (iter > 0)){
			// thresholding
			H_k(d_vec, k, n, d_bin, d_bin_counters, h_bin_counters, &maxChange, &max_value, 
				&slope, &minVal, &alpha, &MaxBin, &k_bin, p_sum, num_bins, 
				numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
			// recompute residual
			computeResidual(d_res, d_y, d_Ax, d_vec, d_rows, d_cols, d_vals, nz, m, n, 
				numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
			cuda_compute_scores_ssmp<<< numBlocks, threadsPerBlock >>>(d_scores, d_medians, d_res, d_rows, d_vals, d_aux_nz, n, p);
		}else{
			// update residual
			cudaUpdateResidual<<<numBlocks, threadsPerBlock>>>(d_res, d_rows, d_vals, d_medians, h_node, p);

			// update scores only in affected columns
			cuda_update_scores_ssmp<<< numBlocksr, threadsPerBlockr >>>(d_scores, d_medians, d_res, d_rows, d_aux_rp, d_rm_cols, d_rm_rows_index, max_nonzero_rows_count, p, h_node);
		}

		err = cublasSnrm2(m, d_res, 1);

		// Stopping conditions
		for(int i = 0; i < 15; i++){
			residNorm_prev[i] = residNorm_prev[i + 1];
			residNorm_evolution[i] = residNorm_evolution[i + 1];
		}
		residNorm_prev[15] = err;
		residNorm_evolution[15] = residNorm_prev[14] - residNorm_prev[15];
		residNorm_diff = max_list(residNorm_evolution, 16);

		// j = j + 1
		iter = iter + 1;
	
		// end timing	
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		time_sum = time_sum + time;

		resRecord[iter] = err;
		timeRecord[iter] = timeRecord[iter - 1] + time;

	}

	H_k(d_vec, k, n, d_bin, d_bin_counters, h_bin_counters, &maxChange, &max_value, 
		&slope, &minVal, &alpha, &MaxBin, &k_bin, p_sum, num_bins, 
		numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);

	*p_iter = iter;
	*p_time_sum = time_sum;

	cudaFree(d_Ax);
	cudaFree(d_scores);
	cudaFree(d_medians);
	cudaFree(d_res);
	cudaFree(d_aux_rp);
	cudaFree(d_aux_nz);

}

