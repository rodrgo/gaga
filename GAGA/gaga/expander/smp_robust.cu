/*
******************************************
**		SMP-Robust		**
** 	Sparse Matching Pursuit		**
******************************************
*/

inline void smp_robust(float *d_vec, float *d_y, float *resid, float *resid_update, 
	int *d_rows, int *d_cols, float *d_vals, int *d_bin, int *d_bin_counters, 
	int *h_bin_counters, 
	float *residNorm_prev, float tol, const int maxiter, const int num_bins, 
	const int k, const int m, const int n, const int p, const int nz, float noise_level, 
	int *p_iter, float err, int *p_sum, float *p_time_sum, 
	dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, 
	dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin,
	float *timeRecord, float *resRecord){


	timeRecord[0] = 0;

	int iter = *p_iter;
	float time_sum = *p_time_sum;
	int k_bin = 0;

	float alpha = 0.25f;
	int MaxBin = (int)(num_bins * (1 - alpha));
	float minVal = 0.0f;
	float maxChange = 1.0f;
	float max_value = 1; 		// MaxMagnitude(d_vec, n);
	float slope = ((num_bins - 1)/(max_value));
	
	cublasInit();

	// SMP variables
	
	float * d_Ax;
	float * d_u;
	float * d_res;
	float * d_aux_nz;
	
	cudaMalloc((void**)&d_Ax, m * sizeof(float));
	cudaMalloc((void**)&d_u, n * sizeof(float));
	cudaMalloc((void**)&d_res, m * sizeof(float));
	cudaMalloc((void**)&d_aux_nz, nz * sizeof(float));

	// res = y - A*x^{0}
	computeResidual(d_res, d_y, d_Ax, d_vec, d_rows, d_cols, d_vals, nz, m, n, 
		numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
	
	err = cublasSnrm2(m, d_res, 1);
	resRecord[0] = err;

	float err_start = err;

	float residNorm_diff = 1.0f;
	float residNorm_evolution[16];

	// Variables to record convergence
	for(int i = 0; i < 16; i++){
		residNorm_prev[i] = 0;
		residNorm_evolution[i] = 1.0f;
	}

	// Some stopping parameters
	int fail = 0;
	float root, convergenceRate;

	// Robust stopping conditions

	float h_sigma2_n = noise_level*noise_level;
	float norm_1_res_mean = ((float) m)*sqrtf(h_sigma2_n)*sqrtf(2/3.1415926535);
	float norm_1_res_sd = sqrtf(((float) m)*h_sigma2_n*(1 - sqrtf(2/3.1415926535)));
	float norm_1_res = cublasSasum(m, d_res, 1);

	while( (iter < maxiter) & (!(norm_1_res - norm_1_res_mean <= tol*norm_1_res_sd)) & (err < (100*err_start)) & (residNorm_diff > 0.01*tol) & (fail == 0) ){

		// time variables
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		// u^* = E_med(res)
	
		zero_vector_float <<< numBlocksnp, threadsPerBlocknp >>>((float*)d_aux_nz, nz);		
		cuda_compute_scores_smp<<<numBlocks, threadsPerBlock>>>(d_u, d_res, d_rows, d_aux_nz, n, p);

		// u^j = H_{2k}[u^*]
		
		H_k(d_u, 2*k, n, d_bin, d_bin_counters, h_bin_counters, &maxChange, &max_value, 
			&slope, &minVal, &alpha, &MaxBin, &k_bin, p_sum, num_bins, 
			numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
	
		// x^j = x^{j - 1} + u^j

		cublasSaxpy(n, 1, d_u, 1, d_vec, 1);
	
		// x^j = H_k[x^j]

		H_k(d_vec, k, n, d_bin, d_bin_counters, h_bin_counters, &maxChange, &max_value, 
			&slope, &minVal, &alpha, &MaxBin, &k_bin, p_sum, num_bins, 
			numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
	
		// res = y - A*x^j
		
		computeResidual(d_res, d_y, d_Ax, d_vec, d_rows, d_cols, d_vals, nz, m, n, 
			numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
		
		err = cublasSnrm2(m, d_res, 1);
		
		// Stopping conditions 
		for(int i = 0; i < 15; i++){
			residNorm_prev[i] = residNorm_prev[i + 1];
			residNorm_evolution[i] = residNorm_evolution[i + 1];
		}
		residNorm_prev[15] = err;
		residNorm_evolution[15] = residNorm_prev[14] - residNorm_prev[15];

		residNorm_diff = max_list(residNorm_evolution, 16);

		// Convergence
		if(iter > ((int)(0.4*maxiter))){
			root = 1/15.0f;
			convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
			convergenceRate = pow(convergenceRate, root);
			if(convergenceRate > 0.999f) fail = 1;
		}

		// end timing
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		time_sum = time_sum + time;

		// j = j + 1
		
		iter = iter + 1;

		resRecord[iter] = err;
		timeRecord[iter] = timeRecord[iter - 1] + time;

	}

	*p_iter = iter;
	*p_time_sum = time_sum;

	// clean GPU	
	cudaFree(d_Ax);
	cudaFree(d_u);
	cudaFree(d_res);
	cudaFree(d_aux_nz);
}

