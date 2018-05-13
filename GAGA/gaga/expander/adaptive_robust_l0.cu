
/*
**************************************************
**		adaptive-robust-l0							
**  	Noise robust version to parallel_l0	
**************************************************
*/

inline void adaptive_robust_l0(float *d_vec, float *d_y, float *d_res, int *d_rows, int *d_cols, float *d_vals, int *d_bin, int *d_bin_counters, int *h_bin_counters, const int num_bins, int *p_sum, float tol, const int maxiter, const int k, const int m, const int n, const int d, int alpha, const int nz, float noise_level, int boost_flag_opt, int adaptive_flag_opt, float *resRecord, float *timeRecord, int *p_iter, int debug_mode, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin, int *p_fail_update_flag){

	int iter = *p_iter;
	int offset = 0;
	timeRecord[0] = 0.0;

	// Boost flag

	cudaMemcpyToSymbol(boost_flag, &boost_flag_opt, sizeof(int));

	// Noise constants

	float h_sigma2_n = noise_level*noise_level;
	float h_sigma2_s = 1.0;
	float h_snr = h_sigma2_s/h_sigma2_n; // Signal-to-noise ratio

	cudaMemcpyToSymbol(sigma2_n, &h_sigma2_n, sizeof(float));
	cudaMemcpyToSymbol(sigma2_s, &h_sigma2_s, sizeof(float));
	cudaMemcpyToSymbol(snr, &h_snr, sizeof(float));

	// Thresholding variables

	int k_bin = 0;
	float alpha_ht = 0.25f;
	int MaxBin = (int)(num_bins * (1 - alpha_ht));
	float minVal = 0.0f;
	float maxChange = 1.0f;
	float max_value = MaxMagnitude(d_vec, n);
	float slope = ((num_bins - 1)/(max_value));
	
	// Other auxiliary variables

	float * d_Ax;
	cudaMalloc((void**)&d_Ax, m * sizeof(float));

	float *d_vec_ind;
	cudaMalloc((void**)&d_vec_ind, n * sizeof(float));
	zero_vector_float<<<numBlocks, threadsPerBlock>>>(d_vec_ind, n);
	thrust::device_ptr<float> thrust_vec_ind(d_vec_ind);

	// Residual
	// Assume initial guess is zero

	cudaMemcpy(d_res, d_y, sizeof(float)*m, cudaMemcpyDeviceToDevice);

	float norm_res;
	float norm_res_start;
	norm_res = cublasSnrm2(m, d_res, 1);
	norm_res_start = norm_res;
	resRecord[0] = norm_res;

	// Stopping condition variables

	int stopping_debug = 0;

	int isCycling = 0;

	float *resCycling;
	int resCyclingLength = 2*d;
	int resRepeatLength = d - 1;
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

	// Create tmp vectors for residual and signal

	float * d_vec_tmp;
	float * d_res_tmp;

	cudaMalloc((void**)&d_vec_tmp, n*sizeof(float));
	cudaMalloc((void**)&d_res_tmp, m*sizeof(float));

	cudaMemcpy(d_vec_tmp, d_vec, sizeof(float)*n, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_res_tmp, d_res, sizeof(float)*m, cudaMemcpyDeviceToDevice);

	float norm_res_tmp;
	norm_res_tmp = norm_res;

	// L1-norm stopping conditions

	float norm_1_res_mean = ((float) m)*sqrtf(h_sigma2_n)*sqrtf(2/3.1415926535);
	float norm_1_res_sd = sqrtf(((float) m)*h_sigma2_n*(1 - sqrtf(2/3.1415926535)));

	float norm_1_res = cublasSasum(m, d_res_tmp, 1);
	float norm_1_res_tmp = norm_1_res;

	if (debug_mode == 1){
		printf("norm_1_res_mean = %5.6f\n", norm_1_res_mean);
		printf("norm_1_res_sd = %5.6f\n", norm_1_res_sd);
	}

	// Other variables for adaptive-robust-l0

	int xhat_k = 0;
	int k_target = k - xhat_k;

	// Get max probability
	// In cuda_adaptive_robust_l0_score_and_update we first compute the probability of 
	// a value in the residual being not zero.
	// prob_not_zero(value) >= prob_thresh.
	// If prob_thresh starts at 1, we might fail a number of times before making an update
	// Hence, we find
	// max_{value} prob_not_zero(value) = max_{value} ( 1 - prob_zero(value) ) 

	float prob_thresh = 1;

	/*
	float *d_probs_nonzero;
	cudaMalloc((void**)&d_probs_nonzero, m*sizeof(float));
	get_probs_nonzero<<<numBlocksm, threadsPerBlockm>>>(d_probs_nonzero, d_res, d, k_target, m);

	thrust::device_ptr<float> thrust_probs_nonzero(d_probs_nonzero);
	prob_thresh = *(thrust::max_element(thrust_probs_nonzero, thrust_probs_nonzero+m));
	cudaFree(d_probs_nonzero);
	*/

	float * d_prob_zero_at_zero;
	cudaMalloc((void**)&d_prob_zero_at_zero, 1*sizeof(float));
	compute_prob_zero_at_zero<<<numBlocks, threadsPerBlock>>>(d_prob_zero_at_zero, d, k_target, m);

	prob_thresh = prob_thresh - 0.01;

	// Select probability decrement 
	float delta = ((float)m)/((float)n); 
	float rho = ((float)k)/((float)m); 
	float prob_decr = 0.0;
	if (delta <= 0.05) prob_decr = 0.01;
	else if (delta > 0.05){
		if (boost_flag_opt == 1){
			if (rho > 0 && rho <= 0.1) prob_decr = 0.05; 
			else if (rho > 0.1 && rho <= 0.2) prob_decr = 0.075; 
			else prob_decr = 0.10; 
		}else{
			prob_decr = 0.025;
		} 
	}

	while( (iter < maxiter) & (!(norm_1_res - norm_1_res_mean <= tol*norm_1_res_sd)) & (norm_res < (100*norm_res_start)) & (residNorm_diff > 0.000001) & (isCycling == 0) & (prob_thresh > 0.0) ){

		// time variables
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		// BEGIN STEP

		offset = iter % d;
		
		// Compute scores and update

		cuda_adaptive_robust_l0_score_and_update<<<numBlocks, threadsPerBlock>>>(d_vec, alpha, d, k_target, m, n, d_res, d_rows, offset, prob_thresh, d_prob_zero_at_zero);

		if (debug_mode == 1){
			printf("k_target=%d, prob_thresh=%1.2f, alpha=%d\n", k_target, prob_thresh, alpha);
		}

		// hard-thresholding
		// Get effective value of xhat_k

		H_k(d_vec_tmp, k, n, d_bin, d_bin_counters, 
			h_bin_counters, &maxChange, &max_value,
			&slope, &minVal, &alpha_ht, &MaxBin,
			&k_bin, p_sum, num_bins, numBlocks,
			threadsPerBlock, numBlocks_bin,
			threadsPerBlock_bin);

		// recompute residual
		computeResidual(d_res_tmp, d_y, d_Ax, d_vec_tmp,
			d_rows, d_cols, d_vals, nz, m, n, numBlocksm,
			threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
		cudaThreadSynchronize();

		norm_1_res_tmp = cublasSasum(m, d_res_tmp, 1);

		// END STEP

		//if (norm_1_res_tmp < 0.99*norm_1_res){
		if (norm_1_res_tmp < norm_1_res){

			cudaMemcpy(d_vec, d_vec_tmp, sizeof(float)*n, cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_res, d_res_tmp, sizeof(float)*m, cudaMemcpyDeviceToDevice);

			norm_res_tmp = cublasSnrm2(m, d_res_tmp, 1);
			norm_res = norm_res_tmp;
			norm_1_res = norm_1_res_tmp;

			// STOPPING CONDITIONS

			// check for no change in residual 
			for (int i = 0; i < residNorm_length - 1; i++){
				residNorm_prev[i] = residNorm_prev[i + 1];
			}
			residNorm_prev[residNorm_length - 1] = norm_1_res;
			for (int i = 0; i < residNorm_length - 1; i++){
				residNorm_evolution[i] = residNorm_evolution[i + 1];
			}
			residNorm_evolution[residNorm_length - 1] = residNorm_prev[residNorm_length - 2] - residNorm_prev[residNorm_length - 1];
			residNorm_diff = max_list(residNorm_evolution, residNorm_length);

			// Check for cycling
			// Check works, but not in all cases. Need more precise check.
			for (int i = 0; i < resCyclingLength - 1; i++)
				resCycling[i] = resCycling[i + 1]; 
			resCycling[resCyclingLength - 1] = norm_1_res;
			if (iter > resCyclingLength){
				isCycling = 1;
				for (int i = 3; i < resCyclingLength; i = i + 2)
					isCycling = isCycling*(resCycling[i] == resCycling[i - 2]);
			}
			if (iter > resRepeatLength){
				isCycling = 1;
				for (int i = resCyclingLength - 1; i >= resCyclingLength - resRepeatLength; i = i - 1)
					isCycling = isCycling*(abs(resCycling[i] - resCycling[i-1]) <= 1e-10);
			}


			// END STOPPING CONDITIONS
			// Get number of nonzeros in xhat

			if (adaptive_flag_opt == 1){
				find_nonzeros<<<numBlocks, threadsPerBlock>>>(d_vec, d_vec_ind, n, d, k_target, m, prob_thresh, d_prob_zero_at_zero);
				cudaThreadSynchronize();

				//xhat_k = thrust::count(thrust_vec_ind, thrust_vec_ind+n, 1);
				xhat_k = cublasSasum(n, d_vec_ind, 1);
				k_target = k - xhat_k;
				k_target = k_target > floor(0.01*m) ? k_target : floor(0.01*m);
				compute_prob_zero_at_zero<<<numBlocks, threadsPerBlock>>>(d_prob_zero_at_zero, d, k_target, m);
			}


		}else{

			cudaMemcpy(d_vec_tmp, d_vec, sizeof(float)*n, cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_res_tmp, d_res, sizeof(float)*m, cudaMemcpyDeviceToDevice);

			prob_thresh -= prob_decr;

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

		if(debug_mode == 1){
			printf("iter = %d, norm_1_res = %5.10f\n", iter, norm_1_res);
			stopping_debug = ( (iter < maxiter) & \
				(!(norm_1_res - norm_1_res_mean <= tol*norm_1_res_sd)) & \
				(norm_res < (100*norm_res_start)) & \
				(residNorm_diff > 0.000001) & \
				(isCycling == 0) & \
				(prob_thresh > 0.05) );
			if (stopping_debug == 0){
				printf("Iterations: %d\n", (iter < maxiter));
				printf("converged: %d\n", (!(norm_1_res - norm_1_res_mean <= tol*norm_1_res_sd)));
				printf("diverged: %d\n", (norm_res < (100*norm_res_start)));
				printf("residNorm_diff: %d\n", (residNorm_diff > 0.000001) );
				printf("isCycling: %d\n", (isCycling == 0) );
				printf("prob_thresh: %d\n", (prob_thresh > 0.05));
			}
		}


	}

	// If norm_xhat == 0.0, then no updates were performed.
	// We flag a fail
	float norm_xhat = 0.0;
	norm_xhat = cublasSnrm2(n, d_vec, 1);

	if (norm_xhat == 0.0){
		*p_fail_update_flag = 1;
	}

	*p_iter = iter;

	free(resCycling);
	free(residNorm_prev);
	free(residNorm_evolution);

	// clean GPU
	cudaFree(d_Ax);
	cudaFree(d_vec_ind);

	// tmp vectors
	cudaFree(d_vec_tmp);
	cudaFree(d_res_tmp);

	cudaFree(d_prob_zero_at_zero);

}

