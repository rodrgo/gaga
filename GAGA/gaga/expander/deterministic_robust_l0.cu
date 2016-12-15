
/*
**************************************************
**	deterministic-robust-l0			**
**  Noise robust version to parallel_l0		**
**************************************************
*/

inline void deterministic_robust_l0(float *d_vec, float *d_y, float *d_res, int *d_rows, int *d_cols, float *d_vals, int *d_bin, int *d_bin_counters, int *h_bin_counters, const int num_bins, int *p_sum, float tol, const int maxiter, const int k, const int m, const int n, const int d, const int alpha, const int nz, float noise_level, float *resRecord, float *timeRecord, int *p_iter, int debug_mode, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin){

	int iter = *p_iter;
	int offset = 0;
	timeRecord[0] = 0.0;

	// Options/Inputs in robust_l0
	int enforce_l1_decrease = 0;
	int do_hard_thresholding = 1; // We always do hard thresholding

	float h_sigma_noise = noise_level;
	float h_sigma_s = 1.0;

	// Thresholding variables
	int k_bin = 0;
	float alpha_ht = 0.25f;
	int MaxBin = (int)(num_bins * (1 - alpha_ht));
	float minVal = 0.0f;
	float maxChange = 1.0f;
	float max_value = MaxMagnitude(d_vec, n);
	float slope = ((num_bins - 1)/(max_value));
	
	float * d_Ax;
	float * d_updates;
	
	cudaMalloc((void**)&d_Ax, m * sizeof(float));
	cudaMalloc((void**)&d_updates, n * sizeof(float));

	computeResidual(d_res, d_y, d_Ax, d_vec, d_rows, d_cols, d_vals, nz, m, n,
		 numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
	
	float norm_res;
	float norm_res_start;
	norm_res = cublasSnrm2(m, d_res, 1);
	norm_res_start = norm_res;
	resRecord[0] = norm_res;


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

        // CUDA constants for p.d.f. construction
	float rho = ((float)k)/((float) m);

	float h_prob_zero_factor = expf(d*rho) - 1;
	float h_prob_equal_factor = expf(2*d*rho) - 1;
	float h_sigma_signal_zero = sqrtf(h_sigma_noise*h_sigma_noise + h_sigma_s*h_sigma_s*d*rho/(1- expf(-d*rho)));
	float h_sigma_signal_equal = sqrtf(2*h_sigma_noise*h_sigma_noise + 2*h_sigma_s*h_sigma_s*d*rho/(1 - expf(-2*d*rho)));

	cudaMemcpyToSymbol(sigma_noise, &h_sigma_noise, sizeof(float));
	cudaMemcpyToSymbol(prob_zero_factor, &h_prob_zero_factor, sizeof(float));
	cudaMemcpyToSymbol(prob_equal_factor, &h_prob_equal_factor, sizeof(float));
	cudaMemcpyToSymbol(sigma_signal_zero, &h_sigma_signal_zero, sizeof(float));
	cudaMemcpyToSymbol(sigma_signal_equal, &h_sigma_signal_equal, sizeof(float));

	// Vector or random variables and scores

	float *d_pz_u;
	int *d_scores;

	cudaMalloc((void**)&d_pz_u, n*sizeof(float));
	cudaMalloc((void**)&d_scores, n*sizeof(int));

	float *d_average_updates;
	cudaMalloc((void**)&d_average_updates, n*sizeof(float));

	float prob_thresh = 0.99;

	float norm_res_mean = (h_sigma_noise*h_sigma_noise)*((float) m);
	float norm_res_sd = (h_sigma_noise*h_sigma_noise)*sqrtf(2 * ((float) m)); 

	// Create tmp vectors for residual and signal

	float * d_vec_tmp;
	float * d_res_tmp;

	cudaMalloc((void**)&d_vec_tmp, n*sizeof(float));
	cudaMalloc((void**)&d_res_tmp, m*sizeof(float));

	cudaMemcpy(d_vec_tmp, d_vec, sizeof(float)*n, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_res_tmp, d_res, sizeof(float)*m, cudaMemcpyDeviceToDevice);

	float norm_res_tmp;
	norm_res_tmp = norm_res;

	int num_failed_attempts = 0;

	while( (iter < maxiter) & (norm_res*norm_res - norm_res_mean > tol*norm_res_sd) & (norm_res < (100*norm_res_start)) & (residNorm_diff > 0.000001) & (isCycling == 0) & (prob_thresh > 0.05) & (num_failed_attempts < d - 1)){

		// time variables
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		// BEGIN STEP
		offset = iter % d;
		
		// compute l0-scores in parallel
		// d_unif has "n" chunks of size "2*d + 1"
		zero_vector_float<<<numBlocks, threadsPerBlock>>>(d_average_updates, n);
		cudaThreadSynchronize();

		// Takes "1" uniform random variable.
		cuda_deterministic_prob_zero_u<<<numBlocks, threadsPerBlock>>>(d_pz_u, d, n, d_res_tmp, d_rows, offset);
		cudaThreadSynchronize();

		// Compute scores
		cuda_compute_scores_det_robust_l0<<<numBlocks, threadsPerBlock>>>(d_scores, d_pz_u, d_average_updates, d, n, d_res_tmp, d_rows, offset, prob_thresh);
		cudaThreadSynchronize();

		// Compute update signal robust l0
		cuda_update_signal_det_robust_l0<<<numBlocks, threadsPerBlock>>>(d_res_tmp, d_vec_tmp, d_updates, d_average_updates, d_rows, n, d, alpha, d_scores, enforce_l1_decrease); 
		cudaThreadSynchronize();

		// compute residual
		computeResidual(d_res_tmp, d_y, d_Ax, d_vec_tmp, d_rows, d_cols, d_vals, nz, m, n,
			 numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
		cudaThreadSynchronize();

		// hard-thresholding

		if (do_hard_thresholding == 1){
			H_k(d_vec_tmp, k, n, d_bin, d_bin_counters, h_bin_counters, &maxChange, &max_value,
				&slope, &minVal, &alpha_ht, &MaxBin, &k_bin, p_sum, num_bins,
				numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);

			// recompute residual
			computeResidual(d_res_tmp, d_y, d_Ax, d_vec_tmp, d_rows, d_cols, d_vals, nz, m, n, 
				numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
		}

		norm_res_tmp = cublasSnrm2(m, d_res_tmp, 1);

		// END STEP

		if (norm_res_tmp < 0.99*norm_res){
			// continue

			cudaMemcpy(d_vec, d_vec_tmp, sizeof(float)*n, cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_res, d_res_tmp, sizeof(float)*m, cudaMemcpyDeviceToDevice);
			norm_res = norm_res_tmp;

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
			if (iter > resRepeatLength){
				isCycling = 1;
				for (int i = resCyclingLength - 1; i >= resCyclingLength - resRepeatLength; i = i - 1)
					isCycling = isCycling*(abs(resCycling[i] - resCycling[i-1]) <= 1e-10);

			}

			num_failed_attempts = 0;

		}else{

			cudaMemcpy(d_vec_tmp, d_vec, sizeof(float)*n, cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_res_tmp, d_res, sizeof(float)*m, cudaMemcpyDeviceToDevice);

			prob_thresh -= 0.05;
			num_failed_attempts += 1;
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
			printf("iter = %d, norm_res = %5.10f, prob_thresh = %1.2f, isCycling = %d\n", iter, norm_res, prob_thresh, isCycling);
		}


	}

	float norm_xhat = 0.0;
	float big_float = 10000.0;
	
	norm_xhat = cublasSnrm2(n, d_vec, 1);

	// If norm_xhat == 0.0, then no updates were performed.
	// We increase the norm so that GAGA doesn't think it is a success (when n is large)
	if (norm_xhat == 0.0){
		cudaMemcpy(d_vec, &big_float, sizeof(float), cudaMemcpyHostToDevice);
	}
	

	*p_iter = iter;

	free(resCycling);
	free(residNorm_prev);
	free(residNorm_evolution);

	// clean GPU
	cudaFree(d_Ax);
	cudaFree(d_updates);
        //cudaFree(randArray);
        //cudaFree(d_state);

	cudaFree(d_average_updates);
	cudaFree(d_pz_u);
	cudaFree(d_scores);

	// tmp vectors
	cudaFree(d_vec_tmp);
	cudaFree(d_res_tmp);

}

