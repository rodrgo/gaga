
/*
**********************************************************
**			ER index			**
** 		Expander Recovery index			**
**********************************************************
*/

inline void er(float *d_vec, float *d_y, float *resid,
	int *d_rows, int *d_cols, float *d_vals,
	int *d_rm_rows_index, int *d_rm_cols, int max_nonzero_rows_count,
	float tol, const int maxiter, const int k, const int m, const int n, const int p, const int nz, 
	float *resRecord, float *timeRecord, int *p_iter, dim3 numBlocks, dim3 threadsPerBlock,
	dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm,
	dim3 numBlocksr, dim3 threadsPerBlockr){

	timeRecord[0] = 0;

	int iter = *p_iter;

	int h_node;
	
	cublasInit();

	float * d_Ax;
	float * d_scores;
	float * d_modes;
	float * d_res;
	float * d_aux_nz;
	float * d_aux_rp;

	int r = p*max_nonzero_rows_count;
	
	cudaMalloc((void**)&d_Ax, m * sizeof(float));
	cudaMalloc((void**)&d_scores, n * sizeof(float));
	cudaMalloc((void**)&d_modes, n * sizeof(float));
	cudaMalloc((void**)&d_res, m * sizeof(float));
	cudaMalloc((void**)&d_aux_nz, nz * sizeof(float));
	cudaMalloc((void**)&d_aux_rp, p * r * sizeof(float));
	
	float err;

	computeResidual(d_res, d_y, d_Ax, d_vec, d_rows, d_cols, d_vals, nz, m, n,
		 numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
	cudaFree(d_Ax);
	
	err = cublasSnrm2(m, d_res, 1);

	resRecord[0] = err;

	float err_start = err;
	float resid_tol = tol;

	int isCycling = 0;
	float *resCycling;
	int resCyclingLength = 2*p;
	resCycling = (float*) malloc(sizeof(float)*resCyclingLength);
	for (int i = 0; i < resCyclingLength; i++)
		resCycling[i] = 0.0;

	cuda_compute_scores_er<<< numBlocks, threadsPerBlock >>>(d_scores, d_modes, d_res, d_rows, d_vals, d_aux_nz, n, p);
	cudaFree(d_aux_nz);

	/*
	float h_score;
	float h_mode;
	*/

	while( (iter < maxiter) & (err > resid_tol) & (err < (100*err_start)) & (isCycling == 0)){
		
		// time variables
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		h_node = cublasIsamax(n, d_scores, 1) - 1;

		/*
		cudaMemcpy(&h_score, d_scores+h_node, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_mode, d_modes+h_node, sizeof(float), cudaMemcpyDeviceToHost);
		printf("iter = %d\terr = %g\th_node = %d\th_score = %g\th_mode = %g\n", iter, err, h_node, h_score, h_mode);
		*/
		
		cudaUpdateSignal<<<numBlocks, threadsPerBlock>>>(d_vec, h_node, d_modes);
	
		cudaUpdateResidual<<<numBlocks, threadsPerBlock>>>(d_res, d_rows, d_vals, d_modes, h_node, p);

		err = cublasSnrm2(m, d_res, 1);

		// Updte scores of columns affected upon updating h_node
		cuda_update_scores_er<<< numBlocksr, threadsPerBlockr >>>(d_scores, d_modes, d_res, d_rows, d_aux_rp, d_rm_cols, d_rm_rows_index, max_nonzero_rows_count, p, h_node);

		// j = j + 1
		iter = iter + 1;
	

		// Check for cycling
		// Check works, but not in all cases. Need more precise check.
		for (int i = 0; i < resCyclingLength - 1; i++)
			resCycling[i] = resCycling[i + 1]; 
		resCycling[resCyclingLength - 1] = err;
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

		resRecord[iter] = err;
		timeRecord[iter] = timeRecord[iter - 1] + time;

	}

	*p_iter = iter;

	cudaFree(d_scores);
	cudaFree(d_modes);
	cudaFree(d_res);
	cudaFree(d_aux_rp);

}




/*
**********************************************************
**			ER naive			**
** 		Expander Recovery naive			**
**********************************************************
*/

inline void er_naive(float *d_vec, float *d_y, float *resid,
	int *d_rows, int *d_cols, float *d_vals, float tol, const int maxiter, 
	const int k, const int m, const int n, const int p, const int nz, 
	float *resRecord, float *timeRecord, int *p_iter, dim3 numBlocks, dim3 threadsPerBlock,
	dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm){

	timeRecord[0] = 0;

	int iter = *p_iter;

	int h_node;
	
	cublasInit();

	float * d_Ax;
	float * d_scores;
	float * d_modes;
	float * d_res;
	float * d_aux_nz;
	
	cudaMalloc((void**)&d_Ax, m * sizeof(float));
	cudaMalloc((void**)&d_scores, n * sizeof(float));
	cudaMalloc((void**)&d_modes, n * sizeof(float));
	cudaMalloc((void**)&d_res, m * sizeof(float));
	cudaMalloc((void**)&d_aux_nz, nz * sizeof(float));

	float err;

	computeResidual(d_res, d_y, d_Ax, d_vec, d_rows, d_cols, d_vals, nz, m, n,
		 numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
	
	err = cublasSnrm2(m, d_res, 1);

	resRecord[0] = err;

	float err_start = err;
	float resid_tol = tol;
	
	/*	
	float h_score;
	float h_mode;
	*/
	

	while( (iter < maxiter) & (err > resid_tol) & (err < (100*err_start)) ){
		
		// time variables
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
	
		cuda_compute_scores_er<<< numBlocks, threadsPerBlock >>>(d_scores, d_modes, d_res, d_rows, d_vals, d_aux_nz, n, p);

		h_node = cublasIsamax(n, d_scores, 1) - 1;

		/*
		cudaMemcpy(&h_score, d_scores+h_node, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_mode, d_modes+h_node, sizeof(float), cudaMemcpyDeviceToHost);
		printf("iter = %d\terr = %g\th_node = %d\th_score = %g\th_mode = %g\n", iter, err, h_node, h_score, h_mode);
		*/

		cudaUpdateSignal<<<numBlocks, threadsPerBlock>>>(d_vec, h_node, d_modes);
	
		cudaUpdateResidual<<<numBlocks, threadsPerBlock>>>(d_res, d_rows, d_vals, d_modes, h_node, p);

		err = cublasSnrm2(m, d_res, 1);

		// j = j + 1
		iter = iter + 1;
	
		// end timing	
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		resRecord[iter] = err;
		timeRecord[iter] = timeRecord[iter - 1] + time;

	}

	*p_iter = iter;

	cudaFree(d_Ax);
	cudaFree(d_scores);
	cudaFree(d_modes);
	cudaFree(d_res);
	cudaFree(d_aux_nz);

}

