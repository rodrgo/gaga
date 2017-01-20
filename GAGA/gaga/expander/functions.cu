int transform_to_row_major_order(const int m, const int n, const int p, int *d_rows, int *d_cols, int *d_rm_rows_index, int *d_rm_cols, int *h_max_nonzero_rows_count){

	cudaDeviceProp dp;
	cudaGetDeviceProperties(&dp,0);
	unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

	int threads_perblocknp = min(n*p, max_threads_per_block);
	dim3 threadsPerBlocknp(threads_perblocknp);
	int num_blocksnp = (int)ceil((float)(n*p)/(float)threads_perblocknp);
	dim3 numBlocksnp(num_blocksnp);

	int threads_perblockm = min(m, max_threads_per_block);
	dim3 threadsPerBlockm(threads_perblockm);
	int num_blocksm = (int)ceil((float)m/(float)threads_perblockm);
	dim3 numBlocksm(num_blocksm);

	int *d_rm_rows;
	int np = ((int) n)*((int) p);
	cudaMalloc((void**)&d_rm_rows, np * sizeof(int));
	cudaMemcpy(d_rm_rows, d_rows, np * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_rm_cols, d_cols, np * sizeof(int), cudaMemcpyDeviceToDevice);

	thrust::device_ptr<int> dev_rm_rows(d_rm_rows);
	thrust::device_ptr<int> dev_rm_cols(d_rm_cols);
	thrust::sort_by_key(dev_rm_rows, dev_rm_rows + np, dev_rm_cols);

	create_row_index<<<numBlocksnp, threadsPerBlocknp>>>(d_rm_rows, d_rm_rows_index, m, np);
	cudaFree(d_rm_rows);

	// Finally, we need to define the size of the kernel blocks for updating.
	// This will be given by the row with the maximum number of elements.

	// For each row, count how many nonzeros there are.

	float * d_nonzero_rows_count;
	cudaMalloc((void**)&d_nonzero_rows_count, m * sizeof(float));
	zero_vector_float<<<numBlocksm, threadsPerBlockm>>>(d_nonzero_rows_count, m);

	count_nonzeros_in_rows_index<<<numBlocksm, threadsPerBlockm>>>(d_nonzero_rows_count, d_rm_rows_index, m);

	if (1 == 0){
		float *h_nonzero_rows_count;
		h_nonzero_rows_count = (float*) malloc(m*sizeof(float));
		cudaMemcpy(h_nonzero_rows_count, d_nonzero_rows_count, m*sizeof(float), cudaMemcpyDeviceToHost); 
		int *h_rm_rows_index;
		h_rm_rows_index = (int*) malloc(2*m*sizeof(int));
		cudaMemcpy(h_rm_rows_index, d_rm_rows_index, 2*m*sizeof(int), cudaMemcpyDeviceToHost);
		for (int i = 0; i < m; i++){
			if (h_nonzero_rows_count[i] <=1 || h_nonzero_rows_count[i] > 30){
				printf("h_rm_rows_index[%d] = %d\n", 2*i, h_rm_rows_index[2*i]);
				printf("h_rm_rows_index[%d] = %d\n", 2*i+1, h_rm_rows_index[2*i+1]);
				printf("h_nonzero_rows_count[%d] = %5.5f\n", i, h_nonzero_rows_count[i]);
			}
		}
		free(h_nonzero_rows_count);
		free(h_rm_rows_index);
	}

	int max_index;
	max_index = cublasIsamax(m, d_nonzero_rows_count, 1) - 1;
	float h_maximum = 0.0;
	cudaMemcpy(&h_maximum, d_nonzero_rows_count+max_index, sizeof(float), cudaMemcpyDeviceToHost); 
	//printf("h_maximum = %5.2f\n", h_maximum);
	h_max_nonzero_rows_count[0] = (int)(h_maximum);
	//printf("h_max_nonzero_rows_count = %d\n", h_max_nonzero_rows_count[0]);

	cudaFree(d_nonzero_rows_count);
	cublasShutdown();

	return 0;
}

inline void computeResidual(float *d_res, float *d_y, float *d_Ax, float *d_vec, int *d_rows, int *d_cols, float *d_vals, int nz, int m, int n, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocksnp, dim3 threadsPerBlocknp){
        cublasScopy(m, d_y, 1, d_res, 1);
        A_smv(d_Ax, d_vec, m, n, d_rows, d_cols, d_vals, nz,
                numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
        cublasSaxpy(m, -1, d_Ax, 1, d_res, 1);
        return;
}

int supportSizeGPU(float * d_vec, int n, dim3 numBlocks, dim3 threadsPerBlock){
        int *c;                      // host copy of c
        int *dev_c;          // device copy of c

        // allocate device copy of c
        cudaMalloc( (void**)&dev_c, sizeof( int ) );
        c = (int *)malloc( sizeof( int ) );

        *c = 0;

        cudaMemcpy( dev_c, c, sizeof( int ), cudaMemcpyHostToDevice );

        countSupport<<< numBlocks, threadsPerBlock >>>( d_vec, n, dev_c );
	cudaThreadSynchronize();

        // copy device result back to host copy of c
        cudaMemcpy( c, dev_c, sizeof( int ) , cudaMemcpyDeviceToHost );
        cudaFree( dev_c );
        return *c;
}


inline void ThresholdK(float *d_vec, int *d_bin, const int k_bin, const int K, const int n, dim3 numBlocks, dim3 threadsPerBlock)
{
        int *supp_size;
        supp_size = (int *)malloc( sizeof(int) );

        threshold <<< numBlocks, threadsPerBlock >>> ((float*)d_vec, (int*)d_bin, k_bin, n);
        cudaDeviceSynchronize();
        *supp_size = supportSizeGPU(d_vec, n, numBlocks, threadsPerBlock);

        int h_offset = *supp_size - K;
        if(h_offset > 0){
                int *d_offset;
                cudaMalloc( (void**)&d_offset, sizeof(int) );
                cudaMemcpy(d_offset, &h_offset, sizeof(int), cudaMemcpyHostToDevice);

                thresholdK <<< numBlocks, threadsPerBlock >>> ((float*)d_vec, (int*)d_bin, k_bin, (int*) d_offset, n);
                cudaDeviceSynchronize();
                cudaFree(d_offset);
        }
        return;
}

inline void H_k(float *d_vec, int K, int n, int *d_bin, int *d_bin_counters, int *h_bin_counters, float *p_maxChange, float *p_max_value, float *p_slope, float *p_minVal, float *p_alpha, int *p_MaxBin, int *p_k_bin, int *p_sum, int num_bins, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin){

        *p_maxChange = 2 * MaxMagnitude(d_vec, n);
        *p_max_value = MaxMagnitude(d_vec, n);
        *p_slope = ((num_bins - 1)/(*p_max_value));
        *p_minVal = 0.0f;
        *p_alpha = 0.25f;
        *p_MaxBin = num_bins; // (int)(num_bins * (1 - alpha));
        *p_k_bin = 0;

        *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters,
                *p_slope, *p_max_value, *p_maxChange, p_minVal, p_alpha, p_MaxBin, p_k_bin,
                n, K, num_bins, numBlocks, threadsPerBlock, numBlocks_bin,
                threadsPerBlock_bin);

        ThresholdK(d_vec, d_bin, *p_k_bin, K, n, numBlocks, threadsPerBlock);

	//*p_sum = supportSizeGPU(d_vec, n, numBlocks, threadsPerBlock);

}

