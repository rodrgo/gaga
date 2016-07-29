

__global__ void cuda_update_residual(float *d_res, int *d_rows, float *d_updates, int n, int p){
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (xIndex < n){
		if (abs(d_updates[xIndex]) > 0){
                	int idx = xIndex*p;
			float update = -d_updates[xIndex]; 
			for (int i = 0; i < p; i++){
				atomicAdd(d_res + d_rows[idx + i], update);
			}
		} 
	}
}

// Not tested yet
/*
__global__ void cuda_update_residual_np(float *d_res, int *d_rows, float *d_updates, int n, int p){
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n*p){
                int idx = (int) (tid/p);
		float update = -d_updates[idx]; 
		if (abs(d_updates[xIndex]) > 0){
			atomicAdd(d_res + d_rows[idx + i], update);
		}
	}
}
*/

/*
*************************************************
* 	For tranforming to row-major order	*
*************************************************
*/

// nonzero_rows_count[i] = number of nonzeros in row i.
__global__ void count_nonzeros_in_rows_index(float *nonzero_rows_count, int *rm_rows_index, int m){
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if ( xIndex < m){
		nonzero_rows_count[xIndex] = (float)(rm_rows_index[2*xIndex + 1] - rm_rows_index[2*xIndex] + 1);
	}
}

	// Create vector 2-m vector [a_1, a_2, b_1, b_2, ..., x_1, x_2] indicating start and end position of columns in row i of matrix
	// i.e. if row_i has only one nonzero, then a_i == b_i
	__global__ void create_row_index(int *rm_rows, int *rm_rows_index, int m, int np){
		unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
		if ( xIndex < np){
			if (xIndex > 0 && xIndex < np - 1){
				if (rm_rows[xIndex - 1] < rm_rows[xIndex]){
					rm_rows_index[2*rm_rows[xIndex]] = xIndex;
				}
				if(rm_rows[xIndex] < rm_rows[xIndex + 1]){
					rm_rows_index[2*rm_rows[xIndex] + 1] = xIndex;
				}
			}
			if (xIndex == 0){
				rm_rows_index[2*rm_rows[xIndex]] = xIndex;
				if(rm_rows[xIndex] < rm_rows[xIndex + 1])
					rm_rows_index[2*rm_rows[xIndex] + 1] = xIndex;
			}
			if (xIndex == np - 1){
				if (rm_rows[xIndex - 1] < rm_rows[xIndex])
					rm_rows_index[2*rm_rows[xIndex]] = xIndex;
				rm_rows_index[2*rm_rows[xIndex] + 1] = xIndex;
			}
		}
	}


	/*
	*************************************************
	*						*
	* 			SMP			*
	*						*
	*************************************************
	*/

	__global__ void cuda_compute_scores_smp(float *d_u, float *d_b, int *d_rows, float *d_aux, int n, int p)
	{
		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		if(tid < n){
			int idx = p*tid;
			float tmp = 0.0;
			float median = 0.0;

			for(int i = 0; i < p; i++)
				d_aux[idx + i] = (float) d_b[d_rows[idx + i]];
		
			// sort chunk in d_aux with bubblesort. O(p^2) work for a thread, but O(1) memory 
			// We exploit computational power

			for(int i = 0; i < p; i++)
				for(int j = 0; j < p - 1; j++)
					if(d_aux[idx + j] > d_aux[idx + j + 1]){
						tmp = d_aux[idx + j + 1];
						d_aux[idx + j + 1] = d_aux[idx + j];
						d_aux[idx + j] = tmp;
					}	

			median = (p % 2) == 0 ? (d_aux[idx + (p/2) - 1] + d_aux[idx + (p/2)])/2 : d_aux[idx + (p - 1)/2];	
			d_u[tid] = median;
		}
	}


	/*
	*************************************************
	*						*
	* 			SSMP			*
	*						*
	*************************************************
	*/

	__global__ void cuda_compute_scores_ssmp(float *d_scores, float *d_medians, float *d_res, int *d_rows, float *d_vals, float *d_aux, int n, int p)
	{
		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		if(tid < n){

			int idx = p*tid;
			float tmp = 0.0;
			float z = 0.0;

			for(int i = 0; i < p; i++)
				d_aux[idx + i] = d_res[d_rows[idx + i]];

			// sort chunk in d_aux with bubblesort. O(p^2) work for a thread, but O(1) memory 
			// We exploit computational power

			for(int i = 0; i < p; i++)
				for(int j = 0; j < p - 1; j++)
					if(d_aux[idx + j] > d_aux[idx + j + 1]){
						tmp = d_aux[idx + j + 1];
						d_aux[idx + j + 1] = d_aux[idx + j];
						d_aux[idx + j] = tmp;
					}

			// z = median
			z = (p % 2) == 0 ? (d_aux[idx + (p/2) - 1] + d_aux[idx + (p/2)])/2 : d_aux[idx + (p - 1)/2];

			d_medians[tid] = z;
			d_scores[tid] = abs(z);

		}
	}

	__global__ void cuda_update_scores_ssmp(float *d_scores, float *d_medians, float *d_res, int *d_rows, float *d_aux, int *d_rm_cols, int *d_rm_rows_index, int max_nonzero_rows_count, int p, int h_node)
	{
		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		if(tid < max_nonzero_rows_count*p){
			int section = (int)(tid/max_nonzero_rows_count); // from 0 to p-1
			int row = d_rows[p*h_node + section];
			int num_nonzeros_in_row = d_rm_rows_index[2*row + 1] - d_rm_rows_index[2*row] + 1; // from 1 to max_nonzero_rows_count
			int max_position = num_nonzeros_in_row - 1; // from 0 to (max_nonzero_rows_count - 1)
			int position_in_section = tid - section*max_nonzero_rows_count; // from 0 to (max_nonzero_rows_count - 1) 
			if (position_in_section <= max_position){

				int node = d_rm_cols[d_rm_rows_index[2*row] + position_in_section];
				int row_idx = p*node;
				int idx = p*tid;
				float tmp = 0.0;
				float z = 0.0;

				for(int i = 0; i < p; i++)
					d_aux[idx + i] = d_res[d_rows[row_idx + i]];

				// sort chunk in d_aux with bubblesort. O(p^2) work for a thread, but O(1) memory 

				for(int i = 0; i < p; i++)
					for(int j = 0; j < p - 1; j++)
						if(d_aux[idx + j] > d_aux[idx + j + 1]){
							tmp = d_aux[idx + j + 1];
							d_aux[idx + j + 1] = d_aux[idx + j];
							d_aux[idx + j] = tmp;
						}

				// z = median
				z = (p % 2) == 0 ? (d_aux[idx + (p/2) - 1] + d_aux[idx + (p/2)])/2 : d_aux[idx + (p - 1)/2];

				d_medians[node] = z;
				d_scores[node] = abs(z);
			}
		}
	}


	/*
	*************************************************
	*						*
	* 			ER			*
	*						*
	*************************************************
	*/

	__global__ void cuda_compute_scores_er(float *d_scores, float *d_modes, float *d_res, int *d_rows, float *d_vals, float *d_aux, int n, int p)
	{
		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		if(tid < n){

			int idx = p*tid;
			float tmp = 0.0;
			float EPS = 0.0000001;

			for(int i = 0; i < p; i++)
				d_aux[idx + i] = d_res[d_rows[idx + i]];

			// sort chunk in d_aux with bubblesort. O(p^2) work for a thread, but O(1) memory 
			// We exploit computational power

			for(int i = 0; i < p; i++)
				for(int j = 0; j < p - 1; j++)
					if(d_aux[idx + j] > d_aux[idx + j + 1]){
						tmp = d_aux[idx + j + 1];
						d_aux[idx + j + 1] = d_aux[idx + j];
						d_aux[idx + j] = tmp;
					}

			// Compute modes

			float mode = d_aux[idx];
			float count = 1;
			float countMode = 1;

			for (int i = 1; i < p; i++){
				if (abs(d_aux[idx + i] - d_aux[idx + i - 1]) <= EPS){
					count = count + 1;
					if (count > countMode){
						countMode = count;
						mode = d_aux[idx + i - 1];
					}
				}else{
					count = 1;
				}
			}

			d_modes[tid] = mode;
			d_scores[tid] = abs(mode) > EPS ? countMode : 0;

		}
	}

	/*
	0 <= tid < max_nonzero_rows_count*p
	[section_0 | section_1 | ... | section_{p-1}]

	*/
	__global__ void cuda_update_scores_er(float *d_scores, float *d_modes, float *d_res, int *d_rows, float *d_aux, int *d_rm_cols, int *d_rm_rows_index, int max_nonzero_rows_count, int p, int h_node)
	{
		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		if(tid < max_nonzero_rows_count*p){
			int section = (int)(tid/max_nonzero_rows_count); // from 0 to p-1
			int row = d_rows[p*h_node + section];
			int num_nonzeros_in_row = d_rm_rows_index[2*row + 1] - d_rm_rows_index[2*row] + 1; // from 1 to max_nonzero_rows_count
			int max_position = num_nonzeros_in_row - 1; // from 0 to (max_nonzero_rows_count - 1)
			int position_in_section = tid - section*max_nonzero_rows_count; // from 0 to (max_nonzero_rows_count - 1) 
			if (position_in_section <= max_position){

				int node = d_rm_cols[d_rm_rows_index[2*row] + position_in_section];
				int row_idx = p*node;
				int idx = p*tid;
				float tmp = 0.0;
				float EPS = 0.0000001;

				for(int i = 0; i < p; i++)
					d_aux[idx + i] = d_res[d_rows[row_idx + i]];

				// sort chunk in d_aux with bubblesort. O(p^2) work for a thread, but O(1) memory 

				for(int i = 0; i < p; i++)
					for(int j = 0; j < p - 1; j++)
						if(d_aux[idx + j] > d_aux[idx + j + 1]){
							tmp = d_aux[idx + j + 1];
							d_aux[idx + j + 1] = d_aux[idx + j];
							d_aux[idx + j] = tmp;
						}

				// Compute modes
				float mode = d_aux[idx];
				float count = 1;
				float countMode = 1;

				for (int i = 1; i < p; i++){
					if (abs(d_aux[idx + i] - d_aux[idx + i - 1]) <= EPS){
						count = count + 1;
						if (count > countMode){
							countMode = count;
							mode = d_aux[idx + i - 1];
						}
					}else{
						count = 1;
					}
				}

				d_modes[node] = mode;
				d_scores[node] = abs(mode) > EPS ? countMode : 0;
			}
		}
	}


	/*
	*************************************************
	*						*
	* 		Parallel-LDDSR			*
	*						*
	*************************************************
	*/

	__global__ void cuda_compute_scores_lddsr(float *d_scores, float *d_res, float *d_vec, float *d_updates, int *d_rows, int *d_foundCandidates, int n, int d, int shift)
	{
		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		
		if (tid < n){
			float omega = 0.0;
			int idx = d*tid;
			float EPS = 0.000001;
			float thresh = ((float) d)/2;

			d_scores[tid] = 0;
			omega = d_res[d_rows[idx + shift]];
			d_updates[tid] = 0;

			if (abs(omega) > EPS){
				for (int i = 0; i < d; i++){
					if (abs(d_res[d_rows[idx + i]] - omega) <= EPS)
						d_scores[tid]++;
				}
				if (d_scores[tid] > thresh){
					d_foundCandidates[0] = 1;
					d_vec[tid] = d_vec[tid] + omega; 
					/*
					// update residual
					for (int i = 0; i < d; i++){
						atomicAdd(d_res + d_rows[idx + i], -omega);
					}
					*/
					d_updates[tid] = omega;
				}
				//d_updates[tid] = d_vec[tid] + omega; // Use this when using h_found_candidates solution
			}

		}
	}

	/*
	*************************************************
	*						*
	* 		Parallel-L0			*
	*						*
	*************************************************
	*/

	__global__ void cuda_compute_scores_l0(float *d_scores, float *d_res, float *d_vec, float *d_updates, int *d_rows, int *d_foundCandidates, int n, int d, int thresh, int shift)
	{
		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		
		if (tid < n){
			float omega = 0.0;
			int idx = d*tid;
			float EPS = 0.000001;
			//float thresh = 1;

			d_scores[tid] = 0;
			omega = d_res[d_rows[idx + shift]];
			d_updates[tid] = 0;

			if (abs(omega) > EPS){
				for (int i = 0; i < d; i++){
					if (abs(d_res[d_rows[idx + i]] - omega) <= EPS)
						d_scores[tid]++;
					if (abs(d_res[d_rows[idx + i]]) <= EPS)
						d_scores[tid]--;
				}
				if (d_scores[tid] > thresh){
					d_foundCandidates[0] = 1;
					d_vec[tid] = d_vec[tid] + omega; 
					/*
					// update residual
					for (int i = 0; i < d; i++){
						atomicAdd(d_res + d_rows[idx + i], -omega);
					}
					*/
					d_updates[tid] = omega;
					//d_updates[tid] = d_vec[tid] + omega;
				}
			}
		}
	}


	/*
	*************************************************
	*						*
	* 		Robust-L0			*
	*						*
	*************************************************
	*/

	__constant__ float sigma_noise; 
	__constant__ float prob_zero_factor;
	__constant__ float prob_equal_factor;
	__constant__ float sigma_signal_zero;
	__constant__ float sigma_signal_equal;

	__device__ int bernoulli(float prob, float unif_rv){
		float value = 0.0;
		if (unif_rv <= prob){
			value = 1.0;
		}
		return value;
	}

	__device__ float norm_pdf_ratio(float x, float sigma_1, float sigma_2){
		return (sigma_2/sigma_1)*expf(x*x/(2*sigma_2*sigma_2) - x*x/(2*sigma_1*sigma_1));
	}

	__device__ float probability_zero(float omega){
		return 1/(1 + prob_zero_factor*norm_pdf_ratio(omega, sigma_signal_zero, sigma_noise));
	}

	__device__ float probability_equal(float omega){
		return 1/(1 + prob_equal_factor*norm_pdf_ratio(omega, sigma_signal_equal, sqrtf(2)*sigma_noise));
	}

	__global__ void cuda_sample_prob_zero_u(int *d_bernoulli_pz_u, float *d_unif, int d, int n, float *d_res, int * d_rows, int shift){
		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		if (tid < n){
			int unif_idx = (2*d + 1)*tid;
			float unif = d_unif[unif_idx]; 
			float omega = d_res[d_rows[d*tid + shift]];
			float prob_zero_u = probability_zero(omega);
			d_bernoulli_pz_u[tid] = bernoulli(prob_zero_u, unif);
		}
	}

	__global__ void cuda_compute_scores(int *d_scores, int* d_bernoulli_pz_u, float *d_average_updates, float *d_unif, int d, int n, float *d_res, int *d_rows, int shift){
		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		if (tid < n){
			if (d_bernoulli_pz_u[tid] == 0){ 
				int unif_idx = (2*d + 1)*tid + 1;
				float omega = d_res[d_rows[d*tid + shift]];
				float v;
				float unif;
				float prob_equal;
				float prob_zero;
				int is_equal = 0;
				int is_zero = 0;

				float num_equal = 0.0;
				float sum_updates = 0.0;
				int score = 0;
				for (int i = 0; i < d; i++){
					// get value
					v = d_res[d_rows[d*tid + i]];

					// compute Prob(v = omega)
					prob_equal = probability_equal(v - omega);
					unif = d_unif[unif_idx++]; 
					is_equal = bernoulli(prob_equal, unif);

					// compute Prob(v = 0)
					prob_zero = probability_zero(v);
					unif = d_unif[unif_idx++]; 
					is_zero = bernoulli(prob_zero, unif);

					if (is_equal == 1){
						score += 1;
						sum_updates += v;
						num_equal += 1;
					}
					if (is_zero == 1){
						score -= 1;
					}
				}
				d_scores[tid] = score;
				if (num_equal > 0){
					d_average_updates[tid] = sum_updates/num_equal;
				}else{
					d_average_updates[tid] = 0;
				}
			}else{
				d_average_updates[tid] = 0;
			}

		}
	}

	__global__ void cuda_update_signal_robust_l0(float *d_res, int *d_bernoulli_pz_u, float *d_vec, float *d_updates, float *d_average_updates, int *d_rows, int n, int d, float alpha, int *d_scores, int enforce_l1_decrease){

		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		if (tid < n){
			if (d_bernoulli_pz_u[tid] == 0){
				int score = d_scores[tid];
				if (((float) score) >= alpha){

					int idx = d*tid;
					//float omega = 0.0;
					float update = 0.0;

					update = d_average_updates[tid];
					//omega = d_res[d_rows[idx + shift]];

					int do_update = 1;

					if (enforce_l1_decrease == 1){
						float new_energy = 0.0;
						float old_energy = 0.0;
						for (int i = 0; i < d; i++){
							old_energy += abs(d_res[d_rows[idx + i]]);
							new_energy += abs(d_res[d_rows[idx + i]] - update);
						}
						if (new_energy > old_energy){
							do_update = 0;
						}
					}

					
					if (do_update == 1){
						d_vec[tid] = d_vec[tid] + update;
						d_updates[tid] = update;
					}
				}
			}
		}
	}

	/*
	*************************************************
	*						*
	* 	Deterministic Robust-L0			*
	*						*
	*************************************************
	*/

	__global__ void cuda_deterministic_prob_zero_u(float *d_pz_u, int d, int n, float *d_res, int * d_rows, int shift){
		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		if (tid < n){
			float omega = d_res[d_rows[d*tid + shift]];
			float prob_zero_u = probability_zero(omega);
			d_pz_u[tid] = prob_zero_u;
		}
	}

	__global__ void cuda_compute_scores_det_robust_l0(int *d_scores, float* d_pz_u, float *d_average_updates, int d, int n, float *d_res, int *d_rows, int shift, float prob_thresh){
		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		if (tid < n){
			if (d_pz_u[tid] <= 1 - prob_thresh){ 
				float omega = d_res[d_rows[d*tid + shift]];
				float v;

				float num_equal = 0.0;
				float sum_updates = 0.0;
				int score = 0;
				for (int i = 0; i < d; i++){
					// get value
					v = d_res[d_rows[d*tid + i]];

					// compute Prob(v = omega)
					if (probability_equal(v - omega) >= prob_thresh){
						score += 1;
						sum_updates += v;
						num_equal += 1;
					}

					// compute Prob(v = 0)
					if (probability_zero(v) >= 1 - prob_thresh){
						score -= 1;
					}

				}
				d_scores[tid] = score;
				if (num_equal > 0){
					d_average_updates[tid] = sum_updates/num_equal;
				}else{
					d_average_updates[tid] = 0;
				}
			}else{
				d_average_updates[tid] = 0;
			}

		}
	}

	__global__ void cuda_update_signal_det_robust_l0(float *d_res, float *d_vec, float *d_updates, float *d_average_updates, int *d_rows, int n, int d, float alpha, int *d_scores, int enforce_l1_decrease){

		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		if (tid < n){
			int score = d_scores[tid];
			if (((float) score) >= alpha){

				int idx = d*tid;
				//float omega = 0.0;
				float update = 0.0;

				update = d_average_updates[tid];
				//omega = d_res[d_rows[idx + shift]];

				int do_update = 1;

				if (enforce_l1_decrease == 1){
					float new_energy = 0.0;
					float old_energy = 0.0;
					for (int i = 0; i < d; i++){
						old_energy += abs(d_res[d_rows[idx + i]]);
						new_energy += abs(d_res[d_rows[idx + i]] - update);
					}
					if (new_energy > old_energy){
						do_update = 0;
					}
				}

				
				if (do_update == 1){
					d_vec[tid] = d_vec[tid] + update;
					d_updates[tid] = update;
				}
			}
		}
	}

/*
*************************************************
*						*
* For exact thresholding and residual update	*
*						*
*************************************************
*/

__global__ void thresholdK(float *d_vec, int *d_bin, int k_bin, int * offset, int n){
        unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

        if( (xIndex < n) & (d_bin[xIndex] == k_bin) ){
                if(atomicSub(offset, 1) - 1 >= 0){
                        d_vec[xIndex] = 0;
                }
        }
}

__global__ void cudaUpdateSignal(float *d_vec, int h_node, float *d_updates){

        int tid = threadIdx.x + blockDim.x*blockIdx.x;

        if(tid == h_node){
                d_vec[tid] += d_updates[h_node];
        }
}

__global__ void cudaUpdateResidual(float *d_res, int *d_rows, float *d_vals, float *d_medians, int h_node, int p){

        int tid = threadIdx.x + blockDim.x*blockIdx.x;

        if(tid == h_node){
                int idx = tid*p;
                for(int i = 0; i < p; i++)
                        d_res[d_rows[idx + i]] -= d_medians[tid]*(d_vals[idx + i]);
        }
}

__global__ void countSupport( float *d_vec, int n, int *dev_c ) { 
 
        int tid = threadIdx.x + blockIdx.x * blockDim.x; 
        if( tid < n ) { 
                if(d_vec[tid] != 0){ 
                        atomicAdd( dev_c , 1 ); 
                } 
        } 
} 

