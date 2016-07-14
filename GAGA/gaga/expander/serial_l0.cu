
float compute_sum_of_squares(float *h_vec, int n){
	float ss = 0.0;
	for (int i = 0; i < n; i++)
		ss = ss + h_vec[i]*h_vec[i];
	return ss;
}

/*
**************************************************
**		serial-l0			**
** 	 Expander Recovery of the Future	**
**************************************************
*/

inline void serial_l0(float *h_vec, float *h_y, float *h_res, int *h_rows, int *h_cols, const int n, const int m, const int k, const int d, const int alpha, float tol, const int maxiter, int *p_iter, float *resRecord, float *timeRecord){

	int iter = *p_iter;
	timeRecord[0] = 0.0;
	
	float sum_of_squares = compute_sum_of_squares(h_res, m);
	float norm_res = sqrt(sum_of_squares);
	float norm_res_start = norm_res;
	resRecord[0] = norm_res;

	int isCycling = 0;

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

	float EPS = 0.000001;
	int idx = 0;

	int score = 0;
	int shift = 0;
	int nIters = 0;

	float omega = 0;
	
	float time;

	//printf("iter = %d, maxiter = %d, norm_res = %g, norm_res_start = %g, tol = %g, resNorm_diff = %g, alpha = %d\n\n", iter, maxiter, norm_res, norm_res_start, tol, residNorm_diff, alpha);

	while( (iter < maxiter) && (norm_res > tol) && (norm_res < (100*norm_res_start)) && (isCycling == 0) && (residNorm_diff > 0.000001) ){

		// time variables
		cudaEvent_t start, stop;
		time = 0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		shift = iter % d;
		for (int node = 0; node < n; node++){
			
			idx = d*node;
			omega = h_res[h_rows[idx + shift]];
			score = 0;
			if (fabs(omega) > EPS){
				for (int i = 0; i < d; i++){
					if (fabs(h_res[h_rows[idx + i]] - omega) <= EPS){
						score++;
					}
					if (fabs(h_res[h_rows[idx + i]]) <= EPS){
						score--;
					}
				}

				if (score > alpha){
					h_vec[node] = h_vec[node] + omega; 
					// update residual
					for (int i = 0; i < d; i++){
						//sum_of_squares = sum_of_squares - pow(h_res[h_rows[idx + i]], 2);
						//sum_of_squares -= pow(h_res[h_rows[idx + i]], 2);
						h_res[h_rows[idx + i]] = h_res[h_rows[idx + i]] - omega;
						//sum_of_squares = sum_of_squares + pow(h_res[h_rows[idx + i]], 2);
						/*
						if (fabs(h_res[h_rows[idx + i]]) <= EPS){
							h_res[h_rows[idx + i]] = 0;
						}else{
							sum_of_squares = sum_of_squares + pow(h_res[h_rows[idx + i]], 2);
						}
						*/
						//sum_of_squares += pow(h_res[h_rows[idx + i]], 2);
						// if roundoff errors make this negative
						/*
						if (sum_of_squares < 0){
							sum_of_squares = compute_sum_of_squares(h_res, m);
						}
						*/
					}
					//norm_res = sqrt(sum_of_squares);
					//norm_res = sqrt(fabs(sum_of_squares));
				}
			}
		}

		// END STEP
		sum_of_squares = compute_sum_of_squares(h_res, m);
		norm_res = sqrt(sum_of_squares);

		// check for no change in residual 
		for (int i = 0; i < residNorm_length - 1; i++){
			residNorm_prev[i] = residNorm_prev[i + 1];
		}
		residNorm_prev[residNorm_length - 1] = norm_res;
		for (int i = 0; i < residNorm_length - 1; i++){
			residNorm_evolution[i] = residNorm_evolution[i + 1];
		}
		residNorm_evolution[residNorm_length - 1] = residNorm_prev[residNorm_length - 2] - residNorm_prev[residNorm_length - 1];
		residNorm_diff = residNorm_evolution[0];
		for (int i = 0; i < residNorm_length; i++)
			residNorm_diff = residNorm_diff > residNorm_evolution[i] ? residNorm_diff : residNorm_evolution[i];

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
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		resRecord[nIters] = norm_res;
		timeRecord[nIters] = timeRecord[nIters - 1] + time;
		nIters++;
		//printf("iter = %d\tshift = %d\tnorm_Res = %g\tnorm_diff = %g\n", iter, shift, norm_res, residNorm_diff);

		iter = iter + 1;
			
	}

	*p_iter = iter;

	free(residNorm_prev);
	free(residNorm_evolution);

}

