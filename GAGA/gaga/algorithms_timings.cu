/* Copyright 2010-2013 Jeffrey D. Blanchard and Jared Tanner
 *   
 * GPU Accelerated Greedy Algorithms for Compressed Sensing
 *
 * Licensed under the GAGA License available at gaga4cs.org and included as GAGA_license.txt.
 *
 * In  order to use the GAGA library, or any of its constituent parts, a user must
 * agree to abide by a set of * conditions of use. The library is available at no cost 
 * for ``Internal'' use. ``Internal'' use of the library * is defined to be use of the 
 * library by a person or institution for academic, educational, or research purposes 
 * under the conditions in the included GAGA_license.txt. Any use of the library implies 
 * that these conditions have been understood, and that the user agrees to abide by all 
 * the listed conditions.
 *     
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Any redistribution or derivatives of this software must contain this header in all files
 * and include a copy of GAGA_license.txt.
 */



/*
*****************************************************************************
**                      NIHT_S_timings_dct                                 **
**      Single Precision Normalized Iterative Hard Thresholding            **
**                    with the DCT                                         **
*****************************************************************************
*/

inline void NIHT_S_timings_dct(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float alpha, int supp_flag, float *time_per_iteration, float *time_supp_set, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{

// alpha is a parameter used to avoid counting unused bins, 
// it should take on values in [0,1) with 0 using the full 
// number of bins from the start and and as 1 is approached 
// the initial number of bins becomes smaller.

// supp_flag = 0 is for adaptively selecting the support size
//           = 1 is for using full binning at each step
//	     = 2 is for using thrust::sort to find the kth largest magnitude with dynamic support detection
//           = 3 is for using thrust::sort at each step


/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


//  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 0.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in IHT initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT initialization");

  float err_start = err;

  float resid_tol = tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }

// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// parameters for FindSupportSet_sort
  thrust::device_ptr<float> d_sort(d_vec_thres);
  float T=0.0f;


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_dct(d_vec, d_y, n, m, d_rows);
  SAFEcuda("AT in IHT loop");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in IHT loop");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in IHT loop");




/* 
*******************
** Main Steepest Descent Loop **
*******************
*/

  while ( (err > resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*tol) & (fail == 0) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  maxChange = RestrictedSD_S_dct(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);

  cudaEvent_t start_supp, stop_supp;
  float time_supp;
  cudaEventCreate(&start_supp);
  cudaEventCreate(&stop_supp);
  cudaEventRecord(start_supp,0);

  if (supp_flag < 2){
	if (minVal <= maxChange) { 
		max_value = MaxMagnitude(d_vec,n);
		slope = ((num_bins-1)/(max_value));
	}
  }

  if (supp_flag == 0){
    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in IHT loop");
    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in IHT loop");
  }
  else if (supp_flag == 1){
    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in IHT loop");
    // note that the second max_value forces binning at each step
    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in IHT loop");
  }
  else if (supp_flag == 2){    
    k_bin = 1;
    T = FindSupportSet_sort(d_vec, d_vec_thres, d_sort, d_bin, T, maxChange, n, k, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet_sort in IHT loop");
    // note that this function includes the threshold
  }
  else if (supp_flag == 3){    
    k_bin = 1;
    T = FindSupportSet_sort(d_vec, d_vec_thres, d_sort, d_bin, T, T, n, k, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet_sort in IHT loop");
    // note that this function includes the threshold
    // note that max_value forces a sort at each iteration
  }


  cudaThreadSynchronize();
  cudaEventRecord(stop_supp,0);
  cudaEventSynchronize(stop_supp);
  cudaEventElapsedTime(&time_supp, start_supp, stop_supp);





  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

  if (iter>749){
	root = 1/16.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1;
  }

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(start_supp);
  cudaEventDestroy(stop_supp);

  time_per_iteration[iter] = time;
  time_supp_set[iter] = time_supp;

  iter++;

  time_sum = time_sum + time;

  }



  *p_iter = iter;
  *p_time_sum = time_sum;

}






/*
*****************************************************************************
**                      NIHT_S_timings_smv                                 **
**      Single Precision Normalized Iterative Hard Thresholding            **
**                    with the DCT                                         **
*****************************************************************************
*/

inline void NIHT_S_timings_smv(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_cols, float *d_vals, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int nz, int *p_iter, float mu, float err, int *p_sum, float alpha, int supp_flag, float *time_per_iteration, float *time_supp_set, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{

// alpha is a parameter used to avoid counting unused bins, 
// it should take on values in [0,1) with 0 using the full 
// number of bins from the start and and as 1 is approached 
// the initial number of bins becomes smaller.

// supp_flag = 0 is for adaptively selecting the support size
//           = 1 is for using full binning at each step
//	     = 2 is for using thrust::sort to find the kth largest magnitude with dynamic support detection
//           = 3 is for using thrust::sort at each step


/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


//  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 0.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in IHT initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT initialization");

  float err_start = err;

  float resid_tol = tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// parameters for FindSupportSet_sort
  thrust::device_ptr<float> d_sort(d_vec_thres);
  float T=0.0f;


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_smv(d_vec, d_y, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT in IHT loop");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in IHT loop");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in IHT loop");




/* 
*******************
** Main Steepest Descent Loop **
*******************
*/

  while ( (err > resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*tol) & (fail == 0) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  maxChange = RestrictedSD_S_smv(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, nz, mu, err, k_bin, numBlocks, threadsPerBlock,  numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);

  cudaEvent_t start_supp, stop_supp;
  float time_supp;
  cudaEventCreate(&start_supp);
  cudaEventCreate(&stop_supp);
  cudaEventRecord(start_supp,0);

  if (supp_flag < 2){
	if (minVal <= maxChange) { 
		max_value = MaxMagnitude(d_vec,n);
		slope = ((num_bins-1)/(max_value));
	}
  }

  if (supp_flag == 0){
    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in IHT loop");
    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in IHT loop");
  }
  else if (supp_flag == 1){
    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in IHT loop");
    // note that the second max_value forces binning at each step
    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in IHT loop");
  }
  else if (supp_flag == 2){    
    k_bin = 1;
    T = FindSupportSet_sort(d_vec, d_vec_thres, d_sort, d_bin, T, maxChange, n, k, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet_sort in IHT loop");
    // note that this function includes the threshold
  }
  else if (supp_flag == 3){    
    k_bin = 1;
    T = FindSupportSet_sort(d_vec, d_vec_thres, d_sort, d_bin, T, T, n, k, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet_sort in IHT loop");
    // note that this function includes the threshold
    // note that max_value forces a sort at each iteration
  }


  cudaThreadSynchronize();
  cudaEventRecord(stop_supp,0);
  cudaEventSynchronize(stop_supp);
  cudaEventElapsedTime(&time_supp, start_supp, stop_supp);



  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

  if (iter>749){
	root = 1/16.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1;
  }

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(start_supp);
  cudaEventDestroy(stop_supp);

  time_per_iteration[iter] = time;
  time_supp_set[iter] = time_supp;

  iter++;

  time_sum = time_sum + time;

  }



  *p_iter = iter;
  *p_time_sum = time_sum;

}



/*
*****************************************************************************
**                      NIHT_S_timings_gen                                 **
**      Single Precision Normalized Iterative Hard Thresholding            **
**                    with the general matrices                            **
*****************************************************************************
*/

inline void NIHT_S_timings_gen(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float alpha, int supp_flag, float *time_per_iteration, float *time_supp_set, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{

// alpha is a parameter used to avoid counting unused bins, 
// it should take on values in [0,1) with 0 using the full 
// number of bins from the start and and as 1 is approached 
// the initial number of bins becomes smaller.

// supp_flag = 0 is for adaptively selecting the support size
//           = 1 is for using full binning at each step
//	     = 2 is for using thrust::sort to find the kth largest magnitude with dynamic support detection
//           = 3 is for using thrust::sort at each step


/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


//  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 0.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in IHT initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT initialization");

  float err_start = err;

  float resid_tol = tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// parameters for FindSupportSet_sort
  thrust::device_ptr<float> d_sort(d_vec_thres);
  float T=0.0f;


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_gen(d_vec, d_y, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in IHT loop");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in IHT loop");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in IHT loop");




/* 
*******************
** Main Steepest Descent Loop **
*******************
*/

  while ( (err > resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*tol) & (fail == 0) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  maxChange = RestrictedSD_S_gen(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);

  cudaEvent_t start_supp, stop_supp;
  float time_supp;
  cudaEventCreate(&start_supp);
  cudaEventCreate(&stop_supp);
  cudaEventRecord(start_supp,0);

  if (supp_flag < 2){
	if (minVal <= maxChange) { 
		max_value = MaxMagnitude(d_vec,n);
		slope = ((num_bins-1)/(max_value));
	}
  }


  if (supp_flag == 0){
    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in IHT loop");
    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in IHT loop");
  }
  else if (supp_flag == 1){
    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in IHT loop");
    // note that the second max_value forces binning at each step
    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in IHT loop");
  }
  else if (supp_flag == 2){    
    k_bin = 1;
    T = FindSupportSet_sort(d_vec, d_vec_thres, d_sort, d_bin, T, maxChange, n, k, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet_sort in IHT loop");
    // note that this function includes the threshold
  }
  else if (supp_flag == 3){    
    k_bin = 1;
    T = FindSupportSet_sort(d_vec, d_vec_thres, d_sort, d_bin, T, T, n, k, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet_sort in IHT loop");
    // note that this function includes the threshold
    // note that max_value forces a sort at each iteration
  }


  cudaThreadSynchronize();
  cudaEventRecord(stop_supp,0);
  cudaEventSynchronize(stop_supp);
  cudaEventElapsedTime(&time_supp, start_supp, stop_supp);



  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

  if (iter>749){
	root = 1/16.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1;
  }

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(start_supp);
  cudaEventDestroy(stop_supp);

  time_per_iteration[iter] = time;
  time_supp_set[iter] = time_supp;

  iter++;

  time_sum = time_sum + time;

  }



  *p_iter = iter;
  *p_time_sum = time_sum;

}




/*
*****************************************************************************
**                      HTP_S_timings_dct                                  **
**      Single Precision Hard Thresholding Pursuit                         **
**                    with the DCT                                         **
*****************************************************************************
*/

inline void HTP_S_timings_dct(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float alpha, int supp_flag, float *time_per_iteration, float *time_supp_set, float *cg_per_iteration, float *time_for_cg, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{

// alpha is a parameter used to avoid counting unused bins, 
// it should take on values in [0,1) with 0 using the full 
// number of bins from the start and and as 1 is approached 
// the initial number of bins becomes smaller.

// supp_flag = 0 is for adaptively selecting the support size
//           = 1 is for using full binning at each step
//	     = 2 is for using thrust::sort to find the kth largest magnitude with dynamic support detection
//           = 3 is for using thrust::sort at each step


/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


//  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 0.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in IHT initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT initialization");

  float err_start = err;

  float resid_tol = tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


  // Some variables for the CG projection step
  float tolCG = resid_tol*resid_tol;    // this is a tolerance for the convergnece of the CG projection step in each iteration
  int iterCG = 0;
  int maxiterCG = 15;       // maximum number of CG iterations per iteration of the algorithm
  float errCG = 1;
  float tol2 = 10*tolCG;

  float residNorm_prevCG[16];
  float residNorm_evolutionCG[16];
  

// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;


// parameters for FindSupportSet_sort
  thrust::device_ptr<float> d_sort(d_vec_thres);
  float T=0.0f;



/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_dct(d_vec, d_y, n, m, d_rows);
  SAFEcuda("AT in IHT loop");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in IHT loop");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in IHT loop");




/* 
*******************
** Main Steepest Descent Loop **
*******************
*/

  while ( (err > resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*tol) & (fail == 0) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  maxChange = RestrictedSD_S_dct(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);


  cudaEvent_t start_supp, stop_supp;
  float time_supp;
  cudaEventCreate(&start_supp);
  cudaEventCreate(&stop_supp);
  cudaEventRecord(start_supp,0);

  if (supp_flag < 2){
	if (minVal <= maxChange) { 
		max_value = MaxMagnitude(d_vec,n);
		slope = ((num_bins-1)/(max_value));
	}
  }


  if (supp_flag == 0){
    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in IHT loop");
    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in IHT loop");
  }
  else if (supp_flag == 1){
    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in IHT loop");
    // note that the second max_value forces binning at each step
    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in IHT loop");
  }
  else if (supp_flag == 2){    
    k_bin = 1;
    T = FindSupportSet_sort(d_vec, d_vec_thres, d_sort, d_bin, T, maxChange, n, k, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet_sort in IHT loop");
    // note that this function includes the threshold
  }
  else if (supp_flag == 3){    
    k_bin = 1;
    T = FindSupportSet_sort(d_vec, d_vec_thres, d_sort, d_bin, T, T, n, k, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet_sort in IHT loop");
    // note that this function includes the threshold
    // note that max_value forces a sort at each iteration
  }


  cudaThreadSynchronize();
  cudaEventRecord(stop_supp,0);
  cudaEventSynchronize(stop_supp);
  cudaEventElapsedTime(&time_supp, start_supp, stop_supp);




/* ********* RESET ALL VARIABLES FOR THE CG PROJECTION ************ */
  iterCG=0;
  for (int j=0; j<16; j++) {
	residNorm_prevCG[j]=0;
	residNorm_evolutionCG[j]=1.0f;
  }

  float residNorm_diffCG = 1.0f;
  residNorm_prevCG[15]=residNorm_prev[15];

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT initializiation");

  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct in CG preparation");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CG prep");

  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in CG prep");

  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CG preparation");

  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CG prep");

  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in CG prep");
  errCG = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in CG prep");
  residNorm_prevCG[15]=errCG;



  cudaEvent_t start_cg, stop_cg;
  float time_cg;
  cudaEventCreate(&start_cg);
  cudaEventCreate(&stop_cg);
  cudaEventRecord(start_cg,0);


  while ((errCG > tolCG) & (iterCG < maxiterCG) & (residNorm_diffCG > tol2))
  	{

  	RestrictedCG_S_dct(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prevCG, num_bins, k, m, n, &mu, errCG, k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);

  	errCG = residNorm_prevCG[15];
  	iterCG++;
       // check for convergence 
  	for (int j = 0; j<15; j++) {
		residNorm_evolutionCG[j]=residNorm_evolutionCG[j+1];
  		}
  	residNorm_evolutionCG[15] = residNorm_prevCG[14]-residNorm_prevCG[15];

	residNorm_diffCG = max_list(residNorm_evolutionCG, 16);

  	}

  err = sqrt(errCG);


  cudaThreadSynchronize();
  cudaEventRecord(stop_cg,0);
  cudaEventSynchronize(stop_cg);
  cudaEventElapsedTime(&time_cg, start_cg, stop_cg);


// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_prev[j] = residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;

  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

  if ( iter > (125) ){
	root = 1/16.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f){ fail = 1;
	}
  }

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(start_supp);
  cudaEventDestroy(stop_supp);

  time_per_iteration[iter] = time;
  time_supp_set[iter] = time_supp;
  cg_per_iteration[iter] = iterCG;
  time_for_cg[iter] = time_cg;

  iter++;

  time_sum = time_sum + time;

  }



  *p_iter = iter;
  *p_time_sum = time_sum;

}












/*
*****************************************************************************
**                      HTP_S_timings_smv                                  **
**      Single Precision Hard Thresholding Pursuit                         **
**                    with the smv                                         **
*****************************************************************************
*/

inline void HTP_S_timings_smv(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_cols, float *d_vals, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int nz, int *p_iter, float mu, float err, int *p_sum, float alpha, int supp_flag, float *time_per_iteration, float *time_supp_set, float *cg_per_iteration, float *time_for_cg, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{

// alpha is a parameter used to avoid counting unused bins, 
// it should take on values in [0,1) with 0 using the full 
// number of bins from the start and and as 1 is approached 
// the initial number of bins becomes smaller.

// supp_flag = 0 is for adaptively selecting the support size
//           = 1 is for using full binning at each step
//	     = 2 is for using thrust::sort to find the kth largest magnitude with dynamic support detection
//           = 3 is for using thrust::sort at each step


/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


//  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 0.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in IHT initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT initialization");

  float err_start = err;

  float resid_tol = tol;


  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


  // Some variables for the CG projection step
  float tolCG = resid_tol*resid_tol;     // this is a tolerance for the convergnece of the CG projection step in each iteration
  int iterCG = 0;
  int maxiterCG = 15;       // maximum number of CG iterations per iteration of the algorithm
  float errCG = 1;
  float tol2 = 10*tolCG;

  float residNorm_prevCG[16];
  float residNorm_evolutionCG[16];

// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;  

// parameters for FindSupportSet_sort
  thrust::device_ptr<float> d_sort(d_vec_thres);
  float T=0.0f;



/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_smv(d_vec, d_y, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT in IHT loop");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in IHT loop");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in IHT loop");




/* 
*******************
** Main Steepest Descent Loop **
*******************
*/

  while ( (err > resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*tol) & (fail == 0) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  maxChange = RestrictedSD_S_smv(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, nz, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);


  cudaEvent_t start_supp, stop_supp;
  float time_supp;
  cudaEventCreate(&start_supp);
  cudaEventCreate(&stop_supp);
  cudaEventRecord(start_supp,0);

  if (supp_flag < 2){
	if (minVal <= maxChange) { 
		max_value = MaxMagnitude(d_vec,n);
		slope = ((num_bins-1)/(max_value));
	}
  }


  if (supp_flag == 0){
    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in IHT loop");
    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in IHT loop");
  }
  else if (supp_flag == 1){
    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in IHT loop");
    // note that the second max_value forces binning at each step
    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in IHT loop");
  }
  else if (supp_flag == 2){    
    k_bin = 1;
    T = FindSupportSet_sort(d_vec, d_vec_thres, d_sort, d_bin, T, maxChange, n, k, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet_sort in IHT loop");
    // note that this function includes the threshold
  }
  else if (supp_flag == 3){    
    k_bin = 1;
    T = FindSupportSet_sort(d_vec, d_vec_thres, d_sort, d_bin, T, T, n, k, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet_sort in IHT loop");
    // note that this function includes the threshold
    // note that max_value forces a sort at each iteration
  }


  cudaThreadSynchronize();
  cudaEventRecord(stop_supp,0);
  cudaEventSynchronize(stop_supp);
  cudaEventElapsedTime(&time_supp, start_supp, stop_supp);




/* ********* RESET ALL VARIABLES FOR THE CG PROJECTION ************ */
  iterCG=0;
  for (int j=0; j<16; j++) {
	residNorm_prevCG[j]=0;
	residNorm_evolutionCG[j]=1.0f;
  }

  float residNorm_diffCG = 1.0f;
  residNorm_prevCG[15]=residNorm_prev[15];

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT CG prep");

  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in CG preparation");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CG prep");

  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in CG prep");

  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CG preparation");

  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CG prep");

  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in CG prep");
  errCG = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in CG prep");
  residNorm_prevCG[15]=errCG;



  cudaEvent_t start_cg, stop_cg;
  float time_cg;
  cudaEventCreate(&start_cg);
  cudaEventCreate(&stop_cg);
  cudaEventRecord(start_cg,0);


  while ((errCG > tolCG) & (iterCG < maxiterCG) & (residNorm_diffCG > tol2))
  	{

  	RestrictedCG_S_smv(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prevCG, num_bins, k, m, n, nz, &mu, errCG, k_bin, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
	SAFEcuda("RestrictedCG in CG loop in HTP_S_timings_smv");

  	errCG = residNorm_prevCG[15];
  	iterCG++;

       // check for convergence 
  	for (int j = 0; j<15; j++) {
		residNorm_evolutionCG[j]=residNorm_evolutionCG[j+1];
  		}
  	residNorm_evolutionCG[15] = residNorm_prevCG[14]-residNorm_prevCG[15];

	residNorm_diffCG = max_list(residNorm_evolutionCG, 16);

  	}

  err = sqrt(errCG);


  cudaThreadSynchronize();
  cudaEventRecord(stop_cg,0);
  cudaEventSynchronize(stop_cg);
  cudaEventElapsedTime(&time_cg, start_cg, stop_cg);


// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_prev[j] = residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;

  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

  if ( iter > (125) ){
	root = 1/16.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f){ fail = 1;
	}
  }


  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(start_supp);
  cudaEventDestroy(stop_supp);

  time_per_iteration[iter] = time;
  time_supp_set[iter] = time_supp;
  cg_per_iteration[iter] = iterCG;
  time_for_cg[iter] = time_cg;

  iter++;

  time_sum = time_sum + time;

  }



  *p_iter = iter;
  *p_time_sum = time_sum;

}




/*
*****************************************************************************
**                      HTP_S_timings_gen                                  **
**      Single Precision Hard Thresholding Pursuit                         **
**                    with general matrices                                **
*****************************************************************************
*/

inline void HTP_S_timings_gen(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float alpha, int supp_flag, float *time_per_iteration, float *time_supp_set, float *cg_per_iteration, float *time_for_cg, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{

// alpha is a parameter used to avoid counting unused bins, 
// it should take on values in [0,1) with 0 using the full 
// number of bins from the start and and as 1 is approached 
// the initial number of bins becomes smaller.

// supp_flag = 0 is for adaptively selecting the support size
//           = 1 is for using full binning at each step
//	     = 2 is for using thrust::sort to find the kth largest magnitude with dynamic support detection
//           = 3 is for using thrust::sort at each step


/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


//  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 0.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in IHT initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT initialization");

  float err_start = err;

  float resid_tol = tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


  // Some variables for the CG projection step
  float tolCG = resid_tol*resid_tol;    // this is a tolerance for the convergnece of the CG projection step in each iteration
  int iterCG = 0;
  int maxiterCG = 15;       // maximum number of CG iterations per iteration of the algorithm
  float errCG = 1;
  float tol2 = 10*tolCG;

  float residNorm_prevCG[16];
  float residNorm_evolutionCG[16];
  
// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// parameters for FindSupportSet_sort
  thrust::device_ptr<float> d_sort(d_vec_thres);
  float T=0.0f;



/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_gen(d_vec, d_y, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in IHT loop");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in IHT loop");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in IHT loop");




/* 
*******************
** Main Steepest Descent Loop **
*******************
*/

  while ( (err > resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*tol) & (fail == 0) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  maxChange = RestrictedSD_S_gen(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedSD_S_gen in HTP_S_timings_gen");


  cudaEvent_t start_supp, stop_supp;
  float time_supp;
  cudaEventCreate(&start_supp);
  cudaEventCreate(&stop_supp);
  cudaEventRecord(start_supp,0);

  if (supp_flag < 2){
	if (minVal <= maxChange) { 
		max_value = MaxMagnitude(d_vec,n);
		slope = ((num_bins-1)/(max_value));
	}
  }


  if (supp_flag == 0){
    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in IHT loop");
    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in IHT loop");
  }
  else if (supp_flag == 1){
    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in IHT loop");
    // note that the second max_value forces binning at each step
    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in IHT loop");
  }
  else if (supp_flag == 2){    
    k_bin = 1;
    T = FindSupportSet_sort(d_vec, d_vec_thres, d_sort, d_bin, T, maxChange, n, k, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet_sort in IHT loop");
    // note that this function includes the threshold
  }
  else if (supp_flag == 3){    
    k_bin = 1;
    T = FindSupportSet_sort(d_vec, d_vec_thres, d_sort, d_bin, T, T, n, k, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet_sort in IHT loop");
    // note that this function includes the threshold
    // note that max_value forces a sort at each iteration
  }


  cudaThreadSynchronize();
  cudaEventRecord(stop_supp,0);
  cudaEventSynchronize(stop_supp);
  cudaEventElapsedTime(&time_supp, start_supp, stop_supp);




/* ********* RESET ALL VARIABLES FOR THE CG PROJECTION ************ */
  iterCG=0;
  for (int j=0; j<16; j++) {
	residNorm_prevCG[j]=0;
	residNorm_evolutionCG[j]=1.0f;
  }

  float residNorm_diffCG = 1.0f;
  residNorm_prevCG[15]=residNorm_prev[15];

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT CG prep");

  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in CG preparation");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CG prep");

  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in CG prep");

  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CG preparation");

  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CG prep");

  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in CG prep");
  errCG = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in CG prep");
  residNorm_prevCG[15]=errCG;



  cudaEvent_t start_cg, stop_cg;
  float time_cg;
  cudaEventCreate(&start_cg);
  cudaEventCreate(&stop_cg);
  cudaEventRecord(start_cg,0);


  while ((errCG > tolCG) & (iterCG < maxiterCG) & (residNorm_diffCG > tol2))
  	{

  	RestrictedCG_S_gen(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, residNorm_prevCG, num_bins, k, m, n, &mu, errCG, k_bin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
	SAFEcuda("RestrictedCG_S_gen in HTP_S_timings_gen");

  	errCG = residNorm_prevCG[15];
  	iterCG++;

       // check for convergence 
  	for (int j = 0; j<15; j++) {
		residNorm_evolutionCG[j]=residNorm_evolutionCG[j+1];
  		}
  	residNorm_evolutionCG[15] = residNorm_prevCG[14]-residNorm_prevCG[15];

	residNorm_diffCG = max_list(residNorm_evolutionCG, 16);

  	}

  err = sqrt(errCG);


  cudaThreadSynchronize();
  cudaEventRecord(stop_cg,0);
  cudaEventSynchronize(stop_cg);
  cudaEventElapsedTime(&time_cg, start_cg, stop_cg);


// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_prev[j] = residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;

  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

  if ( iter > (125) ){
	root = 1/16.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f){ fail = 1;
	}
  }


  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(start_supp);
  cudaEventDestroy(stop_supp);

  time_per_iteration[iter] = time;
  time_supp_set[iter] = time_supp;
  cg_per_iteration[iter] = iterCG;
  time_for_cg[iter] = time_cg;

  iter++;

  time_sum = time_sum + time;

  }



  *p_iter = iter;
  *p_time_sum = time_sum;

}











