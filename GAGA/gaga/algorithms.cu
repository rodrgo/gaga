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
****************************************
**             IHT_S_dct              **
** Single Precision IHT with the DCT  **
****************************************
*/

inline void IHT_S_dct(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err,  int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


  float alpha = 0.25f;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in IHT_S_dct initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT_S_dct initialization");

  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }

// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in IHT_S_dct Set Linear Bins");

  float max_value = MaxMagnitude(grad,n);

  float slope = ((num_bins-1)/(max_value));


/*
*******************
** Main IHT Loop **
*******************
*/

  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  
  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in IHT_S_dct loop");

  maxChange = MaxMagnitude(grad,n);
  maxChange = 2 * mu * maxChange;

  cublasSaxpy(n, mu, grad, 1, d_vec, 1);
  SAFEcublas("First cublasSaxpy in IHT_S_dct loop");


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in IHT_S_dct loop");

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in IHT_S_dct loop");

  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct in IHT_S_dct loop");

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("Second cublasScopy in IHT_S_dct loop");

  cublasSaxpy(m, -1.0f, resid_update, 1, resid, 1);
  SAFEcublas("Second cublasSaxpy in IHT_S_dct loop");

  err=cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT_S_dct loop");


// recording the convergence of the residual
  for (int j = 0; j<15; j++) {
	residNorm_prev[j] = residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_prev[15]=err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

  if (iter>749){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1;
  }


  iter++;

  
  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }
  *p_iter = iter;
  *p_time_sum = time_sum;

}




/*
***********************************************
**              IHT_S_smv                    **
** Single Precision IHT with sparse matrices **
***********************************************
*/


inline void IHT_S_smv(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_cols, float *d_vals, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int nz, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{

  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;

  float alpha = 0.25f;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in IHT_S_smv initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT_S_smv initialization");

  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif

// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in IHT_S_smv Set Linear Bins");

  float max_value = MaxMagnitude(grad,n);

  float slope = ((num_bins-1)/(max_value));


/*
*******************
** Main IHT Loop **
*******************
*/


  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  
  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in IHT_S_smv loop");

  maxChange = MaxMagnitude(grad,n);
  maxChange = 2 * mu * maxChange;

  cublasSaxpy(n, mu, grad, 1, d_vec, 1);
  SAFEcublas("First cublasSaxpy in IHT_S_smv loop");


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in IHT_S_smv loop");

  Threshold(d_vec,d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in IHT_S_smv loop");


  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in IHT_S_smv loop");

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("Second cublasScopy in IHT_S_smv loop");

  cublasSaxpy(m, -1.0f, resid_update, 1, resid, 1);
  SAFEcublas("Second cublasSaxpy in IHT_S_smv loop");

  err=cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT_S_smv loop");



// recording the convergence of the residual
  for (int j = 0; j<15; j++) {
	residNorm_prev[j] = residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_prev[15]=err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

  if (iter>749){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1;
  }


  iter++;

  
  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }

  *p_iter = iter;
  *p_time_sum = time_sum;

}




/*
************************************************
**              IHT_S_gen                     **
** Single Precision IHT with general matrices **
************************************************
*/


inline void IHT_S_gen(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{

  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;

  float alpha = 0.25f;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in IHT_S_gen initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT_S_gen initialization");

  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }

// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif




/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in IHT_S_gen Set Linear Bins");

  float max_value = MaxMagnitude(grad,n);

  float slope = ((num_bins-1)/(max_value));


/*
*******************
** Main IHT Loop **
*******************
*/


  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  
  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in IHT_S_gen loop");

  maxChange = MaxMagnitude(grad,n);
  maxChange = 2 * mu * maxChange;

  cublasSaxpy(n, mu, grad, 1, d_vec, 1);
  SAFEcublas("First cublasSaxpy in IHT_S_gen loop");


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in IHT_S_gen loop");

  Threshold(d_vec,d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in IHT_S_gen loop");


  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in IHT_S_gen loop");

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("Second cublasScopy in IHT_S_gen loop");

  cublasSaxpy(m, -1.0f, resid_update, 1, resid, 1);
  SAFEcublas("Second cublasSaxpy in IHT_S_gen loop");

  err=cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT_S_gen loop");


// recording the convergence of the residual
  for (int j = 0; j<15; j++) {
	residNorm_prev[j] = residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_prev[15]=err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

  if (iter>749){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1;
  }

  iter++;

  
  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }

  *p_iter = iter;
  *p_time_sum = time_sum;

}










/*
******************************************************
**                   HT_SD_S_dct                    **
**      Single Precision One Step Hard Threshold    **
**  followed by subspace restricted stepest descent **
**                with the DCT                      **
******************************************************
*/

inline void HT_SD_S_dct(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


  float alpha = 0.0f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT_SD_S_dct initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in HT_SD_S_dct initialization");

  float resid_tol2 = 0.1*resid_tol; 

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_dct(d_vec, d_y, n, m, d_rows);
  SAFEcuda("AT_dct in HT_SD_S_dct Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in HT_SD_S_dct Set Linear Bins");


  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HT_SD_S_dct Set Linear Bins");


/* 
*******************
** Main Steepest Descent Loop **
*******************
*/


  while ((err > resid_tol) & (iter < maxiter) & (residNorm_diff > resid_tol2))
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  RestrictedSD_S_dct(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedSD in HT_SD_S_dct");

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HT_SD_S_dct");


  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }



  *p_iter = iter;
  *p_time_sum = time_sum;

}






/*
******************************************************
**                   HT_SD_S_smv                    **
**      Single Precision One Step Hard Threshold    **
**  followed by sibspace restricted stepest descent **
**            with the sparse matrices              **
******************************************************
*/

inline void HT_SD_S_smv(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_cols, float *d_vals, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int nz,  int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/



  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;

  float alpha = 0.0f;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT_SD_S_smv initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in HT_SD_S_smv initialization");

  float resid_tol2 = 0.1*resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_smv(d_vec, d_y, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in HT_SD_S_smv Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in HT_SD_S_smv Set Linear Bins");


  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HT_SD_S_smv Set Linear Bins");




/* 
*******************
** Main Steepest Descent Loop **
*******************
*/


  while ((err > resid_tol) & (iter < maxiter) & (residNorm_diff > resid_tol2))
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  RestrictedSD_S_smv(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, nz, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedSD in HT_SD_S_smv loop");

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HT_SD_S_smv loop");


  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }


  *p_iter = iter;
  *p_time_sum = time_sum;

}



/*
*******************************************************
**                   HT_SD_S_gen                     **
**      Single Precision One Step Hard Threshold     **
**  followed by sibspace restricted stepest descent  **
**            with the general matrices              **
*******************************************************
*/

inline void HT_SD_S_gen(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/



  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;

  float alpha = 0.0f;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT_SD_S_gen initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in HT_SD_S_gen initialization");

  float resid_tol2 = 0.1*resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_gen(d_vec, d_y, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in HT_SD_S_gen Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in HT_SD_S_gen Set Linear Bins");


  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in IHT loop");




/* 
*******************
** Main Steepest Descent Loop **
*******************
*/


  while ((err > resid_tol) & (iter < maxiter) & (residNorm_diff > resid_tol2))
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  RestrictedSD_S_gen(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedSD in HT_SD_S_gen loop");

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HT_SD_S_gen loop");


  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }


  *p_iter = iter;
  *p_time_sum = time_sum;

}







/*
*********************************************************************
**                      NIHT_S_dct                                 **
**      Single Precision Normalized Iterative Hard Thresholding    **
**                    with the DCT                                 **
*********************************************************************
*/

inline void NIHT_S_dct(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 0.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in NIHT_S_dct initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in NIHT_S_dct initialization");

  float err_start = err;

  float resid_resid_tol = resid_tol;


  float residNorm_diff = 1.0f;
//  float residNorm_evolution_max = 0.0f;
//  float residNorm_evolution_change = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_dct(d_vec, d_y, n, m, d_rows);
  SAFEcuda("AT_dct in NIHT_S_dct Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in NIHT_S_dct Set Linear Bins");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in NIHT_S_dct Set Linear Bins");




/* 
********************************
** Main Steepest Descent Loop **
********************************
*/

  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  maxChange = RestrictedSD_S_dct(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedSD in NIHT_S_dct loop");

  if (minVal <= maxChange) { 
    max_value = MaxMagnitude(d_vec,n);
    slope = ((num_bins-1)/(max_value));
  }


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in NIHT_S_dct loop");

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in NIHT_S_dct loop");



  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if (iter>749){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1;
  }



  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;

  }


  *p_iter = iter;
  *p_time_sum = time_sum;

}








/*
*********************************************************************
**                      NIHT_S_smv                                 **
**      Single Precision Normalized Iterative Hard Thresholding    **
**                with the sparse matrices                         **
*********************************************************************
*/

inline void NIHT_S_smv(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_cols, float *d_vals, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int nz, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/



  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;

  float alpha = 0.25f;  // normally 0.25f;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in NIHT_S_smv initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in NIHT_S_smv initialization");

  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_smv(d_vec, d_y, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in NIHT_S_smv Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in NIHT_S_smv Set Linear Bins");


  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in NIHT_S_smv Set Linear Bins");




/* 
*******************
** Main Steepest Descent Loop **
*******************
*/

  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  maxChange = RestrictedSD_S_smv(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, nz, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedSD in NIHT_S_smv loop");

  if (minVal <= maxChange) { 
    max_value = MaxMagnitude(d_vec,n);
    slope = ((num_bins-1)/(max_value));
  }

  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in NIHT_S_smv loop");

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in NIHT_S_smv loop");


  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if (iter>749){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1;
  }

  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }



  *p_iter = iter;
  *p_time_sum = time_sum;

}



/*
*********************************************************************
**                      NIHT_S_gen                                 **
**      Single Precision Normalized Iterative Hard Thresholding    **
**                with general matrices                            **
*********************************************************************
*/

inline void NIHT_S_gen(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/





  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;

  float alpha = 0.25f;  // normally 0.25f;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in NIHT_S_gen initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in NIHT_S_gen initialization");

  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// **********************
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// **********************


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_gen(d_vec, d_y, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in NIHT_S_gen Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in NIHT_S_gen Set Linear Bins");


  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in NIHT_S_gen Set Linear Bins");




/* 
*******************
** Main Steepest Descent Loop **
*******************
*/

  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  maxChange = RestrictedSD_S_gen(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedSD in NIHT_S_gen loop");


  if (minVal <= maxChange) { 
    max_value = MaxMagnitude(d_vec,n);
    slope = ((num_bins-1)/(max_value));
  }





  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in NIHT_S_gen loop");



  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in NIHT_S_gen loop");


  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if (iter>749){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1; 
  }


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }


  *p_iter = iter;
  *p_time_sum = time_sum;

}



/*
*********************************************************
**                   HT_CG_S_dct                       **
**      Single Precision One Step Hard Threshold       **
**  followed by subspace restricted conjugate gradient **
**                with the DCT                         **
*********************************************************
*/

inline void HT_CG_S_dct(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


  float alpha = 0.0f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT_CG_S_dct initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in HT_CG_S_dct initialization");

  float resid_tolCG = resid_tol*resid_tol;
  float resid_tol2 = 10*resid_tolCG; 

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_dct(d_vec, d_y, n, m, d_rows);
  SAFEcuda("AT_dct in HT_CG_S_dct Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in HT_CG_S_dct Set Linear Bins");


  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HT_CG_S_dct Set Linear Bins");


  // Our main variables are three n-vectors:
  //   d_vec will store the thresholded vector x
  //   grad will store the descent direction p
  //   grad_previous will store the residual r
  // We also use three working variables
  //    d_vec_thres is a working n-vector
  //    resid and resid_update are working m-vectors

  // Now need to compute the residual and initial descent direction.
  // Using the algorithm listed on page 35 of Greenbaum

  // Initially compute dy - A_dct * d_vec
  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT_CG_S_dct Set Linear Bins");

  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct in CG preparation");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CG prep in HT_CG_S_dct");

  // Compute AT_dct * (dy - A_dct * d_vec) 
  // and then thresholded to d_bin
  // put this final result in grad and grad_previous

  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in CG prep in HT_CG_S_dct");

  // threshold grad to the set used for d_vec
  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CG preparation in HT_CG_S_dct");

  // copy grad onto grad_previous
  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CG prep in HT_CG_S_dct");

  // At this point CG is ready, now setup the main CG loop

  // At this point the input variables are ready for CG.
  // d_bin is our candidate support set.
  // d_vec is our candidate k_bin sparse guess for the solution, 
  // grad is the k_bin restricted search direction.
  // grad_previous is the k_bin restricted residual


  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in HT_CG_S_dct");
  err = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in HT_CG_S_dct");
  residNorm_prev[15]=err;

/* 
*******************
** Main Conjugate Gradient Loop **
*******************
*/


  while ((err > resid_tolCG) & (iter < maxiter) & (residNorm_diff > resid_tol2))
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  RestrictedCG_S_dct(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, &mu, err, k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedSD in CG loop in HT_CG_S_dct");



  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }


  *p_iter = iter;
  *p_time_sum = time_sum;

}





/*
*********************************************************
**                   HT_CG_S_smv                       **
**      Single Precision One Step Hard Threshold       **
**  followed by subspace restricted conjugate gradient **
**                with the SMV                         **
*********************************************************
*/

inline void HT_CG_S_smv(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_cols, float *d_vals, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int nz, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


  float alpha = 0.0f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT_CG_S_smv initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in HT_CG_S_smv initialization");

  float resid_tolCG = resid_tol*resid_tol;
  float resid_tol2 = 10*resid_tolCG; 

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_smv(d_vec, d_y, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in HT_CG_S_smv Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in HT_CG_S_smv Set Linear Bins");


  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HT_CG_S_smv Set Linear Bins");


  // Our main variables are three n-vectors:
  //   d_vec will store the thresholded vector x
  //   grad will store the descent direction p
  //   grad_previous will store the residual r
  // We also use three working variables
  //    d_vec_thres is a working n-vector
  //    resid and resid_update are working m-vectors

  // Now need to compute the residual and initial descent direction.
  // Using the algorithm listed on page 35 of Greenbaum

  // Initially compute dy - A_dct * d_vec
  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT_CG_S_smv Set Linear Bins");

  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in CG preparation in HT_CG_S_smv");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CG prep in HT_CG_S_smv");

  // Compute AT_dct * (dy - A_dct * d_vec) 
  // and then thresholded to d_bin
  // put this final result in grad and grad_previous

  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_dct in CG prep in HT_CG_S_smv");

  // threshold grad to the set used for d_vec
  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CG preparation in HT_CG_S_smv");

  // copy grad onto grad_previous
  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CG prep in HT_CG_S_smv");

  // At this point CG is ready, now setup the main CG loop

  // At this point the input variables are ready for CG.
  // d_bin is our candidate support set.
  // d_vec is our candidate k_bin sparse guess for the solution, 
  // grad is the k_bin restricted search direction.
  // grad_previous is the k_bin restricted residual


  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in HT_CG_S_dct");
  err = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in HT_CG_S_dct");
  residNorm_prev[15]=err;

/* 
*******************
** Main Conjugate Gradient Loop **
*******************
*/


  while ((err > resid_tolCG) & (iter < maxiter) & (residNorm_diff > resid_tol2))
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  RestrictedCG_S_smv(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, nz, &mu, err, k_bin, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedCG in CG loop in HT_CG_S_smv");


  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }


  *p_iter = iter;
  *p_time_sum = time_sum;

}



/*
**********************************************************************
**                   HT_CG_S_gen                                    **
**      Single Precision One Step Hard Threshold                    **
**  followed by subspace restricted conjugate gradient              **
**                with the general matrices                         **
**********************************************************************
*/

inline void HT_CG_S_gen(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


  float alpha = 0.0f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT_CG_S_gen initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in HT_CG_S_gen initialization");

  float resid_tolCG = resid_tol*resid_tol;
  float resid_tol2 = 10*resid_tolCG; 

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_gen(d_vec, d_y, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in HT_CG_S_gen Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in HT_CG_S_gen Set Linear Bins");


  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HT_CG_S_gen Set Linear Bins");


  // Our main variables are three n-vectors:
  //   d_vec will store the thresholded vector x
  //   grad will store the descent direction p
  //   grad_previous will store the residual r
  // We also use three working variables
  //    d_vec_thres is a working n-vector
  //    resid and resid_update are working m-vectors

  // Now need to compute the residual and initial descent direction.
  // Using the algorithm listed on page 35 of Greenbaum

  // Initially compute dy - A_gen * d_vec
  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT_CG_S_gen Set Linear Bins");

  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in CG preparation in HT_CG_S_gen");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CG prep in HT_CG_S_gen");

  // The resid for us is AT_gen * (dy - A_gen * d_vec) 
  // and then thresholded to d_bin
  // put this final result in grad and grad_previous

  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in CG prep in HT_CG_S_gen");

  // threshold grad to the set used for d_vec
  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CG preparation in HT_CG_S_gen");

  // copy grad onto grad_previous
  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CG prep in HT_CG_S_gen");

  // At this point CG is ready, now setup the main CG loop

  // At this point the input variables are ready for CG.
  // d_bin is our candidate support set.
  // d_vec is our candidate k_bin sparse guess for the solution, 
  // grad is the k_bin restricted search direction.
  // grad_previous is the k_bin restricted residual


  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in HT_CG_S_dct");
  err = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in HT_CG_S_dct");
  residNorm_prev[15]=err;

/* 
*******************
** Main Conjugate Gradient Loop **
*******************
*/


  while ((err > resid_tolCG) & (iter < maxiter) & (residNorm_diff > resid_tol2))
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  RestrictedCG_S_gen(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, &mu, err, k_bin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedCG in CG prep in HT_CG_S_gen");


  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }


  *p_iter = iter;
  *p_time_sum = time_sum;

}






/*
*********************************************************************
**                      HTP_S_dct                                  **
**      Single Precision Hard Thresholding Pursuit (aka HITS)      **
**                    with the DCT                                 **
*********************************************************************
*/

inline void HTP_S_dct(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 0.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HTP_S_dct initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in HTP_S_dct initialization");


  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;
  // float residNorm_evolution_max = 0.0f;
  // float residNorm_evolution_change = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


  // Some variables for the CG projection step
  float resid_tolCG = resid_resid_tol*resid_resid_tol;     // this is a resid_tolerance for the convergnece of the CG projection step in each iteration
  int iterCG = 0;
  int maxiterCG = 15;       // maximum number of CG iterations per iteration of the algorithm
  float errCG = 1;
  float resid_tol2 = 10*resid_tolCG;

  float residNorm_prevCG[16];
  float residNorm_evolutionCG[16];
  

// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif



/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_dct(d_vec, d_y, n, m, d_rows);
  SAFEcuda("AT_dct in HTP_S_dct Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in HTP_S_dct Set Linear Bins");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HTP_S_dct Set Linear Bins");




/* 
*******************
** Main Steepest Descent Loop **
*******************
*/

  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  maxChange = RestrictedSD_S_dct(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedSD in HTP_S_dct loop");

  if (minVal <= maxChange) { 
    max_value = MaxMagnitude(d_vec,n);
    slope = ((num_bins-1)/(max_value));
  }

  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in HTP_S_dct loop");

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HTP_S_dct loop");

/* ********* RESET ALL VARIABLES FOR THE CG PROJECTION ************ */
  iterCG=0;
  for (int j=0; j<16; j++) {
	residNorm_prevCG[j]=0;
	residNorm_evolutionCG[j]=1.0f;
  }

  float residNorm_diffCG = 1.0f;
  residNorm_prevCG[15]=residNorm_prev[15];

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CG preparation in HTP_S_dct");

  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct in CG preparation in HTP_S_dct");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CG prep in HTP_S_dct");

  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in CG prep in HTP_S_dct");

  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CG preparation in HTP_S_dct");

  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CG prep in HTP_S_dct");

  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in HTP_S_dct");
  errCG = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in HTP_S_dct");
  residNorm_prevCG[15]=errCG;


  while ((errCG > resid_tolCG) & (iterCG < maxiterCG) & (residNorm_diffCG > resid_tol2))
  	{

  	RestrictedCG_S_dct(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prevCG, num_bins, k, m, n, &mu, errCG, k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  	SAFEcuda("RestrictedCG in CG loop in HTP_S_dct");

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

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_prev[j] = residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;

  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if ( iter > (125) ){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f){ fail = 1;
	}
  }


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }

  *p_iter = iter;
  *p_time_sum = time_sum;

}





/*
*********************************************************************
**                      HTP_S_smv                                  **
**      Single Precision Hard Thresholding Pursuit (aka HITS)      **
**                    with the DCT                                 **
*********************************************************************
*/

inline void HTP_S_smv(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_cols, float *d_vals, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int nz, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 0.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HTP_S_smv initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in HTP_S_smv initialization");
 
  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


  // Some variables for the CG projection step
  float resid_tolCG = resid_resid_tol*resid_resid_tol;     // this is a resid_tolerance for the convergnece of the CG projection step in each iteration
  int iterCG = 0;
  int maxiterCG = 15;       // maximum number of CG iterations per iteration of the algorithm
  float errCG = 1;
  float resid_tol2 = 10*resid_tolCG;

  float residNorm_prevCG[16];
  float residNorm_evolutionCG[16];
  

// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif

/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_smv(d_vec, d_y, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in HTP_S_smv Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in HTP_S_smv Set Linear Bins");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HTP_S_smv Set Linear Bins");




/* 
*******************
** Main Steepest Descent Loop **
*******************
*/


  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  maxChange = RestrictedSD_S_smv(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, nz, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedSD in HTP_S_smv loop");


  if (minVal <= maxChange) { 
    max_value = MaxMagnitude(d_vec,n);
    slope = ((num_bins-1)/(max_value));
  }

  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in HTP_S_smv loop");

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HTP_S_smv loop");

/* ********* RESET ALL VARIABLES FOR THE CG PROJECTION ************ */
  iterCG=0;
  for (int j=0; j<16; j++) {
	residNorm_prevCG[j]=0;
	residNorm_evolutionCG[j]=1.0f;
  }

  float residNorm_diffCG = 1.0f;
  residNorm_prevCG[15]=residNorm_prev[15];

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CG preparation in HTP_S_smv");

  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_dct in CG preparation in HTP_S_smv");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CG prep in HTP_S_smv");

  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in CG prep in HTP_S_smv");

  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CG preparation in HTP_S_smv");

  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CG prep in HTP_S_smv");

  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in HTP_S_smv");
  errCG = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in HTP_S_smv");
  residNorm_prevCG[15]=errCG;


  while ((errCG > resid_tolCG) & (iterCG < maxiterCG) & (residNorm_diffCG > resid_tol2))
  	{

  	RestrictedCG_S_smv(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prevCG, num_bins, k, m, n, nz, &mu, errCG, k_bin, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
  	SAFEcuda("RestrictedCG in CG loop in HTP_S_smv");

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

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_prev[j] = residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if ( iter > (125) ){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f){ fail = 1;
	}
  }


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }

  *p_iter = iter;
  *p_time_sum = time_sum;

}




/*
******************************************************************************
**                      HTP_S_gen                                           **
**      Single Precision Hard Thresholding Pursuit (aka HITS)               **
**                    with general matrices                                 **
******************************************************************************
*/

inline void HTP_S_gen(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 0.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HTP_S_gen initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in HTP_S_gen initialization");

  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


  // Some variables for the CG projection step
  float resid_tolCG = resid_resid_tol*resid_resid_tol;     // this is a resid_tolerance for the convergnece of the CG projection step in each iteration
  int iterCG = 0;
  int maxiterCG = 15;       // maximum number of CG iterations per iteration of the algorithm
  float errCG = 1;
  float resid_tol2 = 10*resid_tolCG;

  float residNorm_prevCG[16];
  float residNorm_evolutionCG[16];
  

// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_gen(d_vec, d_y, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in HTP_S_gen Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in HTP_S_gen Set Linear Bins");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HTP_S_gen Set Linear Bins");




/* 
*******************
** Main Steepest Descent Loop **
*******************
*/


  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  maxChange = RestrictedSD_S_gen(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, num_bins, k, m, n, mu, err, k_bin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedSD in HTP_S_gen loop");


  if (minVal <= maxChange) { 
    max_value = MaxMagnitude(d_vec,n);
    slope = ((num_bins-1)/(max_value));
  }

  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in HTP_S_gen loop");

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in HTP_S_gen loop");

/* ********* RESET ALL VARIABLES FOR THE CG PROJECTION ************ */
  iterCG=0;
  for (int j=0; j<16; j++) {
	residNorm_prevCG[j]=0;
	residNorm_evolutionCG[j]=1.0f;
  }

  float residNorm_diffCG = 1.0f;
  residNorm_prevCG[15]=residNorm_prev[15];

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CG preparation in HTP_S_gen");

  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in CG preparation in HTP_S_gen");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CG prep in HTP_S_gen");

  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in CG prep in HTP_S_gen");

  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CG preparation in HTP_S_gen");

  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CG prep in HTP_S_gen");

  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in HTP_S_gen");
  errCG = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in HTP_S_gen");
  residNorm_prevCG[15]=errCG;


  while ((errCG > resid_tolCG) & (iterCG < maxiterCG) & (residNorm_diffCG > resid_tol2))
  	{

  	RestrictedCG_S_gen(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, residNorm_prevCG, num_bins, k, m, n, &mu, errCG, k_bin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
  	SAFEcuda("RestrictedCG in CG loop in HTP_S_gen");

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

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_prev[j] = residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if ( iter > (125) ){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f){ fail = 1;
	}
  }


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }

  *p_iter = iter;
  *p_time_sum = time_sum;

}





/*
****************************************************************************************
**                                  CSMPSP_S_dct                                      **
**      Single Precision Compressive Sampling Matching Pursuit / Subspace Pursuit     **
**                                  with the DCT                                      **
****************************************************************************************
*/

inline void CSMPSP_S_dct(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int *d_bin_grad, int * d_bin_counters_grad, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;
  int k_bin_grad = 0;


  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CSMPSP_S_dct initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in CSMPSP_S_dct initialization");

  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


  // Some variables for the CG projection step
  float resid_tolCG = resid_resid_tol*resid_resid_tol;     // this is a resid_tolerance for the convergnece of the CG projection step in each iteration
  int iterCG = 0;
  int maxiterCG = 15;       // maximum number of CG iterations per iteration of the algorithm
  float errCG = 1;
  float resid_tol2 = 10*resid_tolCG;

  float residNorm_prevCG[16];
  float residNorm_evolutionCG[16];
  

// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


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
  SAFEcuda("FindSupportSet in CSMPSP_S_dct Set Linear Bins");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CSMPSP_S_dct Set Linear Bins");

/* ********* SET ALL VARIABLES FOR THE CG PROJECTION ************ */
  iterCG=0;
  for (int j=0; j<16; j++) {
	residNorm_prevCG[j]=0;
	residNorm_evolutionCG[j]=1.0f;
  }

  float residNorm_diffCG = 1.0f;
  residNorm_prevCG[15]=residNorm_prev[15];

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CG prep in CSMPSP_S_dct Set Linear Bins");

  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct for resid_update in CG prep in CSMPSP_S_dct Set Linear Bins");  

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CG prep in CSMPSP_S_dct Set Linear Bins");

  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in CG prep in CSMPSP_S_dct Set Linear Bins");

  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CG preparation in CSMPSP_S_dct Set Linear Bins");

  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CG prep in CSMPSP_S_dct Set Linear Bins");

  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in CSMPSP_S_dct Set Linear Bins");
  errCG = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in CSMPSP_S_dct Set Linear Bins");
  residNorm_prevCG[15]=errCG;



  while ((errCG > resid_tolCG) & (iterCG < maxiterCG) & (residNorm_diffCG > resid_tol2))
  	{

  	RestrictedCG_S_dct(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prevCG, num_bins, k, m, n, &mu, errCG, k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("RestrictedCG in CSMPSP_S_dct CG loop in Set Linear Bins");

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
  residNorm_prev[15] = err;
  iter++;		// We consider the initial CG projection of CSMPSP as an iteration.



/* 
*******************
** Main Steepest Descent Loop **
*******************
*/

  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in CG prep in CSMPSP_S_dct loop");

  max_value = MaxMagnitude(grad, n);

  slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(grad, d_bin_grad, d_bin_counters_grad, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin_grad, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet for grad in CSMPSP_S_dct loop");

  joinSupports<<< numBlocks, threadsPerBlock >>>(d_bin, d_bin_grad, k_bin, k_bin_grad, n);
  SAFEcuda("joinSupports in CSMPSP_S_dct loop");


/* ********* RESET ALL VARIABLES FOR THE CG PROJECTION ************ */
  iterCG=0;
  for (int j=0; j<16; j++) {
	residNorm_prevCG[j]=0;
	residNorm_evolutionCG[j]=1.0f;
  }

  float residNorm_diffCG = 1.0f;
  residNorm_prevCG[15]=residNorm_prev[15];

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CG prep in CSMPSP_S_dct loop");

  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct for resid_update in CG prep in CSMPSP_S_dct loop");  

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CG prep in CSMPSP_S_dct loop");

  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in CG prep in CSMPSP_S_dct loop");

  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CG prep in CSMPSP_S_dct loop");

  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CG prep in CSMPSP_S_dct loop");

  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in CSMPSP_S_dct loop");
  errCG = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in CSMPSP_S_dct loop");
  residNorm_prevCG[15]=errCG;

  while ((errCG > resid_tolCG) & (iterCG < maxiterCG) & (residNorm_diffCG > resid_tol2))
  	{

  	RestrictedCG_S_dct(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prevCG, num_bins, k, m, n, &mu, errCG, k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  	SAFEcuda("RestrictedCG in CG loop in CSMPSP_S_dct loop");

  	errCG = residNorm_prevCG[15];
  	iterCG++;

       // check for convergence 
  	for (int j = 0; j<15; j++) {
		residNorm_evolutionCG[j]=residNorm_evolutionCG[j+1];
  		}
  	residNorm_evolutionCG[15] = residNorm_prevCG[14]-residNorm_prevCG[15];

	residNorm_diffCG = max_list(residNorm_evolutionCG, 16);

  	}

  max_value = MaxMagnitude(d_vec,n);
  SAFEcuda("MaxMagnitude in CSMPSP_S_dct loop");

  slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in CSMPSP_S_dct loop");
  // second max_value entry to force binning.

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CSMPSP_S_dct loop");

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CSMPSP_S_dct loop");

  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct in CSMPSP_S_dct loop");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CSMPSP_S_dct loop");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in CSMPSP_S_dct loop");


// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_prev[j]=residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }

  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if ( iter > (125) ){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f){ fail = 1;
	}
  }

  iter++;


  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }

  *p_iter = iter;
  *p_time_sum = time_sum;

}








/*
****************************************************************************************
**                                  CSMPSP_S_smv                                      **
**      Single Precision Compressive Sampling Matching Pursuit / Subspace Pursuit     **
**                                  with the SMV                                      **
****************************************************************************************
*/

inline void CSMPSP_S_smv(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_cols, float *d_vals, int *d_bin, int * d_bin_counters, int *d_bin_grad, int * d_bin_counters_grad, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int nz, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;
  int k_bin_grad = 0;


  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CSMPSP_S_smv initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in CSMPSP_S_smv initialization");

  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


  // Some variables for the CG projection step
  float resid_tolCG = resid_resid_tol*resid_resid_tol;     // this is a resid_tolerance for the convergnece of the CG projection step in each iteration
  int iterCG = 0;
  int maxiterCG = 15;       // maximum number of CG iterations per iteration of the algorithm
  float errCG = 1;
  float resid_tol2 = 10*resid_tolCG;

  float residNorm_prevCG[16];
  float residNorm_evolutionCG[16];
  

// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif

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
  SAFEcuda("FindSupportSet in CSMPSP_S_smv loop");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.


  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CSMPSP_S_smv loop");

/* ********* SET ALL VARIABLES FOR THE CG PROJECTION ************ */
  iterCG=0;
  for (int j=0; j<16; j++) {
	residNorm_prevCG[j]=0;
	residNorm_evolutionCG[j]=1.0f;
  }

  float residNorm_diffCG = 1.0f;
  residNorm_prevCG[15]=residNorm_prev[15];

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in  CG prep");

  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv for resid_update in CG prep");  

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CG prep");

  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in CG prep");

  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CG preparation");

  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CG prep");

  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in CSMPSP_S_smv Set Linear Bins");
  errCG = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in CSMPSP_S_smv Set Linear Bins");
  residNorm_prevCG[15]=errCG;


  while ((errCG > resid_tolCG) & (iterCG < maxiterCG) & (residNorm_diffCG > resid_tol2))
  	{

  	RestrictedCG_S_smv(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prevCG, num_bins, k, m, n, nz, &mu, errCG, k_bin, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
	SAFEcuda("RestrictedCG_S_smv in initial CG loop");

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
  residNorm_prev[15] = err;
  iter++;		// We consider the initial CG projection of CSMPSP as an iteration.

/* 
*******************
** Main Steepest Descent Loop **
*******************
*/

  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  SAFEcuda("Just Before AT_smv in CSMPSP loop");

  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in CSMPSP loop");

  max_value = MaxMagnitude(grad, n);
  minVal = 0.0f;

  slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(grad, d_bin_grad, d_bin_counters_grad, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin_grad, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in CSMPSP loop");

  joinSupports<<< numBlocks, threadsPerBlock >>>(d_bin, d_bin_grad, k_bin, k_bin_grad, n);


/* ********* RESET ALL VARIABLES FOR THE CG PROJECTION ************ */
  iterCG=0;
  for (int j=0; j<16; j++) {
	residNorm_prevCG[j]=0;
	residNorm_evolutionCG[j]=1.0f;
  }

  float residNorm_diffCG = 1.0f;
  residNorm_prevCG[15]=residNorm_prev[15];

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CSMPSP loop CG prep");

  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv for resid_update in CSMPSP loop CG prep");  

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in  CSMPSP loop CG prep");

  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in CG prep");

  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CSMPSP loop CG prep");

  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CSMPSP loop CG prep");

  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in CSMPSP_S_smv loop");
  errCG = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in CSMPSP_S_smv loop");
  residNorm_prevCG[15]=errCG;

  while ((errCG > resid_tolCG) & (iterCG < maxiterCG) & (residNorm_diffCG > resid_tol2))
  	{

  	RestrictedCG_S_smv(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prevCG, num_bins, k, m, n, nz, &mu, errCG, k_bin, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
	SAFEcuda("RestrictedCG_S_smv in CSMPSP loop CG loop");

  	errCG = residNorm_prevCG[15];
  	iterCG++;

       // check for convergence 
  	for (int j = 0; j<15; j++) {
		residNorm_evolutionCG[j]=residNorm_evolutionCG[j+1];
  		}
  	residNorm_evolutionCG[15] = residNorm_prevCG[14]-residNorm_prevCG[15];

	residNorm_diffCG = max_list(residNorm_evolutionCG, 16); //10.0f was here for some reason?

  	}


  max_value = MaxMagnitude(d_vec,n);
  minVal = 0.0f;

  slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in CSMPSP loop");
  // second max_value entry to force binning.

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CSMPSP loop");

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CSMPSP initializiation");

  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in CSMPSP");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CSMPSP");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in CSMPSP");

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_prev[j]=residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }

  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if ( iter > (125) ){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f){ fail = 1;
	}
  }


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }

  *p_iter = iter;
  *p_time_sum = time_sum;

}




/*
****************************************************************************************
**                                  CSMPSP_S_gen                                      **
**      Single Precision Compressive Sampling Matching Pursuit / Subspace Pursuit     **
**                               for general matrices                                 **
****************************************************************************************
*/

inline void CSMPSP_S_gen(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int *d_bin_grad, int * d_bin_counters_grad, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/


  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;
  int k_bin_grad = 0;


  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in IHT initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT initialization");

  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


  // Some variables for the CG projection step
  float resid_tolCG = resid_resid_tol*resid_resid_tol;     // this is a resid_tolerance for the convergnece of the CG projection step in each iteration
  int iterCG = 0;
  int maxiterCG = 15;       // maximum number of CG iterations per iteration of the algorithm
  float errCG = 1;
  float resid_tol2 = 10*resid_tolCG;

  float residNorm_prevCG[16];
  float residNorm_evolutionCG[16];
  

// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif

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

/* ********* SET ALL VARIABLES FOR THE CG PROJECTION ************ */
  iterCG=0;
  for (int j=0; j<16; j++) {
	residNorm_prevCG[j]=0;
	residNorm_evolutionCG[j]=1.0f;
  }

  float residNorm_diffCG = 1.0f;
  residNorm_prevCG[15]=residNorm_prev[15];

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CSMPSP_S_gen Set Linear Bins");

  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_smv for resid_update CSMPSP_S_gen Set Linear Bins");  

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CSMPSP_S_gen Set Linear Bins");

  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in CSMPSP_S_gen Set Linear Bins");

  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CSMPSP_S_gen Set Linear Bins");

  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CSMPSP_S_gen Set Linear Bins");

  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in CSMPSP_S_gen Set Linear Bins");
  errCG = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in CSMPSP_S_gen Set Linear Bins");
  residNorm_prevCG[15]=errCG;


  while ((errCG > resid_tolCG) & (iterCG < maxiterCG) & (residNorm_diffCG > resid_tol2))
  	{

  	RestrictedCG_S_gen(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, residNorm_prevCG, num_bins, k, m, n, &mu, errCG, k_bin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);

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
  residNorm_prev[15] = err;
  iter++;		// We consider the initial CG projection of CSMPSP as an iteration.

/* 
*******************
** Main Steepest Descent Loop **
*******************
*/

  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in CG prep");

  max_value = MaxMagnitude(grad, n);
  minVal = 0.0f;

  slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(grad, d_bin_grad, d_bin_counters_grad, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin_grad, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in IHT loop");

  joinSupports<<< numBlocks, threadsPerBlock >>>(d_bin, d_bin_grad, k_bin, k_bin_grad, n);



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

  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_smv for resid_update in CSMPSP loop CG prep");  

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in  CSMPSP loop CG prep");

  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in CG prep");

  Threshold(grad, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CG preparation");

  cublasScopy(n, grad, 1, grad_previous, 1);
  SAFEcublas("cublasScopy in CG prep");

  mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot in CG prep in CSMPSP_S_gen loop");
  errCG = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot for err in CG prep in CSMPSP_S_gen loop");
  residNorm_prevCG[15]=errCG;

  while ((errCG > resid_tolCG) & (iterCG < maxiterCG) & (residNorm_diffCG > resid_tol2))
  	{

  	RestrictedCG_S_gen(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, residNorm_prevCG, num_bins, k, m, n, &mu, errCG, k_bin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);

  	errCG = residNorm_prevCG[15];
  	iterCG++;
  
       // check for convergence 
  	for (int j = 0; j<15; j++) {
		residNorm_evolutionCG[j]=residNorm_evolutionCG[j+1];
  		}
  	residNorm_evolutionCG[15] = residNorm_prevCG[14]-residNorm_prevCG[15];

	residNorm_diffCG = 10.0f;//max_list(residNorm_evolutionCG, 16);

  	}


  max_value = MaxMagnitude(d_vec,n);
  minVal = 0.0f;

  slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in IHT loop");
  // second max_value entry to force binning.

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in IHT loop");

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT initializiation");

  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in CG preparation");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("cublasSaxpy in CG prep");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in IHT initialization");

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_prev[j]=residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }

  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if ( iter > (125) ){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f){ fail = 1;
	}
  }


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }

  *p_iter = iter;
  *p_time_sum = time_sum;

}




/*
****************************************************************************************
**                                  CGIHT_S_gen                                       **
**               Conjugate Gradient Iterative Hard Thresholding                       **
**                               for general matrices                                 **
****************************************************************************************
*/



inline void CGIHT_S_gen(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float * grad_prev_thres, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int *d_bin_grad, int * d_bin_counters_grad, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int restartFlag, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/



  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;

  float alpha = 0.25f;  // normally 0.25f;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in NIHT_S_gen initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in NIHT_S_gen initialization");

  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// **********************
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// **********************

#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_gen(d_vec, d_y, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in NIHT_S_gen Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in NIHT_S_gen Set Linear Bins");


  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in NIHT_S_gen Set Linear Bins");

/*
*************************************************
**  Calculate the residual A^T (dy - A d_vec)  **
*************************************************
*/

  // calculate r^0 and the thresholded version of r^0 in grad_previous and grad_prev_thres respectively

  // resid already contains d_y
  //cublasScopy(m, d_y, 1, resid, 1);
  //SAFEcublas("cublasScopy in RestrictedSD_S_gen");

  // resid_update is used to store A * d_vec so that it can be used to compute the resid
  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in RestrictedSD_S_gen");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("First cublasSaxpy in RestrictedSD_S_gen");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in RestrictedSD_S_gen");
  residNorm_prev[15]=err;

  // Now need to compute the gradient and its restriction 
  // to where d_bin is larger than k_bin
  AT_gen(grad_previous, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in RestrictedSD_S_gen");

  // grad_prev_thres is being used to store a thresholded version of grad
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)grad_prev_thres, n);
  SAFEcuda("zero_vector_float in RestrictedSD_S_gen");

  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad_previous, (float*)grad_prev_thres, (int*)d_bin, k_bin, n);
  SAFEcuda("threshold_one of grad in RestrictedSD_S_gen");


// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];


/* 
********************************
** Main Steepest Descent Loop **
********************************
*/
  
  // a flag which when equal to 1 indicates beta in CG should be set to zero
  int beta_to_zero_flag = 1;

  // a flag equal to 1 indicates the support has changed, 0 that it is the same
  int suppChange_flag=1;  // this must start with 1.
  int k_bin_prev = 0;
  

  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) ) 
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  if (restartFlag){

    maxChange = RestrictedCGwithSupportEvolution_S_gen(d_vec, d_vec_thres, grad, grad_previous, grad_prev_thres, d_y, resid, resid_update, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, &mu, num_bins, k, m, n, beta_to_zero_flag, suppChange_flag, k_bin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
    SAFEcuda("RestrictedSD in NIHT_S_gen loop");

  } else {  // else for the if (restartFlag)
      if (suppChange_flag){
        maxChange = UnrestrictedCG_S_gen(d_vec, d_vec_thres, grad, grad_previous, grad_prev_thres, d_y, resid, resid_update, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, &mu, num_bins, k, m, n, beta_to_zero_flag, suppChange_flag, k_bin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
      } else {  // else for the if (suppChange_flag)
        maxChange = RestrictedCGwithSupportEvolution_S_gen(d_vec, d_vec_thres, grad, grad_previous, grad_prev_thres, d_y, resid, resid_update, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, &mu, num_bins, k, m, n, beta_to_zero_flag, suppChange_flag, k_bin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
      } // ends the else for if (suppChange_flag)
  }  // ends the else for the if (restartFlag)

  // after the first iteration compute beta
  beta_to_zero_flag = 0;

  if (minVal <= maxChange) { 
    max_value = MaxMagnitude(d_vec,n);
    slope = ((num_bins-1)/(max_value));

    // Save the previous support set information.
    cudaMemcpy(d_bin_grad, d_bin, n*sizeof(int), cudaMemcpyDeviceToDevice);
    k_bin_prev = k_bin;

 //   if (residNorm_prev[15]<0.05f) k_supp=k;

    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in NIHT_S_gen loop");
  
    // We now check if the support changed.  
    // We are using the pre-allocated d_bin_counters_grad as the supp_flag on the device.
    checkSupportEvolution<<< numBlocks, threadsPerBlock >>>(d_bin, d_bin_grad, k_bin, k_bin_prev, d_bin_counters_grad, n);
    cudaMemcpy(&suppChange_flag,d_bin_counters_grad,sizeof(int),cudaMemcpyDeviceToHost);

    // set suppChange_flag to 1 if it is greater than 0
    if (suppChange_flag) {
      suppChange_flag=1;

      // If restarting is active, the next iteration uses beta=0.
      // Otherwise, we must update the restricted search direction.

      if (restartFlag) {
        beta_to_zero_flag = 1;
      }  else {
        // Since the support changed and we are not restarting,
        // the previous search direction in grad_prev_thres is thresholded to the wrong support set.
        // Restrict the CG search direction stored in grad_previous to the new support and put it in grad_prev_thres.
        zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)grad_prev_thres, n);
        SAFEcuda("zero_vector_float in RestrictedSD_S_gen");
  
        threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad_previous, (float*)grad_prev_thres, (int*)d_bin, k_bin, n);
        SAFEcuda("threshold_one of grad in RestrictedSD_S_gen");

      } // ends the else of the if (restartFlag)

    } // ends the if (suppChange_flag)

  } else {   // else for if (minVal <= maxChange)
    suppChange_flag=0;
    minVal = minVal - maxChange; 
  } //ends the else of the if (minVal <= maxChange)

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in NIHT_S_gen loop");

  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if (iter>749){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1; 
  }


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }

  *p_iter = iter;
  *p_time_sum = time_sum;

}



/*
****************************************************************************************
**                                  CGIHT_S_dct                                       **
**               Conjugate Gradient Iterative Hard Thresholding                       **
**                               for general matrices                                 **
****************************************************************************************
*/



inline void CGIHT_S_dct(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float * grad_prev_thres, float *d_y, float *resid, float *resid_update, int *d_rows, int *d_bin, int * d_bin_counters, int *d_bin_grad, int * d_bin_counters_grad, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int restartFlag, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/



  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;

  float alpha = 0.25f;  // normally 0.25f;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in NIHT_S_dct initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in NIHT_S_dct initialization");

  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// **********************
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// **********************

#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_dct(d_vec, d_y, n, m, d_rows);
  SAFEcuda("AT_dct in NIHT_S_dct Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in NIHT_S_dct Set Linear Bins");


  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in NIHT_S_dct Set Linear Bins");

/*
*************************************************
**  Calculate the residual A^T (dy - A d_vec)  **
*************************************************
*/

  // calculate r^0 and the thresholded version of r^0 in grad_previous and grad_prev_thres respectively

  // resid already contains d_y
  //cublasScopy(m, d_y, 1, resid, 1);
  //SAFEcublas("cublasScopy in RestrictedSD_S_dct");

  // resid_update is used to store A * d_vec so that it can be used to compute the resid
  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct in RestrictedSD_S_dct");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("First cublasSaxpy in RestrictedSD_S_dct");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in RestrictedSD_S_dct");
  residNorm_prev[15]=err;

  // Now need to compute the gradient and its restriction 
  // to where d_bin is larger than k_bin
  AT_dct(grad_previous, resid, n, m, d_rows);
  SAFEcuda("AT_dct in RestrictedSD_S_dct");

  // grad_prev_thres is being used to store a thresholded version of grad
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)grad_prev_thres, n);
  SAFEcuda("zero_vector_float in RestrictedSD_S_dct");

  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad_previous, (float*)grad_prev_thres, (int*)d_bin, k_bin, n);
  SAFEcuda("threshold_one of grad in RestrictedSD_S_dct");


// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];



/* 
********************************
** Main Steepest Descent Loop **
********************************
*/
  
  // a flag which when equal to 1 indicates beta in CG should be set to zero
  int beta_to_zero_flag = 1;

  // a flag equal to 1 indicates the support has changed, 0 that it is the same
  int suppChange_flag=1;  // this must start with 1.
  int k_bin_prev = 0;

  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  if (restartFlag){

    maxChange = RestrictedCGwithSupportEvolution_S_dct(d_vec, d_vec_thres, grad, grad_previous, grad_prev_thres, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, &mu, num_bins, k, m, n, beta_to_zero_flag, suppChange_flag, k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
    SAFEcuda("RestrictedSD in NIHT_S_dct loop");

  } else {  // else for the if (restartFlag)
      if (suppChange_flag){
        maxChange = UnrestrictedCG_S_dct(d_vec, d_vec_thres, grad, grad_previous, grad_prev_thres, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, &mu, num_bins, k, m, n, beta_to_zero_flag, suppChange_flag, k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
      } else {  // else for the if (suppChange_flag)
        maxChange = RestrictedCGwithSupportEvolution_S_dct(d_vec, d_vec_thres, grad, grad_previous, grad_prev_thres, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, &mu, num_bins, k, m, n, beta_to_zero_flag, suppChange_flag, k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
      } // ends the else for if (suppChange_flag)
  }  // ends the else for the if (restartFlag)

  // after the first iteration compute beta
  beta_to_zero_flag = 0;

  if (minVal <= maxChange) { 
    max_value = MaxMagnitude(d_vec,n);
    slope = ((num_bins-1)/(max_value));

    // Save the previous support set information.
    cudaMemcpy(d_bin_grad, d_bin, n*sizeof(int), cudaMemcpyDeviceToDevice);
    k_bin_prev = k_bin;

    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in NIHT_S_dct loop");
  
    // We now check if the support changed.  
    // We are using the pre-allocated d_bin_counters_grad as the supp_flag on the device.
    checkSupportEvolution<<< numBlocks, threadsPerBlock >>>(d_bin, d_bin_grad, k_bin, k_bin_prev, d_bin_counters_grad, n);
    cudaMemcpy(&suppChange_flag,d_bin_counters_grad,sizeof(int),cudaMemcpyDeviceToHost);

    // set suppChange_flag to 1 if it is greater than 0
    if (suppChange_flag) {
      suppChange_flag=1;

      // If restarting is active, the next iteration uses beta=0.
      // Otherwise, we must update the restricted search direction.

      if (restartFlag) {
        beta_to_zero_flag = 1;
      }  else {
        // Since the support changed and we are not restarting,
        // the previous search direction in grad_prev_thres is thresholded to the wrong support set.
        // Restrict the CG search direction stored in grad_previous to the new support and put it in grad_prev_thres.
        zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)grad_prev_thres, n);
        SAFEcuda("zero_vector_float in RestrictedSD_S_dct");
  
        threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad_previous, (float*)grad_prev_thres, (int*)d_bin, k_bin, n);
        SAFEcuda("threshold_one of grad in RestrictedSD_S_dct");

      } // ends the else of the if (restartFlag)

    } // ends the if (suppChange_flag)

  } else {   // else for if (minVal <= maxChange)
    suppChange_flag=0;
    minVal = minVal - maxChange; 
  } //ends the else of the if (minVal <= maxChange)

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in NIHT_S_dct loop");

  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if (iter>749){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1;
  }

  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }

  *p_iter = iter;
  *p_time_sum = time_sum;

}



/*
****************************************************************************************
**                                  CGIHT_S_smv                                       **
**               Conjugate Gradient Iterative Hard Thresholding                       **
**                               for general matrices                                 **
****************************************************************************************
*/



inline void CGIHT_S_smv(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float * grad_prev_thres, float *d_y, float *resid, float *resid_update, int *d_rows, int *d_cols, float *d_vals, int *d_bin, int * d_bin_counters, int *d_bin_grad, int * d_bin_counters_grad, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int nz, const int restartFlag, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/



  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;

  float alpha = 0.25f;  // normally 0.25f;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in NIHT_S_smv initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in NIHT_S_smv initialization");

  float err_start = err;

  float resid_resid_tol = resid_tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }


// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;


// **********************
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// **********************

#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_smv(d_vec, d_y, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in NIHT_S_smv Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in NIHT_S_smv Set Linear Bins");


  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in NIHT_S_smv Set Linear Bins");

/*
*************************************************
**  Calculate the residual A^T (dy - A d_vec)  **
*************************************************
*/

  // calculate r^0 and the thresholded version of r^0 in grad_previous and grad_prev_thres respectively

  // resid already contains d_y
  //cublasScopy(m, d_y, 1, resid, 1);
  //SAFEcublas("cublasScopy in RestrictedSD_S_smv");

  // resid_update is used to store A * d_vec so that it can be used to compute the resid
  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in RestrictedSD_S_smv");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("First cublasSaxpy in RestrictedSD_S_smv");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in RestrictedSD_S_smv");
  residNorm_prev[15]=err;

  // Now need to compute the gradient and its restriction 
  // to where d_bin is larger than k_bin
  AT_smv(grad_previous, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in RestrictedSD_S_smv");

  // grad_prev_thres is being used to store a thresholded version of grad
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)grad_prev_thres, n);
  SAFEcuda("zero_vector_float in RestrictedSD_S_smv");

  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad_previous, (float*)grad_prev_thres, (int*)d_bin, k_bin, n);
  SAFEcuda("threshold_one of grad in RestrictedSD_S_smv");


// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];



/* 
********************************
** Main Steepest Descent Loop **
********************************
*/
  
  // a flag which when equal to 1 indicates beta in CG should be set to zero
  int beta_to_zero_flag = 1;

  // a flag equal to 1 indicates the support has changed, 0 that it is the same
  int suppChange_flag=1;  // this must start with 1.
  int k_bin_prev = 0;

  while ( (err > resid_resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  if (restartFlag){

    maxChange = RestrictedCGwithSupportEvolution_S_smv(d_vec, d_vec_thres, grad, grad_previous, grad_prev_thres, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, &mu, num_bins, k, m, n, nz, beta_to_zero_flag, suppChange_flag, k_bin, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
    SAFEcuda("RestrictedSD in NIHT_S_smv loop");

  } else {  // else for the if (restartFlag)
      if (suppChange_flag){
        maxChange = UnrestrictedCG_S_smv(d_vec, d_vec_thres, grad, grad_previous, grad_prev_thres, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, &mu, num_bins, k, m, n, nz, beta_to_zero_flag, suppChange_flag, k_bin, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
      } else {  // else for the if (suppChange_flag)
        maxChange = RestrictedCGwithSupportEvolution_S_smv(d_vec, d_vec_thres, grad, grad_previous, grad_prev_thres, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, &mu, num_bins, k, m, n, nz, beta_to_zero_flag, suppChange_flag, k_bin, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
      } // ends the else for if (suppChange_flag)
  }  // ends the else for the if (restartFlag)

  // after the first iteration compute beta
  beta_to_zero_flag = 0;

  if (minVal <= maxChange) { 
    max_value = MaxMagnitude(d_vec,n);
    slope = ((num_bins-1)/(max_value));

    // Save the previous support set information.
    cudaMemcpy(d_bin_grad, d_bin, n*sizeof(int), cudaMemcpyDeviceToDevice);
    k_bin_prev = k_bin;

    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
    SAFEcuda("FindSupportSet in NIHT_S_smv loop");
  
    // We now check if the support changed.  
    // We are using the pre-allocated d_bin_counters_grad as the supp_flag on the device.
    checkSupportEvolution<<< numBlocks, threadsPerBlock >>>(d_bin, d_bin_grad, k_bin, k_bin_prev, d_bin_counters_grad, n);
    cudaMemcpy(&suppChange_flag,d_bin_counters_grad,sizeof(int),cudaMemcpyDeviceToHost);

    // set suppChange_flag to 1 if it is greater than 0
    if (suppChange_flag) {
      suppChange_flag=1;

      // If restarting is active, the next iteration uses beta=0.
      // Otherwise, we must update the restricted search direction.

      if (restartFlag) {
        beta_to_zero_flag = 1;
      }  else {
        // Since the support changed and we are not restarting,
        // the previous search direction in grad_prev_thres is thresholded to the wrong support set.
        // Restrict the CG search direction stored in grad_previous to the new support and put it in grad_prev_thres.
        zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)grad_prev_thres, n);
        SAFEcuda("zero_vector_float in RestrictedSD_S_smv");
  
        threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad_previous, (float*)grad_prev_thres, (int*)d_bin, k_bin, n);
        SAFEcuda("threshold_one of grad in RestrictedSD_S_smv");

      } // ends the else of the if (restartFlag)

    } // ends the if (suppChange_flag)

  } else {   // else for if (minVal <= maxChange)
    suppChange_flag=0;
    minVal = minVal - maxChange; 
  } //ends the else of the if (minVal <= maxChange)

  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in NIHT_S_smv loop");

  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);


// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if (iter>749){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1;
  }

  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }

  *p_iter = iter;
  *p_time_sum = time_sum;

}





/*
******************************************************************************
**                      CGIHTprojected_S_gen                                **
**      Single Precision CGIHT with projected condition                     **
**                    with general matrices                                 **
******************************************************************************
*/

inline void CGIHTprojected_S_gen(float *d_vec, float *d_vec_thres, float * grad, float * grad_prev_thres, float * d_p_thres, float * d_Ap_thres, float * d_vec_diff, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const float proj_frac_tol, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum,  dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/

  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CGIHTprojected_S_gen initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in CGIHTprojected_S_gen initialization");

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

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_gen(d_vec, d_y, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in CGIHTprojected_S_gen Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in CGIHTprojected_S_gen Set Linear Bins");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CGIHTprojected_S_gen Set Linear Bins");

  // y - Ax -> resid
  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm); // Ax -> resid_update
  SAFEcuda("A_gen in CGIHTprojected_S_gen Set Linear Bins");
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);    //y - Ax -> resid, d_y already copied into resid
  SAFEcuda("cublasSaxpy in CGIHTprojected_S_gen Set Linear Bins");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in in CGIHTprojected_S_gen Set Linear Bins");
  residNorm_prev[15]=err;

  // AT(y-Ax) -> grad
  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in CGIHTprojected_S_gen Set Linear Bins");

  // a thres version of grad -> d_vec_thres
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin,
  k_bin, n);
  SAFEcuda("threshold_one of grad in CGIHTprojected_S_gen Set Linear Bins");

  // Agrad_thres -> resid_update
  A_gen(resid_update, d_vec_thres, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("Agen_thres in CGIHTprojected_S_gen Set Linear Bins");

  for (int j = 0; j < 15; j++) {
    residNorm_evolution[j] = residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

/* 
********************************
** Main Steepest Descent Loop **
********************************
*/
  // a flag which indicates switching to SD step when equal to 1
  int suppChange_flag = 1;

  float beta, time, proj_frac, tmp;

  while ( (err > resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  if (suppChange_flag == 1) { // SD step
    // compute step length
    mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1); // size n
    tmp = cublasSdot(m, resid_update, 1, resid_update, 1); // size m
    SAFEcublas("cublasSdot in CGIHTprojected_S_gen Main Loop");
    if (mu < 400*tmp)
      mu = mu/tmp;
    else
      mu = 1;

    // x = x + mu * grad -> d_vec
    cublasSaxpy(n, mu, grad, 1, d_vec, 1);
    SAFEcublas("cublasSaxpy in CGIHTprojected_S_gen Main Loop");

    // threshold x
    minVal = 0.0f;
    maxChange = 1.0f;
    max_value = MaxMagnitude(d_vec,n);
    slope = (num_bins-1)/max_value;

    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, 
                            &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin,threadsPerBlock_bin);
    SAFEcuda("FindSupportSet in NIHT_S_gen Main Loop");

    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in NIHT_S_gen loop");

    // y - Ax -> resid
    A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm); // Ax -> resid_update
    SAFEcuda("A_gen in CGIHTprojected_S_gen Main Loop");

    cublasScopy(m, d_y, 1, resid, 1); // d_y -> resid, size m
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);    //y - Ax -> resid
    SAFEcublas("cublasScopy&cublasSaxpy in CGIHTprojected_S_gen Main Loop");

    // AT(y-Ax) -> grad
    AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
    SAFEcuda("AT_gen in CGIHTprojected_S_gen Main Loop");

    // grad_thres -> d_vec_thres
    zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
    threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin,
    k_bin, n);
    SAFEcuda("threshold_one of grad in CGIHTprojected_S_gen Main Loop");

    // Agrad_thres -> resid_update
    A_gen(resid_update, d_vec_thres, d_A, m, n, numBlocksm, threadsPerBlockm);
    SAFEcuda("Agen_thres in CGIHTprojected_S_gen Main Loop");

    // p_thres = grad_thres -> d_p_thres, Ap_thres = Ar_thres -> d_Ap_thres
    cublasScopy(n, d_vec_thres, 1, d_p_thres,1);
    cublasScopy(m, resid_update, 1, d_Ap_thres, 1);
    SAFEcublas("cublasScopy in CGIHTprojected Main Loop");
  }
  else { // CG update on the previous support set
    // compute step length
    mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1); // size n
    tmp = cublasSdot(m, d_Ap_thres, 1, d_Ap_thres, 1); // size m
    SAFEcublas("cublasSdot in CGIHTprojected_S_gen Main Loop");
    if (mu < 400*tmp)
      mu = mu/tmp;
    else
      mu = 1;

     // x = x + mu * p in threshold version
    cublasSaxpy(n, mu, d_p_thres, 1, d_vec, 1);
    SAFEcublas("cublasSaxpy in CGIHTprojected_S_gen Main Loop");   

    // y - Ax -> resid
    A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm); // Ax -> resid_update
    SAFEcuda("A_gen in CGIHTprojected_S_gen Main Loop");

    cublasScopy(m, d_y, 1, resid, 1); // d_y -> resid, size m
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);    //y - Ax -> resid
    SAFEcublas("cublasScopy&cublasSaxpy in CGIHTprojected_S_gen Main Loop");

    // grad_thres -> grad_prev_thres: copy d_vec_thres to grad_previous_thres
    cublasScopy(n, d_vec_thres, 1, grad_prev_thres,1);
    SAFEcublas("cublasScopy in CGIHTprojected_S_gen Main Loop");

    // AT(y-Ax) -> grad
    AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
    SAFEcuda("AT_gen in CGIHTprojected_S_gen Main Loop");

    // grad_thres -> d_vec_thres
    zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
    threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin,
    k_bin, n);
    SAFEcuda("threshold_one of grad in CGIHTprojected_S_gen Main Loop");

    // Agrad_thres -> resid_update
    A_gen(resid_update, d_vec_thres, d_A, m, n, numBlocksm, threadsPerBlockm);
    SAFEcuda("Agen_thres in CGIHTprojected_S_gen Main Loop");      

    // compute beta
    beta = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
    tmp = cublasSdot(n, grad_prev_thres, 1, grad_prev_thres, 1);
    SAFEcublas("cublasSdot in CGIHTprojected_S_gen Main Loop");
    if (beta < 400 * tmp)
      beta = beta/tmp;
    else 
      beta = 0;
   
    // update p_thres and Ap_thres
    // p_thres = r_thres + beta * p_thres -> d_p_thres
    cublasSscal(n, beta, d_p_thres, 1); // size n
    cublasSaxpy(n, 1, d_vec_thres, 1, d_p_thres, 1); // size n
    SAFEcublas("cublas updating p_thres");

    // Ap_thres = Ar_thres + beta * Ap_thres
    cublasSscal(m, beta, d_Ap_thres, 1);
    cublasSaxpy(m, 1, resid_update, 1, d_Ap_thres, 1);  
  }

  // compute proj_frac = norm(grad-p_thres,2)/norm(grad_thres,2) 
  // p_thres in d_p_thres, grad_thres in d_vec_thres
  cublasScopy(n, grad, 1, d_vec_diff, 1); // size n
  cublasSaxpy(n, -1, d_p_thres, 1, d_vec_diff, 1); // size n
  proj_frac = cublasSnrm2(n, d_vec_diff, 1);
  tmp = cublasSnrm2(n, d_vec_thres, 1);

  // set suppChange_flag
  if (proj_frac < 1000 * tmp) {
     proj_frac = proj_frac/tmp;   
     if (proj_frac > proj_frac_tol)
       suppChange_flag = 1;
     else
       suppChange_flag = 0;    
  }
  else 
     suppChange_flag = 1; 

  // check for convergence 
  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in CGIHTprojected_S_gen Main Loop");

  for (int j = 0; j<15; j++) {
	residNorm_prev[j] = residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if ( iter > 749 ){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1;
  }


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  } // end of while

  *p_iter = iter;
  *p_time_sum = time_sum;

} // end of CGIHTprojected_S_gen


/*
******************************************************************************
**                      CGIHTprojected_S_dct                                **
**      Single Precision CGIHT with projected condition                     **
**                    with dct matrices                                     **
******************************************************************************
*/

inline void CGIHTprojected_S_dct(float *d_vec, float *d_vec_thres, float * grad, float * grad_prev_thres, float * d_p_thres, float * d_Ap_thres, float * d_vec_diff, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const float proj_frac_tol, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum,  dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/
  //printf("proj_frac_tol = %f\n", proj_frac_tol);  

  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CGIHTprojected_S_dct initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in CGIHTprojected_S_dct initialization");

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

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_dct(d_vec, d_y, n, m, d_rows);
  SAFEcuda("AT_dct in CGIHTprojected_S_dct Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in CGIHTprojected_S_dct Set Linear Bins");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CGIHTprojected_S_dct Set Linear Bins");

  // y - Ax -> resid
  A_dct(resid_update, d_vec, n, m, d_rows); // Ax -> resid_update
  SAFEcuda("A_dct in CGIHTprojected_S_dct Set Linear Bins");
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);    //y - Ax -> resid, d_y already copied into resid
  SAFEcuda("cublasSaxpy in CGIHTprojected_S_dct Set Linear Bins");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in in CGIHTprojected_S_dct Set Linear Bins");
  residNorm_prev[15]=err;

  // AT(y-Ax) -> grad
  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in CGIHTprojected_S_dct Set Linear Bins");

  // a thres version of grad -> d_vec_thres
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin,
  k_bin, n);
  SAFEcuda("threshold_one of grad in CGIHTprojected_S_dct Set Linear Bins");

  // Agrad_thres -> resid_update
  A_dct(resid_update, d_vec_thres, n, m, d_rows);
  SAFEcuda("Agen_thres in CGIHTprojected_S_dct Set Linear Bins");

  for (int j = 0; j < 15; j++) {
    residNorm_evolution[j] = residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

/* 
********************************
** Main Steepest Descent Loop **
********************************
*/
  // a flag which indicates switching to SD step when equal to 1
  int suppChange_flag = 1;

  float beta, time, proj_frac, tmp;

  while ( (err > resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  if (suppChange_flag == 1) { // SD step
    // compute step length
    mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1); // size n
    tmp = cublasSdot(m, resid_update, 1, resid_update, 1); // size m
    SAFEcublas("cublasSdot in CGIHTprojected_S_dct Main Loop");
    if (mu < 400*tmp)
      mu = mu/tmp;
    else
      mu = 1;

    // x = x + mu * grad -> d_vec
    cublasSaxpy(n, mu, grad, 1, d_vec, 1);
    SAFEcublas("cublasSaxpy in CGIHTprojected_S_dct Main Loop");

    // threshold x
    minVal = 0.0f;
    maxChange = 1.0f;
    max_value = MaxMagnitude(d_vec,n);
    slope = (num_bins-1)/max_value;

    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, 
                            &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin,threadsPerBlock_bin);
    SAFEcuda("FindSupportSet in NIHT_S_gen Main Loop");

    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in NIHT_S_gen loop");

    // y - Ax -> resid
    A_dct(resid_update, d_vec, n, m, d_rows); // Ax -> resid_update
    SAFEcuda("A_dct in CGIHTprojected_S_dct Main Loop");

    cublasScopy(m, d_y, 1, resid, 1); // d_y -> resid, size m
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);    //y - Ax -> resid
    SAFEcublas("cublasScopy&cublasSaxpy in CGIHTprojected_S_dct Main Loop");

    // AT(y-Ax) -> grad
    AT_dct(grad, resid, n, m, d_rows);
    SAFEcuda("AT_dct in CGIHTprojected_S_dct Main Loop");

    // grad_thres -> d_vec_thres
    zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
    threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin,
    k_bin, n);
    SAFEcuda("threshold_one of grad in CGIHTprojected_S_dct Main Loop");

    // Agrad_thres -> resid_update
    A_dct(resid_update, d_vec_thres, n, m, d_rows);
    SAFEcuda("Agen_thres in CGIHTprojected_S_dct Main Loop");

    // p_thres = grad_thres -> d_p_thres, Ap_thres = Ar_thres -> d_Ap_thres
    cublasScopy(n, d_vec_thres, 1, d_p_thres,1);
    cublasScopy(m, resid_update, 1, d_Ap_thres, 1);
    SAFEcublas("cublasScopy in CGIHTprojected Main Loop");
  }
  else { // CG update on the previous support set
    // compute step length
    mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1); // size n
    tmp = cublasSdot(m, d_Ap_thres, 1, d_Ap_thres, 1); // size m
    SAFEcublas("cublasSdot in CGIHTprojected_S_dct Main Loop");
    if (mu < 400*tmp)
      mu = mu/tmp;
    else
      mu = 1;

     // x = x + mu * p in threshold version
    cublasSaxpy(n, mu, d_p_thres, 1, d_vec, 1);
    SAFEcublas("cublasSaxpy in CGIHTprojected_S_dct Main Loop");   

    // y - Ax -> resid
    A_dct(resid_update, d_vec, n, m, d_rows); // Ax -> resid_update
    SAFEcuda("A_dct in CGIHTprojected_S_dct Main Loop");

    cublasScopy(m, d_y, 1, resid, 1); // d_y -> resid, size m
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);    //y - Ax -> resid
    SAFEcublas("cublasScopy&cublasSaxpy in CGIHTprojected_S_dct Main Loop");

    // grad_thres -> grad_prev_thres: copy d_vec_thres to grad_previous_thres
    cublasScopy(n, d_vec_thres, 1, grad_prev_thres,1);
    SAFEcublas("cublasScopy in CGIHTprojected_S_dct Main Loop");

    // AT(y-Ax) -> grad
    AT_dct(grad, resid, n, m, d_rows);
    SAFEcuda("AT_dct in CGIHTprojected_S_dct Main Loop");

    // grad_thres -> d_vec_thres
    zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
    threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin,
    k_bin, n);
    SAFEcuda("threshold_one of grad in CGIHTprojected_S_dct Main Loop");

    // Agrad_thres -> resid_update
    A_dct(resid_update, d_vec_thres, n, m, d_rows);
    SAFEcuda("Agen_thres in CGIHTprojected_S_dct Main Loop");      

    // compute beta
    beta = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
    tmp = cublasSdot(n, grad_prev_thres, 1, grad_prev_thres, 1);
    SAFEcublas("cublasSdot in CGIHTprojected_S_dct Main Loop");
    if (beta < 400 * tmp)
      beta = beta/tmp;
    else 
      beta = 0;
   
    // update p_thres and Ap_thres
    // p_thres = r_thres + beta * p_thres -> d_p_thres
    cublasSscal(n, beta, d_p_thres, 1); // size n
    cublasSaxpy(n, 1, d_vec_thres, 1, d_p_thres, 1); // size n
    SAFEcublas("cublas updating p_thres");

    // Ap_thres = Ar_thres + beta * Ap_thres
    cublasSscal(m, beta, d_Ap_thres, 1);
    cublasSaxpy(m, 1, resid_update, 1, d_Ap_thres, 1);  
  }

  // compute proj_frac = norm(grad-p_thres,2)/norm(grad_thres,2) 
  // p_thres in d_p_thres, grad_thres in d_vec_thres
  cublasScopy(n, grad, 1, d_vec_diff, 1); // size n
  cublasSaxpy(n, -1, d_p_thres, 1, d_vec_diff, 1); // size n
  proj_frac = cublasSnrm2(n, d_vec_diff, 1);
  tmp = cublasSnrm2(n, d_vec_thres, 1);
  
  // set suppChange_flag
  if (proj_frac < 1000 * tmp) {
     proj_frac = proj_frac/tmp;   
     if (proj_frac > proj_frac_tol)
       suppChange_flag = 1;
     else
       suppChange_flag = 0;    
  }
  else 
     suppChange_flag = 1; 

  // check for convergence 
  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in CGIHTprojected_S_dct Main Loop");

  for (int j = 0; j<15; j++) {
	residNorm_prev[j] = residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if ( iter > 749 ){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1;
  }


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  } // end of while

  *p_iter = iter;
  *p_time_sum = time_sum;

} // end of CGIHTprojected_S_dct


/*
******************************************************************************
**                      CGIHTprojected_S_smv                                **
**      Single Precision CGIHT with projected condition                     **
**                    with general matrices                                 **
******************************************************************************
*/

inline void CGIHTprojected_S_smv(float *d_vec, float *d_vec_thres, float * grad, float * grad_prev_thres, float * d_p_thres, float * d_Ap_thres, float * d_vec_diff, float *d_y, float *resid, float *resid_update, int * d_rows, int * d_cols, float * d_vals, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const float proj_frac_tol, const int num_bins, const int k, const int m, const int n, const int nz, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum,  dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/ 
  int iter = *p_iter;
  float time_sum=*p_time_sum;
  int k_bin = 0;


  float alpha = 0.25f; // normally 0.25;
  int MaxBin=(int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in CGIHTprojected_S_smv initializiation");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in CGIHTprojected_S_smv initialization");

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

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
*************************
** Set the Linear Bins **
*************************
*/


  AT_smv(d_vec, d_y, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in CGIHTprojected_S_smv Set Linear Bins");

  float max_value = MaxMagnitude(d_vec,n);

  float slope = ((num_bins-1)/(max_value));


  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, max_value, &minVal, &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin); 
  SAFEcuda("FindSupportSet in CGIHTprojected_S_smv Set Linear Bins");
  // max_value input is normally maxChange, but for the first call
  // we want to force binning.

  // Set the initial guess
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in CGIHTprojected_S_smv Set Linear Bins");

  // y - Ax -> resid
  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp); // Ax -> resid_update
  SAFEcuda("A_smv in CGIHTprojected_S_smv Set Linear Bins");
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);    //y - Ax -> resid, d_y already copied into resid
  SAFEcuda("cublasSaxpy in CGIHTprojected_S_smv Set Linear Bins");

  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in in CGIHTprojected_S_smv Set Linear Bins");
  residNorm_prev[15]=err;

  // AT(y-Ax) -> grad
  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in CGIHTprojected_S_smv Set Linear Bins");

  // a thres version of grad -> d_vec_thres
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin,
  k_bin, n);
  SAFEcuda("threshold_one of grad in CGIHTprojected_S_smv Set Linear Bins");

  // Agrad_thres -> resid_update
  A_smv(resid_update, d_vec_thres, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("Agen_thres in CGIHTprojected_S_smv Set Linear Bins");

  for (int j = 0; j < 15; j++) {
    residNorm_evolution[j] = residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

/* 
********************************
** Main Steepest Descent Loop **
********************************
*/
  // a flag which indicates switching to SD step when equal to 1
  int suppChange_flag = 1;

  float beta, time, proj_frac, tmp;

  while ( (err > resid_tol) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) )
  {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  if (suppChange_flag == 1) { // SD step
    // compute step length
    mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1); // size n
    tmp = cublasSdot(m, resid_update, 1, resid_update, 1); // size m
    SAFEcublas("cublasSdot in CGIHTprojected_S_smv Main Loop");
    if (mu < 400*tmp)
      mu = mu/tmp;
    else
      mu = 1;

    // x = x + mu * grad -> d_vec
    cublasSaxpy(n, mu, grad, 1, d_vec, 1);
    SAFEcublas("cublasSaxpy in CGIHTprojected_S_smv Main Loop");
    
    // threshold x
    minVal = 0.0f;
    maxChange = 1.0f;
    max_value = MaxMagnitude(d_vec,n);
    slope = (num_bins-1)/max_value;

    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, 
                            &alpha, &MaxBin, &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin,threadsPerBlock_bin);
    SAFEcuda("FindSupportSet in NIHT_S_gen Main Loop");

    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in NIHT_S_gen loop");

    // y - Ax -> resid
    A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp); // Ax -> resid_update
    SAFEcuda("A_smv in CGIHTprojected_S_smv Main Loop");

    cublasScopy(m, d_y, 1, resid, 1); // d_y -> resid, size m
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);    //y - Ax -> resid
    SAFEcublas("cublasScopy&cublasSaxpy in CGIHTprojected_S_smv Main Loop");

    // AT(y-Ax) -> grad
    AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
    SAFEcuda("AT_smv in CGIHTprojected_S_smv Main Loop");

    // grad_thres -> d_vec_thres
    zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
    threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin,
    k_bin, n);
    SAFEcuda("threshold_one of grad in CGIHTprojected_S_smv Main Loop");

    // Agrad_thres -> resid_update
    A_smv(resid_update, d_vec_thres, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
    SAFEcuda("Agen_thres in CGIHTprojected_S_smv Main Loop");

    // p_thres = grad_thres -> d_p_thres, Ap_thres = Ar_thres -> d_Ap_thres
    cublasScopy(n, d_vec_thres, 1, d_p_thres,1);
    cublasScopy(m, resid_update, 1, d_Ap_thres, 1);
    SAFEcublas("cublasScopy in CGIHTprojected Main Loop");
  }
  else { // CG update on the previous support set
    // compute step length
    mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1); // size n
    tmp = cublasSdot(m, d_Ap_thres, 1, d_Ap_thres, 1); // size m
    SAFEcublas("cublasSdot in CGIHTprojected_S_smv Main Loop");
    if (mu < 400*tmp)
      mu = mu/tmp;
    else
      mu = 1;

     // x = x + mu * p in threshold version
    cublasSaxpy(n, mu, d_p_thres, 1, d_vec, 1);
    SAFEcublas("cublasSaxpy in CGIHTprojected_S_smv Main Loop");   

    // y - Ax -> resid
    A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp); // Ax -> resid_update
    SAFEcuda("A_smv in CGIHTprojected_S_smv Main Loop");

    cublasScopy(m, d_y, 1, resid, 1); // d_y -> resid, size m
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);    //y - Ax -> resid
    SAFEcublas("cublasScopy&cublasSaxpy in CGIHTprojected_S_smv Main Loop");

    // grad_thres -> grad_prev_thres: copy d_vec_thres to grad_previous_thres
    cublasScopy(n, d_vec_thres, 1, grad_prev_thres,1);
    SAFEcublas("cublasScopy in CGIHTprojected_S_smv Main Loop");

    // AT(y-Ax) -> grad
    AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
    SAFEcuda("AT_smv in CGIHTprojected_S_smv Main Loop");
  
    // grad_thres -> d_vec_thres
    zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
    threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin,
    k_bin, n);
    SAFEcuda("threshold_one of grad in CGIHTprojected_S_smv Main Loop");

    // Agrad_thres -> resid_update
    A_smv(resid_update, d_vec_thres, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
    SAFEcuda("Agen_thres in CGIHTprojected_S_smv Main Loop");      

    // compute beta
    beta = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
    tmp = cublasSdot(n, grad_prev_thres, 1, grad_prev_thres, 1);
    SAFEcublas("cublasSdot in CGIHTprojected_S_smv Main Loop");

    if (beta < 400 * tmp)
      beta = beta/tmp;
    else 
      beta = 0;
   
    // update p_thres and Ap_thres
    // p_thres = r_thres + beta * p_thres -> d_p_thres
    cublasSscal(n, beta, d_p_thres, 1); // size n
    cublasSaxpy(n, 1, d_vec_thres, 1, d_p_thres, 1); // size n
    SAFEcublas("cublas updating p_thres");

    // Ap_thres = Ar_thres + beta * Ap_thres
    cublasSscal(m, beta, d_Ap_thres, 1);
    cublasSaxpy(m, 1, resid_update, 1, d_Ap_thres, 1);  
  }

  // compute proj_frac = norm(grad-p_thres,2)/norm(grad_thres,2) 
  // p_thres in d_p_thres, grad_thres in d_vec_thres
  cublasScopy(n, grad, 1, d_vec_diff, 1); // size n
  cublasSaxpy(n, -1, d_p_thres, 1, d_vec_diff, 1); // size n
  proj_frac = cublasSnrm2(n, d_vec_diff, 1);
  tmp = cublasSnrm2(n, d_vec_thres, 1);

  // set suppChange_flag
  if (proj_frac < 1000 * tmp) {
     proj_frac = proj_frac/tmp;   
     if (proj_frac > proj_frac_tol)
       suppChange_flag = 1;
     else
       suppChange_flag = 0;    
  }
  else 
     suppChange_flag = 1; 

  // check for convergence 
  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in CGIHTprojected_S_smv Main Loop");

  for (int j = 0; j<15; j++) {
	residNorm_prev[j] = residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

  if ( iter > 749 ){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1;
  }


  iter++;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;
   
  } // end of while

  *p_iter = iter;
  *p_time_sum = time_sum;

} // end of CGIHTprojected_S_smv




/*
*********************************************************************
**                      FIHT_S_gen                                 **
**      Single Precision Fast Iterative Hard Thresholding          **
**                with general matrices                            **
*********************************************************************
*/

inline void FIHT_S_gen(float *d_vec, float *d_vec_thres,  float *grad, float *d_vec_prev, float *d_Avec_prev, float *d_extra, float *d_Avec_extra,  float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int *d_bin_counters, int *h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/
  int iter = *p_iter;
  float time_sum = *p_time_sum;
  int k_bin = 0;

  float alpha = 0.25f;
  int MaxBin = (int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;


  // initial residuals 
  cublasInit();
  cublasScopy(m, d_y, 1, resid,1);
  SAFEcublas("cublasScopy in FIHT_S_gen initilization");

  err= cublasSnrm2(m,resid,1);
  SAFEcublas("cublasSnrm2 in FIHT_S_gen initilization");

  float err_start = err;

  float resid_tol = tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int i=0; i<16; i++) {// the size of residNorm_prev is 16.
    residNorm_prev[i] = 0;
    residNorm_evolution[i] = 1.0f;
  }

  // some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

  #ifdef VERBOSE
  if (verb > 3) {
    printf("The initial residual error = %f\n",err);
  }
  #endif

/*
*****************************
**   Set the Linear Bins   **
*****************************
*/
  // x = ATy -> d_vec
  AT_gen(d_vec, d_y, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in FIHT_S_gen Set the Linear Bins");

  // find support of x
  float max_value = MaxMagnitude(d_vec, n);

  float slope = (num_bins-1)/max_value;

  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                          &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in FIHT_S_gen Set the Linear Bins");

  // threshold x
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in FIHT_S_gen Set the Linear Bins");

/*
************************
** One more Iteration **
************************
*/

  // Ax -> resid_update
  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in FIHT_S_gen One Iteration");

  //y - Ax -> resid
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  err= cublasSnrm2(m, resid, 1);
  SAFEcuda("cublasSaxpy&cublasSnrm2 in FIHT_S_gen One Iteration");
  residNorm_prev[15] = err;

  // AT(y - Ax) -> grad
  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in FIHT_S_gen One Iteration");
  
  // SAVE d_vec -> d_vec_prev; resid_update -> d_Avec_prev
  cublasScopy(n, d_vec, 1, d_vec_prev, 1);
  cublasScopy(m, resid_update, 1, d_Avec_prev, 1);
  SAFEcuda("cublasScopy in FIHT_S_gen One Iteration");

  // x = x + AT(y - Ax) -> d_vec; 
  cublasSaxpy(n, 1, grad, 1, d_vec, 1);

  // find the support of x
  minVal = 0.0f;
  maxChange = 1.0f;
  max_value = MaxMagnitude(d_vec,n);
  slope = ((num_bins-1)/max_value);
  
  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                          &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in FIHT_S_gen One Iteration");

  // threshold x
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in FIHT_S_gen One Iteration");
  
  // Ax -> resid_update
  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in FIHT_S_gen One Iteration");

  // y - Ax -> resid
  cublasScopy(m, d_y, 1, resid, 1);
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  err= cublasSnrm2(m, resid, 1);
  SAFEcuda("cublasSaxpy&cublasSnrm2 in FIHT_S_gen One Iteration");

  // update residNorm_prev, residNorm_evolution, residNorm_diff
  for (int j=0; j<15; j++) {
    residNorm_prev[j] = residNorm_prev[j+1];
    residNorm_evolution[j] = residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];
  residNorm_diff = max_list(residNorm_evolution, 16);


/*
**************************
**      Main Loop       **
**************************
*/
  float tau, tmp, time;
  while ((err> resid_tol) & (iter < maxiter) & (err< 100*err_start) 
         & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) ) {
    // timing variable [Q]
    cudaEvent_t start, stop;
    //float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    
    // compute and threshold x [Q]   
    maxChange = RestrictedFIHT_S_gen(d_y, d_vec, resid_update, d_vec_prev, d_Avec_prev, d_extra, d_Avec_extra, grad, d_vec_thres, resid, d_A, d_AT, 
                                      k,  m, n, mu, tau, tmp, numBlocks,  threadsPerBlock,  numBlocksm,  threadsPerBlockm);
    SAFEcuda("RestrictedFIHT in FIHT_S_gen main loop");

    minVal = 0.0f;
    //if (minVal <= maxChange) {
    max_value = MaxMagnitude(d_vec, n);
    slope = (num_bins-1)/max_value;
    //}

    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                            &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
    SAFEcuda("FindSupportSet in FIHT_S_gen Compute New Iterate");
   

    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in FIHT_S_gen Compute New Iterate");

    // Ax -> resid_update
    A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
    SAFEcuda("A_gen in FIHT_S_gen Compute New Iterate");

    // y - Ax -> resid
    cublasScopy(m, d_y, 1, resid, 1);
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);
    err= cublasSnrm2(m, resid, 1);
    SAFEcuda("cublasSaxpy&cublasSaxpy in FIHT_S_gen Compute New Iterate");

    /*
     **********************************
     ** SD on Restricted Support Set **
     **********************************
    */
    // AT(y - Ax) -> grad
    AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);    
    SAFEcuda("AT_gen in FIHT_S_gen SD on Restricted Support Set");
    
    // threhold grad -> d_vec_thres
    zero_vector_float<<<numBlocks, threadsPerBlock>>>(d_vec_thres, n);
    threshold_one<<<numBlocks, threadsPerBlock>>>(grad, d_vec_thres, d_bin, k_bin, n);
    
    // Agrad_thres -> d_Avec_extra
    A_gen(d_Avec_extra, d_vec_thres, d_A, m, n, numBlocksm, threadsPerBlockm);
    SAFEcuda("A_gen in FIHT_S_gen SD on Restricted Support Set");

    // compute mu [Q]
    mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
    tmp = cublasSdot(m, d_Avec_extra, 1, d_Avec_extra, 1);
    SAFEcuda("cublasSdot in FIHT_S_gen SD on Restricted Support Set");
    if (mu < 400*tmp)
      mu = mu/tmp;
    else 
      mu = 1;   
    // printf("mu2 = %f\n", mu);

    // x = x + mu * d_vec_thres(!) -> d_vec
    cublasSaxpy(n, mu, d_vec_thres, 1, d_vec, 1);
    SAFEcuda("cublasScopy&cublasSaxpy in FIHT_S_gen SD on Restricted Support Set");

    // Ax = Ax + mu * Agrad_thres -> resid_update
    cublasSaxpy(m, mu, d_Avec_extra, 1, resid_update, 1);
    
    // y - Ax
    cublasScopy(m, d_y, 1, resid, 1);
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);
    err= cublasSnrm2(m, resid, 1);
    SAFEcuda("cublasSaxpy&cublasSnrm2 in FIHT_S_gen SD on Restricted Support Set");   

    /*
     *********************************************
     ** Update Residuals, convergenceRate, Iter **
     *********************************************
    */
    for (int j=0; j<15; j++) {
      residNorm_prev[j] = residNorm_prev[j+1];
      residNorm_evolution[j] = residNorm_evolution[j+1];
    }
    residNorm_prev[15] = err;
    residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];
    residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

    if(iter>749) {
      root = 1/15.0f;
      convergenceRate = residNorm_prev[15]/residNorm_prev[0];
      convergenceRate = pow(convergenceRate, root);
      
      if (convergenceRate > 0.999f) fail = 1;
    }// end of if (iter>749)

    iter++;

    // timing
    cudaThreadSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    time_sum = time_sum + time;
  }// end of Main Loop

  *p_iter = iter;
  *p_time_sum = time_sum;
}// end of FIHT_S_gen



/*
*********************************************************************
**                      FIHT_S_dct                                 **
**      Single Precision Fast Iterative Hard Thresholding          **
**                with general matrices                            **
*********************************************************************
*/

inline void FIHT_S_dct(float *d_vec, float *d_vec_thres,  float *grad, float *d_vec_prev, float *d_Avec_prev, float *d_extra, float *d_Avec_extra, float *d_y, float *resid, float *resid_update, int *d_rows, int *d_bin, int *d_bin_counters, int *h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/
  int iter = *p_iter;
  float time_sum = *p_time_sum;
  int k_bin = 0;

  float alpha = 0.25f;
  int MaxBin = (int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;


  // initial residuals 
  cublasInit();
  cublasScopy(m, d_y, 1, resid,1);
  SAFEcublas("cublasScopy in FIHT_S_dct initilization");

  err= cublasSnrm2(m,resid,1);
  SAFEcublas("cublasSnrm2 in FIHT_S_dct initilization");

  float err_start = err;

  float resid_tol = tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int i=0; i<16; i++) {// the size of residNorm_prev is 16.
    residNorm_prev[i] = 0;
    residNorm_evolution[i] = 1.0f;
  }

  // some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

  #ifdef VERBOSE
  if (verb > 3) {
    printf("The initial residual error = %f\n",err);
  }
  #endif

/*
*****************************
**   Set the Linear Bins   **
*****************************
*/
  // x = ATy -> d_vec
  AT_dct(d_vec, d_y, n, m, d_rows);
  SAFEcuda("AT_dct in FIHT_S_dct Set the Linear Bins");

  // find support of x
  float max_value = MaxMagnitude(d_vec, n);

  float slope = (num_bins-1)/max_value;

  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                          &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in FIHT_S_dct Set the Linear Bins");

  // threshold x
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in FIHT_S_dct Set the Linear Bins");

/*
************************
** One more Iteration **
************************
*/

  // Ax -> resid_update
  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct in FIHT_S_dct One Iteration");

  //y - Ax -> resid
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  err= cublasSnrm2(m, resid, 1);
  SAFEcuda("cublasSaxpy&cublasSnrm2 in FIHT_S_dct One Iteration");
  residNorm_prev[15] = err;

  // AT(y - Ax) -> grad
  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in FIHT_S_dct One Iteration");
  
  // SAVE d_vec -> d_vec_prev; resid_update -> d_Avec_prev
  cublasScopy(n, d_vec, 1, d_vec_prev, 1);
  cublasScopy(m, resid_update, 1, d_Avec_prev, 1);
  SAFEcuda("cublasScopy in FIHT_S_dct One Iteration");

  // x = x + AT(y - Ax) -> d_vec; 
  cublasSaxpy(n, 1, grad, 1, d_vec, 1);

  // find the support of x
  minVal = 0.0f;
  maxChange = 1.0f;
  max_value = MaxMagnitude(d_vec,n);
  slope = ((num_bins-1)/max_value);
  
  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                          &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in FIHT_S_dct One Iteration");

  // threshold x
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in FIHT_S_dct One Iteration");
  
  // Ax -> resid_update
  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct in FIHT_S_dct One Iteration");

  // y - Ax -> resid
  cublasScopy(m, d_y, 1, resid, 1);
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  err= cublasSnrm2(m, resid, 1);
  SAFEcuda("cublasSaxpy&cublasSnrm2 in FIHT_S_dct One Iteration");

  // update residNorm_prev, residNorm_evolution, residNorm_diff
  for (int j=0; j<15; j++) {
    residNorm_prev[j] = residNorm_prev[j+1];
    residNorm_evolution[j] = residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];
  residNorm_diff = max_list(residNorm_evolution, 16);


/*
**************************
**      Main Loop       **
**************************
*/
  float tau, tmp, time;
  while ((err> resid_tol) & (iter < maxiter) & (err< 100*err_start) 
         & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) ) {
    // timing variable [Q]
    cudaEvent_t start, stop;
    //float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    
    // compute and threshold x [Q]   
    maxChange = RestrictedFIHT_S_dct(d_y, d_vec, resid_update, d_vec_prev, d_Avec_prev, d_extra, d_Avec_extra, grad, d_vec_thres, resid, d_rows, 
                                      k,  m, n, mu, tau, tmp, numBlocks,  threadsPerBlock);
    SAFEcuda("RestrictedFIHT in FIHT_S_dct main loop");

    minVal = 0.0f;
    // w = v + mu*grad = x + tau(x_prev-x) + mu*grad
    //   = (1-tau)*x + tau*x_prev + mu*grad
    //if (fabs((1-tau)*minVal)<=maxChange) {
    max_value = MaxMagnitude(d_vec, n);
    slope = (num_bins-1)/max_value;
    //}

    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                            &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
    SAFEcuda("FindSupportSet in FIHT_S_dct Compute New Iterate");
   

    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in FIHT_S_dct Compute New Iterate");

    // Ax -> resid_update
    A_dct(resid_update, d_vec, n, m, d_rows);
    SAFEcuda("A_dct in FIHT_S_dct Compute New Iterate");

    // y - Ax -> resid
    cublasScopy(m, d_y, 1, resid, 1);
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);
    err= cublasSnrm2(m, resid, 1);
    SAFEcuda("cublasSaxpy&cublasSaxpy in FIHT_S_dct Compute New Iterate");

    /*
     **********************************
     ** SD on Restricted Support Set **
     **********************************
    */
    // AT(y - Ax) -> grad
    AT_dct(grad, resid, n, m, d_rows);    
    SAFEcuda("AT_dct in FIHT_S_dct SD on Restricted Support Set");
    
    // threhold grad -> d_vec_thres
    zero_vector_float<<<numBlocks, threadsPerBlock>>>(d_vec_thres, n);
    threshold_one<<<numBlocks, threadsPerBlock>>>(grad, d_vec_thres, d_bin, k_bin, n);
    
    // Agrad_thres -> d_Avec_extra
    A_dct(d_Avec_extra, d_vec_thres, n, m, d_rows);
    SAFEcuda("A_dct in FIHT_S_dct SD on Restricted Support Set");

    // compute mu [Q]
    mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
    tmp = cublasSdot(m, d_Avec_extra, 1, d_Avec_extra, 1);
    SAFEcuda("cublasSdot in FIHT_S_dct SD on Restricted Support Set");
    if (mu < 1000*tmp)
      mu = mu/tmp;
    else 
      mu = 1;   
    // printf("mu2 = %f\n", mu);

    // x = x + mu * d_vec_thres(!) -> d_vec
    cublasSaxpy(n, mu, d_vec_thres, 1, d_vec, 1);
    SAFEcuda("cublasScopy&cublasSaxpy in FIHT_S_dct SD on Restricted Support Set");

    // Ax = Ax + mu * Agrad_thres -> resid_update
    cublasSaxpy(m, mu, d_Avec_extra, 1, resid_update, 1);
    
    // y - Ax
    cublasScopy(m, d_y, 1, resid, 1);
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);
    err= cublasSnrm2(m, resid, 1);
    SAFEcuda("cublasSaxpy&cublasSnrm2 in FIHT_S_dct SD on Restricted Support Set");   

    /*
     *********************************************
     ** Update Residuals, convergenceRate, Iter **
     *********************************************
    */
    for (int j=0; j<15; j++) {
      residNorm_prev[j] = residNorm_prev[j+1];
      residNorm_evolution[j] = residNorm_evolution[j+1];
    }
    residNorm_prev[15] = err;
    residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];
    residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

    if(iter>749) {
      root = 1/15.0f;
      convergenceRate = residNorm_prev[15]/residNorm_prev[0];
      convergenceRate = pow(convergenceRate, root);
      
      if (convergenceRate > 0.999f) fail = 1;
    }// end of if (iter>749)

    iter++;

    // timing
    cudaThreadSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    time_sum = time_sum + time;
  }// end of Main Loop

  *p_iter = iter;
  *p_time_sum = time_sum;
}// end of FIHT_S_dct


/*
*********************************************************************
**                      FIHT_S_smv                                 **
**      Single Precision Fast Iterative Hard Thresholding          **
**                with general matrices                            **
*********************************************************************
*/

inline void FIHT_S_smv(float *d_vec,  float *d_vec_thres,  float *grad, float *d_vec_prev, float *d_Avec_prev, float *d_extra, float *d_Avec_extra, float *d_y, float *resid, float *resid_update, int *d_rows, int *d_cols, float *d_vals, int *d_bin, int *d_bin_counters, int *h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int nz, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock,dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/
  int iter = *p_iter;
  float time_sum = *p_time_sum;
  int k_bin = 0;

  float alpha = 0.25f;
  int MaxBin = (int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;


  // initial residuals 
  cublasInit();
  cublasScopy(m, d_y, 1, resid,1);
  SAFEcublas("cublasScopy in FIHT_S_smv initilization");

  err= cublasSnrm2(m,resid,1);
  SAFEcublas("cublasSnrm2 in FIHT_S_smv initilization");

  float err_start = err;

  float resid_tol = tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int i=0; i<16; i++) {// the size of residNorm_prev is 16.
    residNorm_prev[i] = 0;
    residNorm_evolution[i] = 1.0f;
  }

  // some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

  #ifdef VERBOSE
  if (verb > 3) {
    printf("The initial residual error = %f\n",err);
  }
  #endif

/*
*****************************
**   Set the Linear Bins   **
*****************************
*/
  // x = ATy -> d_vec
  AT_smv(d_vec, d_y, m, n, d_rows, d_cols, d_vals, nz , numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in FIHT_S_smv Set the Linear Bins");

  // find support of x
  float max_value = MaxMagnitude(d_vec, n);

  float slope = (num_bins-1)/max_value;

  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                          &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in FIHT_S_smv Set the Linear Bins");

  // threshold x
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in FIHT_S_smv Set the Linear Bins");

/*
************************
** One more Iteration **
************************
*/

  // Ax -> resid_update
  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in FIHT_S_smv One Iteration");

  //y - Ax -> resid
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  err= cublasSnrm2(m, resid, 1);
  SAFEcuda("cublasSaxpy&cublasSnrm2 in FIHT_S_smv One Iteration");
  residNorm_prev[15] = err;

  // AT(y - Ax) -> grad
  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in FIHT_S_smv One Iteration");
  
  // SAVE d_vec -> d_vec_prev; resid_update -> d_Avec_prev
  cublasScopy(n, d_vec, 1, d_vec_prev, 1);
  cublasScopy(m, resid_update, 1, d_Avec_prev, 1);
  SAFEcuda("cublasScopy in FIHT_S_smv One Iteration");

  // x = x + AT(y - Ax) -> d_vec; 
  cublasSaxpy(n, 1, grad, 1, d_vec, 1);

  // find the support of x
  minVal = 0.0f;
  maxChange = 1.0f;
  max_value = MaxMagnitude(d_vec,n);
  slope = ((num_bins-1)/max_value);
  
  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                          &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in FIHT_S_smv One Iteration");

  // threshold x
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in FIHT_S_smv One Iteration");
  
  // Ax -> resid_update
  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in FIHT_S_smv One Iteration");

  // y - Ax -> resid
  cublasScopy(m, d_y, 1, resid, 1);
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  err= cublasSnrm2(m, resid, 1);
  SAFEcuda("cublasSaxpy&cublasSnrm2 in FIHT_S_smv One Iteration");

  // update residNorm_prev, residNorm_evolution, residNorm_diff
  for (int j=0; j<15; j++) {
    residNorm_prev[j] = residNorm_prev[j+1];
    residNorm_evolution[j] = residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];
  residNorm_diff = max_list(residNorm_evolution, 16);


/*
**************************
**      Main Loop       **
**************************
*/
  float tau, tmp, time;
  while ((err> resid_tol) & (iter < maxiter) & (err< 100*err_start) 
         & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) ) {
    // timing variable [Q]
    cudaEvent_t start, stop;
    //float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    
    // compute and threshold x [Q]   
    maxChange = RestrictedFIHT_S_smv(d_y, d_vec, resid_update, d_vec_prev, d_Avec_prev, d_extra, d_Avec_extra, grad, d_vec_thres, resid, d_rows, d_cols, d_vals, 
                                      k,  m, n, nz, mu, tau, tmp, numBlocks,  threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm,  threadsPerBlockm);
    SAFEcuda("RestrictedFIHT in FIHT_S_smv main loop");

    minVal = 0.0f;
    //if (minVal <= maxChange) {
    max_value = MaxMagnitude(d_vec, n);
    slope = (num_bins-1)/max_value;
    //}

    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                            &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
    SAFEcuda("FindSupportSet in FIHT_S_smv Compute New Iterate");
   

    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in FIHT_S_smv Compute New Iterate");

    // Ax -> resid_update
    A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
    SAFEcuda("A_smv in FIHT_S_smv Compute New Iterate");

    // y - Ax -> resid
    cublasScopy(m, d_y, 1, resid, 1);
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);
    err= cublasSnrm2(m, resid, 1);
    SAFEcuda("cublasSaxpy&cublasSaxpy in FIHT_S_smv Compute New Iterate");

    /*
     **********************************
     ** SD on Restricted Support Set **
     **********************************
    */
    // AT(y - Ax) -> grad
    AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);    
    SAFEcuda("AT_smv in FIHT_S_smv SD on Restricted Support Set");
    
    // threhold grad -> d_vec_thres
    zero_vector_float<<<numBlocks, threadsPerBlock>>>(d_vec_thres, n);
    threshold_one<<<numBlocks, threadsPerBlock>>>(grad, d_vec_thres, d_bin, k_bin, n);
    
    // Agrad_thres -> d_Avec_extra
    A_smv(d_Avec_extra, d_vec_thres, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
    SAFEcuda("A_smv in FIHT_S_smv SD on Restricted Support Set");

    // compute mu [Q]
    mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
    tmp = cublasSdot(m, d_Avec_extra, 1, d_Avec_extra, 1);
    SAFEcuda("cublasSdot in FIHT_S_smv SD on Restricted Support Set");
    if (mu < 400*tmp)
      mu = mu/tmp;
    else 
      mu = 1;   
    // printf("mu2 = %f\n", mu);

    // x = x + mu * d_vec_thres(!) -> d_vec
    cublasSaxpy(n, mu, d_vec_thres, 1, d_vec, 1);
    SAFEcuda("cublasScopy&cublasSaxpy in FIHT_S_smv SD on Restricted Support Set");

    // Ax = Ax + mu * Agrad_thres -> resid_update
    cublasSaxpy(m, mu, d_Avec_extra, 1, resid_update, 1);
    
    // y - Ax
    cublasScopy(m, d_y, 1, resid, 1);
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);
    err= cublasSnrm2(m, resid, 1);
    SAFEcuda("cublasSaxpy&cublasSnrm2 in FIHT_S_smv SD on Restricted Support Set");   

    /*
     *********************************************
     ** Update Residuals, convergenceRate, Iter **
     *********************************************
    */
    for (int j=0; j<15; j++) {
      residNorm_prev[j] = residNorm_prev[j+1];
      residNorm_evolution[j] = residNorm_evolution[j+1];
    }
    residNorm_prev[15] = err;
    residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];
    residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

    if(iter>749) {
      root = 1/15.0f;
      convergenceRate = residNorm_prev[15]/residNorm_prev[0];
      convergenceRate = pow(convergenceRate, root);
      
      if (convergenceRate > 0.999f) fail = 1;
    }// end of if (iter>749)

    iter++;

    // timing
    cudaThreadSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    time_sum = time_sum + time;
  }// end of Main Loop

  *p_iter = iter;
  *p_time_sum = time_sum;
}// end of FIHT_S_smv




/*
*********************************************************************
**                      ALPS_S_gen                                 **
**      Single Precision Fast Iterative Hard Thresholding          **
**                with general matrices                            **
*********************************************************************
*/

inline void ALPS_S_gen(float *d_vec, float *d_vec_thres,  float *grad, float *d_vec_prev, float *d_Avec_prev, float *d_extra, float *d_Avec_extra,  float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int *d_bin_counters, int *h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/
  int iter = *p_iter;
  float time_sum = *p_time_sum;
  int k_bin = 0;

  float alpha = 0.25f;
  int MaxBin = (int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;


  // initial residuals 
  cublasInit();
  cublasScopy(m, d_y, 1, resid,1);
  SAFEcublas("cublasScopy in ALPS_S_gen initilization");

  err= cublasSnrm2(m,resid,1);
  SAFEcublas("cublasSnrm2 in ALPS_S_gen initilization");

  float err_start = err;

  float resid_tol = tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int i=0; i<16; i++) {// the size of residNorm_prev is 16.
    residNorm_prev[i] = 0;
    residNorm_evolution[i] = 1.0f;
  }

  // some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

  #ifdef VERBOSE
  if (verb > 3) {
    printf("The initial residual error = %f\n",err);
  }
  #endif

/*
*****************************
**   Set the Linear Bins   **
*****************************
*/
  // x = ATy -> d_vec
  AT_gen(d_vec, d_y, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in ALPS_S_gen Set the Linear Bins");

  // find support of x
  float max_value = MaxMagnitude(d_vec, n);

  float slope = (num_bins-1)/max_value;

  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                          &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in ALPS_S_gen Set the Linear Bins");

  // threshold x
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in ALPS_S_gen Set the Linear Bins");

/*
************************
** One more Iteration **
************************
*/

  // Ax -> resid_update
  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in ALPS_S_gen One Iteration");

  //y - Ax -> resid
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  err= cublasSnrm2(m, resid, 1);
  SAFEcuda("cublasSaxpy&cublasSnrm2 in ALPS_S_gen One Iteration");
  residNorm_prev[15] = err;

  // AT(y - Ax) -> grad
  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in ALPS_S_gen One Iteration");
  
  // SAVE d_vec -> d_vec_prev; resid_update -> d_Avec_prev
  cublasScopy(n, d_vec, 1, d_vec_prev, 1);
  cublasScopy(m, resid_update, 1, d_Avec_prev, 1);
  SAFEcuda("cublasScopy in ALPS_S_gen One Iteration");

  // x = x + AT(y - Ax) -> d_vec; 
  cublasSaxpy(n, 1, grad, 1, d_vec, 1);

  // find the support of x
  minVal = 0.0f;
  maxChange = 1.0f;
  max_value = MaxMagnitude(d_vec,n);
  slope = ((num_bins-1)/max_value);
  
  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                          &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in ALPS_S_gen One Iteration");

  // threshold x
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in ALPS_S_gen One Iteration");
  
  // Ax -> resid_update
  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in ALPS_S_gen One Iteration");

  // y - Ax -> resid
  cublasScopy(m, d_y, 1, resid, 1);
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  err= cublasSnrm2(m, resid, 1);
  SAFEcuda("cublasSaxpy&cublasSnrm2 in ALPS_S_gen One Iteration");

  // update residNorm_prev, residNorm_evolution, residNorm_diff
  for (int j=0; j<15; j++) {
    residNorm_prev[j] = residNorm_prev[j+1];
    residNorm_evolution[j] = residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];
  residNorm_diff = max_list(residNorm_evolution, 16);


/*
**************************
**      Main Loop       **
**************************
*/
  float tau, tmp, time;
  while ((err> resid_tol) & (iter < maxiter) & (err< 100*err_start) 
         & (residNorm_diff > .01*tol) & (fail == 0) & (cycleFlag) ) {
    // timing variable [Q]
    cudaEvent_t start, stop;
    //float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // compute x -> d_vec
    maxChange = RestrictedALPS_S_gen(d_y, d_vec, resid_update, d_vec_prev, d_Avec_prev, d_extra, d_Avec_extra, grad, d_vec_thres, resid, d_A, d_AT, d_bin, d_bin_counters, h_bin_counters, num_bins, 
 k,  m,  n, p_sum,  mu,  tau,  tmp,  alpha, minVal, maxChange, slope, max_value,  k_bin, MaxBin, numBlocks, threadsPerBlock, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);

    // threshold x [Q]    
    minVal = 0.0f;
    max_value = MaxMagnitude(d_vec, n);
    slope = (num_bins-1)/max_value;

    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                            &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
    SAFEcuda("FindSupportSet in ALPS_S_gen Compute New Iterate");
   

    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in ALPS_S_gen Compute New Iterate");

    // Ax -> resid_update
    A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
    SAFEcuda("A_gen in ALPS_S_gen Compute New Iterate");

    // y - Ax -> resid
    cublasScopy(m, d_y, 1, resid, 1);
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);
    err= cublasSnrm2(m, resid, 1);
    SAFEcuda("cublasSaxpy&cublasSaxpy in ALPS_S_gen Compute New Iterate");

    /*
     **********************************
     ** SD on Restricted Support Set **
     **********************************
    */
    // AT(y - Ax) -> grad
    AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);    
    SAFEcuda("AT_gen in ALPS_S_gen SD on Restricted Support Set");
    
    // threhold grad -> d_vec_thres
    zero_vector_float<<<numBlocks, threadsPerBlock>>>(d_vec_thres, n);
    threshold_one<<<numBlocks, threadsPerBlock>>>(grad, d_vec_thres, d_bin, k_bin, n);
    
    // Agrad_thres -> d_Avec_extra
    A_gen(d_Avec_extra, d_vec_thres, d_A, m, n, numBlocksm, threadsPerBlockm);
    SAFEcuda("A_gen in ALPS_S_gen SD on Restricted Support Set");

    // compute mu [Q]
    mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
    tmp = cublasSdot(m, d_Avec_extra, 1, d_Avec_extra, 1);
    SAFEcuda("cublasSdot in ALPS_S_gen SD on Restricted Support Set");
    if (mu < 400*tmp)
      mu = mu/tmp;
    else 
      mu = 1;    

    // x = x + mu * d_vec_thres(!) -> d_vec
    cublasSaxpy(n, mu, d_vec_thres, 1, d_vec, 1);
    SAFEcuda("cublasScopy&cublasSaxpy in ALPS_S_gen SD on Restricted Support Set");

    // Ax = Ax + mu * Agrad_thres -> resid_update
    cublasSaxpy(m, mu, d_Avec_extra, 1, resid_update, 1);
    
    // y - Ax
    cublasScopy(m, d_y, 1, resid, 1);
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);
    err= cublasSnrm2(m, resid, 1);
    SAFEcuda("cublasSaxpy&cublasSnrm2 in ALPS_S_gen SD on Restricted Support Set");   

    /*
     *********************************************
     ** Update Residuals, convergenceRate, Iter **
     *********************************************
    */
    for (int j=0; j<15; j++) {
      residNorm_prev[j] = residNorm_prev[j+1];
      residNorm_evolution[j] = residNorm_evolution[j+1];
    }
    residNorm_prev[15] = err;
    residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];
    residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

    if(iter>749) {
      root = 1/15.0f;
      convergenceRate = residNorm_prev[15]/residNorm_prev[0];
      convergenceRate = pow(convergenceRate, root);
      
      if (convergenceRate > 0.999f) fail = 1;
    }// end of if (iter>749)
    iter++;

    // timing
    cudaThreadSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    time_sum = time_sum + time;
  }// end of Main Loop

  *p_iter = iter;
  *p_time_sum = time_sum;
}// end of ALPS_S_gen



/*
*********************************************************************
**                      ALPS_S_dct                                 **
**      Single Precision Fast Iterative Hard Thresholding          **
**                with general matrices                            **
*********************************************************************
*/

inline void ALPS_S_dct(float *d_vec, float *d_vec_thres,  float *grad, float *d_vec_prev, float *d_Avec_prev, float *d_extra, float *d_Avec_extra, float *d_y, float *resid, float *resid_update, int *d_rows, int *d_bin, int *d_bin_counters, int *h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const int num_bins, const int k, const int m, const int n, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/
  int iter = *p_iter;
  float time_sum = *p_time_sum;
  int k_bin = 0;

  float alpha = 0.25f;
  int MaxBin = (int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;


  // initial residuals 
  cublasInit();
  cublasScopy(m, d_y, 1, resid,1);
  SAFEcublas("cublasScopy in ALPS_S_dct initilization");

  err= cublasSnrm2(m,resid,1);
  SAFEcublas("cublasSnrm2 in ALPS_S_dct initilization");

  float err_start = err;

  float resid_tol = tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int i=0; i<16; i++) {// the size of residNorm_prev is 16.
    residNorm_prev[i] = 0;
    residNorm_evolution[i] = 1.0f;
  }

  // some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

  #ifdef VERBOSE
  if (verb > 3) {
    printf("The initial residual error = %f\n",err);
  }
  #endif

/*
*****************************
**   Set the Linear Bins   **
*****************************
*/
  // x = ATy -> d_vec
  AT_dct(d_vec, d_y, n, m, d_rows);
  SAFEcuda("AT_dct in ALPS_S_dct Set the Linear Bins");

  // find support of x
  float max_value = MaxMagnitude(d_vec, n);

  float slope = (num_bins-1)/max_value;

  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                          &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in ALPS_S_dct Set the Linear Bins");

  // threshold x
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in ALPS_S_dct Set the Linear Bins");

/*
************************
** One more Iteration **
************************
*/

  // Ax -> resid_update
  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct in ALPS_S_dct One Iteration");

  //y - Ax -> resid
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  err= cublasSnrm2(m, resid, 1);
  SAFEcuda("cublasSaxpy&cublasSnrm2 in ALPS_S_dct One Iteration");
  residNorm_prev[15] = err;

  // AT(y - Ax) -> grad
  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in ALPS_S_dct One Iteration");
  
  // SAVE d_vec -> d_vec_prev; resid_update -> d_Avec_prev
  cublasScopy(n, d_vec, 1, d_vec_prev, 1);
  cublasScopy(m, resid_update, 1, d_Avec_prev, 1);
  SAFEcuda("cublasScopy in ALPS_S_dct One Iteration");

  // x = x + AT(y - Ax) -> d_vec; 
  cublasSaxpy(n, 1, grad, 1, d_vec, 1);

  // find the support of x
  minVal = 0.0f;
  maxChange = 1.0f;
  max_value = MaxMagnitude(d_vec,n);
  slope = ((num_bins-1)/max_value);
  
  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                          &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in ALPS_S_dct One Iteration");

  // threshold x
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in ALPS_S_dct One Iteration");
  
  // Ax -> resid_update
  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct in ALPS_S_dct One Iteration");

  // y - Ax -> resid
  cublasScopy(m, d_y, 1, resid, 1);
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  err= cublasSnrm2(m, resid, 1);
  SAFEcuda("cublasSaxpy&cublasSnrm2 in ALPS_S_dct One Iteration");

  // update residNorm_prev, residNorm_evolution, residNorm_diff
  for (int j=0; j<15; j++) {
    residNorm_prev[j] = residNorm_prev[j+1];
    residNorm_evolution[j] = residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];
  residNorm_diff = max_list(residNorm_evolution, 16);


/*
**************************
**      Main Loop       **
**************************
*/
  float tau, tmp, time;
  while ((err> resid_tol) & (iter < maxiter) & (err< 100*err_start) 
         & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) ) {
    // timing variable [Q]
    cudaEvent_t start, stop;
    //float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    
    // compute x  
    maxChange = RestrictedALPS_S_dct(d_y, d_vec, resid_update, d_vec_prev, d_Avec_prev, d_extra, d_Avec_extra, grad, d_vec_thres, resid, d_rows, d_bin, d_bin_counters, h_bin_counters, num_bins, k, m,
                                     n, p_sum, mu,  tau, tmp,  alpha, minVal,  maxChange,  slope, max_value,  k_bin,  MaxBin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
    SAFEcuda("RestrictedFIHT in ALPS_S_dct main loop");

    // threshold x
    minVal = 0.0f;
    // w = v + mu*grad = x + tau(x_prev-x) + mu*grad
    //   = (1-tau)*x + tau*x_prev + mu*grad
    //if (fabs((1-tau)*minVal)<=maxChange) {
    max_value = MaxMagnitude(d_vec, n);
    slope = (num_bins-1)/max_value;
    //}

    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                            &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
    SAFEcuda("FindSupportSet in ALPS_S_dct Compute New Iterate");
   

    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in ALPS_S_dct Compute New Iterate");

    // Ax -> resid_update
    A_dct(resid_update, d_vec, n, m, d_rows);
    SAFEcuda("A_dct in ALPS_S_dct Compute New Iterate");

    // y - Ax -> resid
    cublasScopy(m, d_y, 1, resid, 1);
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);
    err= cublasSnrm2(m, resid, 1);
    SAFEcuda("cublasSaxpy&cublasSaxpy in ALPS_S_dct Compute New Iterate");

    /*
     **********************************
     ** SD on Restricted Support Set **
     **********************************
    */
    // AT(y - Ax) -> grad
    AT_dct(grad, resid, n, m, d_rows);    
    SAFEcuda("AT_dct in ALPS_S_dct SD on Restricted Support Set");
    
    // threhold grad -> d_vec_thres
    zero_vector_float<<<numBlocks, threadsPerBlock>>>(d_vec_thres, n);
    threshold_one<<<numBlocks, threadsPerBlock>>>(grad, d_vec_thres, d_bin, k_bin, n);
    
    // Agrad_thres -> d_Avec_extra
    A_dct(d_Avec_extra, d_vec_thres, n, m, d_rows);
    SAFEcuda("A_dct in ALPS_S_dct SD on Restricted Support Set");

    // compute mu [Q]
    mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
    tmp = cublasSdot(m, d_Avec_extra, 1, d_Avec_extra, 1);
    SAFEcuda("cublasSdot in ALPS_S_dct SD on Restricted Support Set");
    if (mu < 1000*tmp)
      mu = mu/tmp;
    else 
      mu = 1;   
    // printf("mu2 = %f\n", mu);

    // x = x + mu * d_vec_thres(!) -> d_vec
    cublasSaxpy(n, mu, d_vec_thres, 1, d_vec, 1);
    SAFEcuda("cublasScopy&cublasSaxpy in ALPS_S_dct SD on Restricted Support Set");

    // Ax = Ax + mu * Agrad_thres -> resid_update
    cublasSaxpy(m, mu, d_Avec_extra, 1, resid_update, 1);
    
    // y - Ax
    cublasScopy(m, d_y, 1, resid, 1);
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);
    err= cublasSnrm2(m, resid, 1);
    SAFEcuda("cublasSaxpy&cublasSnrm2 in ALPS_S_dct SD on Restricted Support Set");   

    /*
     *********************************************
     ** Update Residuals, convergenceRate, Iter **
     *********************************************
    */
    for (int j=0; j<15; j++) {
      residNorm_prev[j] = residNorm_prev[j+1];
      residNorm_evolution[j] = residNorm_evolution[j+1];
    }
    residNorm_prev[15] = err;
    residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];
    residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

    if(iter>749) {
      root = 1/15.0f;
      convergenceRate = residNorm_prev[15]/residNorm_prev[0];
      convergenceRate = pow(convergenceRate, root);
      
      if (convergenceRate > 0.999f) fail = 1;
    }// end of if (iter>749)

    iter++;

    // timing
    cudaThreadSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    time_sum = time_sum + time;
  }// end of Main Loop

  *p_iter = iter;
  *p_time_sum = time_sum;
}// end of ALPS_S_dct



/*
*********************************************************************
**                      ALPS_S_smv                                 **
**      Single Precision Fast Iterative Hard Thresholding          **
**                with general matrices                            **
*********************************************************************
*/

inline void ALPS_S_smv(float *d_vec,  float *d_vec_thres,  float *grad, float *d_vec_prev, float *d_Avec_prev, float *d_extra, float *d_Avec_extra, float *d_y, float *resid, float *resid_update, int *d_rows, int *d_cols, float *d_vals, int *d_bin, int *d_bin_counters, int *h_bin_counters, float *residNorm_prev, const float tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int nz, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock,dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/
  int iter = *p_iter;
  float time_sum = *p_time_sum;
  int k_bin = 0;

  float alpha = 0.25f;
  int MaxBin = (int)(num_bins * (1-alpha));
  float minVal = 0.0f;
  float maxChange = 1.0f;


  // initial residuals 
  cublasInit();
  cublasScopy(m, d_y, 1, resid,1);
  SAFEcublas("cublasScopy in ALPS_S_smv initilization");

  err= cublasSnrm2(m,resid,1);
  SAFEcublas("cublasSnrm2 in ALPS_S_smv initilization");

  float err_start = err;

  float resid_tol = tol;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int i=0; i<16; i++) {// the size of residNorm_prev is 16.
    residNorm_prev[i] = 0;
    residNorm_evolution[i] = 1.0f;
  }

  // some stopping parameters
  int fail = 0;
  float root, convergenceRate;

// *********
  int cycleFlag = 1;
  int usecycleFlag = 1;  // set to 0 for cycleFlag =1 in all iterations or set to 1 to use the cycleFlag for iter>4
// *********

  #ifdef VERBOSE
  if (verb > 3) {
    printf("The initial residual error = %f\n",err);
  }
  #endif

/*
*****************************
**   Set the Linear Bins   **
*****************************
*/
  // x = ATy -> d_vec
  AT_smv(d_vec, d_y, m, n, d_rows, d_cols, d_vals, nz , numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in ALPS_S_smv Set the Linear Bins");

  // find support of x
  float max_value = MaxMagnitude(d_vec, n);

  float slope = (num_bins-1)/max_value;

  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                          &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in ALPS_S_smv Set the Linear Bins");

  // threshold x
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in ALPS_S_smv Set the Linear Bins");

/*
************************
** One more Iteration **
************************
*/

  // Ax -> resid_update
  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in ALPS_S_smv One Iteration");

  //y - Ax -> resid
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  err= cublasSnrm2(m, resid, 1);
  SAFEcuda("cublasSaxpy&cublasSnrm2 in ALPS_S_smv One Iteration");
  residNorm_prev[15] = err;

  // AT(y - Ax) -> grad
  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in ALPS_S_smv One Iteration");
  
  // SAVE d_vec -> d_vec_prev; resid_update -> d_Avec_prev
  cublasScopy(n, d_vec, 1, d_vec_prev, 1);
  cublasScopy(m, resid_update, 1, d_Avec_prev, 1);
  SAFEcuda("cublasScopy in ALPS_S_smv One Iteration");

  // x = x + AT(y - Ax) -> d_vec; 
  cublasSaxpy(n, 1, grad, 1, d_vec, 1);

  // find the support of x
  minVal = 0.0f;
  maxChange = 1.0f;
  max_value = MaxMagnitude(d_vec,n);
  slope = ((num_bins-1)/max_value);
  
  *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                          &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in ALPS_S_smv One Iteration");

  // threshold x
  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in ALPS_S_smv One Iteration");
  
  // Ax -> resid_update
  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in ALPS_S_smv One Iteration");

  // y - Ax -> resid
  cublasScopy(m, d_y, 1, resid, 1);
  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  err= cublasSnrm2(m, resid, 1);
  SAFEcuda("cublasSaxpy&cublasSnrm2 in ALPS_S_smv One Iteration");

  // update residNorm_prev, residNorm_evolution, residNorm_diff
  for (int j=0; j<15; j++) {
    residNorm_prev[j] = residNorm_prev[j+1];
    residNorm_evolution[j] = residNorm_evolution[j+1];
  }
  residNorm_prev[15] = err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];
  residNorm_diff = max_list(residNorm_evolution, 16);


/*
**************************
**      Main Loop       **
**************************
*/

  //printf("start the main loop with iter %d, err %f, err_start %f, residNorm_diff %f\n", iter, err, err_start, residNorm_diff);
  float tau, tmp, time;
  while ((err> resid_tol) & (iter < maxiter) & (err< 100*err_start) 
         & (residNorm_diff > .01*resid_tol) & (fail == 0) & (cycleFlag) ) {
    // timing variable [Q]
    cudaEvent_t start, stop;
    //float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    
    // compute x  
    maxChange = RestrictedALPS_S_smv(d_y, d_vec, resid_update, d_vec_prev, d_Avec_prev, d_extra, d_Avec_extra, grad, d_vec_thres, resid, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters,  
                                     num_bins, k, m, n, nz, p_sum, mu, tau, tmp, alpha, minVal, maxChange, slope, max_value, k_bin, MaxBin, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp,      
                                     numBlocksm,  threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
    SAFEcuda("RestrictedFIHT in ALPS_S_smv main loop");
    
    // threshold x
    minVal = 0.0f;
    //if (minVal <= maxChange) {
    max_value = MaxMagnitude(d_vec, n);
    slope = (num_bins-1)/max_value;
    //}

    *p_sum = FindSupportSet(d_vec, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                            &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
    SAFEcuda("FindSupportSet in ALPS_S_smv Compute New Iterate");
   

    Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
    SAFEcuda("Threshold in ALPS_S_smv Compute New Iterate");

    // Ax -> resid_update
    A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
    SAFEcuda("A_smv in ALPS_S_smv Compute New Iterate");

    // y - Ax -> resid
    cublasScopy(m, d_y, 1, resid, 1);
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);
    err= cublasSnrm2(m, resid, 1);
    SAFEcuda("cublasSaxpy&cublasSaxpy in ALPS_S_smv Compute New Iterate");

    /*
     **********************************
     ** SD on Restricted Support Set **
     **********************************
    */
    // AT(y - Ax) -> grad
    AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);    
    SAFEcuda("AT_smv in ALPS_S_smv SD on Restricted Support Set");
    
    // threhold grad -> d_vec_thres
    zero_vector_float<<<numBlocks, threadsPerBlock>>>(d_vec_thres, n);
    threshold_one<<<numBlocks, threadsPerBlock>>>(grad, d_vec_thres, d_bin, k_bin, n);
    
    // Agrad_thres -> d_Avec_extra
    A_smv(d_Avec_extra, d_vec_thres, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
    SAFEcuda("A_smv in ALPS_S_smv SD on Restricted Support Set");

    // compute mu [Q]
    mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
    tmp = cublasSdot(m, d_Avec_extra, 1, d_Avec_extra, 1);
    SAFEcuda("cublasSdot in ALPS_S_smv SD on Restricted Support Set");
    if (mu < 400*tmp)
      mu = mu/tmp;
    else 
      mu = 1;   
    // printf("mu2 = %f\n", mu);

    // x = x + mu * d_vec_thres(!) -> d_vec
    cublasSaxpy(n, mu, d_vec_thres, 1, d_vec, 1);
    SAFEcuda("cublasScopy&cublasSaxpy in ALPS_S_smv SD on Restricted Support Set");

    // Ax = Ax + mu * Agrad_thres -> resid_update
    cublasSaxpy(m, mu, d_Avec_extra, 1, resid_update, 1);
    
    // y - Ax
    cublasScopy(m, d_y, 1, resid, 1);
    cublasSaxpy(m, -1, resid_update, 1, resid, 1);
    err= cublasSnrm2(m, resid, 1);
    SAFEcuda("cublasSaxpy&cublasSnrm2 in ALPS_S_smv SD on Restricted Support Set");   

    /*
     *********************************************
     ** Update Residuals, convergenceRate, Iter **
     *********************************************
    */
    for (int j=0; j<15; j++) {
      residNorm_prev[j] = residNorm_prev[j+1];
      residNorm_evolution[j] = residNorm_evolution[j+1];
    }
    residNorm_prev[15] = err;
    residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];
    residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) && (usecycleFlag) ) {
	  cycleFlag=0;
        } // end if
   } // end for
// **********************************

    if(iter>749) {
      root = 1/15.0f;
      convergenceRate = residNorm_prev[15]/residNorm_prev[0];
      convergenceRate = pow(convergenceRate, root);
      
      if (convergenceRate > 0.999f) fail = 1;
    }// end of if (iter>749)

    iter++;

    // timing
    cudaThreadSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    time_sum = time_sum + time;
  }// end of Main Loop

  *p_iter = iter;
  *p_time_sum = time_sum;
}// end of ALPS_S_smv




