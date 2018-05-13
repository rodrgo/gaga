
/*
****************************************************************************************
**                                  CGIHT_robust                                      **
**               Conjugate Gradient Iterative Hard Thresholding                       **
**                               for general matrices                                 **
****************************************************************************************
*/



inline void CGIHT_robust(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float * grad_prev_thres, float *d_y, float *resid, float *resid_update, int *d_rows, int *d_cols, float *d_vals, int *d_bin, int * d_bin_counters, int *d_bin_grad, int * d_bin_counters_grad, int * h_bin_counters, float *residNorm_prev, const float resid_tol, const int maxiter, const int num_bins, const int k, const int m, const int n, const int nz, float noise_level, const int restartFlag, int *p_iter, float mu, float err, int *p_sum, float *p_time_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
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

  //float err_start = err;

  //float resid_resid_tol = resid_tol;

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

  // Variables for l1-stopping conditions
  float h_sigma2_n = noise_level*noise_level;
  float norm_1_res_mean = ((float) m)*sqrtf(h_sigma2_n)*sqrtf(2/3.1415926535);
  float norm_1_res_sd = sqrtf(((float) m)*h_sigma2_n*(1 - sqrtf(2/3.1415926535)));
  float norm_1_res = cublasSasum(m, resid, 1);

  float tol = resid_tol;
  float err_start_1 = norm_1_res;

  // Consider removing residNorm_diff > 0.01*resid_tol condition
  //while ( (!(norm_1_res - norm_1_res_mean <= tol*norm_1_res_sd)) & (iter < maxiter) & (norm_1_res < (100*err_start_1)) & (residNorm_diff > .01*noise_level*resid_tol) & (fail == 0) & (cycleFlag) )
  while ( (!(norm_1_res - norm_1_res_mean <= tol*norm_1_res_sd)) & (iter < maxiter) & (norm_1_res < (100*err_start_1)) & (residNorm_diff > .01*noise_level*resid_tol) & (fail == 0) & (cycleFlag) )
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
  norm_1_res = cublasSasum(m, resid, 1);

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



