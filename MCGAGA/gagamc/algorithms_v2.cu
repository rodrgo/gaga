/* Copyright 2014 Jeffrey D. Blanchard and Jared Tanner
 *   
 * GPU Accelerated Greedy Algorithms for Matrix Completion
 *
 * Licensed under the GAGAMC License available at gaga4cs.org and included as GAGAMC_license.txt.
 *
 * In  order to use the GAGAMC library, or any of its constituent parts, a user must
 * agree to abide by a set of conditions of use. The library is available at no cost 
 * for ``Internal'' use. ``Internal'' use of the library is defined to be use of the 
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
 * and include a copy of GAGAMC_license.txt.
 */


/*
*********************************************************************
**                      NIHT_MC_S_entry                                 **
**      Single Precision Normalized Iterative Hard Thresholding    **
**                with general matrices                            **
*********************************************************************
*/

inline void NIHT_MC_S_entry(float *d_Mat, float *Grad, float *Grad_proj, float *d_Y, float *d_U, float *d_S, float *d_V, int *d_A, float *d_y, float *d_u, float *d_u_prev, float *d_v, float *residNorm_prev, const int m, const int n, const int r, const int p, const int mn, const int maxiter, const float resid_tol, const int SVDmaxIter, const float SVDTol, curandGenerator_t SVDgen, int * p_iter, float * p_time_sum, dim3 threadsPerBlockp, dim3 numBlocksp, dim3 threadsPerBlockm, dim3 numBlocksm, dim3 threadsPerBlocknr, dim3 numBlocksnr, dim3 threadsPerBlockmn, dim3 numBlocksmn, cublasHandle_t handle)
{
/*
**********************
**  Initialization  **
**********************
*/



  float err;

  int iter = *p_iter;
  float time_sum=*p_time_sum;

  cublasSnrm2(handle, mn, d_Y, 1, &err);
  SAFEcublas("cublasSnrm2 in NIHT_MC_S_entry initialization");

  float err_start = err;

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
//  int cycleFlag = 1;
// **********************


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
************************************************
** Set the Initial Subspace and Approximation **
************************************************
*/

/*
  PartialSVD(d_U, d_S, d_V, d_Y, d_u, d_u_prev, d_v, m, n, r, SVDmaxIter, SVDTol, SVDgen, threadsPerBlockm, numBlocksm);
  SAFEcuda("PartialSVD in initialization of NIHT_MC_S_entry.");

  USVt_product(d_Mat, d_U, d_S, d_V, m, n, r, threadsPerBlocknr, numBlocksnr);
  SAFEcuda("USVt_product in initialization of NIHT_MC_S_entry.");
*/
/*
printf("err = %f and err_start = %f\n",err,err_start);
printf("loop check %d%d%d%d%d\n",(err > resid_tol),(iter < maxiter),(err <(100*err_start)),(residNorm_diff > .01*resid_tol),(fail == 0));
*/

/* 
********************************
** Main Steepest Descent Loop **
********************************
*/

  while ( (err > resid_tol*err_start) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  RestrictedSD_MC_S_entry(d_Mat, Grad, Grad_proj, d_Y, d_U, d_S, d_V, d_A, d_y, d_u, d_u_prev, d_v, residNorm_prev, m, n, r, p, mn, SVDmaxIter, SVDTol, SVDgen, threadsPerBlockp, numBlocksp, threadsPerBlockm, numBlocksm, threadsPerBlocknr, numBlocksnr, threadsPerBlockmn, numBlocksmn, handle);
  SAFEcuda("PartialSVD in initialization of NIHT_MC_S_entry.");

//printf("In the while loop for iteration %d and err = %f before the update",iter,err);
  err = residNorm_prev[15];
//printf(" and err = %f after the update.",err);

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  } 
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
/*
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) ) {
	  cycleFlag=0;
        } // end if
   } // end for
*/
// **********************************

  if (iter>99){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1; 
  } // ends if (iter>149)


  iter++;

//  cout << "iter = " << iter << "   err = " << err << endl;
  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


//printf("loop check %d%d%d%d%d\n",(err > resid_tol),(iter < maxiter),(err <(100*err_start)),(residNorm_diff > .01*resid_tol),(fail == 0));
  } // ends while loop


//cout << "resid_tol = " << resid_tol << endl;

//  printf("\n At the end of NIHT_MC we have iter = %d and time sum = %f\n", iter, time_sum);

  *p_iter = iter;
  *p_time_sum = time_sum;

}







/*
*********************************************************************
**                    CGIHT_MC_S_entry                             **
**      Single Precision Normalized Iterative Hard Thresholding    **
**                with general matrices                            **
*********************************************************************
*/

inline void CGIHT_MC_S_entry(float *d_Mat, float *Grad, float *Grad_proj, float *Grad_prev, float *Grad_prev_proj, float *d_Y, float *d_U, float *d_S, float *d_V, int *d_A, float *d_y, float *d_y_work, float *d_u, float *d_u_prev, float *d_v, float *residNorm_prev, const int m, const int n, const int r, const int p, const int mn, const int maxiter, const float resid_tol, const int SVDmaxIter, const float SVDTol, curandGenerator_t SVDgen, int * p_iter, float * p_time_sum, dim3 threadsPerBlockp, dim3 numBlocksp, dim3 threadsPerBlockm, dim3 numBlocksm, dim3 threadsPerBlocknr, dim3 numBlocksnr, dim3 threadsPerBlockmn, dim3 numBlocksmn,cublasHandle_t handle)
{
/*
**********************
**  Initialization  **
**********************
*/


  float err;

  int iter = *p_iter;
  float time_sum=*p_time_sum;

  cublasSnrm2(handle, m, d_Y, 1, &err);
  SAFEcublas("cublasSnrm2 in CGIHT_MC_S_entry initialization");

  float err_start = err;

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
//  int cycleFlag = 1;
// **********************


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif


/*
************************************************
** Set the Initial Subspace and Approximation **
************************************************
*/

/*
  PartialSVD(d_U, d_S, d_V, d_Y, d_u, d_u_prev, d_v, m, n, r, SVDmaxIter, SVDTol, SVDgen, threadsPerBlockm, numBlocksm);
  SAFEcuda("PartialSVD in initialization of CGIHT_MC_S_entry.");

  USVt_product(d_Mat, d_U, d_S, d_V, m, n, r, threadsPerBlocknr, numBlocksnr);
  SAFEcuda("USVt_product in initialization of CGIHT_MC_S_entry.");
*/

/*
************************************************
** First iteration is a steepest descent step **
************************************************
*/

//  cout << "iter = " << iter << "   err = " << err << endl;
  // Run on step of steepest descent with the output for the search direction given in Grad_prev and Grad_prev_proj for use in CG iterations
  RestrictedSD_MC_S_entry(d_Mat, Grad_prev, Grad_prev_proj, d_Y, d_U, d_S, d_V, d_A, d_y, d_u, d_u_prev, d_v, residNorm_prev, m, n, r, p, mn, SVDmaxIter, SVDTol, SVDgen, threadsPerBlockp, numBlocksp, threadsPerBlockm, numBlocksm, threadsPerBlocknr, numBlocksnr, threadsPerBlockmn, numBlocksmn, handle);
  SAFEcuda("RestrictedSD in first iteration of CGIHT_MC_S_entry.");

  err = residNorm_prev[15];


// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

  iter++;

//  cout << "iter = " << iter << "   err = " << err << endl;
/* 
********************************
** Main Steepest Descent Loop **
********************************
*/

  while ( (err > resid_tol*err_start) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) ) // & (cycleFlag) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  UnrestrictedCG_MC_S_entry(d_Mat, Grad, Grad_proj, Grad_prev, Grad_prev_proj, d_Y, d_U, d_S, d_V, d_A, d_y, d_y_work, d_u, d_u_prev, d_v, residNorm_prev, m, n, r, p, mn, SVDmaxIter, SVDTol, SVDgen, threadsPerBlockp, numBlocksp, threadsPerBlockm, numBlocksm, threadsPerBlocknr, numBlocksnr, threadsPerBlockmn, numBlocksmn, handle);
  SAFEcuda("UnrestrictedCG in initialization of CGIHT_MC_S_entry.");


  err = residNorm_prev[15];

// check for convergence 
  for (int j = 0; j<15; j++) {
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

// **********************************
/*
  for (int j = 0; j<15; j++) {
	if ( (residNorm_prev[15]==residNorm_prev[j]) && (iter > 4) ) {
	  cycleFlag=0;
        } // end if
   } // end for
*/
// **********************************

  if (iter>99){
	root = 1/15.0f;
  	convergenceRate = (residNorm_prev[15]/residNorm_prev[0]);
  	convergenceRate = pow(convergenceRate, root);
	if (convergenceRate > 0.999f) fail = 1; 
  }

  iter++;

//  cout << "iter = " << iter << "   err = " << err << endl;

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  time_sum = time_sum + time;


  }


// cout << "resid_tol = " << resid_tol << endl;
  *p_iter = iter;
  *p_time_sum = time_sum;

}














/*
************************************************
**              IHT_MC_S_entry                     **
** Single Precision IHT with general matrices **
************************************************
*/


inline void IHT_MC_S_entry(float *d_Mat, float *Grad, float *Grad_proj, float *d_Y, float *d_U, float *d_S, float *d_V, int *d_A, float *d_y, float *d_u, float *d_u_prev, float *d_v, float *residNorm_prev, const float mu, const int m, const int n, const int r, const int p, const int mn, const int maxiter, const int resid_tol, const int SVDmaxIter, const float SVDTol, curandGenerator_t SVDgen, int * p_iter, float * p_time_sum, dim3 threadsPerBlockp, dim3 numBlocksp, dim3 threadsPerBlockm, dim3 numBlocksm, dim3 threadsPerBlocknr, dim3 numBlocksnr, dim3 threadsPerBlockmn, dim3 numBlocksmn, cublasHandle_t handle)
{

  int iter = *p_iter;
  float time_sum=*p_time_sum;
  float err;

  cublasScopy(handle, mn, d_Y, 1, Grad, 1);
  SAFEcublas("cublasScopy in IHT_MC_S_entry initializiation");

  cublasSnrm2(handle,mn, Grad, 1, &err);
  SAFEcublas("cublasSnrm2 in IHT_MC_S_entry initialization");

  float err_start = err;

  float residNorm_diff = 1.0f;

  float residNorm_evolution[16];

  for (int j=0; j<16; j++) {
	residNorm_prev[j]=0;
	residNorm_evolution[j]=1.0f;
  }

// Some stopping parameters
  int fail = 0;
  float root, convergenceRate;

	float minus_one= -1.0f;


#ifdef VERBOSE
if (verb>3) {
  printf("The initial residual error = %f\n",err);
}
#endif




/*
************************************************
** Set the Initial Subspace and Approximation **
************************************************
*/

/*

IT SEEMS NO INITIALIZATION IS NEEDED, BUT THAT SEEMS WEIRD.  I GUESS THE INITIALIZATION WAS TO DECLARE THAT GRAD=Y?

  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in IHT_MC_S_entry Set Linear Bins");

  float max_value = MaxMagnitude(grad,n);

  float slope = ((num_bins-1)/(max_value));
*/

/*
*******************
** Main IHT Loop **
*******************
*/
/*
printf("err = %f and err_start = %f\n",err,err_start);
printf("iter = %d and maxiter = %d\n so that (iter<maxiter)=%d",iter,maxiter,(iter<maxiter));
printf("loop check %d%d%d%d%d\n",(err > resid_tol),(iter < maxiter),(err <(100*err_start)),(residNorm_diff > .01*resid_tol),(fail == 0));
*/
  while ( (err > resid_tol*err_start) & (iter < maxiter) & (err < (100*err_start)) & (residNorm_diff > .01*resid_tol) & (fail == 0) )
  {


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  

  cublasSaxpy(handle, mn, &mu, Grad, 1, d_Mat, 1);
  SAFEcublas("First cublasSaxpy in IHT_MC_S_entry loop");

  PartialSVD(d_U, d_S, d_V, d_Mat, d_u, d_u_prev, d_v, m, n, r, SVDmaxIter, SVDTol, SVDgen, threadsPerBlockm, numBlocksm, handle);
  SAFEcuda("PartialSVD in initialization of NIHT_MC_S_entry.");

  USVt_product(d_Mat, d_U, d_S, d_V, m, n, r, threadsPerBlocknr, numBlocksnr, handle);
  SAFEcuda("USVt_product in initialization of NIHT_MC_S_entry.");

  A_entry_mat(Grad_proj, d_Mat, d_A, mn, p, numBlocksmn, threadsPerBlockmn, numBlocksp, threadsPerBlockp);
  SAFEcuda("A_gen in IHT_MC_S_entry loop");

  cublasScopy(handle, mn, d_Y, 1, Grad, 1);
  SAFEcublas("Second cublasScopy in IHT_MC_S_entry loop");

  cublasSaxpy(handle, mn, &minus_one, Grad_proj, 1, Grad, 1);
  SAFEcublas("Second cublasSaxpy in IHT_MC_S_entry loop");

  cublasSnrm2(handle, mn, Grad, 1, &err);
  SAFEcublas("cublasSnrm2 in IHT_MC_S_entry loop");


// recording the convergence of the residual
  for (int j = 0; j<15; j++) {
	residNorm_prev[j] = residNorm_prev[j+1];
	residNorm_evolution[j]=residNorm_evolution[j+1];
  }
  residNorm_prev[15]=err;
  residNorm_evolution[15] = residNorm_prev[14]-residNorm_prev[15];

  residNorm_diff = max_list(residNorm_evolution, 16);

  if (iter>99){
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





