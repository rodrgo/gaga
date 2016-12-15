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



// ******************** Simple Host function calls *********


/* **********************************************************
**  Functions for writing testing data to text files    *****
************************************************************ */

void File_output(FILE* foutput, int k, int m, int n, int vecDistribution, float* errors, float* timings, int iter, float conv_rate, int *supportCheck, unsigned int seed, char* algstr){
  fprintf(foutput,"%s_S_dct output: ",algstr);
  fprintf(foutput,"k %d, m %d, n %d, vecDistribution %d ",k,m,n,vecDistribution);
  fprintf(foutput,"errorl1 %0.7e errorl2 %0.7e errorlinf %0.7e ",errors[0],errors[1],errors[2]);
  fprintf(foutput,"timeALG %0.7e timeIter %0.7e timeTotal %0.7e ",timings[0],timings[1],timings[2]);
  fprintf(foutput,"iterations %d ",iter);
  fprintf(foutput,"converge_rate %0.7e ", conv_rate);
  fprintf(foutput,"TruePos %d FalsePos %d TrueNeg %d FalseNeg %d ", supportCheck[0], supportCheck[1], supportCheck[2], supportCheck[3]);
  fprintf(foutput,"seed %u \n", seed);
}




void File_output_smv(FILE* foutput, int k, int m, int n, int vecDistribution, float* errors, float* timings, int iter, float conv_rate, int *supportCheck, unsigned int seed, char* algstr, int p, int matrixEnsemble, float band_percentage, float noise_level){
  fprintf(foutput,"%s_S_smv output: ",algstr);
  fprintf(foutput,"k %d, m %d, n %d, vecDistribution %d nonzeros_per_column %d matrixEnsemble %d ",k,m,n,vecDistribution,p,matrixEnsemble);
  fprintf(foutput,"errorl1 %0.7e errorl2 %0.7e errorlinf %0.7e ",errors[0],errors[1],errors[2]);
  fprintf(foutput,"timeALG %0.7e timeIter %0.7e timeTotal %0.7e ",timings[0],timings[1],timings[2]);
  fprintf(foutput,"iterations %d ",iter);
  fprintf(foutput,"converge_rate %0.7e ", conv_rate);
  fprintf(foutput,"TruePos %d FalsePos %d TrueNeg %d FalseNeg %d ", supportCheck[0], supportCheck[1], supportCheck[2], supportCheck[3]);
  fprintf(foutput,"seed %u ", seed);

  // parallel-L0 and robust-L0 numerical experiments require to have this additional output 
  fprintf(foutput,"band_percentage %1.2f ", band_percentage);
  fprintf(foutput,"noise_level %2.4f ", noise_level);
  fprintf(foutput,"\n");
}



void File_output_gen(FILE* foutput, int k, int m, int n, int matrixEnsemble, int vecDistribution, float* errors, float* timings, int iter, float conv_rate, int *supportCheck, unsigned int seed, char* algstr){
  fprintf(foutput,"%s_S_gen output: ",algstr);
  fprintf(foutput,"k %d, m %d, n %d, vecDistribution %d matrixEnsemble %d ",k,m,n,vecDistribution,matrixEnsemble);
  fprintf(foutput,"errorl1 %0.7e errorl2 %0.7e errorlinf %0.7e ",errors[0],errors[1],errors[2]);
  fprintf(foutput,"timeALG %0.7e timeIter %0.7e timeTotal %0.7e ",timings[0],timings[1],timings[2]);
  fprintf(foutput,"iterations %d ",iter);
  fprintf(foutput,"converge_rate %0.7e ", conv_rate);
  fprintf(foutput,"TruePos %d FalsePos %d TrueNeg %d FalseNeg %d ", supportCheck[0], supportCheck[1], supportCheck[2], supportCheck[3]);
  fprintf(foutput,"seed %u \n", seed);
}






void File_output_timings(FILE* foutput, int k, int m, int n, int vecDistribution, float* errors, float* timings, int iter, float conv_rate, int *supportCheck, unsigned int seed, float* time_per_iteration, float* time_supp_set, char* algstr, float alpha, int supp_flag){
  fprintf(foutput,"%s_S_dct output: ",algstr);
  fprintf(foutput,"k %d, m %d, n %d, vecDistribution %d ",k,m,n,vecDistribution);
  fprintf(foutput,"errorl1 %0.7e errorl2 %0.7e errorlinf %0.7e ",errors[0],errors[1],errors[2]);
  fprintf(foutput,"timeALG %0.7e timeIter %0.7e timeTotal %0.7e ",timings[0],timings[1],timings[2]);
  fprintf(foutput,"iterations %d ",iter);
  fprintf(foutput,"converge_rate %0.7e ", conv_rate);
  fprintf(foutput,"TruePos %d FalsePos %d TrueNeg %d FalseNeg %d ", supportCheck[0], supportCheck[1], supportCheck[2], supportCheck[3]);
  fprintf(foutput,"seed %u ", seed);
  fprintf(foutput,"alpha %0.7e ", alpha);
  fprintf(foutput,"supp_flag %d \n", supp_flag);

  for (int j=0; j<iter; j++){
    fprintf(foutput,"iteration %d time_iteration %0.7e time_supp_set %0.7e\n",j,time_per_iteration[j],time_supp_set[j]);
  }

}





void File_output_timings_smv(FILE* foutput, int k, int m, int n, int vecDistribution, float* errors, float* timings, int iter, float conv_rate, int *supportCheck, unsigned int seed, float* time_per_iteration, float* time_supp_set, char* algstr, float alpha, int supp_flag, int p, int matrixEnsemble){
  fprintf(foutput,"%s_S_smv output: ",algstr);
  fprintf(foutput,"k %d, m %d, n %d, vecDistribution %d ",k,m,n,vecDistribution);
  fprintf(foutput,"errorl1 %0.7e errorl2 %0.7e errorlinf %0.7e ",errors[0],errors[1],errors[2]);
  fprintf(foutput,"timeALG %0.7e timeIter %0.7e timeTotal %0.7e ",timings[0],timings[1],timings[2]);
  fprintf(foutput,"iterations %d ",iter);
  fprintf(foutput,"converge_rate %0.7e ", conv_rate);
  fprintf(foutput,"TruePos %d FalsePos %d TrueNeg %d FalseNeg %d ", supportCheck[0], supportCheck[1], supportCheck[2], supportCheck[3]);
  fprintf(foutput,"seed %u ", seed);
  fprintf(foutput,"alpha %0.7e ", alpha);
  fprintf(foutput,"supp_flag %d ", supp_flag);
  fprintf(foutput,"NonzerosPerCol %d ", p);
  fprintf(foutput,"matrixEnsemble %d \n", matrixEnsemble);

  for (int j=0; j<iter; j++){
    fprintf(foutput,"iteration %d time_iteration %0.7e time_supp_set %0.7e\n",j,time_per_iteration[j],time_supp_set[j]);
  }

}



void File_output_timings_gen(FILE* foutput, int k, int m, int n, int matrixEnsemble, int vecDistribution, float* errors, float* timings, int iter, float conv_rate, int *supportCheck, unsigned int seed, float* time_per_iteration, float* time_supp_set, char* algstr, float alpha, int supp_flag){
  fprintf(foutput,"%s_S_gen output: ",algstr);
  fprintf(foutput,"k %d, m %d, n %d, vecDistribution %d ",k,m,n,vecDistribution);
  fprintf(foutput,"errorl1 %0.7e errorl2 %0.7e errorlinf %0.7e ",errors[0],errors[1],errors[2]);
  fprintf(foutput,"timeALG %0.7e timeIter %0.7e timeTotal %0.7e ",timings[0],timings[1],timings[2]);
  fprintf(foutput,"iterations %d ",iter);
  fprintf(foutput,"converge_rate %0.7e ", conv_rate);
  fprintf(foutput,"TruePos %d FalsePos %d TrueNeg %d FalseNeg %d ", supportCheck[0], supportCheck[1], supportCheck[2], supportCheck[3]);
  fprintf(foutput,"seed %u ", seed);
  fprintf(foutput,"alpha %0.7e ", alpha);
  fprintf(foutput,"supp_flag %d ", supp_flag);
  fprintf(foutput,"matrixEnsemble %d \n", matrixEnsemble);

  for (int j=0; j<iter; j++){
    fprintf(foutput,"iteration %d time_iteration %0.7e time_supp_set %0.7e\n",j,time_per_iteration[j],time_supp_set[j]);
  }

}


void File_output_timings_HTP(FILE* foutput, int k, int m, int n, int vecDistribution, float* errors, float* timings, int iter, float conv_rate, int *supportCheck, unsigned int seed, float* time_per_iteration, float* time_supp_set, float* cg_per_iteration, float* time_for_cg, char* algstr, float alpha, int supp_flag){
  fprintf(foutput,"%s_S_dct output: ",algstr);
  fprintf(foutput,"k %d, m %d, n %d, vecDistribution %d ",k,m,n,vecDistribution);
  fprintf(foutput,"errorl1 %0.7e errorl2 %0.7e errorlinf %0.7e ",errors[0],errors[1],errors[2]);
  fprintf(foutput,"timeALG %0.7e timeIter %0.7e timeTotal %0.7e ",timings[0],timings[1],timings[2]);
  fprintf(foutput,"iterations %d ",iter);
  fprintf(foutput,"converge_rate %0.7e ", conv_rate);
  fprintf(foutput,"TruePos %d FalsePos %d TrueNeg %d FalseNeg %d ", supportCheck[0], supportCheck[1], supportCheck[2], supportCheck[3]);
  fprintf(foutput,"seed %u ", seed);
  fprintf(foutput,"alpha %0.7e ", alpha);
  fprintf(foutput,"supp_flag %d \n", supp_flag);

  for (int j=0; j<iter; j++){
    fprintf(foutput,"iteration %d time_iteration %0.7e time_supp_set %0.7e cg_per_iteration %0.0f time_for_cg %0.7e\n",j,time_per_iteration[j],time_supp_set[j],cg_per_iteration[j],time_for_cg[j]);
  }

}




void File_output_timings_HTP_smv(FILE* foutput, int k, int m, int n, int vecDistribution, float* errors, float* timings, int iter, float conv_rate, int *supportCheck, unsigned int seed, float* time_per_iteration, float* time_supp_set, float* cg_per_iteration, float* time_for_cg, char* algstr, float alpha, int supp_flag, int p, int matrixEnsemble){
  fprintf(foutput,"%s_S_smv output: ",algstr);
  fprintf(foutput,"k %d, m %d, n %d, vecDistribution %d ",k,m,n,vecDistribution);
  fprintf(foutput,"errorl1 %0.7e errorl2 %0.7e errorlinf %0.7e ",errors[0],errors[1],errors[2]);
  fprintf(foutput,"timeALG %0.7e timeIter %0.7e timeTotal %0.7e ",timings[0],timings[1],timings[2]);
  fprintf(foutput,"iterations %d ",iter);
  fprintf(foutput,"converge_rate %0.7e ", conv_rate);
  fprintf(foutput,"TruePos %d FalsePos %d TrueNeg %d FalseNeg %d ", supportCheck[0], supportCheck[1], supportCheck[2], supportCheck[3]);
  fprintf(foutput,"seed %u ", seed);
  fprintf(foutput,"alpha %0.7e ", alpha);
  fprintf(foutput,"supp_flag %d ", supp_flag);
  fprintf(foutput,"NonzerosPerCol %d ", p);
  fprintf(foutput,"matrixEnsemble %d \n", matrixEnsemble);

  for (int j=0; j<iter; j++){
    fprintf(foutput,"iteration %d time_iteration %0.7e time_supp_set %0.7e cg_per_iteration %0.0f time_for_cg %0.7e\n",j,time_per_iteration[j],time_supp_set[j],cg_per_iteration[j],time_for_cg[j]);
  }

}




void File_output_timings_HTP_gen(FILE* foutput, int k, int m, int n, int matrixEnsemble, int vecDistribution, float* errors, float* timings, int iter, float conv_rate, int *supportCheck, unsigned int seed, float* time_per_iteration, float* time_supp_set, float* cg_per_iteration, float* time_for_cg, char* algstr, float alpha, int supp_flag){
  fprintf(foutput,"%s_S_gen output: ",algstr);
  fprintf(foutput,"k %d, m %d, n %d, vecDistribution %d ",k,m,n,vecDistribution);
  fprintf(foutput,"errorl1 %0.7e errorl2 %0.7e errorlinf %0.7e ",errors[0],errors[1],errors[2]);
  fprintf(foutput,"timeALG %0.7e timeIter %0.7e timeTotal %0.7e ",timings[0],timings[1],timings[2]);
  fprintf(foutput,"iterations %d ",iter);
  fprintf(foutput,"converge_rate %0.7e ", conv_rate);
  fprintf(foutput,"TruePos %d FalsePos %d TrueNeg %d FalseNeg %d ", supportCheck[0], supportCheck[1], supportCheck[2], supportCheck[3]);
  fprintf(foutput,"seed %u ", seed);
  fprintf(foutput,"alpha %0.7e ", alpha);
  fprintf(foutput,"supp_flag %d ", supp_flag);
  fprintf(foutput,"matrixEnsemble %d \n", matrixEnsemble);

  for (int j=0; j<iter; j++){
    fprintf(foutput,"iteration %d time_iteration %0.7e time_supp_set %0.7e cg_per_iteration %0.0f time_for_cg %0.7e\n",j,time_per_iteration[j],time_supp_set[j],cg_per_iteration[j],time_for_cg[j]);
  }

}




// ***************** Misc. functions *************



float max_list(float* list, const int N)
{
  int j=0;
  float max=list[0];

  for (j=1; j<N; j++){
	if (list[j]>max) max = list[j];
  }

  return max;
}










// ************** SAFETY ERROR CHECK FUNCTIONS *************


void check_malloc_int(int *pointer, const char *message)
{
  if ( pointer == NULL ) {
	printf("Malloc failed for %s.\n", message);
  }
}

void check_malloc_float(float *pointer, const char *message)
{
  if ( pointer == NULL ) {
	printf("Malloc failed for %s.\n", message);
  }
}

void check_malloc_double(double *pointer, const char *message)
{
  if ( pointer == NULL ) {
	printf("Malloc failed for %s.\n", message);
  }
}
  
void check_cudaMalloc(const char *message)
{
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
	fprintf(stderr, "Error: cudaMalloc failed for %s: %d\n", message, status);
  }
}

void Check_CUDA_Error(const char *message)
{
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
	fprintf(stderr, "Error: %s: %s\n", message, cudaGetErrorString(error) );
	exit(-1);
  }
}

void Check_CUBLAS_Error(const char *message)
{
  cublasStatus status = cublasGetError();
  if (status != CUBLAS_STATUS_SUCCESS) {
	fprintf (stderr, "Error: %s: %d\n", message, status);
	exit(-1);
  }
}

void Check_CURAND_Error(curandStatus_t curandCheck, const char *message)
{
  if (curandCheck != CURAND_STATUS_SUCCESS) {
	fprintf (stderr, "Error: %s: %d\n", message, curandCheck);
	exit(-1);
  }
}


  
void check_cudaMalloc2(cudaError_t status, const char *message)
{
  if (status != cudaSuccess) {
	fprintf(stderr, "Error: cudaMalloc failed for %s: %d\n", message, status);
  }
}

inline void SAFEcudaMalloc2(cudaError_t status, const char *message)
{
#ifdef SAFE
check_cudaMalloc2(status, message);
#endif
}



inline void SAFEcudaMalloc(const char *message)
{
#ifdef SAFE
check_cudaMalloc(message);
#endif
}

inline void SAFEcuda(const char *message)
{ 
#ifdef SAFE
Check_CUDA_Error(message);
#endif
}

inline void SAFEcublas(const char *message)
{
#ifdef SAFE
Check_CUBLAS_Error(message);
#endif
}

inline void SAFEcurand(curandStatus_t curandCheck, const char *message)
{
#ifdef SAFE
Check_CURAND_Error(curandCheck, message);
#endif
}

inline void SAFEmalloc_int(int * pointer, const char *message)
{
#ifdef SAFE
check_malloc_int(pointer, message);
#endif
}

inline void SAFEmalloc_float(float *pointer, const char *message)
{
#ifdef SAFE
check_malloc_float(pointer, message);
#endif
}

inline void SAFEmalloc_double(double *pointer, const char *message)
{
#ifdef SAFE
check_malloc_double(pointer, message);
#endif
}






/*
******************************************************
**  Subroutines for Sparse Approximation Algorithms **
******************************************************
*/


inline void Threshold(float *d_vec, int *d_bin, const int k_bin, const int n, dim3 numBlocks, dim3 threadsPerBlock) 
{
  threshold <<< numBlocks, threadsPerBlock >>> ((float*)d_vec, (int*)d_bin, k_bin, n);
  SAFEcuda("threshold in Threshold");
}



inline float MaxMagnitude(float * d_vec, const int n)
{
  int ind_abs_max = cublasIsamax(n, d_vec, 1) -1;
  SAFEcublas("cublasIsamax in MaxMagnitude");

  float max_value;
  cudaMemcpy(&max_value, d_vec+ind_abs_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to max_value in MaxMagnitude");

  return abs(max_value);

}



inline int FindSupportSet(float *d_vec, int *d_bin, int *d_bin_counters, int *h_bin_counters, const float slope, const float intercept, const float maxChange, float * minSupportValue, float * alpha, int * Max_Bin, int* p_k_bin, const int n, const int k, const int num_bins, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin) 
{
  float minVal = minSupportValue[0];
  int maxBin = Max_Bin[0];
  int k_bin = p_k_bin[0];
  int sum;
  if (minVal > maxChange) { minVal = minVal - maxChange; }
  else {

	zero_vector_int <<< numBlocks_bin, threadsPerBlock_bin >>>((int*)d_bin_counters,num_bins);
  	SAFEcuda("zero_vector_int in FindSupportSet");

	LinearBinning <<< numBlocks, threadsPerBlock >>>((float*)d_vec, (int*)d_bin, (int*)d_bin_counters, num_bins, maxBin, n, slope, intercept);
  	SAFEcuda("LinearBinning in FindSupportSet");

  	cudaMemcpy(h_bin_counters, d_bin_counters, maxBin * sizeof(int), cudaMemcpyDeviceToHost);
	SAFEcuda("cudaMemcpy in FindSupportSet");
  	k_bin = 0;
  	sum = 0;
  	while ( (sum<k) & (k_bin<maxBin) ) {
		sum = sum + h_bin_counters[k_bin];
		k_bin++;
	}
  	k_bin = k_bin-1;

  	if (sum < k) {
		alpha[0]=alpha[0]*0.5f;
		Max_Bin[0] = (int)(num_bins*(1-alpha[0]));
  	}

  	minVal = intercept - (k_bin+1)/slope;
  
  }


  *minSupportValue = minVal;
  *p_k_bin = k_bin;

  return sum;
}



/* ******************************************
A function to do the support detection and  *
threshold with thrust:sort                  *     
****************************************** */


inline float FindSupportSet_sort(float *d_vec, float *d_vec_thres, thrust::device_ptr<float>& d_sort, int *d_bin, float T, const float maxChange,  const int n, const int k,  dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin) 
{
  if (T > maxChange) { 
	T = T - maxChange;
	threshold_and_support<<< numBlocks, threadsPerBlock >>>(d_vec, d_bin, n, T); 
	SAFEcuda("threshold_and_support in FindSupportSet_sor");
  }
  else {
  	magnitudeCopy<<< numBlocks, threadsPerBlock >>>(d_vec_thres, d_vec, n);
	SAFEcuda("magnitudeCopy in FindSupportSet_sort");
  	thrust::sort(d_sort, d_sort+n);
  	T=d_sort[n-k];
  	one_vector_int<<< numBlocks, threadsPerBlock >>>(d_bin, n);
	SAFEcuda("one_vector_int in FindSupportSet_sort");
  	threshold_and_support<<< numBlocks, threadsPerBlock >>>(d_vec, d_bin, n, T);
	SAFEcuda("threshold_and_support (2nd) in FindSupportSet_sort");
  }

  return T;
}







inline float RestrictedSD_S_dct(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const int num_bins, const int k, const int m, const int n, float mu, float err, int k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/

// This function computes one step of the Steepest Descent solution of Ax=y where 
// x is restricted to being supported on the entries in d_bin that have 
// values larger than k_bin.  Its input is an initial guess to the solution, 
// the vector y, the ability to compute Ax, as well as d_bin and k_bin.
// Various other variables are used along the way, but their input values 
// are not considered valid.
// IMPORTANT: d_vec is considered to be thresholded already.  
//            d_vec_thres is used as working memory for other values



  
  // The following lines compute resid = y - A_dct * d_vec
  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in RestrictedSD_S_dct");


  // resid_update is used to store A * d_vec so that it can be used to compute the resid
  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct in RestrictedSD_S_dct");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("First cublasSaxpy in RestrictedSD_S_dct");



  // recording the convergence of the residual
  for (int j = 0; j<15; j++) residNorm_prev[j] = residNorm_prev[j+1];
  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in RestrictedSD_S_dct");
  residNorm_prev[15]=err;



  // Now need to compute the gradient and its restriction 
  // to where d_bin is larger than k_bin
  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in RestrictedSD_S_dct");

  // d_vec_thres is being used to store a thresholded version of grad
  cublasScopy(n, grad, 1, d_vec_thres, 1);
  SAFEcublas("Copy grad to d_vec_thres in RestrictedSD_S_dct");
  Threshold(d_vec_thres, d_bin, k_bin, n, numBlocks, threadsPerBlock);


  // Now need to compute A_dct * the restricted gradient (stored in d_vec_thres)
  // This vector will only be used momentarily to compute the steepest descent 
  // step size.  resid_update will be used to store this vector.
  A_dct(resid_update, d_vec_thres, n, m, d_rows);
  SAFEcuda("A_dct (2nd) in RestrictedSD_S_dct");
  

  // The Steepest Descent step length is the square of 
  // (the two norm of the restricted gradient divided by 
  //  the two norm of A times the restriced gradient).
  // This variable should be stored in mu.
  // We first use err to temporarily store the denominator.
  err = cublasSnrm2(m, resid_update, 1);
  SAFEcublas("cublasSnrm2 (2nd) in RestrictedSD_S_dct");
  mu = cublasSnrm2(n, d_vec_thres, 1);
  SAFEcublas("cublasSnrm2 (3rd) in RestrictedSD_S_dct");
  if (mu < 400 * err){
    mu = mu / err;
    mu = mu * mu;
  }
  else
    mu = 0;


  // Now need to add mu times the gradient to d_vec
  cublasSaxpy(n, mu, grad, 1, d_vec, 1);
  SAFEcublas("Second cublasSaxpy in Steepest Descent in RestrictedSD_S_dct");

  mu = 2 * mu * MaxMagnitude(grad,n);
  return mu;

}









inline float RestrictedSD_S_smv(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_cols, float *d_vals, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const int num_bins, const int k, const int m, const int n, const int nz, float mu, float err, int k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/

// This function computes the Steepest Descent solution of Ax=y where 
// x is restricted to being supported on the entries in d_bin that have 
// values larger than k_bin.  Its input is an initial guess to the solution, 
// the vector y, the ability to compute Ax, as well as d_bin and k_bin.
// Various other variables are used along the way, but their input values 
// are not considered valid.
// IMPORTANT: d_vec is considered to be thresholded already.  
//            d_vec_thres is used as working memory for other values


  
  // The following lines compute resid = y - A_dct * d_vec
  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in RestrictedSD_S_smv");

  // resid_update is used to store A * d_vec so that it can be used to compute the resid
  A_smv(resid_update, d_vec, m, n,  d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in RestrictedSD_S_smv");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("First cublasSaxpy in RestrictedSD_S_smv");


// recording the convergence of the residual
  for (int j = 0; j<15; j++) residNorm_prev[j] = residNorm_prev[j+1];
  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in RestrictedSD_S_smv");
  residNorm_prev[15]=err;



  // Now need to compute the gradient and its restriction 
  // to where d_bin is larger than k_bin
  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in RestrictedSD_S_smv");

  // d_vec_thres is being used to store a thresholded version of grad
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
  SAFEcuda("zero_vector_float in RestrictedSD_S_smv");

  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin, k_bin, n);
  SAFEcuda("threshold_one of grad in RestrictedSD_S_smv");




  // Now need to compute A_dct * the restricted gradient (stored in d_vec_thres)
  // This vector will only be used momentarily to compute the steepest descent 
  // step size.  resid_update will be used to store this vector.
  A_smv(resid_update, d_vec_thres,  m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv (2nd) in RestrictedSD_S_smv");
  


  // The Steepest Descent step length is the square of 
  // (the two norm of the restricted gradient divided by 
  //  the two norm of A times the restriced gradient).
  // This variable should be stored in mu.
  // We first use err to temporarily store the denominator.
  err = cublasSnrm2(m, resid_update, 1);
  SAFEcublas("cublasSnrm2 (2nd) in RestrictedSD_S_smv");
  mu = cublasSnrm2(n, d_vec_thres, 1);
  SAFEcublas("cublasSnrm2 (3rd) in RestrictedSD_S_smv");
  if (mu < 400 * err){
    mu = mu / err;
    mu = mu * mu;
  }
  else
    mu = 0;

  // Now need to add mu times the gradient to d_vec
  cublasSaxpy(n, mu, grad, 1, d_vec, 1);
  SAFEcublas("Second cublasSaxpy in RestrictedSD_S_smv");

  mu = 2 * mu * MaxMagnitude(grad,n);
  return mu;

}




inline float RestrictedSD_S_gen(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const int num_bins, const int k, const int m, const int n, float mu, float err, int k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/

// This function computes the Steepest Descent solution of Ax=y where 
// x is restricted to being supported on the entries in d_bin that have 
// values larger than k_bin.  Its input is an initial guess to the solution, 
// the vector y, the ability to compute Ax, as well as d_bin and k_bin.
// Various other variables are used along the way, but their input values 
// are not considered valid.
// IMPORTANT: d_vec is considered to be thresholded already.  
//            d_vec_thres is used as working memory for other values


  
  // The following lines compute resid = y - A_gen * d_vec
  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in RestrictedSD_S_gen");

  // resid_update is used to store A * d_vec so that it can be used to compute the resid
  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in RestrictedSD_S_gen");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("First cublasSaxpy in RestrictedSD_S_gen");


// recording the convergence of the residual
  for (int j = 0; j<15; j++) residNorm_prev[j] = residNorm_prev[j+1];
  err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in RestrictedSD_S_gen");
  residNorm_prev[15]=err;



  // Now need to compute the gradient and its restriction 
  // to where d_bin is larger than k_bin
  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in RestrictedSD_S_gen");

  // d_vec_thres is being used to store a thresholded version of grad
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
  SAFEcuda("zero_vector_float in RestrictedSD_S_gen");

  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin, k_bin, n);
  SAFEcuda("threshold_one of grad in RestrictedSD_S_gen");




  // Now need to compute A_gen * the restricted gradient (stored in d_vec_thres)
  // This vector will only be used momentarily to compute the steepest descent 
  // step size.  resid_update will be used to store this vector.
  A_gen(resid_update, d_vec_thres, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen (2nd) in RestrictedSD_S_gen");
  


  // The Steepest Descent step length is the square of 
  // (the two norm of the restricted gradient divided by 
  //  the two norm of A times the restriced gradient).
  // This variable should be stored in mu.
  // We first use err to temporarily store the denominator.
  err = cublasSnrm2(m, resid_update, 1);
  SAFEcublas("cublasSnrm2 (2nd) in RestrictedSD_S_gen");
  mu = cublasSnrm2(n, d_vec_thres, 1);
  SAFEcublas("cublasSnrm2 (3rd) in RestrictedSD_S_gen");
  if (mu < 400 * err){
    mu = mu / err;
    mu = mu * mu;
  }
  else
    mu = 0;

  // Now need to add mu times the gradient to d_vec
  cublasSaxpy(n, mu, grad, 1, d_vec, 1);
  SAFEcublas("Second cublasSaxpy in RestrictedSD_S_gen");

  mu = 2 * mu * MaxMagnitude(grad,n);
  return mu;

}






inline void RestrictedCG_S_dct(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const int num_bins, const int k, const int m, const int n, float *mu, float err, int k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/

// This function computes one step of the Conjugate Gradient solution of Ax=y where 
// x is restricted to being supported on the entries in d_bin that have 
// values larger than k_bin.  Its input is an initial guess to the solution, 
// the vector y, the ability to compute Ax, as well as d_bin and k_bin.
// Various other variables are used along the way, but their input values 
// are not considered valid.
// IMPORTANT: d_vec is considered to be thresholded already.  
//            grad and grad_previously are also thresholded already.
//            d_vec_thres is used as working n-vector
//            resid and resid_update are working m-vectors

  float alpha, beta;

  // compute A_dct * grad and store the value in resid
  A_dct(resid, grad, n, m, d_rows);
  SAFEcuda("A_dct in RestrictedCG_S_dct");

  // update d_vec (the x)
  alpha = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot (1st) in RestrictedCG_S_dct");
  if (*mu < (1000 * alpha))
    alpha = *mu / alpha;
  else alpha = 0.0f;
  cublasSaxpy(n, alpha, grad, 1, d_vec, 1);
  SAFEcublas("cublasSaxpy (1st) in RestrictedCG_S_dct");


  // compute AT_dct * resid, which updates the residual
  AT_dct(d_vec_thres, resid, n, m, d_rows);
  SAFEcuda("AT_dct in RestrictedCG_S_dct");
  Threshold(d_vec_thres, d_bin, k_bin, n, numBlocks, threadsPerBlock);

  // update the residual, which is called grad_previous
  cublasSaxpy(n, -alpha, d_vec_thres, 1, grad_previous, 1);
  SAFEcublas("cublasSaxpy (2nd) in RestrictedCG_S_dct");


  // now need to update the search direction
  beta = *mu;
  *mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot (2nd) in RestrictedCG_S_dct");
  if (*mu < (1000 * beta))
    beta = *mu / beta;
  else beta = 0.0f;


  // want to compute grad_previous + beta * grad and store this in grad. 

  cublasSscal(n, beta, grad, 1);
  SAFEcublas("cublasSscal in RestrictedCG_S_dct");

  cublasSaxpy(n, 1.0f, grad_previous, 1, grad, 1);
  SAFEcublas("cublasSaxpy (3rd) in RestrictedCG_S_dct");

  for (int j = 0; j<15; j++)
    residNorm_prev[j] = residNorm_prev[j+1];

  // We now compute the err for CG in the weighted norm <(vec_input - vec), AT*A(vec_input - vec)> = <y-Avec,y-Avec>
  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct for resid_update in RestrictedCG_S_dct");  

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("Last cublasScopy in RestrictedCG_S_dct");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("Last cublasSaxpy in RestrictedCG_S_dct");

  err = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot (3rd) in RestrictedCG_S_dct");
  residNorm_prev[15]=err;

//cout << "End of one CG step: alpha = " << alpha << " beta = " << beta << " mu = " << *mu << " err = " << err << endl;


}




inline void RestrictedCG_S_smv(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_cols, float *d_vals, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const int num_bins, const int k, const int m, const int n, const int nz, float *mu, float err, int k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/

// This function computes one step of the Conjugate Gradient solution of Ax=y where 
// x is restricted to being supported on the entries in d_bin that have 
// values larger than k_bin.  Its input is an initial guess to the solution, 
// the vector y, the ability to compute Ax, as well as d_bin and k_bin.
// Various other variables are used along the way, but their input values 
// are not considered valid.
// IMPORTANT: d_vec is considered to be thresholded already.  
//            grad and grad_previously are also thresholded already.
//            d_vec_thres is used as working n-vector
//            resid and resid_update are working m-vectors

  float alpha, beta;

  // compute A_dct * grad and store the value in resid
  A_smv(resid, grad, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in RestrictedCG_S_smv");

  // update d_vec (the x)
  alpha = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot in RestrictedCG_S_smv");
  if (*mu < (1000 * alpha))
    alpha = *mu / alpha;
  else alpha = 0.0f;
  cublasSaxpy(n, alpha, grad, 1, d_vec, 1);
  SAFEcublas("cublasSaxpy in RestrictedCG_S_smv");

  // compute AT_dct * resid, which updates the residual
  AT_smv(d_vec_thres, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in RestrictedCG_S_smv");
  Threshold(d_vec_thres, d_bin, k_bin, n, numBlocks, threadsPerBlock);

  // update the residual, which is called grad_previous
  cublasSaxpy(n, -alpha, d_vec_thres, 1, grad_previous, 1);
  SAFEcublas("cublasSaxpy (2nd) in RestrictedCG_S_smv");

  // now need to update the search direction
  beta = *mu;
  *mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot (2nd) in RestrictedCG_S_smv");
  if (*mu < (1000 * beta))
    beta = *mu / beta;
  else beta = 0.0f;


  // want to compute grad_previous + mu * grad and store this in grad. 

  cublasSscal(n, beta, grad, 1);
  SAFEcublas("cublasSscal in RestrictedCG_S_smv");

  cublasSaxpy(n, 1.0f, grad_previous, 1, grad, 1);
  SAFEcublas("cublasSaxpy (3rd) in RestrictedCG_S_smv");

  for (int j = 0; j<15; j++)
    residNorm_prev[j] = residNorm_prev[j+1];

  // We now compute the err for CG in the weighted norm <(vec_input - vec), AT*A(vec_input - vec)> = <y-Avec,y-Avec>
  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv for resid_update in RestrictedCG_S_smv");  

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("Last cublasScopy in RestrictedCG_S_smv");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("Last cublasSaxpy in RestrictedCG_S_smv");

  err = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot (3rd) in RestrictedCG_S_smv");
  residNorm_prev[15]=err;


}




inline void RestrictedCG_S_gen(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, const int num_bins, const int k, const int m, const int n, float *mu, float err, int k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/

// This function computes one step of the Conjugate Gradient solution of Ax=y where 
// x is restricted to being supported on the entries in d_bin that have 
// values larger than k_bin.  Its input is an initial guess to the solution, 
// the vector y, the ability to compute Ax, as well as d_bin and k_bin.
// Various other variables are used along the way, but their input values 
// are not considered valid.
// IMPORTANT: d_vec is considered to be thresholded already.  
//            grad and grad_previous are also thresholded already.
//            d_vec_thres is used as working n-vector
//            resid and resid_update are working m-vectors

  float alpha, beta;

  // compute A_gen * grad and store the value in resid
  A_gen(resid, grad, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in RestrictedCG_S_gen");

  // update d_vec (the x)
  alpha = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot in RestrictedCG_S_gen");
  if (*mu < (1000 * alpha))
    alpha = *mu / alpha;
  else alpha = 0.0f;
  cublasSaxpy(n, alpha, grad, 1, d_vec, 1);
  SAFEcublas("cublasSaxpy in RestrictedCG_S_gen");

  // compute AT_gen * resid, which updates the residual
  AT_gen(d_vec_thres, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in RestrictedCG_S_gen");
  Threshold(d_vec_thres, d_bin, k_bin, n, numBlocks, threadsPerBlock);

  // update the residual, which is called grad_previous
  cublasSaxpy(n, -alpha, d_vec_thres, 1, grad_previous, 1);
  SAFEcublas("cublasSaxpy (2nd) in RestrictedCG_S_gen");

  // now need to update the search direction
  beta = *mu;
  *mu = cublasSdot(n, grad_previous, 1, grad_previous, 1);
  SAFEcublas("cublasSdot (2nd) in RestrictedCG_S_gen");
  if (*mu < (1000 * beta))
    beta = *mu / beta;
  else beta = 0.0f;


  // want to compute grad_previous + mu * grad and store this in grad. 

  cublasSscal(n, beta, grad, 1);
  SAFEcublas("cublasSscal in RestrictedCG_S_gen");

  cublasSaxpy(n, 1.0f, grad_previous, 1, grad, 1);
  SAFEcublas("cublasSaxpy (3rd) in RestrictedCG_S_gen");

  for (int j = 0; j<15; j++)
    residNorm_prev[j] = residNorm_prev[j+1];

  // We now compute the err for CG in the weighted norm <(vec_input - vec), AT*A(vec_input - vec)> = <y-Avec,y-Avec>
  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen for resid_update in RestrictedCG_S_gen");  

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("Last cublasScopy in RestrictedCG_S_gen");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("Last cublasSaxpy in RestrictedCG_S_gen");

  err = cublasSdot(m, resid, 1, resid, 1);
  SAFEcublas("cublasSdot (3rd) in RestrictedCG_S_gen");
  residNorm_prev[15]=err;


}



inline float RestrictedCGwithSupportEvolution_S_gen(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float * grad_prev_thres, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, float *mu, const int num_bins, const int k, const int m, const int n, int beta_to_zero_flag, int suppChange_flag, int k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/

// This function computes a step of Conjugate Gradient descent of Ax=y where 
// x is restricted to being supported on the entries in d_bin that have 
// values larger than k_bin.  Its input is an initial guess to the solution, 
// the past search direction, the vector y, the ability to compute Ax, as well 
// as d_bin and k_bin.
// Various other variables are used along the way, but their input values 
// are not considered valid.
// IMPORTANT: d_vec is considered to be thresholded already.  
//            grad_previous is the prior search direction and 
//            grad_previous_thresh is the thresholded version.
//            d_vec_thres is used as working n-vector
//            resid and resid_update are working m-vectors

  float alpha, beta;

  // The following lines compute resid = y - A_gen * d_vec
  // resid_update is used to store A * d_vec so that it can be used to compute the resid
  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen for resid_update in RestrictedCG_S_gen");  

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("Last cublasScopy in RestrictedCG_S_gen");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("Last cublasSaxpy in RestrictedCG_S_gen");

  // Compute the err for CG in the weighted norm <(vec_input - vec), AT*A(vec_input - vec)> = <y-Avec,y-Avec>
  for (int j = 0; j<15; j++) residNorm_prev[j] = residNorm_prev[j+1];
  float err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in RestrictedSD_S_gen");
  residNorm_prev[15]=err;

  // Now compute the CG residual r=AT*(y-Ax) and store in grad
  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in RestrictedSD_S_gen");


  // d_vec_thres is being used to store a thresholded version of grad, 
  // namely the restriction of the CG residual r to the support set
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
  SAFEcuda("zero_vector_float in RestrictedSD_S_gen");

  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin, k_bin, n);
  SAFEcuda("threshold_one of grad in RestrictedSD_S_gen");


  // compute beta to orthogonalize the search direction
  if (beta_to_zero_flag == 1){
    beta = 0.0f;
    // we must compute the inner product of the restricted CG residual for later use when computing alpha
    *mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
  } else {
    beta = *mu;
    *mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
    if (*mu < (1000 * beta))
      beta = *mu / beta;
    else beta = 0.0f;
  }

  // update the search direction
  // multiply the past search direction by beta
  cublasSscal(n, beta, grad_previous, 1);
  // add the residual
  cublasSaxpy(n, 1, grad, 1, grad_previous, 1);
  
  // compute a thresholded version of grad_previous, the current search direction
  // set grad_prev_thres to be zero
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)grad_prev_thres, n);
  // threshold_one copies the thresholded points of grad_previous onto grad_prev_thres
  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad_previous, (float*)grad_prev_thres, (int*)d_bin, k_bin, n);

  // resid_update is used to store A * grad_prev_thres 
  A_gen(resid_update, grad_prev_thres, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen for grad_previous in RestrictedSD_S_gen");

  // compute alpha for update stepsize
  alpha = cublasSdot(m, resid_update, 1, resid_update, 1);
  SAFEcublas("cublasSdot in RestrictedCG_S_gen");
  if (*mu < (1000 * alpha))
    alpha = *mu / alpha;
  else alpha = 0.0f;

  // add alpha grad_prev to d_vec
  cublasSaxpy(n, alpha, grad_previous, 1, d_vec, 1);


  // compute the maximum amount by which values could have changed
  alpha = 2 * alpha * MaxMagnitude(grad_previous,n);
  return alpha;

}




inline float UnrestrictedCG_S_gen(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float * grad_prev_thres, float *d_y, float *resid, float *resid_update, float *d_A, float *d_AT, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, float *mu, const int num_bins, const int k, const int m, const int n, int beta_to_zero_flag, int suppChange_flag, int k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/

// This function computes a step of Conjugate Gradient descent of Ax=y where 
// x is restricted to being supported on the entries in d_bin that have 
// values larger than k_bin.  Its input is an initial guess to the solution, 
// the past search direction, the vector y, the ability to compute Ax, as well 
// as d_bin and k_bin.
// Various other variables are used along the way, but their input values 
// are not considered valid.
// IMPORTANT: d_vec is considered to be thresholded already.  
//            grad_previous is the prior search direction and 
//            grad_previous_thresh is the thresholded version.
//            d_vec_thres is used as working n-vector
//            resid and resid_update are working m-vectors

  
  // The following lines compute resid = y - A_gen * d_vec
//  cublasInit();
  // resid_update is used to store A * d_vec so that it can be used to compute the resid
  A_gen(resid_update, d_vec, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in RestrictedSD_S_gen");

  cublasSaxpy(m, -1, d_y, 1, resid_update, 1);
  SAFEcublas("First cublasSaxpy in RestrictedSD_S_gen");

  cublasSscal(m, -1, resid_update, 1);
  SAFEcublas("cublasScopy in RestrictedSD_S_gen");

  for (int j = 0; j<15; j++) residNorm_prev[j] = residNorm_prev[j+1];
  float err = cublasSnrm2(m, resid_update, 1);
  SAFEcublas("cublasSnrm2 in RestrictedSD_S_gen");
  residNorm_prev[15]=err;

  // Now need to compute the gradient and its restriction 
  // to where d_bin is larger than k_bin
  AT_gen(grad, resid_update, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in RestrictedSD_S_gen");


  // d_vec_thres is being used to store a thresholded version of grad
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
  SAFEcuda("zero_vector_float in RestrictedSD_S_gen");

  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin, k_bin, n);
  SAFEcuda("threshold_one of grad in RestrictedSD_S_gen");


  // resid is used to store A * grad_prev_thres
  A_gen(resid, grad_prev_thres, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen for grad_prev_thres in RestrictedSD_S_gen");

  // resid_update is used to store A times the projection of the residual 
  A_gen(resid_update, d_vec_thres, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen for d_vec_thres in RestrictedSD_S_gen");

  // compute beta to orthogonalize the search direction
  float beta;
  if (beta_to_zero_flag == 1){
    beta = 0;
  } else {
    float beta_num = cublasSdot(m, resid, 1, resid_update, 1);
    SAFEcuda("cublasSdot for beta_num in RestrictedSD_S_gen");
    float beta_denom = cublasSdot(m, resid, 1, resid, 1);
    SAFEcuda("cublasSnrm2 for beta_denom in RestrictedSD_S_gen");
    beta = - beta_num / beta_denom;
  }

  // update the search direction
  // multiply the past search direction by beta
  cublasSscal(n, beta, grad_previous, 1);
  // add the residual
  cublasSaxpy(n, 1, grad, 1, grad_previous, 1);
  
  // compute a thresholded version of grad_previous, the current search direction
  // set grad_prev_thres to be zero
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)grad_prev_thres, n);
  // threshold_one copies the thresholded points of grad_previous onto grad_prev_thres
  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad_previous, (float*)grad_prev_thres, (int*)d_bin, k_bin, n);

  // resid is used to store A * grad_prev_thres 
  // A_gen(resid, grad_prev_thres, d_A, m, n, numBlocksm, threadsPerBlockm);
  // SAFEcuda("A_gen for grad_previous in RestrictedSD_S_gen");
 
  // update resid, which is used to store A * grad_prev_thres
  cublasSscal(m, beta, resid, 1);
  cublasSaxpy(m, 1, resid_update, 1, resid, 1);
  SAFEcuda("A_gen for grad_previous in UnrestrictedCG_S_gen");

  // compute alpha for update stepsize
  float alpha_num = cublasSdot(n, grad_prev_thres, 1, d_vec_thres, 1);
  SAFEcuda("cublasSdot for alpha_num in RestrictedSD_S_gen");
  float alpha_denom = cublasSdot(m, resid, 1, resid, 1);
  SAFEcuda("cublasSnrm2 for alpha_denom in RestrictedSD_S_gen");
  float alpha = alpha_num / alpha_denom;

  // add alpha grad_prev to d_vec
  cublasSaxpy(n, alpha, grad_previous, 1, d_vec, 1);

  // compute the value of mu to pass out in case the support remains unchanged
  // This way, both UnrestrictedCG and RestrictedCGwithSupportEvolution have the same starting points
  *mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);


  // compute the maximum amount by which values could have changed
  alpha = 2 * alpha * MaxMagnitude(grad_previous,n);
  return alpha;

}




inline float RestrictedCGwithSupportEvolution_S_dct(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float * grad_prev_thres, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, float *mu, const int num_bins, const int k, const int m, const int n, int beta_to_zero_flag, int suppChange_flag, int k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/

// This function computes a step of Conjugate Gradient descent of Ax=y where 
// x is restricted to being supported on the entries in d_bin that have 
// values larger than k_bin.  Its input is an initial guess to the solution, 
// the past search direction, the vector y, the ability to compute Ax, as well 
// as d_bin and k_bin.
// Various other variables are used along the way, but their input values 
// are not considered valid.
// IMPORTANT: d_vec is considered to be thresholded already.  
//            grad_previous is the prior search direction and 
//            grad_previous_thresh is the thresholded version.
//            d_vec_thres is used as working n-vector
//            resid and resid_update are working m-vectors

  float alpha, beta;

  // The following lines compute resid = y - A_dct * d_vec
  // resid_update is used to store A * d_vec so that it can be used to compute the resid
  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct for resid_update in RestrictedCG_S_dct");  

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("Last cublasScopy in RestrictedCG_S_dct");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("Last cublasSaxpy in RestrictedCG_S_dct");

  // Compute the err for CG in the weighted norm <(vec_input - vec), AT*A(vec_input - vec)> = <y-Avec,y-Avec>
  for (int j = 0; j<15; j++) residNorm_prev[j] = residNorm_prev[j+1];
  float err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in RestrictedSD_S_dct");
  residNorm_prev[15]=err;

  // Now compute the CG residual r=AT*(y-Ax) and store in grad
  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in RestrictedSD_S_dct");


  // d_vec_thres is being used to store a thresholded version of grad, 
  // namely the restriction of the CG residual r to the support set
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
  SAFEcuda("zero_vector_float in RestrictedSD_S_dct");

  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin, k_bin, n);
  SAFEcuda("threshold_one of grad in RestrictedSD_S_dct");


  // compute beta to orthogonalize the search direction
  if (beta_to_zero_flag == 1){
    beta = 0.0f;
    // we must compute the inner product of the restricted CG residual for later use when computing alpha
    *mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
  } else {
    beta = *mu;
    *mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
    if (*mu < (1000 * beta))
      beta = *mu / beta;
    else beta = 0.0f;
  }

  // update the search direction
  // multiply the past search direction by beta
  cublasSscal(n, beta, grad_previous, 1);
  // add the residual
  cublasSaxpy(n, 1, grad, 1, grad_previous, 1);
  
  // compute a thresholded version of grad_previous, the current search direction
  // set grad_prev_thres to be zero
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)grad_prev_thres, n);
  // threshold_one copies the thresholded points of grad_previous onto grad_prev_thres
  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad_previous, (float*)grad_prev_thres, (int*)d_bin, k_bin, n);

  // resid_update is used to store A * grad_prev_thres 
  A_dct(resid_update, grad_prev_thres, n, m, d_rows);
  SAFEcuda("A_dct for grad_previous in RestrictedSD_S_dct");

  // compute alpha for update stepsize
  alpha = cublasSdot(m, resid_update, 1, resid_update, 1);
  SAFEcublas("cublasSdot in RestrictedCG_S_dct");
  if (*mu < (1000 * alpha))
    alpha = *mu / alpha;
  else alpha = 0.0f;

  // add alpha grad_prev to d_vec
  cublasSaxpy(n, alpha, grad_previous, 1, d_vec, 1);


  // compute the maximum amount by which values could have changed
  alpha = 2 * alpha * MaxMagnitude(grad_previous,n);
  return alpha;

}




inline float UnrestrictedCG_S_dct(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float * grad_prev_thres, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, float *mu, const int num_bins, const int k, const int m, const int n, int beta_to_zero_flag, int suppChange_flag, int k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/

// This function computes a step of Conjugate Gradient descent of Ax=y where 
// x is restricted to being supported on the entries in d_bin that have 
// values larger than k_bin.  Its input is an initial guess to the solution, 
// the past search direction, the vector y, the ability to compute Ax, as well 
// as d_bin and k_bin.
// Various other variables are used along the way, but their input values 
// are not considered valid.
// IMPORTANT: d_vec is considered to be thresholded already.  
//            grad_previous is the prior search direction and 
//            grad_previous_thresh is the thresholded version.
//            d_vec_thres is used as working n-vector
//            resid and resid_update are working m-vectors

  
  // The following lines compute resid = y - A_dct * d_vec
//  cublasInit();
  // resid_update is used to store A * d_vec so that it can be used to compute the resid
  A_dct(resid_update, d_vec, n, m, d_rows);
  SAFEcuda("A_dct in RestrictedSD_S_dct");

  cublasSaxpy(m, -1, d_y, 1, resid_update, 1);
  SAFEcublas("First cublasSaxpy in RestrictedSD_S_dct");

  cublasSscal(m, -1, resid_update, 1);
  SAFEcublas("cublasScopy in RestrictedSD_S_dct");

  for (int j = 0; j<15; j++) residNorm_prev[j] = residNorm_prev[j+1];
  float err = cublasSnrm2(m, resid_update, 1);
  SAFEcublas("cublasSnrm2 in RestrictedSD_S_dct");
  residNorm_prev[15]=err;

  // Now need to compute the gradient and its restriction 
  // to where d_bin is larger than k_bin
  AT_dct(grad, resid_update, n, m, d_rows);
  SAFEcuda("AT_dct in RestrictedSD_S_dct");


  // d_vec_thres is being used to store a thresholded version of grad
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
  SAFEcuda("zero_vector_float in RestrictedSD_S_dct");

  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin, k_bin, n);
  SAFEcuda("threshold_one of grad in RestrictedSD_S_dct");


  // resid is used to store A * grad_prev_thres
  A_dct(resid, grad_prev_thres, n, m, d_rows);
  SAFEcuda("A_dct for grad_prev_thres in RestrictedSD_S_dct");

  // resid_update is used to store A times the projection of the residual 
  A_dct(resid_update, d_vec_thres, n, m, d_rows);
  SAFEcuda("A_dct for d_vec_thres in RestrictedSD_S_dct");

  // compute beta to orthogonalize the search direction
  float beta;
  if (beta_to_zero_flag == 1){
    beta = 0;
  } else {
    float beta_num = cublasSdot(m, resid, 1, resid_update, 1);
    SAFEcuda("cublasSdot for beta_num in RestrictedSD_S_dct");
    float beta_denom = cublasSdot(m, resid, 1, resid, 1);
    SAFEcuda("cublasSnrm2 for beta_denom in RestrictedSD_S_dct");
    beta = - beta_num / beta_denom;
  }

  // update the search direction
  // multiply the past search direction by beta
  cublasSscal(n, beta, grad_previous, 1);
  // add the residual
  cublasSaxpy(n, 1, grad, 1, grad_previous, 1);
  
  // compute a thresholded version of grad_previous, the current search direction
  // set grad_prev_thres to be zero
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)grad_prev_thres, n);
  // threshold_one copies the thresholded points of grad_previous onto grad_prev_thres
  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad_previous, (float*)grad_prev_thres, (int*)d_bin, k_bin, n);

  // resid is used to store A * grad_prev_thres 
  // A_dct(resid, grad_prev_thres, n, m, d_rows);
  // SAFEcuda("A_dct for grad_previous in RestrictedSD_S_dct");

  // update resid, which is used to store A * grad_prev_thres
  cublasSscal(m, beta, resid, 1);
  cublasSaxpy(m, 1, resid_update, 1, resid, 1);
  SAFEcuda("A_dct for grad_previous in UnrestrictedCG_S_dct");

  // compute alpha for update stepsize
  float alpha_num = cublasSdot(n, grad_prev_thres, 1, d_vec_thres, 1);
  SAFEcuda("cublasSdot for alpha_num in RestrictedSD_S_dct");
  float alpha_denom = cublasSdot(m, resid, 1, resid, 1);
  SAFEcuda("cublasSnrm2 for alpha_denom in RestrictedSD_S_dct");
  float alpha = alpha_num / alpha_denom;

  // add alpha grad_prev to d_vec
  cublasSaxpy(n, alpha, grad_previous, 1, d_vec, 1);

  // compute the value of mu to pass out in case the support remains unchanged
  // This way, both UnrestrictedCG and RestrictedCGwithSupportEvolution have the same starting points
  *mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);


  // compute the maximum amount by which values could have changed
  alpha = 2 * alpha * MaxMagnitude(grad_previous,n);
  return alpha;

}





inline float RestrictedCGwithSupportEvolution_S_smv(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float * grad_prev_thres, float *d_y, float *resid, float *resid_update, int * d_rows, int * d_cols, float * d_vals, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, float *mu, const int num_bins, const int k, const int m, const int n, const int nz, int beta_to_zero_flag, int suppChange_flag, int k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/

// This function computes a step of Conjugate Gradient descent of Ax=y where 
// x is restricted to being supported on the entries in d_bin that have 
// values larger than k_bin.  Its input is an initial guess to the solution, 
// the past search direction, the vector y, the ability to compute Ax, as well 
// as d_bin and k_bin.
// Various other variables are used along the way, but their input values 
// are not considered valid.
// IMPORTANT: d_vec is considered to be thresholded already.  
//            grad_previous is the prior search direction and 
//            grad_previous_thresh is the thresholded version.
//            d_vec_thres is used as working n-vector
//            resid and resid_update are working m-vectors

  float alpha, beta;

  // The following lines compute resid = y - A_smv * d_vec
  // resid_update is used to store A * d_vec so that it can be used to compute the resid
  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv for resid_update in RestrictedCG_S_smv");  

  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("Last cublasScopy in RestrictedCG_S_smv");

  cublasSaxpy(m, -1, resid_update, 1, resid, 1);
  SAFEcublas("Last cublasSaxpy in RestrictedCG_S_smv");

  // Compute the err for CG in the weighted norm <(vec_input - vec), AT*A(vec_input - vec)> = <y-Avec,y-Avec>
  for (int j = 0; j<15; j++) residNorm_prev[j] = residNorm_prev[j+1];
  float err = cublasSnrm2(m, resid, 1);
  SAFEcublas("cublasSnrm2 in RestrictedSD_S_smv");
  residNorm_prev[15]=err;

  // Now compute the CG residual r=AT*(y-Ax) and store in grad
  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in RestrictedSD_S_smv");


  // d_vec_thres is being used to store a thresholded version of grad, 
  // namely the restriction of the CG residual r to the support set
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
  SAFEcuda("zero_vector_float in RestrictedSD_S_smv");

  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin, k_bin, n);
  SAFEcuda("threshold_one of grad in RestrictedSD_S_smv");


  // compute beta to orthogonalize the search direction
  if (beta_to_zero_flag == 1){
    beta = 0.0f;
    // we must compute the inner product of the restricted CG residual for later use when computing alpha
    *mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
  } else {
    beta = *mu;
    *mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
    if (*mu < (1000 * beta))
      beta = *mu / beta;
    else beta = 0.0f;
  }

  // update the search direction
  // multiply the past search direction by beta
  cublasSscal(n, beta, grad_previous, 1);
  // add the residual
  cublasSaxpy(n, 1, grad, 1, grad_previous, 1);
  
  // compute a thresholded version of grad_previous, the current search direction
  // set grad_prev_thres to be zero
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)grad_prev_thres, n);
  // threshold_one copies the thresholded points of grad_previous onto grad_prev_thres
  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad_previous, (float*)grad_prev_thres, (int*)d_bin, k_bin, n);

  // resid_update is used to store A * grad_prev_thres 
  A_smv(resid_update, grad_prev_thres, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv for grad_previous in RestrictedSD_S_smv");

  // compute alpha for update stepsize
  alpha = cublasSdot(m, resid_update, 1, resid_update, 1);
  SAFEcublas("cublasSdot in RestrictedCG_S_smv");
  if (*mu < (1000 * alpha))
    alpha = *mu / alpha;
  else alpha = 0.0f;

  // add alpha grad_prev to d_vec
  cublasSaxpy(n, alpha, grad_previous, 1, d_vec, 1);


  // compute the maximum amount by which values could have changed
  alpha = 2 * alpha * MaxMagnitude(grad_previous,n);
  return alpha;

}




inline float UnrestrictedCG_S_smv(float *d_vec, float *d_vec_thres, float * grad, float * grad_previous, float * grad_prev_thres, float *d_y, float *resid, float *resid_update, int * d_rows, int * d_cols, float * d_vals, int *d_bin, int * d_bin_counters, int * h_bin_counters, float *residNorm_prev, float *mu, const int num_bins, const int k, const int m, const int n, const int nz, int beta_to_zero_flag, int suppChange_flag, int k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
/*
**********************
**  Initialization  **
**********************
*/

// This function computes a step of Conjugate Gradient descent of Ax=y where 
// x is restricted to being supported on the entries in d_bin that have 
// values larger than k_bin.  Its input is an initial guess to the solution, 
// the past search direction, the vector y, the ability to compute Ax, as well 
// as d_bin and k_bin.
// Various other variables are used along the way, but their input values 
// are not considered valid.
// IMPORTANT: d_vec is considered to be thresholded already.  
//            grad_previous is the prior search direction and 
//            grad_previous_thresh is the thresholded version.
//            d_vec_thres is used as working n-vector
//            resid and resid_update are working m-vectors

  
  // The following lines compute resid = y - A_smv * d_vec
//  cublasInit();
  // resid_update is used to store A * d_vec so that it can be used to compute the resid
  A_smv(resid_update, d_vec, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in RestrictedSD_S_smv");

  cublasSaxpy(m, -1, d_y, 1, resid_update, 1);
  SAFEcublas("First cublasSaxpy in RestrictedSD_S_smv");

  cublasSscal(m, -1, resid_update, 1);
  SAFEcublas("cublasScopy in RestrictedSD_S_smv");

  for (int j = 0; j<15; j++) residNorm_prev[j] = residNorm_prev[j+1];
  float err = cublasSnrm2(m, resid_update, 1);
  SAFEcublas("cublasSnrm2 in RestrictedSD_S_smv");
  residNorm_prev[15]=err;

  // Now need to compute the gradient and its restriction 
  // to where d_bin is larger than k_bin
  AT_smv(grad, resid_update, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in RestrictedSD_S_smv");


  // d_vec_thres is being used to store a thresholded version of grad
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)d_vec_thres, n);
  SAFEcuda("zero_vector_float in RestrictedSD_S_smv");

  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad, (float*)d_vec_thres, (int*)d_bin, k_bin, n);
  SAFEcuda("threshold_one of grad in RestrictedSD_S_smv");


  // resid is used to store A * grad_prev_thres
  A_smv(resid, grad_prev_thres, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv for grad_prev_thres in RestrictedSD_S_smv");

  // resid_update is used to store A times the projection of the residual 
  A_smv(resid_update, d_vec_thres, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv for d_vec_thres in RestrictedSD_S_smv");

  // compute beta to orthogonalize the search direction
  float beta;
  if (beta_to_zero_flag == 1){
    beta = 0;
  } else {
    float beta_num = cublasSdot(m, resid, 1, resid_update, 1);
    SAFEcuda("cublasSdot for beta_num in RestrictedSD_S_smv");
    float beta_denom = cublasSdot(m, resid, 1, resid, 1);
    SAFEcuda("cublasSnrm2 for beta_denom in RestrictedSD_S_smv");
    beta = - beta_num / beta_denom;
  }

  // update the search direction
  // multiply the past search direction by beta
  cublasSscal(n, beta, grad_previous, 1);
  // add the residual
  cublasSaxpy(n, 1, grad, 1, grad_previous, 1);
  
  // compute a thresholded version of grad_previous, the current search direction
  // set grad_prev_thres to be zero
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)grad_prev_thres, n);
  // threshold_one copies the thresholded points of grad_previous onto grad_prev_thres
  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)grad_previous, (float*)grad_prev_thres, (int*)d_bin, k_bin, n);

  // resid is used to store A * grad_prev_thres 
  // A_smv(resid, grad_prev_thres, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  // SAFEcuda("A_smv for grad_previous in RestrictedSD_S_smv");

  // update resid, which is used to store A * grad_prev_thres
  cublasSscal(m, beta, resid, 1);
  cublasSaxpy(m, 1, resid_update, 1, resid, 1);
  SAFEcuda("A_dct for grad_previous in UnrestrictedCG_S_smv");


  // compute alpha for update stepsize
  float alpha_num = cublasSdot(n, grad_prev_thres, 1, d_vec_thres, 1);
  SAFEcuda("cublasSdot for alpha_num in RestrictedSD_S_smv");
  float alpha_denom = cublasSdot(m, resid, 1, resid, 1);
  SAFEcuda("cublasSnrm2 for alpha_denom in RestrictedSD_S_smv");
  float alpha = alpha_num / alpha_denom;

  // add alpha grad_prev to d_vec
  cublasSaxpy(n, alpha, grad_previous, 1, d_vec, 1);

  // compute the value of mu to pass out in case the support remains unchanged
  // This way, both UnrestrictedCG and RestrictedCGwithSupportEvolution have the same starting points
  *mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);


  // compute the maximum amount by which values could have changed
  alpha = 2 * alpha * MaxMagnitude(grad_previous,n);
  return alpha;

}



inline float RestrictedFIHT_S_gen(float *d_y, float *d_vec, float *resid_update, float *d_vec_prev, float *d_Avec_prev, float *d_extra, float *d_Avec_extra, float *grad, float *d_vec_thres, float *resid, float *d_A, float *d_AT, const int k, const int m, const int n, float mu, float tau, float tmp, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksm, dim3 threadsPerBlockm)
{
  cublasInit(); // Is it necessary?
  /*
   ********************************
   ** Compute Extrapolated Point **
   ********************************
  */
  // compute tau 
  // Ax_prev - Ax -> Ax_prev
  cublasSaxpy(m, -1, resid_update, 1, d_Avec_prev, 1);
  tau = cublasSdot(m, resid, 1, d_Avec_prev, 1);
  tmp = cublasSdot(m, d_Avec_prev, 1, d_Avec_prev, 1);
  SAFEcuda("cublasSdot in FIHT_S_gen Compute Extrapolated Point");
  if (fabs(tau) < 400*fabs(tmp))
    tau = tau/tmp;
  else 
    tau = 0;
  //printf("tau = %f\n", tau);

  // v = x + tau * (x_prev - x) -> d_extra
  // 1. x -> v; 2. -x + x_prev -> x_prev; 3. tau * x_prev + v -> v
  cublasScopy(n, d_vec, 1, d_extra, 1);
  cublasSaxpy(n, -1, d_vec, 1, d_vec_prev, 1);   
  cublasSaxpy(n, tau, d_vec_prev, 1, d_extra, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in FIHT_S_gen Compute Extrapolated Point");

  // Av = Ax + tau * (Ax_prev - Ax) -> d_Avec_extra
  // 1. Ax -> Av; 2. Ax_prev - Ax -> Ax_prev (done!); 3. tau * Ax_prev + Av -> Av
  cublasScopy(m, resid_update, 1, d_Avec_extra, 1);
  cublasSaxpy(m, tau, d_Avec_prev, 1, d_Avec_extra, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in FIHT_S_gen Compute Extrapolated Point");

  // y - Av -> resid
  cublasScopy(m, d_y, 1, resid, 1);
  cublasSaxpy(m, -1, d_Avec_extra, 1, resid, 1);    
  SAFEcuda("cublasSaxpy in FIHT_S_gen Compute Extrapolated Point");   

  // AT(y - Av) -> grad
  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in FIHT_S_gen Compute Extrapolated Point");
   
  /* 
   *******************************
   **    Compute New Iterate    **
   *******************************
  */
  // before computing new x, Ax, SAVE d_vec -> d_vec_prev, resid_update -> d_Avec_prev
  cublasScopy(n, d_vec, 1, d_vec_prev, 1);
  cublasScopy(m, resid_update, 1, d_Avec_prev, 1);
  SAFEcuda("cublasScopy in FIHT_S_gen Compute New Iterate");
  
  // restrict grad to the support of v
  cublasScopy(n, grad, 1, d_vec_thres, 1);
  SAFEcuda("cublasScopy in FIHT_S_gen Compute New Iterate");
  threshold_two<<<numBlocks, threadsPerBlock>>>(d_vec_thres, d_extra, n);

  // Agrad_thres -> d_Avec_extra
  A_gen(d_Avec_extra, d_vec_thres, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in FIHT_S_gen Compute New Iterate");

  // compute mu 
  mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
  tmp = cublasSdot(m, d_Avec_extra, 1, d_Avec_extra, 1);
  SAFEcuda("cublasSdot in FIHT_S_gen Compute New Iterate");
  if (mu < 400*tmp)
    mu = mu/tmp;
  else 
    mu = 1;
  //printf("mu1 = %f\n", mu);

  // x = v + mu * AT(y-Av) -> d_vec
  // 1. copy v -> x; 2. mu * AT(y-Av) + x -> x 
  cublasScopy(n, d_extra, 1, d_vec, 1);
  cublasSaxpy(n, mu, grad, 1, d_vec, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in FIHT_S_gen Compute New Iterate");

  return 1.0f; // not quite right here, should involve x_prev!
}

inline float RestrictedALPS_S_gen(float *d_y, float *d_vec, float *resid_update, float *d_vec_prev, float *d_Avec_prev, float *d_extra, float *d_Avec_extra, 
float *grad, float *d_vec_thres, float *resid, float *d_A, float *d_AT, int *d_bin, int *d_bin_counters, int *h_bin_counters, const int num_bins, const int k, const int m, const int n,
 int *p_sum, float mu, float tau, float tmp, float alpha, float minVal, float maxChange, float slope, float max_value, int k_bin, int MaxBin, dim3 numBlocks, dim3 threadsPerBlock, 
dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
  cublasInit(); // Is it necessary?
  /*
   ********************************
   ** Compute Extrapolated Point **
   ********************************
  */
  // compute tau [Q]
  // Ax_prev - Ax -> Ax_prev
  cublasSaxpy(m, -1, resid_update, 1, d_Avec_prev, 1);
  tau = cublasSdot(m, resid, 1, d_Avec_prev, 1);
  tmp = cublasSdot(m, d_Avec_prev, 1, d_Avec_prev, 1);
  SAFEcuda("cublasSdot in ALPS_S_gen Compute Extrapolated Point");
  if (fabs(tau) < 400*fabs(tmp))
    tau = tau/tmp;
  else 
    tau = 0;

  // v = x + tau * (x_prev - x) -> d_extra
  // 1. x -> v; 2. -x + x_prev -> x_prev; 3. tau * x_prev + v -> v
  cublasScopy(n, d_vec, 1, d_extra, 1);
  cublasSaxpy(n, -1, d_vec, 1, d_vec_prev, 1);   
  cublasSaxpy(n, tau, d_vec_prev, 1, d_extra, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in ALPS_S_gen Compute Extrapolated Point");

  // Av = Ax + tau * (Ax_prev - Ax) -> d_Avec_extra
  // 1. Ax -> Av; 2. Ax_prev - Ax -> Ax_prev (done!); 3. tau * Ax_prev + Av -> Av
  cublasScopy(m, resid_update, 1, d_Avec_extra, 1);
  cublasSaxpy(m, tau, d_Avec_prev, 1, d_Avec_extra, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in ALPS_S_gen Compute Extrapolated Point");

  // y - Av -> resid
  cublasScopy(m, d_y, 1, resid, 1);
  cublasSaxpy(m, -1, d_Avec_extra, 1, resid, 1);    
  SAFEcuda("cublasSaxpy in ALPS_S_gen Compute Extrapolated Point");   

  // AT(y - Av) -> grad
  AT_gen(grad, resid, d_AT, m, n, numBlocks, threadsPerBlock);
  SAFEcuda("AT_gen in ALPS_S_gen Compute Extrapolated Point");
   
  /* 
   *******************************
   **    Compute New Iterate    **
   *******************************
  */
  // before computing new x, Ax, SAVE d_vec -> d_vec_prev, resid_update -> d_Avec_prev
  cublasScopy(n, d_vec, 1, d_vec_prev, 1);
  cublasScopy(m, resid_update, 1, d_Avec_prev, 1);
  SAFEcuda("cublasScopy in ALPS_S_gen Compute New Iterate");

  // ============ difference between FIHT and ALPS: start ===============

  // restrict grad to the union support of v and k largest one
  // set up a zero vector 
  zero_vector_float<<<numBlocks, threadsPerBlock>>>(d_vec_thres, n);
  SAFEcuda("zero_vector_float in ALPS_S_gen Compute New Iterate");
    
  // d_vec_thres[ind] = grad[ind], if d_extra[ind] == 0;
  threshold_three<<<numBlocks, threadsPerBlock>>>(d_vec_thres, grad, d_extra, n);
  SAFEcuda("threshold_three in ALPS_S_gen Compute New Iterate");

  // find the support of d_vec_thres
  minVal = 0.0f;
  maxChange = 1.0f;
  max_value = MaxMagnitude(d_vec_thres, n);
  slope = (num_bins-1)/max_value;
  *p_sum = FindSupportSet(d_vec_thres, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                            &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in ALPS_S_gen Compute New Iterate");

  // threshold d_vec_thres
  threshold<<<numBlocks, threadsPerBlock>>>(d_vec_thres, d_bin, k_bin, n);
  SAFEcuda("threshold in ALPS_S_gen Compute New Iterate");
   
  // d_vec_thres[ind] = grad[ind], if d_extra[ind] != 0;
  threshold_four<<<numBlocks, threadsPerBlock>>>(d_vec_thres, grad, d_extra, n);
  SAFEcuda("threshold_four in ALPS_S_gen Compute New Iterate");

  // ============ difference between FIHT and ALPS: end =================

  // Agrad_thres -> d_Avec_extra
  A_gen(d_Avec_extra, d_vec_thres, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in ALPS_S_gen Compute New Iterate");

  // compute mu [Q]
  mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
  tmp = cublasSdot(m, d_Avec_extra, 1, d_Avec_extra, 1);
  SAFEcuda("cublasSdot in ALPS_S_gen Compute New Iterate");
  if (mu < 400*tmp)
    mu = mu/tmp;
  else 
    mu = 1;
  // x = v + mu * AT(y-Av) -> d_vec
  // 1. copy v -> x; 2. mu * AT(y-Av) + x -> x 
  cublasScopy(n, d_extra, 1, d_vec, 1);
  cublasSaxpy(n, mu, grad, 1, d_vec, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in ALPS_S_gen Compute New Iterate");

  return 1.0f;
}

inline float RestrictedFIHT_S_dct(float *d_y, float *d_vec, float *resid_update, float *d_vec_prev, float *d_Avec_prev, float *d_extra, float *d_Avec_extra, float *grad, float *d_vec_thres, float *resid, int *d_rows, const int k, const int m, const int n, float mu, float tau, float tmp, dim3 numBlocks, dim3 threadsPerBlock)
{
  cublasInit(); // Is it necessary?
  /*
   ********************************
   ** Compute Extrapolated Point **
   ********************************
  */
  // compute tau
  // Ax_prev - Ax -> Ax_prev
  cublasSaxpy(m, -1, resid_update, 1, d_Avec_prev, 1);
  tau = cublasSdot(m, resid, 1, d_Avec_prev, 1);
  tmp = cublasSdot(m, d_Avec_prev, 1, d_Avec_prev, 1);
  SAFEcuda("cublasSdot in FIHT_S_dct Compute Extrapolated Point");
  if (fabs(tau) < 1000*fabs(tmp))
    tau = tau/tmp;
  else 
    tau = 0;
  //printf("tau = %f\n", tau);

  // v = x + tau * (x_prev - x) -> d_extra
  // 1. x -> v; 2. -x + x_prev -> x_prev; 3. tau * x_prev + v -> v
  cublasScopy(n, d_vec, 1, d_extra, 1);
  cublasSaxpy(n, -1, d_vec, 1, d_vec_prev, 1);   
  cublasSaxpy(n, tau, d_vec_prev, 1, d_extra, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in FIHT_S_gen Compute Extrapolated Point");

  // Av = Ax + tau * (Ax_prev - Ax) -> d_Avec_extra
  // 1. Ax -> Av; 2. Ax_prev - Ax -> Ax_prev (done!); 3. tau * Ax_prev + Av -> Av
  cublasScopy(m, resid_update, 1, d_Avec_extra, 1);
  cublasSaxpy(m, tau, d_Avec_prev, 1, d_Avec_extra, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in FIHT_S_dct Compute Extrapolated Point");

  // y - Av -> resid
  cublasScopy(m, d_y, 1, resid, 1);
  cublasSaxpy(m, -1, d_Avec_extra, 1, resid, 1);    
  SAFEcuda("cublasSaxpy in FIHT_S_dct Compute Extrapolated Point");   

  // AT(y - Av) -> grad
  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in FIHT_S_dct Compute Extrapolated Point");
   
  /* 
   *******************************
   **    Compute New Iterate    **
   *******************************
  */
  // before computing new x, Ax, SAVE d_vec -> d_vec_prev, resid_update -> d_Avec_prev
  cublasScopy(n, d_vec, 1, d_vec_prev, 1);
  cublasScopy(m, resid_update, 1, d_Avec_prev, 1);
  SAFEcuda("cublasScopy in FIHT_S_dct Compute New Iterate");
  
  // restrict grad to the support of v
  cublasScopy(n, grad, 1, d_vec_thres, 1);
  SAFEcuda("cublasScopy in FIHT_S_dct Compute New Iterate");
  threshold_two<<<numBlocks, threadsPerBlock>>>(d_vec_thres, d_extra, n);

  // Agrad_thres -> d_Avec_extra
  A_dct(d_Avec_extra, d_vec_thres, n, m, d_rows);
  SAFEcuda("A_dct in FIHT_S_dct Compute New Iterate");

  // compute mu 
  mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
  tmp = cublasSdot(m, d_Avec_extra, 1, d_Avec_extra, 1);
  SAFEcuda("cublasSdot in FIHT_S_dct Compute New Iterate");
  if (mu < 1000*tmp)
    mu = mu/tmp;
  else 
    mu = 1;
  //printf("mu1 = %f\n", mu);

  // x = v + mu * AT(y-Av) -> d_vec
  // 1. copy v -> x; 2. mu * AT(y-Av) + x -> x 
  cublasScopy(n, d_extra, 1, d_vec, 1);
  cublasSaxpy(n, mu, grad, 1, d_vec, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in FIHT_S_dct Compute New Iterate");

  return 1.0f;
}

inline float RestrictedALPS_S_dct(float *d_y, float *d_vec, float *resid_update, float *d_vec_prev, float *d_Avec_prev, float *d_extra, float *d_Avec_extra, float *grad, float *d_vec_thres, float *resid, int *d_rows, int *d_bin, int *d_bin_counters, int *h_bin_counters, const int num_bins, const int k, const int m, const int n, int *p_sum, float mu, float tau, float tmp, float alpha,
float minVal, float maxChange, float slope, float max_value, int k_bin, int MaxBin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
  cublasInit(); // Is it necessary?
  /*
   ********************************
   ** Compute Extrapolated Point **
   ********************************
  */
  // compute tau
  // Ax_prev - Ax -> Ax_prev
  cublasSaxpy(m, -1, resid_update, 1, d_Avec_prev, 1);
  tau = cublasSdot(m, resid, 1, d_Avec_prev, 1);
  tmp = cublasSdot(m, d_Avec_prev, 1, d_Avec_prev, 1);
  SAFEcuda("cublasSdot in ALPS_S_dct Compute Extrapolated Point");
  if (fabs(tau) < 1000*fabs(tmp))
    tau = tau/tmp;
  else 
    tau = 0;
  //printf("tau = %f\n", tau);

  // v = x + tau * (x_prev - x) -> d_extra
  // 1. x -> v; 2. -x + x_prev -> x_prev; 3. tau * x_prev + v -> v
  cublasScopy(n, d_vec, 1, d_extra, 1);
  cublasSaxpy(n, -1, d_vec, 1, d_vec_prev, 1);   
  cublasSaxpy(n, tau, d_vec_prev, 1, d_extra, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in FIHT_S_gen Compute Extrapolated Point");

  // Av = Ax + tau * (Ax_prev - Ax) -> d_Avec_extra
  // 1. Ax -> Av; 2. Ax_prev - Ax -> Ax_prev (done!); 3. tau * Ax_prev + Av -> Av
  cublasScopy(m, resid_update, 1, d_Avec_extra, 1);
  cublasSaxpy(m, tau, d_Avec_prev, 1, d_Avec_extra, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in ALPS_S_dct Compute Extrapolated Point");

  // y - Av -> resid
  cublasScopy(m, d_y, 1, resid, 1);
  cublasSaxpy(m, -1, d_Avec_extra, 1, resid, 1);    
  SAFEcuda("cublasSaxpy in ALPS_S_dct Compute Extrapolated Point");   

  // AT(y - Av) -> grad
  AT_dct(grad, resid, n, m, d_rows);
  SAFEcuda("AT_dct in ALPS_S_dct Compute Extrapolated Point");
   
  /* 
   *******************************
   **    Compute New Iterate    **
   *******************************
  */
  // before computing new x, Ax, SAVE d_vec -> d_vec_prev, resid_update -> d_Avec_prev
  cublasScopy(n, d_vec, 1, d_vec_prev, 1);
  cublasScopy(m, resid_update, 1, d_Avec_prev, 1);
  SAFEcuda("cublasScopy in ALPS_S_dct Compute New Iterate");
  
  // ================ difference between FIHT and ALPS: start =====================
  // restrict grad to the union support of v and k largest one
  // set up a zero vector 
  zero_vector_float<<<numBlocks, threadsPerBlock>>>(d_vec_thres, n);
  SAFEcuda("zero_vector_float in ALPS_S_dct Compute New Iterate");
    
  // d_vec_thres[ind] = grad[ind], if d_extra[ind] == 0;
  threshold_three<<<numBlocks, threadsPerBlock>>>(d_vec_thres, grad, d_extra, n);
  SAFEcuda("threshold_three in ALPS_S_dct Compute New Iterate");

  // find the support of d_vec_thres
  minVal = 0.0f;
  maxChange = 1.0f;
  max_value = MaxMagnitude(d_vec_thres, n);
  slope = (num_bins-1)/max_value;
  *p_sum = FindSupportSet(d_vec_thres, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                            &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in ALPS_S_dct Compute New Iterate");

  // threshold d_vec_thres
  threshold<<<numBlocks, threadsPerBlock>>>(d_vec_thres, d_bin, k_bin, n);
  SAFEcuda("threshold in ALPS_S_dct Compute New Iterate");
   
  // d_vec_thres[ind] = grad[ind], if d_extra[ind] != 0;
  threshold_four<<<numBlocks, threadsPerBlock>>>(d_vec_thres, grad, d_extra, n);
  SAFEcuda("threshold_four in ALPS_S_dct Compute New Iterate");
  // ================ difference between FIHT and ALPS: start =====================

  // Agrad_thres -> d_Avec_extra
  A_dct(d_Avec_extra, d_vec_thres, n, m, d_rows);
  SAFEcuda("A_dct in ALPS_S_dct Compute New Iterate");

  // compute mu 
  mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
  tmp = cublasSdot(m, d_Avec_extra, 1, d_Avec_extra, 1);
  SAFEcuda("cublasSdot in ALPS_S_dct Compute New Iterate");
  if (mu < 1000*tmp)
    mu = mu/tmp;
  else 
    mu = 1;
  //printf("mu1 = %f\n", mu);

  // x = v + mu * AT(y-Av) -> d_vec
  // 1. copy v -> x; 2. mu * AT(y-Av) + x -> x 
  cublasScopy(n, d_extra, 1, d_vec, 1);
  cublasSaxpy(n, mu, grad, 1, d_vec, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in ALPS_S_dct Compute New Iterate");

  return 1.0f;
}

inline float RestrictedFIHT_S_smv(float *d_y, float *d_vec, float *resid_update, float *d_vec_prev, float *d_Avec_prev, float *d_extra, float *d_Avec_extra, float *grad, float *d_vec_thres, float *resid, int *d_rows, int *d_cols, float *d_vals, const int k, const int m, const int n, const int nz, float mu, float tau, float tmp, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm)
{
  cublasInit(); // Is it necessary?
  /*
   ********************************
   ** Compute Extrapolated Point **
   ********************************
  */
  // compute tau [Q]
  // Ax_prev - Ax -> Ax_prev
  cublasSaxpy(m, -1, resid_update, 1, d_Avec_prev, 1);
  tau = cublasSdot(m, resid, 1, d_Avec_prev, 1);
  tmp = cublasSdot(m, d_Avec_prev, 1, d_Avec_prev, 1);
  SAFEcuda("cublasSdot in FIHT_S_smv Compute Extrapolated Point");
  if (fabs(tau) < 400*fabs(tmp))
    tau = tau/tmp;
  else 
    tau = 0;
  //printf("tau = %f\n", tau);

  // v = x + tau * (x_prev - x) -> d_extra
  // 1. x -> v; 2. -x + x_prev -> x_prev; 3. tau * x_prev + v -> v
  cublasScopy(n, d_vec, 1, d_extra, 1);
  cublasSaxpy(n, -1, d_vec, 1, d_vec_prev, 1);   
  cublasSaxpy(n, tau, d_vec_prev, 1, d_extra, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in FIHT_S_smv Compute Extrapolated Point");

  // Av = Ax + tau * (Ax_prev - Ax) -> d_Avec_extra
  // 1. Ax -> Av; 2. Ax_prev - Ax -> Ax_prev (done!); 3. tau * Ax_prev + Av -> Av
  cublasScopy(m, resid_update, 1, d_Avec_extra, 1);
  cublasSaxpy(m, tau, d_Avec_prev, 1, d_Avec_extra, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in FIHT_S_smv Compute Extrapolated Point");

  // y - Av -> resid
  cublasScopy(m, d_y, 1, resid, 1);
  cublasSaxpy(m, -1, d_Avec_extra, 1, resid, 1);    
  SAFEcuda("cublasSaxpy in FIHT_S_smv Compute Extrapolated Point");   

  // AT(y - Av) -> grad
  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in FIHT_S_smv Compute Extrapolated Point");
   
  /* 
   *******************************
   **    Compute New Iterate    **
   *******************************
  */
  // before computing new x, Ax, SAVE d_vec -> d_vec_prev, resid_update -> d_Avec_prev
  cublasScopy(n, d_vec, 1, d_vec_prev, 1);
  cublasScopy(m, resid_update, 1, d_Avec_prev, 1);
  SAFEcuda("cublasScopy in FIHT_S_smv Compute New Iterate");
  
  // restrict grad to the support of v
  cublasScopy(n, grad, 1, d_vec_thres, 1);
  SAFEcuda("cublasScopy in FIHT_S_smv Compute New Iterate");
  threshold_two<<<numBlocks, threadsPerBlock>>>(d_vec_thres, d_extra, n);

  // Agrad_thres -> d_Avec_extra
  A_smv(d_Avec_extra, d_vec_thres, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in FIHT_S_smv Compute New Iterate");

  // compute mu [Q]
  mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
  tmp = cublasSdot(m, d_Avec_extra, 1, d_Avec_extra, 1);
  SAFEcuda("cublasSdot in FIHT_S_smv Compute New Iterate");
  if (mu < 400*tmp)
    mu = mu/tmp;
  else 
    mu = 1;
  //printf("mu1 = %f\n", mu);

  // x = v + mu * AT(y-Av) -> d_vec
  // 1. copy v -> x; 2. mu * AT(y-Av) + x -> x 
  cublasScopy(n, d_extra, 1, d_vec, 1);
  cublasSaxpy(n, mu, grad, 1, d_vec, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in FIHT_S_smv Compute New Iterate");

  return 1.0f; 
}


inline float RestrictedALPS_S_smv(float *d_y, float *d_vec, float *resid_update, float *d_vec_prev, float *d_Avec_prev, float *d_extra, float *d_Avec_extra, float *grad, float *d_vec_thres, float *resid, int *d_rows, int *d_cols, float *d_vals, int *d_bin, int *d_bin_counters, int *h_bin_counters, const int num_bins, const int k, const int m, const int n, const int nz,  int *p_sum, float mu, float tau, float tmp, float alpha, float minVal, float maxChange, float slope, float max_value, int k_bin, int MaxBin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{
  cublasInit(); // Is it necessary?
  /*
   ********************************
   ** Compute Extrapolated Point **
   ********************************
  */
  // compute tau [Q]
  // Ax_prev - Ax -> Ax_prev
  cublasSaxpy(m, -1, resid_update, 1, d_Avec_prev, 1);
  tau = cublasSdot(m, resid, 1, d_Avec_prev, 1);
  tmp = cublasSdot(m, d_Avec_prev, 1, d_Avec_prev, 1);
  SAFEcuda("cublasSdot in ALPS_S_smv Compute Extrapolated Point");
  if (fabs(tau) < 400*fabs(tmp))
    tau = tau/tmp;
  else 
    tau = 0;
  //printf("tau = %f\n", tau);

  // v = x + tau * (x_prev - x) -> d_extra
  // 1. x -> v; 2. -x + x_prev -> x_prev; 3. tau * x_prev + v -> v
  cublasScopy(n, d_vec, 1, d_extra, 1);
  cublasSaxpy(n, -1, d_vec, 1, d_vec_prev, 1);   
  cublasSaxpy(n, tau, d_vec_prev, 1, d_extra, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in ALPS_S_smv Compute Extrapolated Point");

  // Av = Ax + tau * (Ax_prev - Ax) -> d_Avec_extra
  // 1. Ax -> Av; 2. Ax_prev - Ax -> Ax_prev (done!); 3. tau * Ax_prev + Av -> Av
  cublasScopy(m, resid_update, 1, d_Avec_extra, 1);
  cublasSaxpy(m, tau, d_Avec_prev, 1, d_Avec_extra, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in ALPS_S_smv Compute Extrapolated Point");

  // y - Av -> resid
  cublasScopy(m, d_y, 1, resid, 1);
  cublasSaxpy(m, -1, d_Avec_extra, 1, resid, 1);    
  SAFEcuda("cublasSaxpy in ALPS_S_smv Compute Extrapolated Point");   

  // AT(y - Av) -> grad
  AT_smv(grad, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("AT_smv in ALPS_S_smv Compute Extrapolated Point");
   
  /* 
   *******************************
   **    Compute New Iterate    **
   *******************************
  */
  // before computing new x, Ax, SAVE d_vec -> d_vec_prev, resid_update -> d_Avec_prev
  cublasScopy(n, d_vec, 1, d_vec_prev, 1);
  cublasScopy(m, resid_update, 1, d_Avec_prev, 1);
  SAFEcuda("cublasScopy in ALPS_S_smv Compute New Iterate");
  
  // ================ difference between FIHT and ALPS: start =====================
  // restrict grad to the union support of v and k largest one
  // set up a zero vector 
  zero_vector_float<<<numBlocks, threadsPerBlock>>>(d_vec_thres, n);
  SAFEcuda("zero_vector_float in ALPS_S_smv Compute New Iterate");
    
  // d_vec_thres[ind] = grad[ind], if d_extra[ind] == 0;
  threshold_three<<<numBlocks, threadsPerBlock>>>(d_vec_thres, grad, d_extra, n);
  SAFEcuda("threshold_three in ALPS_S_dct Compute New Iterate");

  // find the support of d_vec_thres
  minVal = 0.0f;
  maxChange = 1.0f;
  max_value = MaxMagnitude(d_vec_thres, n);
  slope = (num_bins-1)/max_value;
  *p_sum = FindSupportSet(d_vec_thres, d_bin, d_bin_counters, h_bin_counters, slope, max_value, maxChange, &minVal, &alpha, &MaxBin,
                            &k_bin, n, k, num_bins, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  SAFEcuda("FindSupportSet in ALPS_S_dct Compute New Iterate");

  // threshold d_vec_thres
  threshold<<<numBlocks, threadsPerBlock>>>(d_vec_thres, d_bin, k_bin, n);
  SAFEcuda("threshold in ALPS_S_dct Compute New Iterate");
   
  // d_vec_thres[ind] = grad[ind], if d_extra[ind] != 0;
  threshold_four<<<numBlocks, threadsPerBlock>>>(d_vec_thres, grad, d_extra, n);
  SAFEcuda("threshold_four in ALPS_S_dct Compute New Iterate");
  // ================ difference between FIHT and ALPS: end =====================

  // Agrad_thres -> d_Avec_extra
  A_smv(d_Avec_extra, d_vec_thres, m, n, d_rows, d_cols, d_vals, nz, numBlocksm, threadsPerBlockm, numBlocksnp, threadsPerBlocknp);
  SAFEcuda("A_smv in ALPS_S_smv Compute New Iterate");

  // compute mu [Q]
  mu = cublasSdot(n, d_vec_thres, 1, d_vec_thres, 1);
  tmp = cublasSdot(m, d_Avec_extra, 1, d_Avec_extra, 1);
  SAFEcuda("cublasSdot in ALPS_S_smv Compute New Iterate");
  if (mu < 400*tmp)
    mu = mu/tmp;
  else 
    mu = 1;
  //printf("mu1 = %f\n", mu);

  // x = v + mu * AT(y-Av) -> d_vec
  // 1. copy v -> x; 2. mu * AT(y-Av) + x -> x 
  cublasScopy(n, d_extra, 1, d_vec, 1);
  cublasSaxpy(n, mu, grad, 1, d_vec, 1);
  SAFEcuda("cublasScopy&cublasSaxpy in ALPS_S_smv Compute New Iterate");

  return 1.0f; 
}


// ************************* Functions that were used in development and are no longer active, **********************
// ************************* but will be used for timings or kept for completness **********




/*

inline void halveNumbins(int *d_bin, int *d_bin_counters, int *d_int_workspace, const int num_bins, const int n, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin) 
{
  // This function is designed to halve the number of bins used.  
  // Before it is called, num_bins should be divided by 2 and 
  // numBlocks_bin and threadsPerBlock_bin should be adjusted. 
  // k_bin should also be divided by two outside this function.

  // It assumes that the number of bins used is a power of 2, 
  // and that the number of bins is never more than 2n in lengh.
  // The former assumption allows us to easily halve, and the 
  // second ensures that we can use an n-vector as workspace.
  // num_bins should be entered in its original size, and it and 
  // 

  // shrink d_bin
  halve_bins <<< numBlocks, threadsPerBlock >>>((int*)d_bin, n);
  SAFEcuda("halve_bins loop");

  // add the adjacent values in d_bin_counters to the left of the pair
  // as num_bins has already been halved this is the number of threads needed.
  add_adjacent <<< numBlocks_bin, threadsPerBlock_bin >>>((int*)d_bin_counters, (int*)d_int_workspace, num_bins);

  // copy added pairs from d_int_workspace to the first num_bin entries in d_bin_counters
  int_copy <<< numBlocks_bin, threadsPerBlock_bin >>>((int*)d_bin_counters, (int*)d_int_workspace, num_bins);

}

*/

/*
void MakeCountBins(float *vec, int *bin, int *counter, int *h_counter, int *segcounter, const int n, const int NC, const int k, int *sum, int *k_bin)
{  int NC2 = 2*NC;
  float high, low, slope;
  int LowSegLength, HighSegLength, HighLength, threadsHigh;
  int NC_blocks, NC_threads;
  NC_threads = min(896, NC*NC);
  NC_blocks = (int)ceil((float)NC*NC/((float)NC_threads));
  dim3 NCthreads(NC_threads);
  dim3 NCblocks(NC_blocks);  

  LowSegLength = n/NC;
  HighSegLength = LowSegLength;
  HighLength = n;
  threadsHigh = n - NC*LowSegLength;

  if (threadsHigh != 0) {
	HighSegLength = LowSegLength + 1;
	HighLength = threadsHigh * HighSegLength;
  }

  high = MaxMagnitude(vec, n);
  low = .000001*high;
  slope = (NC-2)/(high-low);

  zero_vector_int<<<NCblocks, NCthreads>>>(segcounter, NC*NC);
  zero_vector_int<<<1,NC>>>(counter,NC);

  make_and_count_seg<<<1, NC>>>(vec, bin, segcounter, n, NC, HighLength, HighSegLength, threadsHigh, LowSegLength, low, high, slope);
//  make_and_count_seg_sharedAtomic<<<1, NC2, NC*sizeof(int)>>>(vec, bin, segcounter, n, NC, HighLength, HighSegLength, threadsHigh, LowSegLength, low, high, slope);

  
  segCountSum<<<1, NC>>>(counter, segcounter, NC);
//  segCountSum_shared<<<1, NC, NC*sizeof(int)>>>(counter, segcounter, NC);

  FindKbin(counter, h_counter, NC, k, sum, k_bin);

  cout << "Kbin = " << *k_bin << " and sum = " << *sum << endl;

  int excess = (*sum-k);
  int allowed = k/500;
  int ii=0;
//  cout << "excess = " << excess << " and allowed = " << allowed << endl;

 

  while ( (excess > allowed) & ii<2) {
  
	low = high - (*k_bin)/slope;
	high = high - (*k_bin-1)/slope;
	slope = (float)(NC-2)/((float)(high-low));

	zero_vector_int<<<NCblocks, NCthreads>>>(segcounter, NC*NC);
  	zero_vector_int<<<1,NC>>>(counter,NC);

  make_and_count_seg_sharedAtomic<<<1, NC2, NC*sizeof(int)>>>(vec, bin, segcounter, n, NC, HighLength, HighSegLength, threadsHigh, LowSegLength, low, high, slope);

 // 	make_and_count_seg<<<1, NC>>>(vec, bin, segcounter, n, NC, HighLength, HighSegLength, threadsHigh, LowSegLength, low, high, slope);

  
 // 	segCountSum<<<1,NC>>>(counter, segcounter, NC);
        segCountSum_shared<<<1, NC, NC>>>(counter, segcounter, NC);

  	FindKbin(counter, h_counter, NC, k, sum, k_bin);

  	cout << "Kbin = " << *k_bin << " and sum = " << *sum << endl;

  	excess = (*sum-k);
	ii++;


 // cout << "excess = " << excess << endl;
  }


  return;
}




inline void MakeCountAlt(float * d_vec, float * grad, int * d_bin, int * d_bin_counters, const int n, const int num_bins, const int kbin, const float mu, const float max_value, const float slope, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin) 
{

  zero_vector_int <<< numBlocks_bin, threadsPerBlock_bin >>>((int*)d_bin_counters,num_bins);

  float maxUpdate, minKbinVal;
  int maxBin;

  minKbinVal = max_value - (kbin+1)/slope;
  maxUpdate = MaxMagnitude(grad, n);
  float temp = max(0.0f, minKbinVal - mu*maxUpdate);
  maxBin = (int)( slope*(max_value-temp) );


 // cout << " slope and maxvalue are " << slope << " and " << max_value << endl;
 // cout << " minKbinVal and maxUpdate are " << minKbinVal << " and " << maxUpdate << endl;
 // cout << " kbin and maxBin are " << kbin << " and " << maxBin << endl;

  make_and_count_bins_alt <<< numBlocks, threadsPerBlock >>>((float*)d_vec, (int*)d_bin, (int*)d_bin_counters, num_bins, maxBin, n, slope, max_value);

}

*/











/*

inline void FindKbin(int *d_counter, int *h_counter, const int num_bins, const int k, int * sum, int * Kbin)
{
  cudaMemcpy(sum, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  Kbin[0]=0;

  if (sum[0]<k) {
	cudaMemcpy(h_counter, d_counter, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
  	SAFEcuda("cudaMemcpy to h_bin_counters");

  	while ( (sum[0]<k) & (Kbin[0]<num_bins) ) {
		Kbin[0] +=1; // = Kbin[0]+1;
		sum[0] = sum[0] + h_counter[Kbin[0]];
	}
  }

}

*/


/*
inline void FindKbin(int * d_bin, int *d_counter, int *h_counter, const int num_bins, const int maxBin, const int n, const int k, int * sum, int * Kbin)
{
  int tempSum = 0;
  int tempBin = 0;
  int preBin=Kbin[0]/2;

//cout << " preBin = " << preBin << endl;

  thrust::device_ptr<int> d_precount(d_counter);

if (preBin == 0) { 
	tempSum = d_precount[0];
	if (tempSum >= k) tempBin = 1;
}
else {

  

  // We want to sum on the GPU here up to Kbin[0]/2.  
  // cublasSasum is only for floats and won't work.  We have to write our own int sum.  
  // Right now we do the serial search up to here, but don't want to count this as time.

  tempSum = thrust::reduce(d_precount, d_precount + preBin, 0, thrust::plus<int>());
  tempBin = preBin;
/ *  <- fix this comment if you uncomment the whole function
  cudaMemcpy(h_counter, d_counter, preBin * sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_bin_counters");
  tempBin=0;
  while ( (tempBin < preBin) ) {
	tempSum = tempSum + h_counter[tempBin];
	tempBin++;
  }
* / <- fix this comment if you uncomment the whole function
//  cout << "In the precount: The count up to bin number " << tempBin << " is " << tempSum << endl;
  // now we have counted up to kbin/2, which we plan to do on the gpu 

}

  if ( tempSum < k ) {
	cudaMemcpy(h_counter+preBin, d_counter+preBin, (maxBin - preBin) * sizeof(int), cudaMemcpyDeviceToHost);
	while ( (tempSum < k) & (tempBin < maxBin) ) {
 		tempSum = tempSum + h_counter[tempBin];
		tempBin++;
  	}
//  cout << "In the maxBin count: The count up to bin number " << tempBin << " is " << tempSum << endl;
	if ( tempSum < k ) {
   		cout <<"Inside the countRest loop"<<endl;
		dim3 Threads = 896;
		int recountBlocks = (int)ceil((float)(num_bins-maxBin)/(float)896);
		dim3 Blocks(recountBlocks);
		countRest <<< Blocks, Threads >>>(d_bin, d_counter, num_bins, maxBin, n);
		cudaMemcpy(h_counter+maxBin, d_counter+maxBin, (num_bins - maxBin) * sizeof(int), cudaMemcpyDeviceToHost);
  		while ( (tempSum < k) & (tempBin < num_bins) ) {
 			tempSum += h_counter[tempBin];
			tempBin++; 
  		}
//  cout << "In the countRest count: The count up to bin number " << tempBin << " is " << tempSum << endl;
	}
  }
  else if (tempSum > k) {
	cudaMemcpy(h_counter, d_counter, preBin * sizeof(int), cudaMemcpyDeviceToHost);
  	SAFEcuda("cudaMemcpy to h_bin_counters");
  	tempBin=0;
  	while ( (tempSum < k) & (tempBin < preBin) ) {
		tempSum = tempSum + h_counter[tempBin];
		tempBin++; 
 	 }
//  cout << "INSIDE ELSE PRE GPU COUNT TOO HIGH: The count up to bin number " << tempBin << " is " << tempSum << endl;
  }

  sum[0]=tempSum;
  Kbin[0]=tempBin-1;

}

*/

/*
int find_kbin(int * d_counter, int *h_counter, int *k_bin, const int length, const int k, unsigned int max_threads)
{
  int sum;
  int kbin;

  cudaMemcpy(&sum, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

  if (sum<k) {
    int threads_perblock, num_blocks;
    int shift, adds, new_length;
    shift = 1;

    while ( (sum < k) & (shift < length) ) {
	
	adds = 2*shift;

  	new_length = length/(adds);

	threads_perblock = min(new_length, max_threads);
  	dim3 threadsPerBlock(threads_perblock);
  	num_blocks = (int)ceil((float)new_length/(float)threads_perblock);
  	dim3 numBlocks(num_blocks);

  	dyadicAdd<<<numBlocks, threadsPerBlock>>>(d_counter, length, shift);

  	cudaMemcpy(&sum, d_counter+(adds-1), sizeof(int), cudaMemcpyDeviceToHost);
	
	
	//cout << "cummulative sum = "<< sum << endl;
	//cout << "cummulative sum bin = "<< (adds-1) << endl;
	

	shift = adds;
	
    }

    cudaMemcpy(h_counter, d_counter, length * sizeof(int), cudaMemcpyDeviceToHost);

    int L, R, M, Lsum, Rsum, tempSum;
    R=adds-1;
    L=(adds/2)-1;
    Lsum=h_counter[L];
    Rsum=h_counter[R];

    while ( (R-L)>1 ) {
	M = (R+L)/2;
	tempSum = Lsum + h_counter[M];

	if (tempSum<k) {
		Lsum = tempSum;
		L = M;
	}
	else {
		Rsum = tempSum;
		R = M;
	}
    }
    kbin = R;
    sum = Rsum;
  }

  else { kbin=0; }

  *k_bin = kbin;

  return sum;
}

*/
/*
void MaxMagnitudeGPU(float * d_vec, float *maxValue, const int n)
{

  float *segmentMaxes;

if (n>45000){
  int numCores = 448;
  int LowSegmentLength = n/numCores;
  int numBlocks = LowSegmentLength/100;
  numBlocks = max(numBlocks, 1);
  LowSegmentLength = LowSegmentLength/numBlocks;
  int HighSegmentLength = LowSegmentLength;
  int HighLength = n;
  
  int segMaxLength = numBlocks*numCores;

  int threadsHigh = n - segMaxLength*LowSegmentLength;

  if (threadsHigh != 0) {
	HighSegmentLength = LowSegmentLength+1;
	HighLength = threadsHigh * HighSegmentLength;
  }

//  float *segmentMaxes;
  cudaMalloc((void**)&segmentMaxes, segMaxLength*sizeof(float));

  dim3 Blocks(numBlocks);
  dim3 numThreads(numCores);

  segmentMax<<<Blocks,numThreads>>>(d_vec, segmentMaxes, n, HighLength, HighSegmentLength, threadsHigh, LowSegmentLength);


  segmentMax<<<1,numThreads>>>(segmentMaxes, segmentMaxes, segMaxLength, numBlocks, numCores, segMaxLength, numBlocks);


  segmentMax<<<1,1>>>(segmentMaxes, segmentMaxes, 448, 448, 448, 1, 448);

  cudaMemcpy(maxValue, segmentMaxes, sizeof(float), cudaMemcpyDeviceToHost);
} 
else {
  int HighLength = n;
  int HighSegmentLength = n/448;
  int LowSegmentLength = n/448;
  int threadsHigh = n - 448*LowSegmentLength;

  if (threadsHigh != 0) {
	HighSegmentLength = LowSegmentLength+1;
	HighLength = threadsHigh * HighSegmentLength;
  }

  cudaMalloc((void**)&segmentMaxes, 448*sizeof(float));

  segmentMax<<<1,448>>>(d_vec, segmentMaxes, n, HighLength, HighSegmentLength, threadsHigh, LowSegmentLength);


  segmentMax<<<1,1>>>(segmentMaxes, segmentMaxes, 448, 448, 448, 1, 448);

  cudaMemcpy(maxValue, segmentMaxes, sizeof(float), cudaMemcpyDeviceToHost);


}
  
  cudaFree(segmentMaxes);

  return;
  
}
*/

/*
void MaxMagnitudeGPU(float * d_vec, float *maxValue, const int n)
{
  int HighLength = n;
  int HighSegmentLength = n/448;
  int LowSegmentLength = n/448;
  int threadsHigh = n - 448*LowSegmentLength;

  if (threadsHigh != 0) {
	HighSegmentLength = LowSegmentLength+1;
	HighLength = threadsHigh * HighSegmentLength;
  }

  float *segmentMaxes;
  cudaMalloc((void**)&segmentMaxes, 448*sizeof(float));

  segmentMax<<<1,448>>>(d_vec, segmentMaxes, n, HighLength, HighSegmentLength, threadsHigh, LowSegmentLength);


  segmentMax<<<1,1>>>(segmentMaxes, segmentMaxes, 448, 448, 448, 1, 448);

  cudaMemcpy(maxValue, segmentMaxes, sizeof(float), cudaMemcpyDeviceToHost);


  cudaFree(segmentMaxes);
  return;
  
}
*/



/*

inline void MaxMagnitude2(float * d_vec, float *max_value, int *ind_abs_max, const int n)
{
  ind_abs_max[0] = cublasIsamax(n, d_vec, 1) -1;
  SAFEcublas("cublasIsamax in IHT loop");

  cudaMemcpy(max_value, d_vec+ind_abs_max[0], sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to max_value");

  max_value[0] = abs(max_value[0]);

  return;

}



inline int binThreshold(float *d_vec, float *d_vec_thres, int *d_bin, int *d_bin_counters, int *h_bin_counters, const int n, const int k, const int num_bins, int* p_k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin) 
{


  zero_vector_int <<< numBlocks_bin, threadsPerBlock_bin >>>((int*)d_bin_counters,num_bins);
  SAFEcuda("zero_vector_int in IHT loop");


  int ind_abs_max = cublasIsamax(n, d_vec, 1) -1;
  SAFEcublas("cublasIsamax in IHT loop");

  float max_value;
  cudaMemcpy(&max_value, d_vec+ind_abs_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to max_value");


  max_value = abs(max_value);
  float slope = ((num_bins-1)/(max_value));

  make_and_count_bins <<< numBlocks, threadsPerBlock >>>((float*)d_vec, (int*)d_bin, (int*)d_bin_counters, num_bins, n, slope, max_value);
  SAFEcuda("make_and_count_bins in IHT loop");
  


  cudaMemcpy(h_bin_counters, d_bin_counters, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_bin_counters");

  int k_bin = 0;
  int sum=0;

  while ( (sum<k) & (k_bin<num_bins) ) {
	sum = sum + h_bin_counters[k_bin];
	k_bin++;
	}
  k_bin = k_bin-1;


  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);
  SAFEcuda("Threshold in binThreshold loop");


//  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)d_vec, (float*)d_vec_thres, (int*)d_bin, k_bin, n);
//  SAFEcuda("threshold_one in IHT loop");
//  cublasScopy(n, d_vec_thres, 1, d_vec, 1);
//  SAFEcublas("First cublasScopy in IHT loop");  


  *p_k_bin = k_bin;

  return sum;
}





inline int binThreshold_update(float *d_vec, float *d_vec_thres, int *d_bin, int *d_bin_counters, int *h_bin_counters, const int n, const int k, const int num_bins, const float max_value, const float slope, int* p_k_bin, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin) 
{



  update_bins <<< numBlocks, threadsPerBlock >>>((float*)d_vec, (int*)d_bin, (int*)d_bin_counters, num_bins, n, slope, max_value);
  SAFEcuda("update_bins in IHT loop");
  


  cudaMemcpy(h_bin_counters, d_bin_counters, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_bin_counters");

  int k_bin = 0;
  int sum=0;


  while ( (sum<k) & (k_bin<num_bins) ) {
	sum = sum + h_bin_counters[k_bin];
	k_bin++;
	}
  k_bin = k_bin-1;


  Threshold(d_vec, d_bin, k_bin, n, numBlocks, threadsPerBlock);

  *p_k_bin = k_bin;

  return sum;
}

*/


/*
inline void HT_S_dct(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_bin, int * d_bin_counters, int * h_bin_counters, const int num_bins, const int k, const int m, const int n, int * p_k_bin, int *p_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT initializiation");

  AT_dct(d_vec, resid, n, m, d_rows);
//  cudaThreadSynchronize();
  SAFEcuda("AT in HT loop");

  *p_sum = binThreshold(d_vec, d_vec_thres, d_bin, d_bin_counters, h_bin_counters, n, k, num_bins, p_k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);


}
*/


/*
inline void HT_S_smv(float *d_vec, float *d_vec_thres, float * grad, float *d_y, float *resid, float *resid_update, int * d_rows, int *d_cols, float *d_vals, int *d_bin, int * d_bin_counters, int * h_bin_counters, const int num_bins, const int k, const int m, const int n, const int nz,  int * p_k_bin, int *p_sum, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp, dim3 numBlocks_bin, dim3 threadsPerBlock_bin)
{

  cublasInit();
  cublasScopy(m, d_y, 1, resid, 1);
  SAFEcublas("cublasScopy in HT initializiation");

  AT_smv(d_vec, resid, m, n, d_rows, d_cols, d_vals, nz, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp);
//  cudaThreadSynchronize();
  SAFEcuda("AT in HT loop");

  *p_sum = binThreshold(d_vec, d_vec_thres, d_bin, d_bin_counters, h_bin_counters, n, k, num_bins, p_k_bin, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);


}
*/


