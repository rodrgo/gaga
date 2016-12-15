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


inline void results(float *d_vec, float *d_vec_input, int vecDistribution, float *residNorm_prev, float *h_norms, float *h_out, float *h_times, float *convergence_rate, int *total_iter, int *checkSupport, int iter, float timeIHT, float time_sum, int *p_sum, const int k, const int m, const int n, unsigned int seed, cudaEvent_t *p_startTest, cudaEvent_t *p_stopTest, char* algstr, dim3 numBlocks, dim3 threadsPerBlock)
{
  float convRate, root;
  int temp = min(iter, 16);
  if (temp <= 1) {
	convRate = 0.0f;
  }
  else {
	temp = temp-1;
  	root = 1/(float)temp;
  	temp=15-temp;
  	convRate = (residNorm_prev[15]/residNorm_prev[temp]);
  	convRate = pow(convRate, root);
  }

  convergence_rate[0]=convRate;

// debugging, this is unnecessary as we won't return the actual output.
// this should be deleletd along with the plhs[5] scripts above.

  cudaMemcpy(h_out, d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_out");



// record IHT times and iterations


  h_times[0]=timeIHT;

  h_times[1]=time_sum/(float)iter;

  total_iter[0] = iter;



// check support identification counting
// true_positive: correctly identified as in the support
// false_positive: incorrectly identified as in the support
// true_negative: correctly identified as not in support
// flase_negative: incorrectly identified as not in support
// From the last iteration, sum identifies total number of 
// nonzeros in the output vector.

  int * h_support_counter;
  h_support_counter = (int*)malloc(2*sizeof(int));
  h_support_counter[0]=0;
  h_support_counter[1]=0;

  int * d_support_counter;
  cudaMalloc((void**)&d_support_counter, 2*sizeof(int));
  SAFEcudaMalloc("d_support_counter");

  cudaMemcpy(d_support_counter, h_support_counter, 2*sizeof(int), cudaMemcpyHostToDevice);
  SAFEcuda("cudaMemcpy to d_support_counter");

  check_support<<< numBlocks, threadsPerBlock >>>(d_vec_input, d_vec, n, d_support_counter);
  SAFEcuda("check_support");

  cudaMemcpy(h_support_counter, d_support_counter, 2* sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_support_counter");

  int true_pos = h_support_counter[0];
  int false_pos = k- true_pos;
  int true_neg = h_support_counter[1];
  int false_neg = n - k - true_neg;


  checkSupport[0] = true_pos;
  checkSupport[1] = false_pos;
  checkSupport[2] = true_neg;
  checkSupport[3] = false_neg;


// Norms of input vector for computing relative norms
  float vec_input_2norm, vec_input_1norm, vec_input_supnorm;
  vec_input_1norm = cublasSasum(n, d_vec_input, 1);
  SAFEcublas("cublasSasum for vec_input_1norm in results.cu");
  vec_input_2norm = cublasSnrm2(n, d_vec_input, 1);
  SAFEcublas("cublasSnrm2 for vec_input_2norm in results.cu");
  
  int ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&vec_input_supnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  vec_input_supnorm = abs(vec_input_supnorm); 

// determine norms of the difference  
// between the input and output vectors  
  cublasSaxpy(n, -1.0f, d_vec, 1, d_vec_input, 1);
  SAFEcublas("cublasSaxpy in Results");

// Now d_vec_input is the difference between the 
// input and output vectors.
// We now compute the relative errors.
  float rel1norm, rel2norm, relsupnorm;

// relative ell-1 error
  rel1norm = cublasSasum(n, d_vec_input, 1);
  rel1norm = rel1norm/vec_input_1norm;
  h_norms[0] = rel1norm;
  SAFEcublas("h_norms[0]");


// relative ell-2 error

  rel2norm = cublasSnrm2(n, d_vec_input, 1);
  rel2norm = rel2norm/vec_input_2norm;
  h_norms[1] = rel2norm;
  SAFEcublas("h_norms[1]");

// relative ell-infinity error
  ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&relsupnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  relsupnorm = abs(relsupnorm);
  relsupnorm = relsupnorm/vec_input_supnorm;
  h_norms[2] = relsupnorm;


  cudaEvent_t startTest = *p_startTest;
  cudaEvent_t stopTest = *p_stopTest;

// record the final time of the entire test including
// memory allocation, problem generation, and error computations
  
  float timeTest;
  cudaThreadSynchronize();
  cudaEventRecord(stopTest,0);
  cudaEventSynchronize(stopTest);
  cudaEventElapsedTime(&timeTest, startTest, stopTest);
  cudaEventDestroy(startTest);
  cudaEventDestroy(stopTest);

  h_times[2]=timeTest;


// write all data to the big file

  FILE *foutput;

  time_t rawtime;
  time(&rawtime);
  struct tm * time_format;
  time_format = localtime(&rawtime);
  char fname[80]; 
  sprintf(fname,"gpu_data_dct%d%02d%02d.txt",time_format->tm_year+1900,time_format->tm_mon+1,time_format->tm_mday);

  foutput = fopen(fname, "a+");
  if (foutput == NULL) {
	printf("WARNING: Output file did not open!\n");
  }
  else {
	File_output(foutput, k, m, n, vecDistribution, h_norms, h_times, iter, convRate, checkSupport, seed, algstr);
  }
  fclose(foutput);


  cudaFree(d_support_counter);
  free(h_support_counter);

}




inline void results_smv(float *d_vec, float *d_vec_input, int vecDistribution, float *residNorm_prev, float *h_norms, float *h_out, float *h_times, float *convergence_rate, int *total_iter, int *checkSupport, int iter, float timeIHT, float time_sum, int *p_sum, const int k, const int m, const int n, const int p, const int matrixEnsemble, unsigned int seed, cudaEvent_t *p_startTest, cudaEvent_t *p_stopTest, char* algstr, dim3 numBlocks, dim3 threadsPerBlock, float band_percentage)
{
  float convRate, root;
  int temp = min(iter, 16);
  if (temp <= 1) {
	convRate = 0.0f;
  }
  else {
	temp = temp-1;
  	root = 1/(float)temp;
  	temp=15-temp;
  	convRate = (residNorm_prev[15]/residNorm_prev[temp]);
  	convRate = pow(convRate, root);
  }

  convergence_rate[0]=convRate;


// debugging, this is unnecessary as we won't return the actual output.
// this should be deleletd along with the plhs[5] scripts above.

  cudaMemcpy(h_out, d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_out");



// record IHT times and iterations


  h_times[0]=timeIHT;

  h_times[1]=time_sum/(float)iter;

  total_iter[0] = iter;



// check support identification counting
// true_positive: correctly identified as in the support
// false_positive: incorrectly identified as in the support
// true_negative: correctly identified as not in support
// flase_negative: incorrectly identified as not in support
// From the last iteration, sum identifies total number of 
// nonzeros in the output vector.

  int * h_support_counter;
  h_support_counter = (int*)malloc(2*sizeof(int));
  h_support_counter[0]=0;
  h_support_counter[1]=0;

  int * d_support_counter;
  cudaMalloc((void**)&d_support_counter, 2*sizeof(int));
  SAFEcudaMalloc("d_support_counter");

  cudaMemcpy(d_support_counter, h_support_counter, 2*sizeof(int), cudaMemcpyHostToDevice);
  SAFEcuda("cudaMemcpy to d_support_counter");

  check_support<<< numBlocks, threadsPerBlock >>>(d_vec_input, d_vec, n, d_support_counter);
  SAFEcuda("check_support");

  cudaMemcpy(h_support_counter, d_support_counter, 2* sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_support_counter");

  int true_pos = h_support_counter[0];
  int false_pos = k - true_pos;
  int true_neg = h_support_counter[1];
  int false_neg = n - k - true_neg;


  checkSupport[0] = true_pos;
  checkSupport[1] = false_pos;
  checkSupport[2] = true_neg;
  checkSupport[3] = false_neg;




// Norms of input vector for computing relative norms
  float vec_input_2norm, vec_input_1norm, vec_input_supnorm;
  vec_input_1norm = cublasSasum(n, d_vec_input, 1);
  SAFEcublas("cublasSasum for vec_input_1norm in results.cu");
  vec_input_2norm = cublasSnrm2(n, d_vec_input, 1);
  SAFEcublas("cublasSnrm2 for vec_input_2norm in results.cu");
  
  int ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&vec_input_supnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  vec_input_supnorm = abs(vec_input_supnorm); 

// determine norms of the difference  
// between the input and output vectors  
  cublasSaxpy(n, -1.0f, d_vec, 1, d_vec_input, 1);
  SAFEcublas("cublasSaxpy in Results");

// Now d_vec_input is the difference between the 
// input and output vectors.
// We now compute the relative errors.
  float rel1norm, rel2norm, relsupnorm;

// relative ell-1 error
  rel1norm = cublasSasum(n, d_vec_input, 1);
  rel1norm = rel1norm/vec_input_1norm;
  h_norms[0] = rel1norm;
  SAFEcublas("h_norms[0]");


// relative ell-2 error

  rel2norm = cublasSnrm2(n, d_vec_input, 1);
  rel2norm = rel2norm/vec_input_2norm;
  h_norms[1] = rel2norm;
  SAFEcublas("h_norms[1]");

// relative ell-infinity error
  ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&relsupnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  relsupnorm = abs(relsupnorm);
  relsupnorm = relsupnorm/vec_input_supnorm;
  h_norms[2] = relsupnorm;


  cudaEvent_t startTest = *p_startTest;
  cudaEvent_t stopTest = *p_stopTest;

// record the final time of the entire test including
// memory allocation, problem generation, and error computations
  
  float timeTest;
  cudaThreadSynchronize();
  cudaEventRecord(stopTest,0);
  cudaEventSynchronize(stopTest);
  cudaEventElapsedTime(&timeTest, startTest, stopTest);
  cudaEventDestroy(startTest);
  cudaEventDestroy(stopTest);

  h_times[2]=timeTest;


// write all data to the big file

  FILE *foutput;

  time_t rawtime;
  time(&rawtime);
  struct tm * time_format;
  time_format = localtime(&rawtime);
  char fname[80]; 
  sprintf(fname,"gpu_data_smv%d%02d%02d.txt",time_format->tm_year+1900,time_format->tm_mon+1,time_format->tm_mday);

  foutput = fopen(fname, "a+");
  if (foutput == NULL) {
	printf("WARNING: Output file did not open!\n");
  }
  else {
	File_output_smv(foutput, k, m, n, vecDistribution, h_norms, h_times, iter, convRate, checkSupport, seed, algstr, p, matrixEnsemble, band_percentage, 0.0);
  }
  fclose(foutput);


  cudaFree(d_support_counter);
  free(h_support_counter);

}




inline void results_gen(float *d_vec, float *d_vec_input, int vecDistribution, float *residNorm_prev, float *h_norms, float *h_out, float *h_times, float *convergence_rate, int *total_iter, int *checkSupport, int iter, float timeIHT, float time_sum, int *p_sum, const int k, const int m, const int n, const int matrixEnsemble, unsigned int seed, cudaEvent_t *p_startTest, cudaEvent_t *p_stopTest, char* algstr, dim3 numBlocks, dim3 threadsPerBlock)
{
  float convRate, root;
  int temp = min(iter, 16);
  if (temp <= 1) {
	convRate = 0.0f;
  }
  else {
	temp = temp-1;
  	root = 1/(float)temp;
  	temp=15-temp;
  	convRate = (residNorm_prev[15]/residNorm_prev[temp]);
  	convRate = pow(convRate, root);
  }

  convergence_rate[0]=convRate;


// debugging, this is unnecessary as we won't return the actual output.
// this should be deleletd along with the plhs[5] scripts above.

  cudaMemcpy(h_out, d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_out");



// record IHT times and iterations


  h_times[0]=timeIHT;

  h_times[1]=time_sum/(float)iter;

  total_iter[0] = iter;



// check support identification counting
// true_positive: correctly identified as in the support
// false_positive: incorrectly identified as in the support
// true_negative: correctly identified as not in support
// flase_negative: incorrectly identified as not in support
// From the last iteration, sum identifies total number of 
// nonzeros in the output vector.

  int * h_support_counter;
  h_support_counter = (int*)malloc(2*sizeof(int));
  h_support_counter[0]=0;
  h_support_counter[1]=0;

  int * d_support_counter;
  cudaMalloc((void**)&d_support_counter, 2*sizeof(int));
  SAFEcudaMalloc("d_support_counter");

  cudaMemcpy(d_support_counter, h_support_counter, 2*sizeof(int), cudaMemcpyHostToDevice);
  SAFEcuda("cudaMemcpy to d_support_counter");

  check_support<<< numBlocks, threadsPerBlock >>>(d_vec_input, d_vec, n, d_support_counter);
  SAFEcuda("check_support");

  cudaMemcpy(h_support_counter, d_support_counter, 2* sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_support_counter");

  int true_pos = h_support_counter[0];
  int false_pos = k - true_pos;
  int true_neg = h_support_counter[1];
  int false_neg = n - k - true_neg;



  checkSupport[0] = true_pos;
  checkSupport[1] = false_pos;
  checkSupport[2] = true_neg;
  checkSupport[3] = false_neg;


  cudaFree(d_support_counter);
  SAFEcuda("in results_dct after cudaFree");


// Norms of input vector for computing relative norms
  float vec_input_2norm, vec_input_1norm, vec_input_supnorm;
  vec_input_1norm = cublasSasum(n, d_vec_input, 1);
  SAFEcublas("cublasSasum for vec_input_1norm in results.cu");
  vec_input_2norm = cublasSnrm2(n, d_vec_input, 1);
  SAFEcublas("cublasSnrm2 for vec_input_2norm in results.cu");
  
  int ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&vec_input_supnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  vec_input_supnorm = abs(vec_input_supnorm); 

// determine norms of the difference  
// between the input and output vectors  
  cublasSaxpy(n, -1.0f, d_vec, 1, d_vec_input, 1);
  SAFEcublas("cublasSaxpy in Results");

// Now d_vec_input is the difference between the 
// input and output vectors.
// We now compute the relative errors.
  float rel1norm, rel2norm, relsupnorm;

// relative ell-1 error
  rel1norm = cublasSasum(n, d_vec_input, 1);
  rel1norm = rel1norm/vec_input_1norm;
  h_norms[0] = rel1norm;
  SAFEcublas("h_norms[0]");


// relative ell-2 error

  rel2norm = cublasSnrm2(n, d_vec_input, 1);
  rel2norm = rel2norm/vec_input_2norm;
  h_norms[1] = rel2norm;
  SAFEcublas("h_norms[1]");

// relative ell-infinity error
  ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&relsupnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  relsupnorm = abs(relsupnorm);
  relsupnorm = relsupnorm/vec_input_supnorm;
  h_norms[2] = relsupnorm;


  cudaEvent_t startTest = *p_startTest;
  cudaEvent_t stopTest = *p_stopTest;

// record the final time of the entire test including
// memory allocation, problem generation, and error computations
  
  float timeTest;
  cudaThreadSynchronize();
  cudaEventRecord(stopTest,0);
  cudaEventSynchronize(stopTest);
  cudaEventElapsedTime(&timeTest, startTest, stopTest);
  cudaEventDestroy(startTest);
  cudaEventDestroy(stopTest);

  h_times[2]=timeTest;


// write all data to the big file

  FILE *foutput;

  time_t rawtime;
  time(&rawtime);
  struct tm * time_format;
  time_format = localtime(&rawtime);
  char fname[80]; 
  sprintf(fname,"gpu_data_gen%d%02d%02d.txt",time_format->tm_year+1900,time_format->tm_mon+1,time_format->tm_mday);

  foutput = fopen(fname, "a+");
  if (foutput == NULL) {
	printf("WARNING: Output file did not open!\n");
  }
  else {
	File_output_gen(foutput, k, m, n, matrixEnsemble, vecDistribution, h_norms, h_times, iter, convRate, checkSupport, seed, algstr);
  }
  fclose(foutput);


  free(h_support_counter);

}



inline void results_dct_noise(float *d_vec, float *d_vec_input, int vecDistribution, float *residNorm_prev, float *h_norms, float *h_out, float *h_times, float *convergence_rate, int *total_iter, int *checkSupport, int iter, float timeIHT, float time_sum, int *p_sum, const int k, const int m, const int n, unsigned int seed, float noise_level, cudaEvent_t *p_startTest, cudaEvent_t *p_stopTest, char* algstr, dim3 numBlocks, dim3 threadsPerBlock)
{ 

  float convRate, root;
  int temp = min(iter, 16);
  if (temp <= 1) {
	convRate = 0.0f;
  }
  else {
	temp = temp-1;
  	root = 1/(float)temp;
  	temp=15-temp;
  	convRate = (residNorm_prev[15]/residNorm_prev[temp]);
  	convRate = pow(convRate, root);
  }

  convergence_rate[0]=convRate;


// debugging, this is unnecessary as we won't return the actual output.
// this should be deleletd along with the plhs[5] scripts above.

  cudaMemcpy(h_out, d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_out");



// record IHT times and iterations


  h_times[0]=timeIHT;

  h_times[1]=time_sum/(float)iter;

  total_iter[0] = iter;



// check support identification counting
// true_positive: correctly identified as in the support
// false_positive: incorrectly identified as in the support
// true_negative: correctly identified as not in support
// flase_negative: incorrectly identified as not in support
// From the last iteration, sum identifies total number of 
// nonzeros in the output vector.

  int * h_support_counter;
  h_support_counter = (int*)malloc(2*sizeof(int));
  h_support_counter[0]=0;
  h_support_counter[1]=0;


  int * d_support_counter;
  cudaMalloc((void**)&d_support_counter, 2*sizeof(int));
  SAFEcudaMalloc("d_support_counter");

  cudaMemcpy(d_support_counter, h_support_counter, 2*sizeof(int), cudaMemcpyHostToDevice);
  SAFEcuda("cudaMemcpy to d_support_counter");

  check_support<<< numBlocks, threadsPerBlock >>>(d_vec_input, d_vec, n, d_support_counter);
  SAFEcuda("check_support");

  cudaMemcpy(h_support_counter, d_support_counter, 2* sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_support_counter");

  int true_pos = h_support_counter[0];
  int false_pos = k - true_pos;
  int true_neg = h_support_counter[1];
  int false_neg = n - k - true_neg;


  checkSupport[0] = true_pos;
  checkSupport[1] = false_pos;
  checkSupport[2] = true_neg;
  checkSupport[3] = false_neg;



// Norms of input vector for computing relative norms
  float vec_input_2norm, vec_input_1norm, vec_input_supnorm;
  vec_input_1norm = cublasSasum(n, d_vec_input, 1);
  SAFEcublas("cublasSasum for vec_input_1norm in results.cu");
  vec_input_2norm = cublasSnrm2(n, d_vec_input, 1);
  SAFEcublas("cublasSnrm2 for vec_input_2norm in results.cu");
  
  int ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&vec_input_supnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  vec_input_supnorm = abs(vec_input_supnorm); 

// determine norms of the difference  
// between the input and output vectors  
  cublasSaxpy(n, -1.0f, d_vec, 1, d_vec_input, 1);
  SAFEcublas("cublasSaxpy in Results");

// Now d_vec_input is the difference between the 
// input and output vectors.
// We now compute the relative errors.
  float rel1norm, rel2norm, relsupnorm;

// relative ell-1 error
  rel1norm = cublasSasum(n, d_vec_input, 1);
  rel1norm = rel1norm/vec_input_1norm;
  h_norms[0] = rel1norm;
  SAFEcublas("h_norms[0]");


// relative ell-2 error

  rel2norm = cublasSnrm2(n, d_vec_input, 1);
  rel2norm = rel2norm/vec_input_2norm;
  h_norms[1] = rel2norm;
  SAFEcublas("h_norms[1]");

// relative ell-infinity error
  ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&relsupnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  relsupnorm = abs(relsupnorm);
  relsupnorm = relsupnorm/vec_input_supnorm;
  h_norms[2] = relsupnorm;


  cudaEvent_t startTest = *p_startTest;
  cudaEvent_t stopTest = *p_stopTest;

// record the final time of the entire test including
// memory allocation, problem generation, and error computations
  
  float timeTest;
  cudaThreadSynchronize();
  cudaEventRecord(stopTest,0);
  cudaEventSynchronize(stopTest);
  cudaEventElapsedTime(&timeTest, startTest, stopTest);
  cudaEventDestroy(startTest);
  cudaEventDestroy(stopTest);

  h_times[2]=timeTest;



// write all data to the big file

  FILE *foutput;

  time_t rawtime;
  time(&rawtime);
  struct tm * time_format;
  time_format = localtime(&rawtime);
  char fname[80]; 
  sprintf(fname,"gpu_data_dct%d%02d%02d_noise%0.3f.txt",time_format->tm_year+1900,time_format->tm_mon+1,time_format->tm_mday,noise_level);

  foutput = fopen(fname, "a+");
  if (foutput == NULL) {
	printf("WARNING: Output file did not open!\n");
  }
  else {  
	File_output(foutput, k, m, n, vecDistribution, h_norms, h_times, iter, convRate, checkSupport, seed, algstr);
  }
  fclose(foutput);

  cudaFree(d_support_counter);
  SAFEcuda("in results_dct_noise after cudaFree");

  free(h_support_counter);


}



inline void results_smv_noise(float *d_vec, float *d_vec_input, int vecDistribution, float *residNorm_prev, float *h_norms, float *h_out, float *h_times, float *convergence_rate, int *total_iter, int *checkSupport, int iter, float timeIHT, float time_sum, int *p_sum, const int k, const int m, const int n, const int p, const int matrixEnsemble, unsigned int seed, float noise_level, cudaEvent_t *p_startTest, cudaEvent_t *p_stopTest, char* algstr, dim3 numBlocks, dim3 threadsPerBlock, float band_percentage)
{
  float convRate, root;
  int temp = min(iter, 16);
  if (temp <= 1) {
	convRate = 0.0f;
  }
  else {
	temp = temp-1;
  	root = 1/(float)temp;
  	temp=15-temp;
  	convRate = (residNorm_prev[15]/residNorm_prev[temp]);
  	convRate = pow(convRate, root);
  }

  convergence_rate[0]=convRate;


// debugging, this is unnecessary as we won't return the actual output.
// this should be deleletd along with the plhs[5] scripts above.

  cudaMemcpy(h_out, d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_out");



// record IHT times and iterations


  h_times[0]=timeIHT;

  h_times[1]=time_sum/(float)iter;

  total_iter[0] = iter;



// check support identification counting
// true_positive: correctly identified as in the support
// false_positive: incorrectly identified as in the support
// true_negative: correctly identified as not in support
// flase_negative: incorrectly identified as not in support
// From the last iteration, sum identifies total number of 
// nonzeros in the output vector.

  int * h_support_counter;
  h_support_counter = (int*)malloc(2*sizeof(int));
  h_support_counter[0]=0;
  h_support_counter[1]=0;

  int * d_support_counter;
  cudaMalloc((void**)&d_support_counter, 2*sizeof(int));
  SAFEcudaMalloc("d_support_counter");

  cudaMemcpy(d_support_counter, h_support_counter, 2*sizeof(int), cudaMemcpyHostToDevice);
  SAFEcuda("cudaMemcpy to d_support_counter");

  check_support<<< numBlocks, threadsPerBlock >>>(d_vec_input, d_vec, n, d_support_counter);
  SAFEcuda("check_support");

  cudaMemcpy(h_support_counter, d_support_counter, 2* sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_support_counter");

  int true_pos = h_support_counter[0];
  int false_pos = k - true_pos;
  int true_neg = h_support_counter[1];
  int false_neg = n - k - true_neg;


  checkSupport[0] = true_pos;
  checkSupport[1] = false_pos;
  checkSupport[2] = true_neg;
  checkSupport[3] = false_neg;




// Norms of input vector for computing relative norms
  float vec_input_2norm, vec_input_1norm, vec_input_supnorm;
  vec_input_1norm = cublasSasum(n, d_vec_input, 1);
  SAFEcublas("cublasSasum for vec_input_1norm in results.cu");
  vec_input_2norm = cublasSnrm2(n, d_vec_input, 1);
  SAFEcublas("cublasSnrm2 for vec_input_2norm in results.cu");
  
  int ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&vec_input_supnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  vec_input_supnorm = abs(vec_input_supnorm); 

// determine norms of the difference  
// between the input and output vectors  
  cublasSaxpy(n, -1.0f, d_vec, 1, d_vec_input, 1);
  SAFEcublas("cublasSaxpy in Results");

// Now d_vec_input is the difference between the 
// input and output vectors.
// We now compute the relative errors.
  float rel1norm, rel2norm, relsupnorm;

// relative ell-1 error
  rel1norm = cublasSasum(n, d_vec_input, 1);
  rel1norm = rel1norm/vec_input_1norm;
  h_norms[0] = rel1norm;
  SAFEcublas("h_norms[0]");


// relative ell-2 error

  rel2norm = cublasSnrm2(n, d_vec_input, 1);
  rel2norm = rel2norm/vec_input_2norm;
  h_norms[1] = rel2norm;
  SAFEcublas("h_norms[1]");

// relative ell-infinity error
  ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&relsupnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  relsupnorm = abs(relsupnorm);
  relsupnorm = relsupnorm/vec_input_supnorm;
  h_norms[2] = relsupnorm;


  cudaEvent_t startTest = *p_startTest;
  cudaEvent_t stopTest = *p_stopTest;

// record the final time of the entire test including
// memory allocation, problem generation, and error computations
  
  float timeTest;
  cudaThreadSynchronize();
  cudaEventRecord(stopTest,0);
  cudaEventSynchronize(stopTest);
  cudaEventElapsedTime(&timeTest, startTest, stopTest);
  cudaEventDestroy(startTest);
  cudaEventDestroy(stopTest);

  h_times[2]=timeTest;


// write all data to the big file

  FILE *foutput;

  time_t rawtime;
  time(&rawtime);
  struct tm * time_format;
  time_format = localtime(&rawtime);
  char fname[80]; 
  sprintf(fname,"gpu_data_smv%d%02d%02d_noise%0.3f.txt",time_format->tm_year+1900,time_format->tm_mon+1,time_format->tm_mday,noise_level);

  foutput = fopen(fname, "a+");
  if (foutput == NULL) {
	printf("WARNING: Output file did not open!\n");
  }
  else {
	File_output_smv(foutput, k, m, n, vecDistribution, h_norms, h_times, iter, convRate, checkSupport, seed, algstr, p, matrixEnsemble, band_percentage, noise_level);
  }
  fclose(foutput);


  cudaFree(d_support_counter);
  free(h_support_counter);

}




inline void results_gen_noise(float *d_vec, float *d_vec_input, int vecDistribution, float *residNorm_prev, float *h_norms, float *h_out, float *h_times, float *convergence_rate, int *total_iter, int *checkSupport, int iter, float timeIHT, float time_sum, int *p_sum, const int k, const int m, const int n, const int matrixEnsemble, unsigned int seed, float noise_level, cudaEvent_t *p_startTest, cudaEvent_t *p_stopTest, char* algstr, dim3 numBlocks, dim3 threadsPerBlock)
{
  float convRate, root;
  int temp = min(iter, 16);
  if (temp <= 1) {
	convRate = 0.0f;
  }
  else {
	temp = temp-1;
  	root = 1/(float)temp;
  	temp=15-temp;
  	convRate = (residNorm_prev[15]/residNorm_prev[temp]);
  	convRate = pow(convRate, root);
  }

  convergence_rate[0]=convRate;


// debugging, this is unnecessary as we won't return the actual output.
// this should be deleletd along with the plhs[5] scripts above.

  cudaMemcpy(h_out, d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_out");



// record IHT times and iterations


  h_times[0]=timeIHT;

  h_times[1]=time_sum/(float)iter;

  total_iter[0] = iter;



// check support identification counting
// true_positive: correctly identified as in the support
// false_positive: incorrectly identified as in the support
// true_negative: correctly identified as not in support
// flase_negative: incorrectly identified as not in support
// From the last iteration, sum identifies total number of 
// nonzeros in the output vector.

  int * h_support_counter;
  h_support_counter = (int*)malloc(2*sizeof(int));
  h_support_counter[0]=0;
  h_support_counter[1]=0;

  int * d_support_counter;
  cudaMalloc((void**)&d_support_counter, 2*sizeof(int));
  SAFEcudaMalloc("d_support_counter");

  cudaMemcpy(d_support_counter, h_support_counter, 2*sizeof(int), cudaMemcpyHostToDevice);
  SAFEcuda("cudaMemcpy to d_support_counter");

  check_support<<< numBlocks, threadsPerBlock >>>(d_vec_input, d_vec, n, d_support_counter);
  SAFEcuda("check_support");

  cudaMemcpy(h_support_counter, d_support_counter, 2* sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_support_counter");

  int true_pos = h_support_counter[0];
  int false_pos = k - true_pos;
  int true_neg = h_support_counter[1];
  int false_neg = n - k - true_neg;


  checkSupport[0] = true_pos;
  checkSupport[1] = false_pos;
  checkSupport[2] = true_neg;
  checkSupport[3] = false_neg;



// Norms of input vector for computing relative norms
  float vec_input_2norm, vec_input_1norm, vec_input_supnorm;
  vec_input_1norm = cublasSasum(n, d_vec_input, 1);
  SAFEcublas("cublasSasum for vec_input_1norm in results.cu");
  vec_input_2norm = cublasSnrm2(n, d_vec_input, 1);
  SAFEcublas("cublasSnrm2 for vec_input_2norm in results.cu");
  
  int ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&vec_input_supnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  vec_input_supnorm = abs(vec_input_supnorm); 

// determine norms of the difference  
// between the input and output vectors  
  cublasSaxpy(n, -1.0f, d_vec, 1, d_vec_input, 1);
  SAFEcublas("cublasSaxpy in Results");

// Now d_vec_input is the difference between the 
// input and output vectors.
// We now compute the relative errors.
  float rel1norm, rel2norm, relsupnorm;

// relative ell-1 error
  rel1norm = cublasSasum(n, d_vec_input, 1);
  rel1norm = rel1norm/vec_input_1norm;
  h_norms[0] = rel1norm;
  SAFEcublas("h_norms[0]");


// relative ell-2 error

  rel2norm = cublasSnrm2(n, d_vec_input, 1);
  rel2norm = rel2norm/vec_input_2norm;
  h_norms[1] = rel2norm;
  SAFEcublas("h_norms[1]");

// relative ell-infinity error
  ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&relsupnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  relsupnorm = abs(relsupnorm);
  relsupnorm = relsupnorm/vec_input_supnorm;
  h_norms[2] = relsupnorm;


  cudaEvent_t startTest = *p_startTest;
  cudaEvent_t stopTest = *p_stopTest;

// record the final time of the entire test including
// memory allocation, problem generation, and error computations
  
  float timeTest;
  cudaThreadSynchronize();
  cudaEventRecord(stopTest,0);
  cudaEventSynchronize(stopTest);
  cudaEventElapsedTime(&timeTest, startTest, stopTest);
  cudaEventDestroy(startTest);
  cudaEventDestroy(stopTest);

  h_times[2]=timeTest;


// write all data to the big file

  FILE *foutput;

  time_t rawtime;
  time(&rawtime);
  struct tm * time_format;
  time_format = localtime(&rawtime);
  char fname[80]; 
  sprintf(fname,"gpu_data_gen%d%02d%02d_noise%0.3f.txt",time_format->tm_year+1900,time_format->tm_mon+1,time_format->tm_mday,noise_level);

  foutput = fopen(fname, "a+");
  if (foutput == NULL) {
	printf("WARNING: Output file did not open!\n");
  }
  else {
	File_output_gen(foutput, k, m, n, matrixEnsemble, vecDistribution, h_norms, h_times, iter, convRate, checkSupport, seed, algstr);
  }
  fclose(foutput);


  cudaFree(d_support_counter);
  free(h_support_counter);

}








inline void results_timings(float *d_vec, float *d_vec_input, int vecDistribution, float *residNorm_prev, float *h_norms, float *h_out, float *h_times, float *convergence_rate, int *total_iter, int *checkSupport, int iter, float timeIHT, float time_sum, float *time_per_iteration, float *time_supp_set, int *p_sum, float alpha, int supp_flag, const int k, const int m, const int n, unsigned int seed, cudaEvent_t *p_startTest, cudaEvent_t *p_stopTest, char* algstr, dim3 numBlocks, dim3 threadsPerBlock)
{
  float convRate, root;
  int temp = min(iter, 16);
  if (temp <= 1) {
	convRate = 0.0f;
  }
  else {
	temp = temp-1;
  	root = 1/(float)temp;
  	temp=15-temp;
  	convRate = (residNorm_prev[15]/residNorm_prev[temp]);
  	convRate = pow(convRate, root);
  }

  convergence_rate[0]=convRate;


// debugging, this is unnecessary as we won't return the actual output.
// this should be deleletd along with the plhs[5] scripts above.

  cudaMemcpy(h_out, d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_out");



// record IHT times and iterations


  h_times[0]=timeIHT;

  h_times[1]=time_sum/(float)iter;

  total_iter[0] = iter;



// check support identification counting
// true_positive: correctly identified as in the support
// false_positive: incorrectly identified as in the support
// true_negative: correctly identified as not in support
// flase_negative: incorrectly identified as not in support
// From the last iteration, sum identifies total number of 
// nonzeros in the output vector.

  int * h_support_counter;
  h_support_counter = (int*)malloc(2*sizeof(int));
  h_support_counter[0]=0;
  h_support_counter[1]=0;

  int * d_support_counter;
  cudaMalloc((void**)&d_support_counter, 2*sizeof(int));
  SAFEcudaMalloc("d_support_counter");

  cudaMemcpy(d_support_counter, h_support_counter, 2*sizeof(int), cudaMemcpyHostToDevice);
  SAFEcuda("cudaMemcpy to d_support_counter");

  check_support<<< numBlocks, threadsPerBlock >>>(d_vec_input, d_vec, n, d_support_counter);
  SAFEcuda("check_support");

  cudaMemcpy(h_support_counter, d_support_counter, 2* sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_support_counter");

  int true_pos = h_support_counter[0];
  int false_pos = k - true_pos;
  int true_neg = h_support_counter[1];
  int false_neg = n - k - true_neg;


  checkSupport[0] = true_pos;
  checkSupport[1] = false_pos;
  checkSupport[2] = true_neg;
  checkSupport[3] = false_neg;




// Norms of input vector for computing relative norms
  float vec_input_2norm, vec_input_1norm, vec_input_supnorm;
  vec_input_1norm = cublasSasum(n, d_vec_input, 1);
  SAFEcublas("cublasSasum for vec_input_1norm in results.cu");
  vec_input_2norm = cublasSnrm2(n, d_vec_input, 1);
  SAFEcublas("cublasSnrm2 for vec_input_2norm in results.cu");
  
  int ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&vec_input_supnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  vec_input_supnorm = abs(vec_input_supnorm); 

// determine norms of the difference  
// between the input and output vectors  
  cublasSaxpy(n, -1.0f, d_vec, 1, d_vec_input, 1);
  SAFEcublas("cublasSaxpy in Results");

// Now d_vec_input is the difference between the 
// input and output vectors.
// We now compute the relative errors.
  float rel1norm, rel2norm, relsupnorm;

// relative ell-1 error
  rel1norm = cublasSasum(n, d_vec_input, 1);
  rel1norm = rel1norm/vec_input_1norm;
  h_norms[0] = rel1norm;
  SAFEcublas("h_norms[0]");


// relative ell-2 error

  rel2norm = cublasSnrm2(n, d_vec_input, 1);
  rel2norm = rel2norm/vec_input_2norm;
  h_norms[1] = rel2norm;
  SAFEcublas("h_norms[1]");

// relative ell-infinity error
  ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&relsupnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  relsupnorm = abs(relsupnorm);
  relsupnorm = relsupnorm/vec_input_supnorm;
  h_norms[2] = relsupnorm;


  cudaEvent_t startTest = *p_startTest;
  cudaEvent_t stopTest = *p_stopTest;

// record the final time of the entire test including
// memory allocation, problem generation, and error computations
  
  float timeTest;
  cudaThreadSynchronize();
  cudaEventRecord(stopTest,0);
  cudaEventSynchronize(stopTest);
  cudaEventElapsedTime(&timeTest, startTest, stopTest);
  cudaEventDestroy(startTest);
  cudaEventDestroy(stopTest);

  h_times[2]=timeTest;


// write all data to the big file

  FILE *foutput;

  time_t rawtime;
  time(&rawtime);
  struct tm * time_format;
  time_format = localtime(&rawtime);
  char fname[80]; 
  sprintf(fname,"gpu_timings_data_dct%d%02d%02d.txt",time_format->tm_year+1900,time_format->tm_mon+1,time_format->tm_mday);

  foutput = fopen(fname, "a+");
  if (foutput == NULL) {
	printf("WARNING: Output file did not open!\n");
  }
  else {
	File_output_timings(foutput, k, m, n, vecDistribution, h_norms, h_times, iter, convRate, checkSupport, seed, time_per_iteration, time_supp_set, algstr, alpha, supp_flag);
  }
  fclose(foutput);


  cudaFree(d_support_counter);
  free(h_support_counter);

}




inline void results_timings_smv(float *d_vec, float *d_vec_input, int vecDistribution, float *residNorm_prev, float *h_norms, float *h_out, float *h_times, float *convergence_rate, int *total_iter, int *checkSupport, int iter, float timeIHT, float time_sum, float *time_per_iteration, float *time_supp_set, int *p_sum, float alpha, int supp_flag, const int k, const int m, const int n, const int p, const int matrixEnsemble, unsigned int seed, cudaEvent_t *p_startTest, cudaEvent_t *p_stopTest, char* algstr, dim3 numBlocks, dim3 threadsPerBlock)
{
  float convRate, root;
  int temp = min(iter, 16);
  if (temp <= 1) {
	convRate = 0.0f;
  }
  else {
	temp = temp-1;
  	root = 1/(float)temp;
  	temp=15-temp;
  	convRate = (residNorm_prev[15]/residNorm_prev[temp]);
  	convRate = pow(convRate, root);
  }

  convergence_rate[0]=convRate;


// debugging, this is unnecessary as we won't return the actual output.
// this should be deleletd along with the plhs[5] scripts above.

  cudaMemcpy(h_out, d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_out");



// record IHT times and iterations


  h_times[0]=timeIHT;

  h_times[1]=time_sum/(float)iter;

  total_iter[0] = iter;



// check support identification counting
// true_positive: correctly identified as in the support
// false_positive: incorrectly identified as in the support
// true_negative: correctly identified as not in support
// flase_negative: incorrectly identified as not in support
// From the last iteration, sum identifies total number of 
// nonzeros in the output vector.

  int * h_support_counter;
  h_support_counter = (int*)malloc(2*sizeof(int));
  h_support_counter[0]=0;
  h_support_counter[1]=0;

  int * d_support_counter;
  cudaMalloc((void**)&d_support_counter, 2*sizeof(int));
  SAFEcudaMalloc("d_support_counter");

  cudaMemcpy(d_support_counter, h_support_counter, 2*sizeof(int), cudaMemcpyHostToDevice);
  SAFEcuda("cudaMemcpy to d_support_counter");

  check_support<<< numBlocks, threadsPerBlock >>>(d_vec_input, d_vec, n, d_support_counter);
  SAFEcuda("check_support");

  cudaMemcpy(h_support_counter, d_support_counter, 2* sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_support_counter");

  int true_pos = h_support_counter[0];
  int false_pos = k - true_pos;
  int true_neg = h_support_counter[1];
  int false_neg = n - k - true_neg;


  checkSupport[0] = true_pos;
  checkSupport[1] = false_pos;
  checkSupport[2] = true_neg;
  checkSupport[3] = false_neg;



// Norms of input vector for computing relative norms
  float vec_input_2norm, vec_input_1norm, vec_input_supnorm;
  vec_input_1norm = cublasSasum(n, d_vec_input, 1);
  SAFEcublas("cublasSasum for vec_input_1norm in results.cu");
  vec_input_2norm = cublasSnrm2(n, d_vec_input, 1);
  SAFEcublas("cublasSnrm2 for vec_input_2norm in results.cu");
  
  int ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&vec_input_supnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  vec_input_supnorm = abs(vec_input_supnorm); 

// determine norms of the difference  
// between the input and output vectors  
  cublasSaxpy(n, -1.0f, d_vec, 1, d_vec_input, 1);
  SAFEcublas("cublasSaxpy in Results");

// Now d_vec_input is the difference between the 
// input and output vectors.
// We now compute the relative errors.
  float rel1norm, rel2norm, relsupnorm;

// relative ell-1 error
  rel1norm = cublasSasum(n, d_vec_input, 1);
  rel1norm = rel1norm/vec_input_1norm;
  h_norms[0] = rel1norm;
  SAFEcublas("h_norms[0]");


// relative ell-2 error

  rel2norm = cublasSnrm2(n, d_vec_input, 1);
  rel2norm = rel2norm/vec_input_2norm;
  h_norms[1] = rel2norm;
  SAFEcublas("h_norms[1]");

// relative ell-infinity error
  ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&relsupnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  relsupnorm = abs(relsupnorm);
  relsupnorm = relsupnorm/vec_input_supnorm;
  h_norms[2] = relsupnorm;


  cudaEvent_t startTest = *p_startTest;
  cudaEvent_t stopTest = *p_stopTest;

// record the final time of the entire test including
// memory allocation, problem generation, and error computations
  
  float timeTest;
  cudaThreadSynchronize();
  cudaEventRecord(stopTest,0);
  cudaEventSynchronize(stopTest);
  cudaEventElapsedTime(&timeTest, startTest, stopTest);
  cudaEventDestroy(startTest);
  cudaEventDestroy(stopTest);

  h_times[2]=timeTest;


// write all data to the big file

  FILE *foutput;

  time_t rawtime;
  time(&rawtime);
  struct tm * time_format;
  time_format = localtime(&rawtime);
  char fname[80]; 
  sprintf(fname,"gpu_timings_data_smv%d%02d%02d.txt",time_format->tm_year+1900,time_format->tm_mon+1,time_format->tm_mday);

  foutput = fopen(fname, "a+");
  if (foutput == NULL) {
	printf("WARNING: Output file did not open!\n");
  }
  else {
	File_output_timings_smv(foutput, k, m, n, vecDistribution, h_norms, h_times, iter, convRate, checkSupport, seed, time_per_iteration, time_supp_set, algstr, alpha, supp_flag, p, matrixEnsemble);
  }
  fclose(foutput);


  cudaFree(d_support_counter);
  free(h_support_counter);

}




inline void results_timings_gen(float *d_vec, float *d_vec_input, int vecDistribution, float *residNorm_prev, float *h_norms, float *h_out, float *h_times, float *convergence_rate, int *total_iter, int *checkSupport, int iter, float timeIHT, float time_sum, float *time_per_iteration, float *time_supp_set, int *p_sum, float alpha, int supp_flag, const int k, const int m, const int n, const int matrixEnsemble, unsigned int seed, cudaEvent_t *p_startTest, cudaEvent_t *p_stopTest, char* algstr, dim3 numBlocks, dim3 threadsPerBlock)
{
  float convRate, root;
  int temp = min(iter, 16);
  if (temp <= 1) {
	convRate = 0.0f;
  }
  else {
	temp = temp-1;
  	root = 1/(float)temp;
  	temp=15-temp;
  	convRate = (residNorm_prev[15]/residNorm_prev[temp]);
  	convRate = pow(convRate, root);
  }

  convergence_rate[0]=convRate;


// debugging, this is unnecessary as we won't return the actual output.
// this should be deleletd along with the plhs[5] scripts above.

  cudaMemcpy(h_out, d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_out");



// record IHT times and iterations


  h_times[0]=timeIHT;

  h_times[1]=time_sum/(float)iter;

  total_iter[0] = iter;



// check support identification counting
// true_positive: correctly identified as in the support
// false_positive: incorrectly identified as in the support
// true_negative: correctly identified as not in support
// flase_negative: incorrectly identified as not in support
// From the last iteration, sum identifies total number of 
// nonzeros in the output vector.

  int * h_support_counter;
  h_support_counter = (int*)malloc(2*sizeof(int));
  h_support_counter[0]=0;
  h_support_counter[1]=0;

  int * d_support_counter;
  cudaMalloc((void**)&d_support_counter, 2*sizeof(int));
  SAFEcudaMalloc("d_support_counter");

  cudaMemcpy(d_support_counter, h_support_counter, 2*sizeof(int), cudaMemcpyHostToDevice);
  SAFEcuda("cudaMemcpy to d_support_counter");

  check_support<<< numBlocks, threadsPerBlock >>>(d_vec_input, d_vec, n, d_support_counter);
  SAFEcuda("check_support");

  cudaMemcpy(h_support_counter, d_support_counter, 2* sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_support_counter");

  int true_pos = h_support_counter[0];
  int false_pos = k - true_pos;
  int true_neg = h_support_counter[1];
  int false_neg = n - k - true_neg;


  checkSupport[0] = true_pos;
  checkSupport[1] = false_pos;
  checkSupport[2] = true_neg;
  checkSupport[3] = false_neg;




// Norms of input vector for computing relative norms
  float vec_input_2norm, vec_input_1norm, vec_input_supnorm;
  vec_input_1norm = cublasSasum(n, d_vec_input, 1);
  SAFEcublas("cublasSasum for vec_input_1norm in results.cu");
  vec_input_2norm = cublasSnrm2(n, d_vec_input, 1);
  SAFEcublas("cublasSnrm2 for vec_input_2norm in results.cu");
  
  int ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&vec_input_supnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  vec_input_supnorm = abs(vec_input_supnorm); 

// determine norms of the difference  
// between the input and output vectors  
  cublasSaxpy(n, -1.0f, d_vec, 1, d_vec_input, 1);
  SAFEcublas("cublasSaxpy in Results");

// Now d_vec_input is the difference between the 
// input and output vectors.
// We now compute the relative errors.
  float rel1norm, rel2norm, relsupnorm;

// relative ell-1 error
  rel1norm = cublasSasum(n, d_vec_input, 1);
  rel1norm = rel1norm/vec_input_1norm;
  h_norms[0] = rel1norm;
  SAFEcublas("h_norms[0]");


// relative ell-2 error

  rel2norm = cublasSnrm2(n, d_vec_input, 1);
  rel2norm = rel2norm/vec_input_2norm;
  h_norms[1] = rel2norm;
  SAFEcublas("h_norms[1]");

// relative ell-infinity error
  ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&relsupnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  relsupnorm = abs(relsupnorm);
  relsupnorm = relsupnorm/vec_input_supnorm;
  h_norms[2] = relsupnorm;


  cudaEvent_t startTest = *p_startTest;
  cudaEvent_t stopTest = *p_stopTest;

// record the final time of the entire test including
// memory allocation, problem generation, and error computations
  
  float timeTest;
  cudaThreadSynchronize();
  cudaEventRecord(stopTest,0);
  cudaEventSynchronize(stopTest);
  cudaEventElapsedTime(&timeTest, startTest, stopTest);
  cudaEventDestroy(startTest);
  cudaEventDestroy(stopTest);

  h_times[2]=timeTest;


// write all data to the big file

  FILE *foutput;

  time_t rawtime;
  time(&rawtime);
  struct tm * time_format;
  time_format = localtime(&rawtime);
  char fname[80]; 
  sprintf(fname,"gpu_timings_data_gen%d%02d%02d.txt",time_format->tm_year+1900,time_format->tm_mon+1,time_format->tm_mday);

  foutput = fopen(fname, "a+");
  if (foutput == NULL) {
	printf("WARNING: Output file did not open!\n");
  }
  else {
	File_output_timings_gen(foutput, k, m, n, matrixEnsemble, vecDistribution, h_norms, h_times, iter, convRate, checkSupport, seed, time_per_iteration, time_supp_set, algstr, alpha, supp_flag);
  }
  fclose(foutput);


  cudaFree(d_support_counter);
  free(h_support_counter);

}






inline void results_timings_HTP(float *d_vec, float *d_vec_input, int vecDistribution, float *residNorm_prev, float *h_norms, float *h_out, float *h_times, float *convergence_rate, int *total_iter, int *checkSupport, int iter, float timeIHT, float time_sum, float *time_per_iteration, float *time_supp_set, float *cg_per_iteration, float *time_for_cg, int *p_sum, float alpha, int supp_flag, const int k, const int m, const int n, unsigned int seed, cudaEvent_t *p_startTest, cudaEvent_t *p_stopTest, char* algstr, dim3 numBlocks, dim3 threadsPerBlock)
{
  float convRate, root;
  int temp = min(iter, 16);
  if (temp <= 1) {
	convRate = 0.0f;
  }
  else {
	temp = temp-1;
  	root = 1/(float)temp;
  	temp=15-temp;
  	convRate = (residNorm_prev[15]/residNorm_prev[temp]);
  	convRate = pow(convRate, root);
  }

  convergence_rate[0]=convRate;


// debugging, this is unnecessary as we won't return the actual output.
// this should be deleletd along with the plhs[5] scripts above.

  cudaMemcpy(h_out, d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_out");



// record IHT times and iterations


  h_times[0]=timeIHT;

  h_times[1]=time_sum/(float)iter;

  total_iter[0] = iter;



// check support identification counting
// true_positive: correctly identified as in the support
// false_positive: incorrectly identified as in the support
// true_negative: correctly identified as not in support
// flase_negative: incorrectly identified as not in support
// From the last iteration, sum identifies total number of 
// nonzeros in the output vector.

  int * h_support_counter;
  h_support_counter = (int*)malloc(2*sizeof(int));
  h_support_counter[0]=0;
  h_support_counter[1]=0;

  int * d_support_counter;
  cudaMalloc((void**)&d_support_counter, 2*sizeof(int));
  SAFEcudaMalloc("d_support_counter");

  cudaMemcpy(d_support_counter, h_support_counter, 2*sizeof(int), cudaMemcpyHostToDevice);
  SAFEcuda("cudaMemcpy to d_support_counter");

  check_support<<< numBlocks, threadsPerBlock >>>(d_vec_input, d_vec, n, d_support_counter);
  SAFEcuda("check_support");

  cudaMemcpy(h_support_counter, d_support_counter, 2* sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_support_counter");

  int true_pos = h_support_counter[0];
  int false_pos = k - true_pos;
  int true_neg = h_support_counter[1];
  int false_neg = n - k - true_neg;


  checkSupport[0] = true_pos;
  checkSupport[1] = false_pos;
  checkSupport[2] = true_neg;
  checkSupport[3] = false_neg;



// Norms of input vector for computing relative norms
  float vec_input_2norm, vec_input_1norm, vec_input_supnorm;
  vec_input_1norm = cublasSasum(n, d_vec_input, 1);
  SAFEcublas("cublasSasum for vec_input_1norm in results.cu");
  vec_input_2norm = cublasSnrm2(n, d_vec_input, 1);
  SAFEcublas("cublasSnrm2 for vec_input_2norm in results.cu");
  
  int ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&vec_input_supnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  vec_input_supnorm = abs(vec_input_supnorm); 

// determine norms of the difference  
// between the input and output vectors  
  cublasSaxpy(n, -1.0f, d_vec, 1, d_vec_input, 1);
  SAFEcublas("cublasSaxpy in Results");

// Now d_vec_input is the difference between the 
// input and output vectors.
// We now compute the relative errors.
  float rel1norm, rel2norm, relsupnorm;

// relative ell-1 error
  rel1norm = cublasSasum(n, d_vec_input, 1);
  rel1norm = rel1norm/vec_input_1norm;
  h_norms[0] = rel1norm;
  SAFEcublas("h_norms[0]");


// relative ell-2 error

  rel2norm = cublasSnrm2(n, d_vec_input, 1);
  rel2norm = rel2norm/vec_input_2norm;
  h_norms[1] = rel2norm;
  SAFEcublas("h_norms[1]");

// relative ell-infinity error
  ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&relsupnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  relsupnorm = abs(relsupnorm);
  relsupnorm = relsupnorm/vec_input_supnorm;
  h_norms[2] = relsupnorm;


  cudaEvent_t startTest = *p_startTest;
  cudaEvent_t stopTest = *p_stopTest;

// record the final time of the entire test including
// memory allocation, problem generation, and error computations
  
  float timeTest;
  cudaThreadSynchronize();
  cudaEventRecord(stopTest,0);
  cudaEventSynchronize(stopTest);
  cudaEventElapsedTime(&timeTest, startTest, stopTest);
  cudaEventDestroy(startTest);
  cudaEventDestroy(stopTest);

  h_times[2]=timeTest;


// write all data to the big file

  FILE *foutput;

  time_t rawtime;
  time(&rawtime);
  struct tm * time_format;
  time_format = localtime(&rawtime);
  char fname[80]; 
  sprintf(fname,"gpu_timings_HTP_data_dct%d%02d%02d.txt",time_format->tm_year+1900,time_format->tm_mon+1,time_format->tm_mday);

  foutput = fopen(fname, "a+");
  if (foutput == NULL) {
	printf("WARNING: Output file did not open!\n");
  }
  else {
	File_output_timings_HTP(foutput, k, m, n, vecDistribution, h_norms, h_times, iter, convRate, checkSupport, seed, time_per_iteration, time_supp_set, cg_per_iteration, time_for_cg, algstr, alpha, supp_flag);
  }
  fclose(foutput);


  cudaFree(d_support_counter);
  free(h_support_counter);

}








inline void results_timings_HTP_smv(float *d_vec, float *d_vec_input, int vecDistribution, float *residNorm_prev, float *h_norms, float *h_out, float *h_times, float *convergence_rate, int *total_iter, int *checkSupport, int iter, float timeIHT, float time_sum, float *time_per_iteration, float *time_supp_set, float *cg_per_iteration, float *time_for_cg, int *p_sum, float alpha, int supp_flag, const int k, const int m, const int n, const int p, const int matrixEnsemble, unsigned int seed, cudaEvent_t *p_startTest, cudaEvent_t *p_stopTest, char* algstr, dim3 numBlocks, dim3 threadsPerBlock)
{
  float convRate, root;
  int temp = min(iter, 16);
  if (temp <= 1) {
	convRate = 0.0f;
  }
  else {
	temp = temp-1;
  	root = 1/(float)temp;
  	temp=15-temp;
  	convRate = (residNorm_prev[15]/residNorm_prev[temp]);
  	convRate = pow(convRate, root);
  }

  convergence_rate[0]=convRate;


// debugging, this is unnecessary as we won't return the actual output.
// this should be deleletd along with the plhs[5] scripts above.

  cudaMemcpy(h_out, d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_out");



// record IHT times and iterations


  h_times[0]=timeIHT;

  h_times[1]=time_sum/(float)iter;

  total_iter[0] = iter;



// check support identification counting
// true_positive: correctly identified as in the support
// false_positive: incorrectly identified as in the support
// true_negative: correctly identified as not in support
// flase_negative: incorrectly identified as not in support
// From the last iteration, sum identifies total number of 
// nonzeros in the output vector.

  int * h_support_counter;
  h_support_counter = (int*)malloc(2*sizeof(int));
  h_support_counter[0]=0;
  h_support_counter[1]=0;

  int * d_support_counter;
  cudaMalloc((void**)&d_support_counter, 2*sizeof(int));
  SAFEcudaMalloc("d_support_counter");

  cudaMemcpy(d_support_counter, h_support_counter, 2*sizeof(int), cudaMemcpyHostToDevice);
  SAFEcuda("cudaMemcpy to d_support_counter");

  check_support<<< numBlocks, threadsPerBlock >>>(d_vec_input, d_vec, n, d_support_counter);
  SAFEcuda("check_support");

  cudaMemcpy(h_support_counter, d_support_counter, 2* sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_support_counter");

  int true_pos = h_support_counter[0];
  int false_pos = k - true_pos;
  int true_neg = h_support_counter[1];
  int false_neg = n - k - true_neg;


  checkSupport[0] = true_pos;
  checkSupport[1] = false_pos;
  checkSupport[2] = true_neg;
  checkSupport[3] = false_neg;




// Norms of input vector for computing relative norms
  float vec_input_2norm, vec_input_1norm, vec_input_supnorm;
  vec_input_1norm = cublasSasum(n, d_vec_input, 1);
  SAFEcublas("cublasSasum for vec_input_1norm in results.cu");
  vec_input_2norm = cublasSnrm2(n, d_vec_input, 1);
  SAFEcublas("cublasSnrm2 for vec_input_2norm in results.cu");
  
  int ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&vec_input_supnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  vec_input_supnorm = abs(vec_input_supnorm); 

// determine norms of the difference  
// between the input and output vectors  
  cublasSaxpy(n, -1.0f, d_vec, 1, d_vec_input, 1);
  SAFEcublas("cublasSaxpy in Results");

// Now d_vec_input is the difference between the 
// input and output vectors.
// We now compute the relative errors.
  float rel1norm, rel2norm, relsupnorm;

// relative ell-1 error
  rel1norm = cublasSasum(n, d_vec_input, 1);
  rel1norm = rel1norm/vec_input_1norm;
  h_norms[0] = rel1norm;
  SAFEcublas("h_norms[0]");


// relative ell-2 error

  rel2norm = cublasSnrm2(n, d_vec_input, 1);
  rel2norm = rel2norm/vec_input_2norm;
  h_norms[1] = rel2norm;
  SAFEcublas("h_norms[1]");

// relative ell-infinity error
  ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&relsupnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  relsupnorm = abs(relsupnorm);
  relsupnorm = relsupnorm/vec_input_supnorm;
  h_norms[2] = relsupnorm;


  cudaEvent_t startTest = *p_startTest;
  cudaEvent_t stopTest = *p_stopTest;

// record the final time of the entire test including
// memory allocation, problem generation, and error computations
  
  float timeTest;
  cudaThreadSynchronize();
  cudaEventRecord(stopTest,0);
  cudaEventSynchronize(stopTest);
  cudaEventElapsedTime(&timeTest, startTest, stopTest);
  cudaEventDestroy(startTest);
  cudaEventDestroy(stopTest);

  h_times[2]=timeTest;


// write all data to the big file

  FILE *foutput;

  time_t rawtime;
  time(&rawtime);
  struct tm * time_format;
  time_format = localtime(&rawtime);
  char fname[80]; 
  sprintf(fname,"gpu_timings_HTP_data_smv%d%02d%02d.txt",time_format->tm_year+1900,time_format->tm_mon+1,time_format->tm_mday);

  foutput = fopen(fname, "a+");
  if (foutput == NULL) {
	printf("WARNING: Output file did not open!\n");
  }
  else {
	File_output_timings_HTP_smv(foutput, k, m, n, vecDistribution, h_norms, h_times, iter, convRate, checkSupport, seed, time_per_iteration, time_supp_set, cg_per_iteration, time_for_cg, algstr, alpha, supp_flag, p, matrixEnsemble);
  }
  fclose(foutput);


  cudaFree(d_support_counter);
  free(h_support_counter);

}





inline void results_timings_HTP_gen(float *d_vec, float *d_vec_input, int vecDistribution, float *residNorm_prev, float *h_norms, float *h_out, float *h_times, float *convergence_rate, int *total_iter, int *checkSupport, int iter, float timeIHT, float time_sum, float *time_per_iteration, float *time_supp_set, float *cg_per_iteration, float *time_for_cg, int *p_sum, float alpha, int supp_flag, const int k, const int m, const int n, const int matrixEnsemble, unsigned int seed, cudaEvent_t *p_startTest, cudaEvent_t *p_stopTest, char* algstr, dim3 numBlocks, dim3 threadsPerBlock)
{
  float convRate, root;
  int temp = min(iter, 16);
  if (temp <= 1) {
	convRate = 0.0f;
  }
  else {
	temp = temp-1;
  	root = 1/(float)temp;
  	temp=15-temp;
  	convRate = (residNorm_prev[15]/residNorm_prev[temp]);
  	convRate = pow(convRate, root);
  }

  convergence_rate[0]=convRate;


// debugging, this is unnecessary as we won't return the actual output.
// this should be deleletd along with the plhs[5] scripts above.

  cudaMemcpy(h_out, d_vec, n * sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_out");



// record IHT times and iterations


  h_times[0]=timeIHT;

  h_times[1]=time_sum/(float)iter;

  total_iter[0] = iter;



// check support identification counting
// true_positive: correctly identified as in the support
// false_positive: incorrectly identified as in the support
// true_negative: correctly identified as not in support
// flase_negative: incorrectly identified as not in support
// From the last iteration, sum identifies total number of 
// nonzeros in the output vector.

  int * h_support_counter;
  h_support_counter = (int*)malloc(2*sizeof(int));
  h_support_counter[0]=0;
  h_support_counter[1]=0;

  int * d_support_counter;
  cudaMalloc((void**)&d_support_counter, 2*sizeof(int));
  SAFEcudaMalloc("d_support_counter");

  cudaMemcpy(d_support_counter, h_support_counter, 2*sizeof(int), cudaMemcpyHostToDevice);
  SAFEcuda("cudaMemcpy to d_support_counter");

  check_support<<< numBlocks, threadsPerBlock >>>(d_vec_input, d_vec, n, d_support_counter);
  SAFEcuda("check_support");

  cudaMemcpy(h_support_counter, d_support_counter, 2* sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_support_counter");

  int true_pos = h_support_counter[0];
  int false_pos = k - true_pos;
  int true_neg = h_support_counter[1];
  int false_neg = n - k - true_neg;


  checkSupport[0] = true_pos;
  checkSupport[1] = false_pos;
  checkSupport[2] = true_neg;
  checkSupport[3] = false_neg;




// Norms of input vector for computing relative norms
  float vec_input_2norm, vec_input_1norm, vec_input_supnorm;
  vec_input_1norm = cublasSasum(n, d_vec_input, 1);
  SAFEcublas("cublasSasum for vec_input_1norm in results.cu");
  vec_input_2norm = cublasSnrm2(n, d_vec_input, 1);
  SAFEcublas("cublasSnrm2 for vec_input_2norm in results.cu");
  
  int ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&vec_input_supnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  vec_input_supnorm = abs(vec_input_supnorm); 

// determine norms of the difference  
// between the input and output vectors  
  cublasSaxpy(n, -1.0f, d_vec, 1, d_vec_input, 1);
  SAFEcublas("cublasSaxpy in Results");

// Now d_vec_input is the difference between the 
// input and output vectors.
// We now compute the relative errors.
  float rel1norm, rel2norm, relsupnorm;

// relative ell-1 error
  rel1norm = cublasSasum(n, d_vec_input, 1);
  rel1norm = rel1norm/vec_input_1norm;
  h_norms[0] = rel1norm;
  SAFEcublas("h_norms[0]");


// relative ell-2 error

  rel2norm = cublasSnrm2(n, d_vec_input, 1);
  rel2norm = rel2norm/vec_input_2norm;
  h_norms[1] = rel2norm;
  SAFEcublas("h_norms[1]");

// relative ell-infinity error
  ind_abs_vec_max = cublasIsamax(n, d_vec_input, 1)-1;
  SAFEcublas("cublasIsamax in Results");
  cudaMemcpy(&relsupnorm, d_vec_input+ind_abs_vec_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to h_norms+2");
  relsupnorm = abs(relsupnorm);
  relsupnorm = relsupnorm/vec_input_supnorm;
  h_norms[2] = relsupnorm;


  cudaEvent_t startTest = *p_startTest;
  cudaEvent_t stopTest = *p_stopTest;

// record the final time of the entire test including
// memory allocation, problem generation, and error computations
  
  float timeTest;
  cudaThreadSynchronize();
  cudaEventRecord(stopTest,0);
  cudaEventSynchronize(stopTest);
  cudaEventElapsedTime(&timeTest, startTest, stopTest);
  cudaEventDestroy(startTest);
  cudaEventDestroy(stopTest);

  h_times[2]=timeTest;


// write all data to the big file

  FILE *foutput;

  time_t rawtime;
  time(&rawtime);
  struct tm * time_format;
  time_format = localtime(&rawtime);
  char fname[80]; 
  sprintf(fname,"gpu_timings_HTP_data_gen%d%02d%02d.txt",time_format->tm_year+1900,time_format->tm_mon+1,time_format->tm_mday);

  foutput = fopen(fname, "a+");
  if (foutput == NULL) {
	printf("WARNING: Output file did not open!\n");
  }
  else {
	File_output_timings_HTP_gen(foutput, k, m, n, matrixEnsemble, vecDistribution, h_norms, h_times, iter, convRate, checkSupport, seed, time_per_iteration, time_supp_set, cg_per_iteration, time_for_cg, algstr, alpha, supp_flag);
  }
  fclose(foutput);


  cudaFree(d_support_counter);
  free(h_support_counter);

}







