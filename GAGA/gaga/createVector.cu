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

__global__ void lookForValue(float *d_vec, float *d_value, int n){
        unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
        if( (xIndex < n) ){
                if( abs(d_vec[xIndex]) > 1e-5 ){
			d_value[0] = d_vec[xIndex];
                }
        }
}

__global__ void createBandedStructure(float *d_vec, int *d_bin, int k_bin, int * band_size, float val, int n){
        unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
        if( (xIndex < n) & (d_bin[xIndex] <= k_bin) ){
                if(atomicSub(band_size, 1) > 0){
                        d_vec[xIndex] = val;
                }
        }
}


/*
**********************************
** MAIN FUNCTION (SMV): createVector_smv  **
**********************************
*/

void  createVector_smv(int * k_pointer, const int m, const int n, int vecDistribution, float band_percentage, float *d_vec_input, float *values_rand, float *support_rand)
{
/*
  cudaEvent_t startHost, stopHost;
  float hostTime;
  cudaEventCreate(&startHost);
  cudaEventCreate(&stopHost);
  cudaEventRecord(startHost,0);
*/

  int threads_perblock = min(n, 512);
  dim3 threadsPerBlock(threads_perblock);
  int num_blocks = (int)ceil((float)n/(float)threads_perblock);
  dim3 numBlocks(num_blocks);


  // determine the random support with fast binning

  
  float *bin_vec, *d_bin_counter;

  cudaMalloc((void**)&bin_vec, n * sizeof(int));
  SAFEcudaMalloc("bin_vector in createVector");

  cudaMalloc((void**)&d_bin_counter, n * sizeof(int)); 
  SAFEcudaMalloc("d_bin_counter in createVector");

  int * h_bin_counter = (int*)malloc( sizeof(int) * n );
  SAFEmalloc_int(h_bin_counter, "h_bin_counter in createVector");

  zero_vector_int<<< numBlocks, threadsPerBlock >>>((int*)bin_vec, n);
  SAFEcuda("zero_vector_int in createVector (d_bin)");
  //cudaThreadSynchronize();

  zero_vector_int<<< numBlocks, threadsPerBlock >>>((int*)d_bin_counter, n);
  SAFEcuda("zero_vector_int in createVector (d_bin_counter)");
  //cudaThreadSynchronize();

  zero_vector_float<<< numBlocks, threadsPerBlock >>>((float*)d_vec_input, n);
  SAFEcuda("zero_vector_float in createVector");
  //cudaThreadSynchronize();

  int ind_abs_max = cublasIsamax(n, support_rand, 1) - 1;
  SAFEcublas("cublasIsamax in createVector");

  float max_value;
  cudaMemcpy(&max_value, support_rand + ind_abs_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy max_value in createVector");

  max_value = abs(max_value);
  float slope = ((n-1)/max_value);

  LinearBinning <<< numBlocks, threadsPerBlock >>>((float*)support_rand, (int*)bin_vec, (int*)d_bin_counter, n, n, n, slope, max_value);
  SAFEcuda("LinearBinning in createVector");


  cudaMemcpy(h_bin_counter, d_bin_counter, n * sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy h_bin_counter in createVector");

  int kk=*k_pointer;

  int k_bin = 0;
  int sum=0;

  while ( (sum<kk) & (k_bin<n) ) {
	sum = sum + h_bin_counter[k_bin];
	k_bin++;
	}
  k_bin = k_bin-1;

 // printf("Inside createVector, kbin is: %d \n", k_bin);

  free(h_bin_counter);
  cudaFree(d_bin_counter);

  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)values_rand, (float*)d_vec_input, (int*)bin_vec, k_bin, n);
  SAFEcuda("threshold_one in createVector");

  *k_pointer = sum;

  //cudaThreadSynchronize();

  if (0 < band_percentage & band_percentage <= 1){
        //printf("band_percentage = %g\n", band_percentage);
        int k = *k_pointer;
	int band_size = min((int) (( (float) k )*band_percentage), k);
        //printf("band_size = %d\n", band_size);
	int * d_band_size;
        cudaMalloc((void**)&d_band_size, 1 * sizeof(int));
	cudaMemcpy(d_band_size, &band_size, sizeof(int), cudaMemcpyHostToDevice);

	float * d_value;
	float zero = 0.0;
        cudaMalloc((void**)&d_value, 1 * sizeof(float));
	cudaMemcpy(d_value, &zero, sizeof(float), cudaMemcpyHostToDevice);
	lookForValue<<<numBlocks, threadsPerBlock>>>((float*)d_vec_input, (float *)d_value, n);

	float h_value = 0.0;
	cudaMemcpy(&h_value, d_value, sizeof(float), cudaMemcpyDeviceToHost);
	//printf("h_value = %g\n", h_value);

	//createBandedStructure(float *d_vec, int *d_bin, int k_bin, int * band_size, float val, int n){
	//printf("About to createBandedStructure\n");
	createBandedStructure<<< numBlocks, threadsPerBlock >>>((float*)d_vec_input, (int*)bin_vec, k_bin, d_band_size, h_value, n);

	// Check banded Structure
	/*
	float value = 0;
	int bin = 0;
	for (int i = 0; i < n; i++){
		cudaMemcpy(&bin, bin_vec+i, sizeof(int), cudaMemcpyDeviceToHost);
		if (bin <= k_bin){
			cudaMemcpy(&value, d_vec_input+i, sizeof(int), cudaMemcpyDeviceToHost);
			printf("%g -> ", value);
		}
	}
	printf("\n", value);
	*/

  }
 
/*
// ******  checking that there is something in the vector *******

  float * h_vec_input = (float*)malloc( sizeof(float) * n );
  cudaMemcpy(h_vec_input, d_vec_input, n * sizeof(float), cudaMemcpyDeviceToHost);
  printf(" the first 25 entries of the new input vector: \n");
  for (j = 0; j<25; j++) printf(" %f ", h_vec_input[j]);
  printf("\n");

  free(h_vec_input);
*/

  cudaFree(bin_vec);

/*
  cudaThreadSynchronize();
  cudaEventRecord(stopHost,0);
  cudaEventSynchronize(stopHost);
  cudaEventElapsedTime(&hostTime, startHost, stopHost);
  cudaEventDestroy(startHost);
  cudaEventDestroy(stopHost);

  printf("The function createVector takes %f ms.\n", hostTime);
*/

  return;
}



/*
**********************************
** MAIN FUNCTION: createVector  **
**********************************
*/

void  createVector(int * k_pointer, const int m, const int n, int vecDistribution, float *d_vec_input, float *values_rand, float *support_rand)
{
/*
  cudaEvent_t startHost, stopHost;
  float hostTime;
  cudaEventCreate(&startHost);
  cudaEventCreate(&stopHost);
  cudaEventRecord(startHost,0);
*/

  int threads_perblock = min(n, 512);
  dim3 threadsPerBlock(threads_perblock);
  int num_blocks = (int)ceil((float)n/(float)threads_perblock);
  dim3 numBlocks(num_blocks);


  // determine the random support with fast binning

  
  float *bin_vec, *d_bin_counter;

  cudaMalloc((void**)&bin_vec, n * sizeof(int));
  SAFEcudaMalloc("bin_vector in createVector");

  cudaMalloc((void**)&d_bin_counter, n * sizeof(int)); 
  SAFEcudaMalloc("d_bin_counter in createVector");

  int * h_bin_counter = (int*)malloc( sizeof(int) * n );
  SAFEmalloc_int(h_bin_counter, "h_bin_counter in createVector");

  zero_vector_int<<< numBlocks, threadsPerBlock >>>((int*)bin_vec, n);
  SAFEcuda("zero_vector_int in createVector (d_bin)");
  //cudaThreadSynchronize();

  zero_vector_int<<< numBlocks, threadsPerBlock >>>((int*)d_bin_counter, n);
  SAFEcuda("zero_vector_int in createVector (d_bin_counter)");
  //cudaThreadSynchronize();

  zero_vector_float<<< numBlocks, threadsPerBlock >>>((float*)d_vec_input, n);
  SAFEcuda("zero_vector_float in createVector");
  //cudaThreadSynchronize();

  int ind_abs_max = cublasIsamax(n, support_rand, 1) - 1;
  SAFEcublas("cublasIsamax in createVector");

  float max_value;
  cudaMemcpy(&max_value, support_rand + ind_abs_max, sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy max_value in createVector");

  max_value = abs(max_value);
  float slope = ((n-1)/max_value);

  LinearBinning <<< numBlocks, threadsPerBlock >>>((float*)support_rand, (int*)bin_vec, (int*)d_bin_counter, n, n, n, slope, max_value);
  SAFEcuda("LinearBinning in createVector");


  cudaMemcpy(h_bin_counter, d_bin_counter, n * sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy h_bin_counter in createVector");

  int kk=*k_pointer;

  int k_bin = 0;
  int sum=0;

  while ( (sum<kk) & (k_bin<n) ) {
	sum = sum + h_bin_counter[k_bin];
	k_bin++;
	}
  k_bin = k_bin-1;

 // printf("Inside createVector, kbin is: %d \n", k_bin);

  free(h_bin_counter);
  cudaFree(d_bin_counter);

  threshold_one<<< numBlocks, threadsPerBlock >>>((float*)values_rand, (float*)d_vec_input, (int*)bin_vec, k_bin, n);
  SAFEcuda("threshold_one in createVector");

  *k_pointer = sum;

  //cudaThreadSynchronize();

/*
// ******  checking that there is something in the vector *******

  float * h_vec_input = (float*)malloc( sizeof(float) * n );
  cudaMemcpy(h_vec_input, d_vec_input, n * sizeof(float), cudaMemcpyDeviceToHost);
  printf(" the first 25 entries of the new input vector: \n");
  for (j = 0; j<25; j++) printf(" %f ", h_vec_input[j]);
  printf("\n");

  free(h_vec_input);
*/

  cudaFree(bin_vec);

/*
  cudaThreadSynchronize();
  cudaEventRecord(stopHost,0);
  cudaEventSynchronize(stopHost);
  cudaEventElapsedTime(&hostTime, startHost, stopHost);
  cudaEventDestroy(startHost);
  cudaEventDestroy(stopHost);

  printf("The function createVector takes %f ms.\n", hostTime);
*/

  return;
}






