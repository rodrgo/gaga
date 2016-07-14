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
***************************************
** MAIN FUNCTION: createMeasurements **
***************************************
*/

void createMeasurements_gen(const int m, const int n, float *d_vec_input, float *d_y, float *d_A, int ensemble, unsigned int *p_seed)
{

// THE INPUTS ARE AS FOLLOWS:
// m and n are the size of the matrix being created
// d_vec_input is the vector that the matrix will multiply
// d_y is the result of the matrix times d_vec_input
// d_A defines the m by n matrix
// ensemble determines how the values in d_A should be drawn.
// ensemble 1 is gaussian and 2 is random \pm 1.


  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
//  unsigned int global_mem = dp.totalGlobalMem;
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

/*
  cudaEvent_t startHost, stopHost;
  float hostTime;
  cudaEventCreate(&startHost);
  cudaEventCreate(&stopHost);
  cudaEventRecord(startHost,0);
*/

  int mn = m * n;

  int threads_perblockm = min(m, max_threads_per_block);
  dim3 threadsPerBlockm(threads_perblockm);
  int num_blocksm = (int)ceil((float)m/(float)threads_perblockm);
  dim3 numBlocksm(num_blocksm);

  int threads_perblockmn = min(mn, max_threads_per_block);
  dim3 threadsPerBlockmn(threads_perblockmn);
  int num_blocksmn = (int)ceil((float)mn/(float)threads_perblockmn);
  dim3 numBlocksmn(num_blocksmn);

  curandStatus_t curandCheck;

  curandGenerator_t gen2;
  curandCheck = curandCreateGenerator(&gen2,CURAND_RNG_PSEUDO_DEFAULT);
  SAFEcurand(curandCheck, "curandCreateGenerator in createMeasurements_gen");
  // note, CURAND_RNG_PSEUDO_DEFAULT selects the random number generator type
  curandCheck = curandSetPseudoRandomGeneratorSeed(gen2,*p_seed);
  SAFEcurand(curandCheck, "curandSet...Seed in createMeasurements_gen");


  if( ensemble == 1 ){
    // set n*m values to be gaussian
   curandCheck = curandGenerateNormal(gen2,d_A,mn,0,1);
   SAFEcurand(curandCheck, "curandGenerateNormal in createMeasurements_gen");
  }
  else if( ensemble == 2 ){
    // set n*m values to be uniform \pm 1
   curandCheck = curandGenerateUniform(gen2,d_A,mn);
   SAFEcurand(curandCheck, "curandGenerateUniform in createMeasurements_gen");
   sign_pattern<<< numBlocksmn, threadsPerBlockmn >>>(d_A,mn);
   SAFEcuda("sign_pattern in createMeasurements_gen");
}



  // Scale the entries by m to create approximate unit l2-norm columns

  float scale = sqrt((float)m);
  scale = 1/scale;
  cublasSscal(mn,scale,d_A,1);
  SAFEcublas("cublasSscal in createMeasurements_gen");


// creat the measurements
  zero_vector_float <<< numBlocksm, threadsPerBlockm >>>((float*)d_y, m);
  SAFEcuda("zero_vector_float in createMeasurements_gen");

  A_gen(d_y, d_vec_input, d_A, m, n, numBlocksm, threadsPerBlockm);
  SAFEcuda("A_gen in createMeasurements_gen");

  curandCheck = curandDestroyGenerator(gen2);
  SAFEcurand(curandCheck, "curandDestroyGenerator in createMeasurements_gen");

/*
 
  cudaThreadSynchronize();
  cudaEventRecord(stopMatvec,0);
  cudaEventSynchronize(stopMatvec);
  cudaEventElapsedTime(&MatvecTime, startMatvec, stopMatvec);
  cudaEventDestroy(startMatvec);
  cudaEventDestroy(stopMatvec);

  cudaThreadSynchronize();
  cudaEventRecord(stopHost,0);
  cudaEventSynchronize(stopHost);
  cudaEventElapsedTime(&hostTime, startHost, stopHost);
  cudaEventDestroy(startHost);
  cudaEventDestroy(stopHost);

  printf("The function createMeasurements_en takes %f ms of which %f is the Mat-vc .\n", hostTime, MatvecTime);

*/

}



