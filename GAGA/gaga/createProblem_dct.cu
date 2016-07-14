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
**************************************
** MAIN FUNCTION: createProblem_dct **
**************************************
*/

void  createProblem_dct(int *k_pointer, const int m, const int n, const int vecDistribution, float *d_vec_input, float *d_y, int *d_rows, unsigned int *p_seed)
{
// k_pointer provides k and lets it be updated
// m and n are the size of the matrix
// vecDistribution: 0 = uniform, 1=random{-1,1}, 2=Gaussian
// d_vec_input is the address for the vector
// d_y is the vector of measurements
// d_rows describes the rows of the DCT to retain
// p_seed allows us to store the seed and therefore recreate the problem

/*
  cudaEvent_t startHost, stopHost;
  float hostTime;
  cudaEventCreate(&startHost);
  cudaEventCreate(&stopHost);
  cudaEventRecord(startHost,0);
*/

  // To create this problem we need
  //	n random numbers to determine the random values of the input vector
  //	n random numbers to determine the support of the input vector
  //	n random numbers to determine the random rows of the dct

  const int L = 3*n;


// ********* Create L random numbers needed to generate the problem ******

  // Allocate device variable for random numbers
  float * random;
  cudaMalloc((void**)&random, L * sizeof(float));
  SAFEcudaMalloc("random in createVector");

  curandStatus_t curandCheck;

  curandGenerator_t gen;
  curandCheck = curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  SAFEcurand(curandCheck, "curandCreateGenerator in createProblem_dct");
  // note, CURAND_RNG_PSEUDO_DEFAULT selects the random number generator type
  curandCheck = curandSetPseudoRandomGeneratorSeed(gen,*p_seed);
  SAFEcurand(curandCheck, "curandSet...Seed in createProblem_dct");
  curandCheck = curandGenerateUniform(gen,random,L);
  SAFEcurand(curandCheck, "curandGenerateUniform in createProblem_dct");
  SAFEcuda("createRandomNumbers uniform in createProblem_dct"); 

  // if vecDistribution == 1 then need uniform sign pattern for the 
  // nonzeros in the vector measured.
  if(vecDistribution == 1){

        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp,0);
        unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

        int threads_perblock = min(n, max_threads_per_block);
        dim3 threadsPerBlock(threads_perblock);
        int num_blocks = (int)ceil((float)n/(float)threads_perblock);
        dim3 numBlocks(num_blocks);

        sign_pattern<<< numBlocks, threadsPerBlock >>>(random,n);
        SAFEcuda("sign_pattern in createProblem_dct");
  }

  // if vecDistribution == 2 then need gaussian random numbers for the
  // nonzeros in the vector measured.
  if(vecDistribution == 2){

        curandCheck = curandGenerateNormal(gen,random,n,0,1);
  	SAFEcurand(curandCheck, "curandGenerateNormal in createProblem_dct");
  }

  curandCheck = curandDestroyGenerator(gen);
  SAFEcurand(curandCheck, "curandDestroyGenerator in createProblem_dct");

/*
  The random numbers are used to create various things:
  values_rand = random;
  support_rand = random + n;
  rows_rand = random + 2*n;
*/

//  createVector(k_pointer, m, n, vecDistribution, d_vec_input, values_rand, support_rand);
  createVector(k_pointer, m, n, vecDistribution, d_vec_input, random, random+n);
  SAFEcuda("createVector in createProblem_dct"); 

//  createMeasurements_dct(m, n, d_vec_input, d_y, d_rows, rows_rand);
  createMeasurements_dct(m, n, d_vec_input, d_y, d_rows, random+2*n);
  SAFEcuda("createMeasurements_dct in createProblem_dct"); 


  cudaFree(random);
/*
  cudaFree(values_rand);
  cudaFree(support_rand);
  cudaFree(rows_rand);
*/

/*
  cudaThreadSynchronize();
  cudaEventRecord(stopHost,0);
  cudaEventSynchronize(stopHost);
  cudaEventElapsedTime(&hostTime, startHost, stopHost);
  cudaEventDestroy(startHost);
  cudaEventDestroy(stopHost);

  printf("The function createProblem takes %f ms.\n", hostTime);
*/

  return;
}



void  createProblem_dct_noise(int *k_pointer, const int m, const int n, const int vecDistribution, float *d_vec_input, float *d_y, int *d_rows, unsigned int *p_seed, float noise_level)
{
// k_pointer provides k and lets it be updated
// m and n are the size of the matrix
// vecDistribution: 0 = uniform, 1=random{-1,1}, 2=Gaussian
// d_vec_input is the address for the vector
// d_y is the vector of measurements
// d_rows describes the rows of the DCT to retain
// p_seed allows us to store the seed and therefore recreate the problem

/*
  cudaEvent_t startHost, stopHost;
  float hostTime;
  cudaEventCreate(&startHost);
  cudaEventCreate(&stopHost);
  cudaEventRecord(startHost,0);
*/

  // To create this problem we need
  //	n random numbers to determine the random values of the input vector
  //	n random numbers to determine the support of the input vector
  //	n random numbers to determine the random rows of the dct
  //    m random numbers for noise to measurements
  //    Since curandGenerateNormal requires an even length, we add m mod 2.

  const int L = 3*n+ (m + m%2);



// ********* Create L random numbers needed to generate the problem ******

  // Allocate device variable for random numbers
  float * random;
  cudaMalloc((void**)&random, L * sizeof(float));
  SAFEcudaMalloc("random in createVector");

  curandStatus_t curandCheck;

  curandGenerator_t gen;
  curandCheck = curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  SAFEcurand(curandCheck, "curandCreateGenerator in createProblem_dct_noise");
  // note, CURAND_RNG_PSEUDO_DEFAULT selects the random number generator type
  curandCheck = curandSetPseudoRandomGeneratorSeed(gen,*p_seed);
  SAFEcurand(curandCheck, "curandSet...Seed in createProblem_dct_noise");
  curandCheck = curandGenerateUniform(gen,random,L-m);
  SAFEcurand(curandCheck, "curandGenerateUniform in createProblem_dct_noise");
  curandCheck = curandGenerateNormal(gen,random+L-(m+m%2),m+m%2,0.0f,1.0f);
  SAFEcurand(curandCheck, "curandGenerateNormal (1st) in createProblem_dct_noise");
  SAFEcuda("createRandomNumbers in createProblem_dct_noise"); 

  // if vecDistribution == 1 then need uniform sign pattern for the 
  // nonzeros in the vector measured.
  if(vecDistribution == 1){

        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp,0);
        unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

        int threads_perblock = min(n, max_threads_per_block);
        dim3 threadsPerBlock(threads_perblock);
        int num_blocks = (int)ceil((float)n/(float)threads_perblock);
        dim3 numBlocks(num_blocks);

        sign_pattern<<< numBlocks, threadsPerBlock >>>(random,n);
        SAFEcuda("createRandomNumbers sign in createProblem_dct_noise");
  }


  // if vecDistribution == 2 then need gaussian random numbers for the
  // nonzeros in the vector measured.
  if(vecDistribution == 2){

        curandCheck = curandGenerateNormal(gen,random,n,0,1);
  	SAFEcurand(curandCheck, "curandGenerateNormal (2nd) in createProblem_dct_noise");
  }

  curandCheck = curandDestroyGenerator(gen);
  SAFEcurand(curandCheck, "curandDestroyGenerator in createProblem_dct_noise");



/*
  The random numbers are used to create various things:
  values_rand = random;
  support_rand = random + n;
  rows_rand = random + 2*n;
  noise = random + L - (m+m%2);
*/

  createVector(k_pointer, m, n, vecDistribution, d_vec_input, random, random + n);
  SAFEcuda("createVector in createProblem_dct_noise"); 

  createMeasurements_dct(m, n, d_vec_input, d_y, d_rows, random + 2*n);
  SAFEcuda("createMeasurements_dect in createProblem_dct_noise"); 


//  Here we want to add a noise vector that is scaled so that it's norm is noise_level*norm2(y).
  float noise_norm, y_norm, noise_scale;
  noise_norm = cublasSnrm2(m, random+L-(m+m%2), 1);
  SAFEcublas("cublasSnrm2 computing noise_norm in createProblem_dct_noise");

  y_norm = cublasSnrm2(m, d_y, 1);
  SAFEcublas("cublasSnrm2 computing y_norm in createProblem_dct_noise");

  noise_scale = noise_level*(y_norm/noise_norm);

  cublasSaxpy(m, noise_scale, random+L-(m+m%2), 1, d_y, 1);
  SAFEcublas("cublasSaxpy adding noise to measurements in createProblem_dct_noise");

/*
  createVector(k_pointer, m, n, vecDistribution, d_vec_input, values_rand, support_rand);
  SAFEcuda("createVector in createProblem_dct"); 

  createMeasurements_dct(m, n, d_vec_input, d_y, d_rows, rows_rand);
  SAFEcuda("createMeasurements_dect in createProblem_dct"); 

  cublasSaxpy(m, 1.0f, noise, 1, d_y, 1);
  SAFEcublas("cublasSaxpy adding noise to measurements");
*/


  cudaFree(random);
/*
  cudaFree(values_rand);
  cudaFree(support_rand);
  cudaFree(rows_rand);
  cudaFree(noise);
*/

/*
  cudaThreadSynchronize();
  cudaEventRecord(stopHost,0);
  cudaEventSynchronize(stopHost);
  cudaEventElapsedTime(&hostTime, startHost, stopHost);
  cudaEventDestroy(startHost);
  cudaEventDestroy(stopHost);

  printf("The function createProblem takes %f ms.\n", hostTime);
*/

  return;
}





