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
** MAIN FUNCTION: createProblem_smv **
**************************************
*/

int createProblem_smv(int *k_pointer, const int m, const int n, const int vecDistribution, float band_percentage, float *d_vec_input, float *d_y, int *d_rows, int *d_cols, float *d_vals, const int p, const int ensemble, unsigned int *p_seed)
{
// k_pointer is used rather than k as we may allow k to vary somewhat.
// m and n are the size of the matrix
// vecDistribution: 0 = uniform, 1=random{-1,1}, 2 = Gaussian
// d_vec_input is the vector the matrix multiplies "x"
// d_rows, d_cols, and d_vals are the row,col location and value of 
// the nonzeros in the sparse matrix
// p is the number of nonzeros per column
// ensemble (matrix): 1=nonzeros all 1, 2=nonzeros random{-1,1}.
// p_seed allows us to store the seed and thus recreate the problem.

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
  //	n*p random numbers to determine the random locations of the p 
  //        nonzeros per column.  the method used to select these locations 
  //        may have duplicates so we compute  1.3*p per column.
  //    and possibly n*p random numbers to determine the values of the 
  //        nonzeros in the sparse matrix.
  //        ensemble=1 is all ones (no need for random values to generate these)
  //        ensemble=2 is uniform {-1,1}. (need n*p random numbers)


  // ensemble=1 is our first, and has all entries in d_vals equal to 1.
  // ensemble=2 is random {-1,1}
  // ensemble=3 is all ones too. 

  int l;
  int L;

  if (ensemble==1 || ensemble==3){
    l = (int)(1.3f*p);
    if( l < p+2 )
      l=p+2;
    L = (2+l)*n;
  }
  if (ensemble==2){
    l = (int)(1.3f*p);
    if( l < p+2 )
      l=p+2;
    L = (2+l+p)*n;
  }


// ********* Create L random numbers needed to generate the problem ******

  // Allocate device variable for random numbers
  float * random;
  cudaMalloc((void**)&random, L * sizeof(float));
  SAFEcudaMalloc("random in createProblem_smv");

  curandStatus_t curandCheck;

  curandGenerator_t gen;
  curandCheck = curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  SAFEcurand(curandCheck, "curandCreateGenerator in createProblem_smv");
  // note, CURAND_RNG_PSEUDO_DEFAULT selects the random number generator type
  curandCheck = curandSetPseudoRandomGeneratorSeed(gen,*p_seed);
  SAFEcurand(curandCheck, "curandSetPsuedoRandomGeneratorSeed in createProblem_smv");
  curandCheck = curandGenerateUniform(gen,random,L);
  SAFEcurand(curandCheck, "curandGenerateUniform in createProblem_smv");
  SAFEcuda("createRandomNumbers uniform in createProblem_smv"); 

  // if vecDistribution == 1 then need uniform sign pattern for the
  // nonzeros in the vector measured.

  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

  int threads_perblock = min(n, max_threads_per_block);
  dim3 threadsPerBlock(threads_perblock);
  int num_blocks = (int)ceil((float)n/(float)threads_perblock);
  dim3 numBlocks(num_blocks);

  if(vecDistribution == 1){
        sign_pattern<<< numBlocks, threadsPerBlock >>>(random,n);
        SAFEcuda("createRandomNumbers sign in createProblem_smv");
  }

  // if vecDistribution == 2 then need gaussian random numbers for the
  // nonzeros in the vector measured.
  if(vecDistribution == 2){       

        curandCheck = curandGenerateNormal(gen,random,n,0,1);
        SAFEcurand(curandCheck, "curandGenerateNormal in createProblem_smv");
  }

  curandCheck = curandDestroyGenerator(gen);
  SAFEcurand(curandCheck, "curandDestroyGenerator in createProblem_smv");

/*
  The random numbers are used to create various things:
  values_rand = random;
  support_rand = random + n;
  smv_rand = random + 2*n;
*/

//  createVector(k_pointer, m, n, vecDistribution, d_vec_input, values_rand, support_rand);
  createVector_smv(k_pointer, m, n, vecDistribution, band_percentage, d_vec_input, random, random + n);
  SAFEcuda("createVector in createProblem_smv"); 


//  int error_flagCM = createMeasurements_smv(m, n, d_vec_input, d_y, d_rows, d_cols, d_vals, p, l, smv_rand, ensemble);
  int error_flagCM = createMeasurements_smv(m, n, d_vec_input, d_y, d_rows, d_cols, d_vals, p, l, random+2*n, ensemble);
  SAFEcuda("createMeasurements_smv in createProblem_smv"); 

  cudaFree(random);
/*
  cudaFree(values_rand);
  cudaFree(support_rand);
  cudaFree(smv_rand);
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

  return error_flagCM;
}



int createProblem_smv_noise(int *k_pointer, const int m, const int n, const int vecDistribution, float *d_vec_input, float *d_y, int *d_rows, int *d_cols, float *d_vals, const int p, const int ensemble, unsigned int *p_seed, float noise_level)
{
// k_pointer is used rather than k as we may allow k to vary somewhat.
// m and n are the size of the matrix
// vecDistribution: 0 = uniform, 1=random{-1,1}, 2 = Gaussian
// d_vec_input is the vector the matrix multiplies "x"
// d_rows, d_cols, and d_vals are the row,col location and value of 
// the nonzeros in the sparse matrix
// p is the number of nonzeros per column
// ensemble (matrix): 1=nonzeros all 1, 2=nonzeros random{-1,1}.
// p_seed allows us to store the seed and thus recreate the problem.

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
  //	n*p random numbers to determine the random locations of the p 
  //        nonzeros per column.  the method used to select these locations 
  //        may have duplicates so we compute  1.3*p per column.
  //    and possibly n*p random numbers to determine the values of the 
  //        nonzeros in the sparse matrix.
  //        ensemble=1 is all ones (no need for random values to generate these)
  //        ensemble=2 is uniform {-1,1}. (need n*p random numbers)
  //    and m entries for gaussian noise
  //    Since curandGenerateNormal requires an even length, we add m mod 2.




  // ensemble=1 is our first, and has all entries in d_vals equal to 1.
  // ensemble=2 is random {-1,1}

  int l;
  int L;

  if (ensemble==1 || ensemble==3){
    l = (int)(1.3f*p);
    if( l < p+2 )
      l=p+2;
    L = (2+l)*n;
  }
  if (ensemble==2){
    l = (int)(1.3f*p);
    if( l < p+2 )
      l=p+2;
    L = (2+l+p)*n;
  }
  L=L+(m + m%2); // extra m entries for noise


// ********* Create L random numbers needed to generate the problem ******



  // Allocate device variable for random numbers
  float * random;
  cudaMalloc((void**)&random, L * sizeof(float));
  SAFEcudaMalloc("random in createProblem_smv");

  curandStatus_t curandCheck;

  curandGenerator_t gen;
  curandCheck = curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  SAFEcurand(curandCheck, "curandCreateGenerator in createProblem_smv_noise");
  // note, CURAND_RNG_PSEUDO_DEFAULT selects the random number generator type
  curandCheck = curandSetPseudoRandomGeneratorSeed(gen,*p_seed);
  SAFEcurand(curandCheck, "curandSet...Seed in createProblem_smv_noise");
  curandCheck = curandGenerateUniform(gen,random,L-m);
  SAFEcurand(curandCheck, "curandGenerateUniform in createProblem_smv_noise");
  curandCheck = curandGenerateNormal(gen,random+L-(m+m%2),m+m%2,0,1);
  SAFEcurand(curandCheck, "curandGenerateNormal (1st) in createProblem_smv_noise");
  SAFEcuda("createRandomNumbers in createProblem_smv_noise"); 

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
        SAFEcuda("sign_pattern in createProblem_smv");
  }

  // if vecDistribution == 2 then need gaussian random numbers for the
  // nonzeros in the vector measured.
  if(vecDistribution == 2){       

        curandCheck = curandGenerateNormal(gen,random,n,0,1);
  	SAFEcurand(curandCheck, "curandGenerateNormal (2nd) in createProblem_smv_noise");
  }

  curandCheck = curandDestroyGenerator(gen);
  SAFEcurand(curandCheck, "curandDestroyGenerator in createProblem_smv_noise");


/* 
  The random numbers are used to create various things:
   values_rand = random;
  support_rand = random + n;
  smv_rand = random + 2*n;
  noise = random + L - (m+m%2);
*/

  createVector(k_pointer, m, n, vecDistribution, d_vec_input, random, random+n);
  SAFEcuda("createVector in createProblem_smv_noise"); 


  int error_flagCM = createMeasurements_smv(m, n, d_vec_input, d_y, d_rows, d_cols, d_vals, p, l, random+2*n, ensemble);
  SAFEcuda("createMeasurements_smv in createProblem_smv_noise"); 

//  Here we want to add a noise vector that is scaled so that it's norm is noise_level*norm2(y).
  float noise_norm, y_norm, noise_scale;
  noise_norm = cublasSnrm2(m, random+L-(m+m%2), 1);
  SAFEcublas("cublasSnrm2 computing noise_norm in createProblem_smv_noise");

  y_norm = cublasSnrm2(m, d_y, 1);
  SAFEcublas("cublasSnrm2 computing y_norm in createProblem_smv_noise");

  noise_scale = noise_level*(y_norm/noise_norm);

  cublasSaxpy(m, noise_scale, random+L-(m+m%2), 1, d_y, 1);
  SAFEcublas("cublasSaxpy adding noise to measurements in createProblem_smv_noise");

  cudaFree(random);
/*
  cudaFree(values_rand);
  cudaFree(support_rand);
  cudaFree(smv_rand);
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


  return error_flagCM;
}



