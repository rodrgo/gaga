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
**************************************************
** createProblem Kernels **
**************************************************
*/

// make a vector of length L filled iwth integers from 0 to L-1.
__global__ void makeIndexset(int *vector, int L)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < L ) {
    vector[xIndex]= xIndex;
  }
}

// Random selection of p integers from 0 to L-1
__global__ void RandSelectInts(int *indexSet, float *randValues, const int L, const int p)
{
  for (int j=0; j<p; j++){
	
	int k = (int)(L * randValues[j]);
    	indexSet[k] = atomicExch(indexSet+j,indexSet[k]);
  }
}

/*
***************************************
** MAIN FUNCTION: createDataMatrix **
***************************************
*/

void createProblem_entry(float *d_Y, float *d_y, float *d_Mat, int *d_A, const int m, const int n, const int p, unsigned int *p_seed, curandGenerator_t gen, dim3 threadsPerBlockp, dim3 numBlocksp, dim3 threadsPerBlockmn, dim3 numBlocksmn)
{

/*
  cudaEvent_t startHost, stopHost;
  float hostTime;
  cudaEventCreate(&startHost);
  cudaEventCreate(&stopHost);
  cudaEventRecord(startHost,0);
*/
  int mn = m * n;

  // allocate and create a vector of integers from 0 to mn-1.
  int *indexSet;
  cudaMalloc((void**)&indexSet, mn * sizeof(float));
  SAFEcudaMalloc("indexSet in createProblem_entry");

  makeIndexSet<<<threadPerBlockmn, numBlocksmn>>>(indexSet, mn);
  SAFEcuda("makeIndexSet in createProblem_entry");

/*
  // write p random floats to d_y (used here as temporary storage)
  curandStatus_t curandCheck;
  curandCheck = curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  SAFEcurand(curandCheck, "curandCreateGenerator in createProblem_entry");
  curandCheck = curandSetPseudoRandomGeneratorSeed(gen,*p_seed);
  SAFEcurand(curandCheck, "curandSet...Seed in createProblem_entry");
*/  
  curandCheck = curandGenerateUniform(gen,d_y,p);
  SAFEcurand(curandCheck, "curandGenerateUniform in createProblem_entry");

  // use the random floats to rendomly select p of the integersfrom o to mn-1 stored in d_y
  randSelectInts<<<threadPerBlockp, numBlocksp>>>(indexSet, d_y, mn, p);
  SAFEcuda("randSelectInts in createProblem_entry");

  // create the radom entry sensor d_A by copying the first p randomly selected values in indexSet to d_A
  cudaMemcpy(d_A, indexSet, p*sizeof(int), cudaMemcpyDeviceToDevice);
  SAFEcuda("cudaMemcy(d_A,d_y...) in createProblem_entry");
  
  // create the measurements in matrix and vector formats
  A_entry_mat(d_Y, d_Mat, d_A, mn, p, threadPerBlockmn, numBlocksmn, threadPerBlockp, numBlocksp);
  SAFEcuda("A_entry_mat in createProblem_entry");

  A_entry_vec(d_y, d_Mat, d_A, p, threadPerBlockp, numBlocksp);
  SAFEcuda("A_entry_vec in createProblem_entry");

  cudaFree(indexSet);

  return;
}
