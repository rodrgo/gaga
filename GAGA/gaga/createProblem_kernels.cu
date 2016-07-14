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




__global__ void makeIndexset(int *vector, int L)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < L ) {
    vector[xIndex]= xIndex;
  }
}



__global__ void sign_pattern(float *values, const int k)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < k ) {
	if (signbit(values[xIndex]-0.5f)){
		values[xIndex] = -1.0f;
	}
	else {values[xIndex] = 1.0f;}
  }

}

__global__ void sign_pattern_alternative(float *vec, const int n, float *rand_vals)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < n ) {
	if (signbit(rand_vals[xIndex]-0.5f)){
		vec[xIndex] = -1.0f;
	}
	else {vec[xIndex] = 1.0f;}
  }
}


// ********* Although not a kernel, this function is used in createMeasurements_dct 
// to select the subset of rows 

void MRandShuffle(int *indexSet, float *randValues, const int L, const int M)
{
  for (int j=L-1; j>L-M; j--){
	
	int k = (int)(j * randValues[j]);
  	int tmp = indexSet[j];
    	indexSet[j] = indexSet[k];
	indexSet[k] = tmp;
  }
  return;
}


/*  
************************************************************************************************
**  This is used when using the Mersenne Twister for the random number generator.              *
**  The Mersenne Twister versions of createProblem were depricated with the release of cuRand. *
************************************************************************************************

void MTseeds(int * seed){
    int i;
    //Need to be thread-safe
    mt_struct_stripped *MT = (mt_struct_stripped *)malloc(MT_RNG_COUNT * sizeof(mt_struct_stripped));

    for(i = 0; i < MT_RNG_COUNT; i++){
        MT[i]      = h_MT[i];
        MT[i].seed = seed[i];
    }
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(ds_MT, MT, sizeof(h_MT)) );

    free(MT);
}
*/




