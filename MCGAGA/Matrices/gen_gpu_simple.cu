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


#include <math.h>

#define BLOCK_HEIGHT 1024
#define BLOCK_WIDTH 256
#define TYPE float

/***************MatMulKernel*****************/

__global__ void MatMulKernel(TYPE *out, TYPE *in, TYPE *a, const int arrayCols, const int arrayRows) {
  // get variables for loop
  // copy section of b into shared mem
  // go through the threads vertically and sum them into a variable
  // atomic add these variables to the corresponding c index

  // get variables for loop
  // variable for loop length: blockEltHeight
  __shared__ int blockElt;
  __shared__ int blockColInd;
  __shared__ int blockRowInd;
  if (threadIdx.x == 0) {
    if ((blockIdx.y + 1) * BLOCK_WIDTH <= arrayRows)
      blockElt = BLOCK_WIDTH;
    else blockElt = arrayRows % BLOCK_WIDTH;
    blockColInd = blockIdx.x * BLOCK_HEIGHT;
    blockRowInd = blockIdx.y * BLOCK_WIDTH;
  }
  
  __syncthreads();
  
  // copy section of b into shared mem
  // use the first BLOCK_WIDTH of thread
  __shared__ TYPE b[BLOCK_WIDTH];

  if (threadIdx.x < blockElt)
    b[threadIdx.x] = in[blockRowInd + threadIdx.x];
  
  __syncthreads();

  // summing variable
  TYPE cSum = (TYPE) 0;
  int threadColInd = blockColInd + threadIdx.x;

  // make sure we are inside the array horizontally
  if (threadColInd < arrayCols) {
  
    // go through the threads vertically and sum them into a variable
    for (int i=0; i<blockElt; i++)
      // A cols : arrayCols
      // A rows : arrayRows
      // A row index : blockIdx.y * BLOCK_WIDTH + i : blockRowInd + i
      // A col index : blockIdx.x * BLOCK_HEIGHT + threadIdx.x : blockColInd + threadIdx.x : threadColInd
      // B index : b[i]

      // cSum = B index * ( A row index * A cols + A col index)
      cSum += b[i] * a[(blockRowInd + i) * (arrayCols) + (threadColInd)];
      //printf("csum = %f\n", cSum);
    
    // atomic add these variables to the corresponding c index
    atomicAdd(out + threadColInd, cSum);
    //atomicAdd(out + blockIdx.x * BLOCK_HEIGHT + threadIdx.x, cSum);
    //printf("el[%d%d;%d] csum = %f tot = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, cSum, *(out + blockIdx.x * BLOCK_HEIGHT + threadIdx.x));
  }
  
}

__global__ void MatMulKernelT(TYPE *out, TYPE *in, TYPE *a, const int arrayCols, const int arrayRows) {
  // get variables for loop
  // copy section of b into shared mem
  // go through the threads vertically and sum them into a variable
  // atomic add these variables to the corresponding c index

  // get variables for loop
  // variable for loop length: blockElt
  __shared__ int blockElt;
  __shared__ int blockColInd;
  __shared__ int blockRowInd;
  if (threadIdx.x == 0) {
    if ((blockIdx.y + 1) * BLOCK_WIDTH <= arrayCols)
      blockElt = BLOCK_WIDTH;
    else blockElt = arrayCols % BLOCK_WIDTH;
    blockColInd = blockIdx.y * BLOCK_WIDTH;
    blockRowInd = blockIdx.x * BLOCK_HEIGHT;
  }
  
  __syncthreads();
  
  // copy section of b into shared mem
  // use the first BLOCK_WIDTH of thread
  __shared__ TYPE b[BLOCK_WIDTH];

  if (threadIdx.x < blockElt)
    b[threadIdx.x] = in[blockColInd + threadIdx.x];
  
  __syncthreads();

  // summing variable
  TYPE cSum = (TYPE) 0;
  int threadRowInd = blockRowInd + threadIdx.x;

  // make sure we are inside the array horizontally
  if (threadRowInd < arrayRows) {
  
    // go through the threads vertically and sum them into a variable
    for (int i=0; i<blockElt; i++)
      // A cols : arrayCols
      // A rows : arrayRows
      // A col index : blockIdx.y * BLOCK_WIDTH + i : blockColInd + i
      // A row index : blockIdx.x * BLOCK_HEIGHT + threadIdx.x : blockRowInd + threadIdx.x : threadRowInd
      // B index : b[i]

      // cSum = B index * ( A row index * A cols + A col index)
      cSum += b[i] * a[(threadRowInd) * (arrayCols) + (blockColInd + i)];
      //printf("csum = %f\n", cSum);
    
    // atomic add these variables to the corresponding c index
    atomicAdd(out + threadRowInd , cSum);
    //printf("el[%d%d;%d] csum = %f tot = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, cSum, *(out + blockIdx.x * BLOCK_HEIGHT + threadIdx.x));
  }
  
}

/*
/ ***********createRandomMatrix*************** /


void createRandomMatrix(TYPE *A, int size, int seed) {
  float *d_A;
  float *h_A = (float *) malloc (size * sizeof(float));
  curandGenerator_t gen;
  size_t size_d_A = size * sizeof(TYPE);

  cudaMalloc((void **) &d_A, size_d_A);

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  curandGenerateUniform(gen, d_A, size);

  cudaMemcpy(h_A, d_A, size_d_A, cudaMemcpyDeviceToHost);

  // for (int j = 0; j < 10; j++) 
  //  printf("h_A[%d] = %l=f\n", j, 10* h_A[j]);

  for (int j = 0; j < size; j++) 
    A[j] = h_A[j] / sqrt (size); 

  curandDestroyGenerator(gen);
  cudaFree(d_A);
  free(h_A);
}
*/

void A_gen(float * out, float * in, float * A, const int m, const int n, dim3 dimGridm, dim3 dimBlocksm)
{
  int blockCols = (int) ceil(n / (double) BLOCK_WIDTH);
  int blockRows = (int) ceil(m / (double) BLOCK_HEIGHT);
  dim3 dimBlock(BLOCK_HEIGHT);
  dim3 dimGrid(blockRows, blockCols);

  zero_vector_float<<<dimGridm, dimBlocksm>>>(out, m);
  MatMulKernel<<<dimGrid, dimBlock>>>(out, in, A, m, n);
}

void AT_gen(float * out, float * in, float * A, const int m, const int n, dim3 dimGridn, dim3 dimBlocksn)
{
  int blockCols = (int) ceil(m / (double) BLOCK_WIDTH);
  int blockRows = (int) ceil(n / (double) BLOCK_HEIGHT);
  dim3 dimBlock(BLOCK_HEIGHT);
  dim3 dimGrid(blockRows, blockCols);

  zero_vector_float<<<dimGridn, dimBlocksn>>>(out, n);
  MatMulKernelT<<<dimGrid, dimBlock>>>(out, in, A, m, n);
}
