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


#define BLOCK_SIZE 32

/***************MatMulKernels*****************/


/*************** gen_matvec for y=Ax where A is dense and x is dense *****************/

__global__ void gen_matvec(float *A, float *x, float *y, const int m, const int n) 
{

  // get variables for loop
  __shared__ int numberBlockCols; // the number of columns in the block
  __shared__ int offset;


  if (threadIdx.y == 0){
    if ((blockIdx.x + 1) * BLOCK_SIZE <= n){
      numberBlockCols = BLOCK_SIZE;
    }
    else { 
      numberBlockCols = n % BLOCK_SIZE;
    }
    offset = ((blockIdx.x * BLOCK_SIZE) * m) + blockIdx.y * BLOCK_SIZE;
  }

  __syncthreads();

  // copy part of x into shared memory, call it xblock
  // use the first BLOCK_SIZE of thread
  __shared__ float xblock[BLOCK_SIZE];

  if (threadIdx.y < numberBlockCols)
    xblock[threadIdx.y] = x[blockIdx.x * BLOCK_SIZE + threadIdx.y];
  
  __syncthreads();

  // summing variable
  float c = 0.0f;

  // make sure we are inside the array vertically
  if (blockIdx.y * BLOCK_SIZE + threadIdx.y < m) {
  
    // go through the threads vertically and sum them into a variable
    for (int i=0; i<numberBlockCols; i++)
      c = c + xblock[i] * A[offset + threadIdx.y + (m * i)];
    atomicAdd(y + blockIdx.y * BLOCK_SIZE + threadIdx.y, c);

  }
}


/*************** gen_matvec for y=Ax where A is dense and x is sparse *****************/

__global__ void gen_matvec_sparse(float *A, float *x, float *y, const int m, const int n) 
{

  // get variables for loop
  __shared__ int numberBlockCols; // the number of columns in the block
  __shared__ int offset;
  __shared__ uint nonzero_size;


  if (threadIdx.y == 0){
    if ((blockIdx.x + 1) * BLOCK_SIZE <= n){
      numberBlockCols = BLOCK_SIZE;
    }
    else { 
      numberBlockCols = n % BLOCK_SIZE;
    }
    offset = ((blockIdx.x * BLOCK_SIZE) * m) + blockIdx.y * BLOCK_SIZE;
    nonzero_size = 0;
  }

  __syncthreads();

  // copy part of x into shared memory, call it xblock
  // use the first BLOCK_SIZE of thread
  __shared__ float xblock[BLOCK_SIZE];
  __shared__ int x_index[BLOCK_SIZE];
  float temp=0.0f;

  if (threadIdx.y < numberBlockCols) {
    temp = x[blockIdx.x * BLOCK_SIZE + threadIdx.y];
      if (temp !=0) {
       xblock[threadIdx.y] = temp;
       x_index[atomicAdd(&nonzero_size,1)]=threadIdx.y;
      } 
  }
  __syncthreads();


  // summing variable
  float c = 0.0f;
  

  // make sure we are inside the array vertically
  if (blockIdx.y * BLOCK_SIZE + threadIdx.y < m) {
  
    // go through the threads vertically and sum them into a variable
    // while only multiiplying the nonzero values of x and only reading
    // the corresponding locations of A from global memory.
      for (int i=0; i<nonzero_size; i++){
        c = c + xblock[x_index[i]] * A[offset + threadIdx.y + (m * x_index[i])];
      }
    atomicAdd(y + blockIdx.y * BLOCK_SIZE + threadIdx.y, c);
  }
}


/*
*******************************
** The matrix multiplication **
*******************************
*/

// A_gen is updated to a dense matrix - sparse vector multiplication.
// Throughout the recovery region for all algorithms in GAGA, this results in a speedup.
// For problems where the algorithms fail, this could be slower as one may be multiplying a dense-dense matrix-vector multiplication.
void A_gen(float * out, float * in, float * A, const int m, const int n, dim3 numBlocksm, dim3 threadsPerBlockm)
{
  int blockCols = (int) ceil(n / (double) BLOCK_SIZE);
  int blockRows = (int) ceil(m / (double) BLOCK_SIZE);
  dim3 dimBlock(1,BLOCK_SIZE);
  dim3 dimGrid(blockCols, blockRows);

// perform the multiplication
  zero_vector_float<<<numBlocksm, threadsPerBlockm>>>(out, m);
  gen_matvec_sparse <<< dimGrid, dimBlock >>>((float*)A, (float*)in, (float*)out, m, n);  // If (float*)in is not sparse, this will be slower.
//  gen_matvec <<< dimGrid, dimBlock >>>((float*)A, (float*)in, (float*)out, m, n);
}


/*
*****************************************
** The matrix Transpose multiplication **
*****************************************
*/

void AT_gen(float * out, float * in, float * AT, const int m, const int n, dim3 numBlocks, dim3 threadsPerBlock)
{
  int blockCols = (int) ceil(m / (double) BLOCK_SIZE);
  int blockRows = (int) ceil(n / (double) BLOCK_SIZE);
  dim3 dimBlock(1,BLOCK_SIZE);
  dim3 dimGrid(blockCols, blockRows);

// perform the multiplication
  zero_vector_float<<<numBlocks, threadsPerBlock>>>(out, n);
  gen_matvec <<< dimGrid, dimBlock >>>((float*)AT, (float*)in, (float*)out, n, m);
}

