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


// ******************** Sparse Mat-Vec Functions ******************


__global__ void sparse_matvec(int *row, int *col, float *val, float *x, float *y, const int m, const int n, const int nz) 
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < nz ){
    float val_to_add = val[xIndex] * x[col[xIndex]];
    if ( val_to_add )
      atomicAdd(y+row[xIndex],val_to_add);
  }
}


__global__ void sparse_matvecT(int *row, int *col, float *val, float *x, float *y, const int m, const int n, const int nz) 
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < nz ){
    float val_to_add = val[xIndex] * y[row[xIndex]];
    atomicAdd(x+col[xIndex],val_to_add);
  }
}



//*********** Sparse Mat Vec Creation Functions ******************

__global__ void make_smv_cols(int *vec, const int n, const int p)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < n ){
    for(int j=xIndex*p; j < ((xIndex+1)*p); j++)
      vec[j]=xIndex;
  }
}



__global__ void make_smv_rows(int *vec, const int m, const int n, const int p, const int l, float *error_flag, float *randvals)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < n ){
    int j_vec = xIndex * p;  // there are p random nonzeros 
    int j_rand = xIndex * l;  // there are l random values per column
    int duplicate_flag; 
    while( (j_vec < ((xIndex+1)*p)) & (j_rand < ((xIndex+1)*l)) ){
      vec[j_vec] = (int)(m * randvals[j_rand]);
      duplicate_flag=0;  //assume not a duplicate
      for( int jj = xIndex * p; jj<j_vec; jj++){
        if( vec[jj] == vec[j_vec] ){
	  duplicate_flag=1;
        }
      }
      if( duplicate_flag == 0 ){
        j_vec++; j_rand++;
      }
      else{
        j_rand++;
      }

    for (int jj = j_vec; jj < ((xIndex+1)*p); jj++)
      vec[jj]=-1;
    }

    if( (duplicate_flag==1) || (j_vec < ((xIndex+1)*p) ) )
      atomicAdd(error_flag,1.0f);
  }
}


/*
*******************************
** The matrix multiplication **
*******************************
*/

void A_smv(float * out, float * in, const int m, const int n, int * rows, int * cols, float * vals, const int nz, dim3 numBlocksm, dim3 threadsPerBlockm, dim3 numBlocksnp, dim3 threadsPerBlocknp)
{
// zero out the output vector
  zero_vector_float <<< numBlocksm, threadsPerBlockm >>>((float*)out, m);

// perform the multiplication
  sparse_matvec <<< numBlocksnp, threadsPerBlocknp >>>((int*)rows, (int*)cols, (float*)vals, (float*)in, (float*)out, m, n, nz);
  return;
}



/*
******************************************
** The matrix Transpose  multiplication **
******************************************
*/

inline void AT_smv(float * out, float * in, const int m, const int n, int * rows, int * cols, float * vals, const int nz, dim3 numBlocks, dim3 threadsPerBlock, dim3 numBlocksnp, dim3 threadsPerBlocknp)
{
// zero out the output vector
  zero_vector_float <<< numBlocks, threadsPerBlock >>>((float*)out, n);

// perform the multiplication
  sparse_matvecT <<< numBlocksnp, threadsPerBlocknp >>>((int*)rows, (int*)cols, (float*)vals, (float*)out, (float*)in, m, n, nz);

  return;
}


