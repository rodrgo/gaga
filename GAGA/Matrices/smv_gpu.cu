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

__global__ void make_smv_rows(int *rows, int *nonzero_rows, const int m, const int n, const int p, const int l, float *error_flag, float *randvals){
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if ( xIndex < n ){
		int j_rows = xIndex * p;  // there are p random nonzeros per column 
		int j_rand = xIndex * l;  // there are l random values per column
		int duplicate_flag; 
		while( (j_rows < ((xIndex+1)*p)) && (j_rand < ((xIndex+1)*l)) ){
			rows[j_rows] = (int)(m * randvals[j_rand]);
			// correct for getting an "m"
			rows[j_rows] = rows[j_rows] == m ? m - 1 : rows[j_rows];
			duplicate_flag = 0;  //assume not a duplicate
			for( int jj = xIndex * p; jj<j_rows; jj++ ){
				if( rows[jj] == rows[j_rows] ){
					duplicate_flag = 1;
				}
			}
			if( duplicate_flag == 0 ){
				j_rows++; j_rand++;
			}else{
				j_rand++;
			}
			// prepares for next iteration.
			// sets to -1 all entries from j_rows to limit.
			for( int jj = j_rows; jj < ((xIndex+1)*p); jj++ )
				rows[jj] = -1;
		}
		// It is possible to have j_rows == (xIndex + 1)*p with duplicate_flag == 0, but rows incorrectly constructed
		// See node=41949 for problem n=2^18, m = 5243, k = 158, d = 7, seed = 838  
		if( (duplicate_flag==1) || (j_rows < ((xIndex+1)*p)) ){
			atomicAdd(error_flag, 1.0f);
		}
		if (error_flag[0] < 0.5f){
		// Finally, record that row has a zero
			for( int i = xIndex*p; i < (xIndex + 1)*p; i++ )
				nonzero_rows[rows[i]] = 1;
		}
		
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


