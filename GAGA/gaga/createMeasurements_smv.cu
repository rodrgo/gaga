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
************************************************
** KERNELS check nonzeros: createMeasurements **
************************************************
*/

__device__ int lock = 0;

__global__ void check_redundancy_in_columns(int *rows, int *exists_inconsistent_column, int n, int p){
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if ( xIndex < n){
		int idx = xIndex*p;
		for (int i = idx; i < (xIndex + 1)*p - 1; i++)
			for (int j = i + 1; j < (xIndex + 1)*p; j++)
				if (rows[i] == rows[j])
					exists_inconsistent_column[0] = 1; 
	}
}

__global__ void flag_nonzero_rows(int *rows, int *nonzero_rows, int n, int p){
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if ( xIndex < n*p){
		nonzero_rows[rows[xIndex]] = 1;
	}
}

/* 
If column has nonzero in row with many nonzeros, move one nonzero to a zero-row.
rows = np vector of row indices.
nonzero_rows_count = number of nonzero elements in each row.
num_zero_rows = number of zero rows in measurement vector.
zero_rows_list = list of "num_zero_rows" rows that are zero.

__global__ void move_nonzeros(int *rows, int *nonzero_rows_count, int *nonzero_rows_index, int *num_zero_rows, int *zero_rows_list){
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if ( xIndex < num_zero_rows[0] ){
		int zero_row = zero_rows_list[xIndex];
		int row_index = nonzero_rows_index[xIndex];
		nonzero_rows_count[rows[xIndex]]--;
		rows[row_index] = zero_row;
		nonzero_rows_count[zero_row]++;
	}
}
*/

__global__ void move_nonzeros(int *rows, int *nonzero_rows_index, int *zero_rows_list, int num_zero_rows){
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if ( xIndex < num_zero_rows ){
		int zero_row = zero_rows_list[xIndex];
		int row_index = nonzero_rows_index[xIndex];
		rows[row_index] = zero_row;
	}
}

// Get number of zero rows
__global__ void get_num_zero_rows(int *nonzero_rows, int *num_zero_rows, int m){
		unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
		if ( xIndex < m){
			if (nonzero_rows[xIndex] == 0){
				atomicAdd(num_zero_rows, 1);
			}
		}
}

// nonzero_rows_index contains the indices in rows of thos elements that occur more than twice.
__global__ void count_nonzeros_in_rows(int *rows, int *nonzero_rows_count, int n, int p){
		unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
		if ( xIndex < n*p){
			atomicAdd(nonzero_rows_count+rows[xIndex], 1);
			/*
			atomicAdd(nonzero_rows_count+rows[xIndex], 1);
			if(nonzero_rows_count[rows[xIndex]] >= 2 && track_index[0] < m){
				nonzero_rows_index[atomicAdd(track_index, 1)] = xIndex;
			}
			*/
		}
}

// Create an m-vector [r1, r2, r3, ..., r_{num_zero_rows}, 0, ..., 0] containing zero-rows in first "num_zero_rows" entries.
__global__ void get_zero_rows_list(int *nonzero_rows, int *num_zero_rows, int *zero_rows_list, int m){
		unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
		if ( xIndex < m ){
			if (nonzero_rows[xIndex] == 0){
				zero_rows_list[atomicAdd(num_zero_rows, 1)] = xIndex;
			}
		}
	}

// checks if there is a zero row in vector
// nonzero_rows[i] == 1 iff row i has a nonzero
__global__ void check_zero_rows(int *nonzero_rows, int *exists_zero_row, int m){
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if ( xIndex < m ){
		if (nonzero_rows[xIndex] == 0)
			exists_zero_row[0] = 1;
	}
}

/*
***************************************
** MAIN FUNCTION: createMeasurements **
***************************************
*/

int createMeasurements_smv(const int m, const int n, float *d_vec_input, float *d_y, int *d_rows, int *d_cols, float *d_vals, const int p, const int l, float *smv_rand, int ensemble)
{

// THE INPUTS ARE AS FOLLOWS:
// m and n are the size of the spars matrix being created
// d_vec_input is the vector that the matrix will multiply
// d_y is the result of the matrix times d_vec_input
// d_rows, d_cols, and d_vals define the sparse matrix through 
// the row/col index and value in that location which is nonzero.
// p is the number of nonzeros per column
// l is the number of random numbers we have to crease the locations 
// of the p random numbers per row.  l is slightly larger than p in 
// case the binning used has repititions.
// smv_rand is a list of random numbers uniform from [0,1] which 
// we use to define d_rows, d_cols, and d_vals.  
// ensemble determines how the values in d_vals should be drawn.
// for now we just have ensemble=1 which is all entries in d_vals=1, 
// so the sparse matrix has entries 0 or 1.


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


  int threads_perblock = min(n, max_threads_per_block);
  dim3 threadsPerBlock(threads_perblock);
  int num_blocks = (int)ceil((float)n/(float)threads_perblock);
  dim3 numBlocks(num_blocks);

  int threads_perblocknp = min(n*p, max_threads_per_block);
  dim3 threadsPerBlocknp(threads_perblocknp);
  int num_blocksnp = (int)ceil((float)(n*p)/(float)threads_perblock);
  dim3 numBlocksnp(num_blocksnp);

  int threads_perblockm = min(m, max_threads_per_block);
  dim3 threadsPerBlockm(threads_perblockm);
  int num_blocksm = (int)ceil((float)m/(float)threads_perblockm);
  dim3 numBlocksm(num_blocksm);

  if(ensemble==1 || ensemble==3){
    // Set the n*p values in d_vals to be 1
    one_vector_float<<<numBlocksnp, threadsPerBlocknp>>>(d_vals,n*p);
    SAFEcuda("one_vector_float in createMeasurements_smv");
  }
  if(ensemble==2){
    // Set the n*p values in d_vals to be a random sign
    // use the random numbers in smv_rand starting n*p later
    sign_pattern_alternative<<<numBlocksnp, threadsPerBlocknp>>>(d_vals,n*p,smv_rand+l*n);  // use first l*n rand values later.
    SAFEcuda("sign_pattern in createMeasurements_smv");
  }

  // Scale the entries by the number of nonzeros per column to 
  // create unit l2-norm columns

  if (ensemble==1 || ensemble==2){
	  float scale = sqrt((float)p);
	  scale = 1/scale;
	  cublasSscal(n*p,scale,d_vals,1);
	  SAFEcublas("cublasSscal in createMeashurements_smv");
  }

  // Set the values in d_cols.
  // the first p entries in d_cols is 1, the second p entries 
  // in d_cols is 2, and so on.
  make_smv_cols<<<numBlocks, threadsPerBlock>>>(d_cols,n,p);
  SAFEcuda("make_smv_cols in createMeasurements_smv");

  // Set the n*p values in d_rows.
  int error_flag=0;
  float * d_error_flag;
  cudaMalloc((void**)&d_error_flag,sizeof(float));

  float * h_error_flag = (float*)malloc( sizeof(float));
  h_error_flag[0] = 1.0f;

  int error_count = 0;
  int max_error_count = 5;

  cublasInit();

  int * d_nonzero_rows;
  cudaMalloc((void**)&d_nonzero_rows, m * sizeof(int));
  zero_vector_int<<<numBlocksm, threadsPerBlockm>>>(d_nonzero_rows, m);

  while ( (h_error_flag[0] > 0.5f) & (error_count < max_error_count) ) {
	zero_vector_float<<<1,1>>>(d_error_flag,1);
	zero_vector_int<<<numBlocksm, threadsPerBlockm>>>(d_nonzero_rows, m);
	make_smv_rows<<<numBlocks, threadsPerBlock>>>(d_rows,d_nonzero_rows,m,n,p,l,d_error_flag,smv_rand);
    	cudaDeviceSynchronize();
	SAFEcuda("make_smv_rows in createMeasurements_smv");

	cudaMemcpy(h_error_flag,d_error_flag,sizeof(float),cudaMemcpyDeviceToHost);

	error_count++;
  }
  
  /*
  if(1 == 1){
    int *h_rows;
    int np = n*p;
    h_rows = (int*)malloc( np*sizeof(int) );
    cudaMemcpy(h_rows, d_rows, np*sizeof(int), cudaMemcpyDeviceToHost); 
	//printf("n = %d, m = %d, p = %d, np = %d\n", n, m, p, np);
	for (int i = 0; i < np; i++){
		if (h_rows[i] < 0 || h_rows[i] > m -1){
			//printf("p = %d, l = %d\n", p, l);
			printf("After make_smv_rows: Inconsistent row detected  h_rows[%d] = %d, error_flag = %1.1f\n", i, h_rows[i], h_error_flag[0]);
			//printf("h_error_flag = %1.1f\n", h_error_flag[0]);
		}
	}
    free(h_rows);
  }
  */

  // Starts checking for zero rows
  int zero = 0;
  int h_exists_zero_row = 0;
  int * d_exists_zero_row;
  cudaMalloc((void**)&d_exists_zero_row, 1 * sizeof(int));
  cudaMemcpy(d_exists_zero_row, &zero, sizeof(int), cudaMemcpyHostToDevice); 

  check_zero_rows<<<numBlocksm, threadsPerBlockm>>>(d_nonzero_rows, d_exists_zero_row, m);
  cudaMemcpy(&h_exists_zero_row, d_exists_zero_row, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_exists_zero_row);

  // If exists a zero row, modify matrix
  if ( h_exists_zero_row == 1 && h_error_flag[0] <= 0.5f){
    //printf("Exists zero row...\n");

    // Get number of zero-rows
    int * d_num_zero_rows;
    cudaMalloc((void**)&d_num_zero_rows, 1*sizeof(int));
    cudaMemcpy(d_num_zero_rows, &zero, sizeof(int), cudaMemcpyHostToDevice); 

    get_num_zero_rows<<<numBlocksm, threadsPerBlockm>>>(d_nonzero_rows, d_num_zero_rows, m);
    cudaDeviceSynchronize();

    int num_zero_rows = 0; 
    cudaMemcpy(&num_zero_rows, d_num_zero_rows, sizeof(int), cudaMemcpyDeviceToHost); 
    cudaFree(d_num_zero_rows);

    // Get a "list" of zero rows and count how many there are.

    int * d_tracker;
    cudaMalloc((void**)&d_tracker, 1*sizeof(int));
    cudaMemcpy(d_tracker, &zero, sizeof(int), cudaMemcpyHostToDevice); 

    int * d_zero_rows_list;
    cudaMalloc((void**)&d_zero_rows_list, num_zero_rows * sizeof(int));
    zero_vector_int<<<numBlocksm, threadsPerBlockm>>>(d_zero_rows_list, num_zero_rows);

    get_zero_rows_list<<<numBlocksm, threadsPerBlockm>>>(d_nonzero_rows, d_tracker, d_zero_rows_list, m);
    cudaDeviceSynchronize();
    cudaFree(d_tracker);
    //printf("passed: get_zero_rows_list...\n\n");

    //After counting, pass array to host and create an m-list of rows with more than one nonzero.
    // h_nonzero_rows_count[i] = number of nonzeros in row i
    // h_nonzero_rows_index = list of m indices of d_rows that contain rows with more than one nonzero.

    int *h_rows;
    int np = n*p;
    h_rows = (int*)malloc( np*sizeof(int) );
    cudaMemcpy(h_rows, d_rows, np*sizeof(int), cudaMemcpyDeviceToHost); 

    int *h_nonzero_rows_index;
    h_nonzero_rows_index = (int*)malloc( num_zero_rows*sizeof(int) );

    int *h_nonzero_rows_count;
    h_nonzero_rows_count = (int*)malloc( m*sizeof(int) );
    for (int i = 0; i < m; i++)
      h_nonzero_rows_count[i] = 0;
    //cudaMemcpy(h_nonzero_rows_count, d_nonzero_rows_count, m*sizeof(int), cudaMemcpyDeviceToHost); 

    int i = 0;
    int j = 0;
    while (j < num_zero_rows){
      h_nonzero_rows_count[h_rows[i]]++;
      if (h_nonzero_rows_count[h_rows[i]] > 1){
        h_nonzero_rows_index[j] = i;
        j++;
        h_nonzero_rows_count[h_rows[i]]--;
      }
      i = (i == (np - 1)) ? 0 : i + 1;
    }

    int * d_nonzero_rows_index;
    cudaMalloc((void**)&d_nonzero_rows_index, num_zero_rows * sizeof(int));
    cudaMemcpy(d_nonzero_rows_index, h_nonzero_rows_index, num_zero_rows*sizeof(int), cudaMemcpyHostToDevice); 

    free(h_rows);
    free(h_nonzero_rows_index);
    free(h_nonzero_rows_count);
    //printf("passed: d_nonzero_rows_index creation...\n\n");

    // Redistribute nonzeros from rows with many to rows to none. 
    //int h_num_zero_rows = 0;
    //cudaMemcpy(&h_num_zero_rows, d_num_zero_rows, sizeof(int), cudaMemcpyDeviceToHost); 
    //move_nonzeros<<<numBlocksm, threadsPerBlockm>>>(d_rows, d_nonzero_rows_count, d_nonzero_rows_index, d_num_zero_rows, d_zero_rows_list);
    move_nonzeros<<<numBlocksm, threadsPerBlockm>>>(d_rows, d_nonzero_rows_index, d_zero_rows_list, num_zero_rows);
    cudaDeviceSynchronize();
    //printf("passed: move_nonzeros...\n\n");

    // Perform checks to see matrix is now left-d-regular and has no zero rows.
    h_exists_zero_row = 0;
    int * d_exists_zero_row;
    cudaMalloc((void**)&d_exists_zero_row, 1 * sizeof(int));
    cudaMemcpy(d_exists_zero_row, &zero, sizeof(int), cudaMemcpyHostToDevice); 

    zero_vector_int<<<numBlocksm, threadsPerBlockm>>>(d_nonzero_rows, m);
    cudaDeviceSynchronize();
    flag_nonzero_rows<<<numBlocksnp, threadsPerBlocknp>>>(d_rows, d_nonzero_rows, n, p);
    cudaDeviceSynchronize();

    check_zero_rows<<<numBlocksm, threadsPerBlockm>>>(d_nonzero_rows, d_exists_zero_row, m);
    cudaMemcpy(&h_exists_zero_row, d_exists_zero_row, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_exists_zero_row);
    cudaDeviceSynchronize();

    // check for repeated rows in columns (this should not happen, but we still perform the check)
    int h_exists_inconsistent_column = 0;
    int * d_exists_inconsistent_column;
    cudaMalloc((void**)&d_exists_inconsistent_column, 1 * sizeof(int));
    cudaMemcpy(d_exists_inconsistent_column, &zero, sizeof(int), cudaMemcpyHostToDevice); 

    check_redundancy_in_columns<<<numBlocks, threadsPerBlock>>>(d_rows, d_exists_inconsistent_column, n, p);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_exists_inconsistent_column, d_exists_inconsistent_column, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_exists_inconsistent_column);

    if (h_exists_zero_row == 1){
      printf("FAIL: (zero-row correction) There was a zero row and could not be corrected!\n\n");
      error_flag = 1;
      }

    if (h_exists_inconsistent_column == 1){
      printf("FAIL: (zero-row correction) There is an inconsistent column in matrix!\n\n");
      error_flag = 1;
    }

    cudaFree(d_nonzero_rows_index);
    cudaFree(d_num_zero_rows);
    cudaFree(d_zero_rows_list);
    //cudaFree(d_nonzero_rows_count);

  }
  cudaFree(d_nonzero_rows);
  // Finishes check for zero-rows

  cublasShutdown();

  cudaFree(d_error_flag);
  free(h_error_flag);

  if( error_count >= max_error_count ) {
//    printf("WARNING: Sparse Matrix Construction failed %d consecutive times.\n",max_error_count);
//    printf("The parameter p is too high for the problem size.\n");
    error_flag=1;
  }

// create the measurements
  zero_vector_float <<< numBlocksm, threadsPerBlockm >>>((float*)d_y, m);
  SAFEcuda("zero_vector_float in createMeasurements_smv");

  if (error_flag==0){
    sparse_matvec <<< numBlocksnp, threadsPerBlocknp >>>((int*)d_rows, (int*)d_cols, (float*)d_vals, (float*)d_vec_input, (float*)d_y,m,n,p*n);
    SAFEcuda("sparse_matvec in createMeasurements_smv");
  }

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

  printf("The function createMeasurements_smv takes %f ms of which %f is the Mat-vc .\n", hostTime, MatvecTime);

*/

  return error_flag;
}


