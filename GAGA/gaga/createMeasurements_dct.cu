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

void  createMeasurements_dct(const int m, const int n, float *d_vec_input, float *d_y, int *d_rows, float *rows_rand)
{
/*
  cudaEvent_t startHost, stopHost;
  float hostTime;
  cudaEventCreate(&startHost);
  cudaEventCreate(&stopHost);
  cudaEventRecord(startHost,0);
*/

  int threads_perblock = min(n, 512);
  dim3 threadsPerBlock(threads_perblock);
  int num_blocks = (int)ceil((float)n/(float)threads_perblock);
  dim3 numBlocks(num_blocks);

  int j;
  


  int *rows_index;  
  cudaMalloc((void**)&rows_index, n * sizeof(int));
  SAFEcudaMalloc("rows_index in createMeasurements_Dct");

  makeIndexset<<<numBlocks, threadsPerBlock>>>(rows_index, n);
  SAFEcuda("makeIndexset in createMeasurements_dct");

 // printf("n is %d\n",n);




// ********** do the shuffle on the host ************

  int * h_rows = (int*)malloc( sizeof(int) * m );
  SAFEmalloc_int(h_rows, "h_rows in createMeasurements_Dct");


  int * supportHost;
  supportHost=(int*)malloc(n * sizeof(int));
  SAFEmalloc_int(supportHost, "supportHost in createMeasurements_Dct");

  float * valuesHost;
  valuesHost=(float*)malloc(n * sizeof(float));
  SAFEmalloc_float(valuesHost, "valuesHost in createMeasurements_Dct");


  cudaMemcpy(supportHost, rows_index, n * sizeof(int), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to supportHost in createMeasurements_dct");

  cudaMemcpy(valuesHost, rows_rand, n * sizeof(float), cudaMemcpyDeviceToHost);
  SAFEcuda("cudaMemcpy to valuesHost in createMeasurements_dct");

  MRandShuffle(supportHost, valuesHost, n, m);

  for (j=0; j<m; j++){
	h_rows[j]=supportHost[n-1-j];
  }

  cudaFree(rows_index);
 
  free(supportHost);
  free(valuesHost);

// ************ shuffle complete **************




// copy the data to the appropriate device variables

  cudaMemcpy(d_rows, h_rows, m * sizeof(int), cudaMemcpyHostToDevice);
  SAFEcuda("cudaMemcpy to d_rows in createMeasurements_dct");


/*
  cudaEvent_t startMatvec, stopMatvec;
  float MatvecTime;
  cudaEventCreate(&startMatvec);
  cudaEventCreate(&stopMatvec);
  cudaEventRecord(startMatvec,0);
*/

// create the measurements
  A_dct(d_y, d_vec_input, n, m, d_rows);
  SAFEcuda("dct_gpu in createMeasurements_dct");

  free(h_rows); 

 
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



  printf("The function createMeasurements_dct takes %f ms of which %f is the Mat-vc .\n", hostTime, MatvecTime);

*/

  return;
}



