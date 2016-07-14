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




#include "greedyheader.cu"


/*
*********** VERBOSE or SAFE ***************
**     IF you want verbose or safe,      **
**     change it in greedyheader.cu      **
*******************************************
*/



// Host function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

#ifdef VERBOSE
 cout <<"This is create_dct_noise."<<endl;
#endif



// inputs 
    int k = (int)mxGetScalar(prhs[0]);
    int m = (int)mxGetScalar(prhs[1]);
    int n = (int)mxGetScalar(prhs[2]);
    int vecDistribution = (int)mxGetScalar(prhs[3]);
    float noise_level = (float)mxGetScalar(prhs[4]);



// outputs    

  float *h_vec_input;
  float *h_y;
  int *h_rows;

    plhs[0] = mxCreateNumericMatrix(n, 1, mxSINGLE_CLASS, mxREAL);
    h_vec_input = (float*) mxGetData(plhs[0]);

    plhs[1] = mxCreateNumericMatrix(m, 1, mxSINGLE_CLASS, mxREAL);
    h_y = (float*) mxGetData(plhs[1]);

    plhs[2] = mxCreateNumericMatrix(m, 1, mxINT32_CLASS, mxREAL);
    h_rows = (int*) mxGetData(plhs[2]);

// allocate device memory

  float *d_vec_input;
  float *d_y;
  int *d_rows;

  cudaMalloc((void**)&d_vec_input, n * sizeof(float));
  SAFEcudaMalloc("d_vec_input");

  cudaMalloc((void**)&d_y, m * sizeof(float));
  SAFEcudaMalloc("d_y");

  cudaMalloc((void**)&d_rows, m * sizeof(int));
  SAFEcudaMalloc("d_rows");

// create seed

  unsigned int seed = clock();


// initialize dct
  dct_gpu_init(n,1);
  SAFEcuda("dct_gpu_init");

// run create problem

  createProblem_dct_noise(&k, m, n, vecDistribution, d_vec_input, d_y, d_rows, &seed, noise_level);
  SAFEcuda("createProblem");

printf("\nGot back to main from createProblem_dct_noise.\n");

// copy everything to the host
  cudaMemcpy(h_rows, d_rows, sizeof(int)*m, cudaMemcpyDeviceToHost);
  SAFEcuda("createProblem copy h_rows");
  cudaMemcpy(h_vec_input, d_vec_input, sizeof(float)*n, cudaMemcpyDeviceToHost);
  SAFEcuda("createProblem copy h_vec_input");
  cudaMemcpy(h_y, d_y, sizeof(float)*m, cudaMemcpyDeviceToHost);
  SAFEcuda("createProblem copy h_y");


  int q=8;
  printf("Create Problem has provided the following:\n");
  printf("The matrix rows:\n");
  for (int jj=0; jj<min(m,q); jj++){ printf("%d\n ", h_rows[jj]); }
  printf("The initial target vector (x):\n");
  for (int jj=0; jj<min(n,q); jj++){ printf("%f\n", h_vec_input[jj]); }
  printf("The input measurements:\n");
  for (int jj=0; jj<min(m,q); jj++){ printf("%f\n", h_y[jj]); }


printf("\nJust after cudaMemcpy\n");
// free up the allocated memory on the device

  cudaFree(d_vec_input);
  cudaFree(d_y);
  cudaFree(d_rows);

printf("Just after cudaFree\n");
  return;
}

