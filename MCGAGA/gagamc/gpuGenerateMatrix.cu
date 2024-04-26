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
 cout <<"This is gpuGenerateMatrix."<<endl;
#endif


  if ( (nlhs!=2) && (nrhs!=4) )
    printf("[gpuGenerateMatrix] Error: This functions takes 4 input arguments: m (number of rows), n (number of columns), r (rank), ens (type of random data used to generate matrix).  This function returns two output arguments: Mat (the mxn matrix), TimeElapsed (the elapsed time).  Use the function as: \n [Mat, TimeElapsed] = gpuGenerateMatrix(m,n,r,ens)\n");
  else {

// possible inputs
    int r, m, n, ens;

// possible outputs
    float *h_Mat, *h_timeOut;


    m = (int)mxGetScalar(prhs[0]);
    n = (int)mxGetScalar(prhs[1]);
    r = (int)mxGetScalar(prhs[2]);
    ens = (int)mxGetScalar(prhs[3]);
   

    plhs[0] = mxCreateNumericMatrix(m, n, mxSINGLE_CLASS, mxREAL);
    h_Mat = (float*) mxGetData(plhs[0]);

    plhs[1] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    h_timeOut = (float*) mxGetData(plhs[1]);

// size of Mat
    int mn = m*n;

// generate variables for cuda timings
    cudaEvent_t startCreateMAT, stopCreateMAT;
    float timeCreateMAT;
    cudaEventCreate(&startCreateMAT);
    cudaEventCreate(&stopCreateMAT);
    cudaEventRecord(startCreateMAT,0);

// Allocate variables on the device
    float * d_Mat;

    cudaMalloc((void**)&d_Mat, mn * sizeof(float));
    SAFEcudaMalloc("d_vec");


// allocate memory on the host 
//      h_Mat = (float*)malloc( sizeof(float) * mn );

// define seed    
    unsigned int seed = clock();


    cublasInit();
    SAFEcublas("cublasInit");

    createDataMatrix(m, n, r, d_Mat, ens, &seed);
    SAFEcuda("createDataMatrix"); 

    cudaMemcpy(h_Mat, d_Mat, sizeof(float)*mn, cudaMemcpyDeviceToHost);

    cudaFree(d_Mat);

    cublasShutdown();
    SAFEcublas("cublasShutdown");

    cudaThreadSynchronize();
    cudaEventRecord(stopCreateMAT,0);
    cudaEventSynchronize(stopCreateMAT);
    cudaEventElapsedTime(&timeCreateMAT, startCreateMAT, stopCreateMAT);
    cudaEventDestroy(startCreateMAT);
    cudaEventDestroy(stopCreateMAT);

    *h_timeOut = timeCreateMAT;

  } // ends the else of if ((nlhs!=1) && (nrhs!=4)) else {
  return;
}



