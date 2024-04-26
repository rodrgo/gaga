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
 cout <<"This is gpuPartialSVD_SPI"<<endl;
#endif


  if ( (nlhs!=4) && (nrhs!=7) )
    printf("[gpuPartialSVD_SPI] Error: This function takes seven input arguments:\n Mat (the matrix), m (rows of Mat), n (#cols of Mat), r (# desired singular vectors/values), U0 (initial guess for U), Tol (accuracy tolerance for power iteration), maxIter (maximum iterations per power iteration).\n  This function returns four output arguments:\n U (r left singular vectors), S (r singular values), V (r right singular vectors), Timings (total time with and without memory allocation/free.\n  Use as: [U,S,V,Timings]=gpuPartialSVD(Mat,m,n,r, U0, Tol,maxIter);\n");
  else {
	
    // start timing of entire function
    cudaEvent_t startTotalTime, stopTotalTime;
    float timeTotal;
    cudaEventCreate(&startTotalTime);
    cudaEventCreate(&stopTotalTime);
    cudaEventRecord(startTotalTime,0);

    // possible inputs
    float *h_Mat, *h_U0, Tol;
    int m, n, r, maxIter;

    // possible outputs
    float *h_U, *h_S, *h_V, *h_timeOut;

    // read input variables
    h_Mat = (float*)mxGetData(prhs[0]);
    m = (int)mxGetScalar(prhs[1]);
    n = (int)mxGetScalar(prhs[2]);
    r = (int)mxGetScalar(prhs[3]);
    h_U0 = (float*)mxGetData(prhs[4]);
    Tol = (float)mxGetScalar(prhs[5]);
    maxIter = (int)mxGetScalar(prhs[6]);

    // check input validity
    int m_check = (int)mxGetM(prhs[0]);
    int n_check = (int)mxGetN(prhs[0]);
    if ( ( m != m_check ) || ( n != n_check ) ){
      printf("[gpuPartialSVD] Error: The dimensions of the input matrix do not match the dimensions provided as input arguments.\n");
    }

    m_check = (int)mxGetM(prhs[4]);
    int r_check = (int)mxGetN(prhs[4]);
    if ( ( m != m_check ) || ( r != r_check ) ){
      printf("[gpuPartialSVD] Error: The dimensions of the initial guess U0 do not match the dimensions provided as input arguments.\n");
    }

    // create output variables
    plhs[0] = mxCreateNumericMatrix(m, r, mxSINGLE_CLASS, mxREAL);
    h_U = (float*) mxGetData(plhs[0]);

    plhs[1] = mxCreateNumericMatrix(r, 1, mxSINGLE_CLASS, mxREAL);
    h_S = (float*) mxGetData(plhs[1]);

    

    plhs[2] = mxCreateNumericMatrix(n, r, mxSINGLE_CLASS, mxREAL);
    h_V = (float*) mxGetData(plhs[2]);

    plhs[3] = mxCreateNumericMatrix(2, 1, mxSINGLE_CLASS, mxREAL);
    h_timeOut = (float*) mxGetData(plhs[3]);

    // create some additional dimension variables
    int mr = m * r;
    int nr = n * r;
    int mn = m * n;

    // create and allocate device variables
    float *d_Mat, *d_U0, *d_U,  *d_V, *d_A, *d_R;

    cudaMalloc((void**)&d_Mat, mn * sizeof(float));
    SAFEcudaMalloc("d_Mat");

    cudaMalloc((void**)&d_U0,  mr * sizeof(float));
    SAFEcudaMalloc("d_U0");

    cudaMalloc((void**)&d_U, mr * sizeof(float));
    SAFEcudaMalloc("d_U");

    cudaMalloc((void**)&d_R, r * sizeof(float));
    SAFEcudaMalloc("d_R");

    cudaMalloc((void**)&d_V, nr * sizeof(float));
    SAFEcudaMalloc("d_V");

    cudaMalloc((void**)&d_A, (m*m)*sizeof(float));
    SAFEcudaMalloc("d_A");


    // fill d_Mat with input matrix
    cudaMemcpy(d_Mat, h_Mat, mn*sizeof(float), cudaMemcpyHostToDevice);
		SAFEcuda("cudaMemcpy d_Mat");
    // fill d_U0 with input initial guess
    cudaMemcpy(d_U0, h_U0, mr*sizeof(float), cudaMemcpyHostToDevice);
    SAFEcuda("cudaMemcpy d_U0");


    float *h_S_prev;
    h_S_prev = new float[r];
    
    
    cublasInit();
		cublasHandle_t handle;
		cublasCreate(&handle);
    SAFEcublas("cublasInit");

    // start timing of PSVD only
    cudaEvent_t startPSVD, stopPSVD;
    float timePSVD;
    cudaEventCreate(&startPSVD);
    cudaEventCreate(&stopPSVD);
    cudaEventRecord(startPSVD,0);

    // execute the partial svd via power iteration
    PartialSVD_SPI(d_U, h_S, d_V, d_Mat, d_A, d_U0, h_S_prev, d_R, m, n, r, maxIter, Tol, handle);
		SAFEcuda("PartialSVD_SPI");

		// record time for PSVD
    cudaThreadSynchronize();
    cudaEventRecord(stopPSVD,0);
    cudaEventSynchronize(stopPSVD);
    cudaEventElapsedTime(&timePSVD, startPSVD, stopPSVD);
    cudaEventDestroy(startPSVD);
    cudaEventDestroy(stopPSVD);

    h_timeOut[1] = timePSVD;

    // copy results to host     
    cudaMemcpy(h_U, d_U, mr*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V, d_V, nr*sizeof(float), cudaMemcpyDeviceToHost);
		SAFEcuda("Copy U and V");
    //cudaMemcpy(h_S, d_S, r*sizeof(float), cudaMemcpyDeviceToHost);




    

    // free device variables, shutcown cublas, destroy curand generator
    cudaFree(d_Mat);
    cudaFree(d_U0);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_A);
    //cudaFree(d_S_prev);
    cudaFree(d_R);
		SAFEcuda("cudaFree");
		free(h_S_prev);

		cublasDestroy(handle);
    cublasShutdown();
    SAFEcublas("cublasShutdown");



    // record total time
    cudaThreadSynchronize();
    cudaEventRecord(stopTotalTime,0);
    cudaEventSynchronize(stopTotalTime);
    cudaEventElapsedTime(&timeTotal, startTotalTime, stopTotalTime);
    cudaEventDestroy(startTotalTime);
    cudaEventDestroy(stopTotalTime);

    h_timeOut[0] = timeTotal;

  } // ends the else of if ((nlhs!=4 && (nrhs!=6)) else {
  return;
}



