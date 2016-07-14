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
 cout <<"This is gaga_dct."<<endl;
#endif


  if ( (nlhs!=3) && (nlhs!=6)  && ((nrhs!=4) || (nrhs!=5) || (nrhs!=6) || (nrhs!=7) ) )
    printf("[gaga_dct] Error: There are two possible usages for this funtion.\n Four or five (with options) input arguments with six output arguments. \n [norms times iterations support convRate vec_out] = gaga_dct(algstring,k,m,n,options).\n Six or seven (with options) input arguments with three output arguments.\n [outputVector iterations convRate] = gaga_dct(algstring,y,dct_rows,k,m,n,options).\n");
  else {


// reading in the string to determine the algorithm
    int strlen = mxGetN(prhs[0])+1;
    char algstr[strlen+100];
    int algerr = mxGetString(prhs[0], algstr, strlen);

// creating the switch variable defining the algorithm
    int alg;

// check that algstr indicates one of the valid algorithms:
// NIHT, HTP, IHT, ThresholdSD, ThresholdCG, CSMPSP, CGIHT
    int valid_alg = 0;
    if ( (strcmp(algstr, "NIHT")==0) || (strcmp(algstr, "HTP")==0) || (strcmp(algstr, "IHT")==0) || (strcmp(algstr, "ThresholdSD")==0) || (strcmp(algstr, "ThresholdCG")==0) || (strcmp(algstr, "CSMPSP")==0) || (strcmp(algstr, "CGIHT")==0) || (strcmp(algstr, "FIHT")==0) || (strcmp(algstr, "ALPS")==0) || (strcmp(algstr, "CGIHTprojected")==0)) valid_alg = 1;


// possible inputs
    int k, m, n;
    float *h_y;
    int * h_rows;


// possible outputs
    int *total_iter, *checkSupport;
    float *h_norms, *h_times, *convergence_rate, *h_out;

// make variables to store properties of the GPU (device)
    unsigned int max_threads_per_block;
    cudaDeviceProp dp;



    if (valid_alg==0){
      printf("[gaga_dct] Error: The possible (case sensitive) input strings for algorithms using gaga_dct are:\n NIHT\n IHT\n HTP\n ThresholdSD\n ThresholdCG\n CSMPSP\n");
    }
    else {

    if (nlhs == 6){
      k = (int)mxGetScalar(prhs[1]);
      m = (int)mxGetScalar(prhs[2]);
      n = (int)mxGetScalar(prhs[3]);

      plhs[0] = mxCreateNumericMatrix(3, 1, mxSINGLE_CLASS, mxREAL);
      h_norms = (float*) mxGetData(plhs[0]);
    
      plhs[1] = mxCreateNumericMatrix(3, 1, mxSINGLE_CLASS, mxREAL);
      h_times = (float*) mxGetData(plhs[1]);
    
      plhs[2] = mxCreateNumericMatrix(1,1, mxINT32_CLASS, mxREAL);
      total_iter = (int*) mxGetData(plhs[2]);
    
      plhs[3] = mxCreateNumericMatrix(4,1, mxINT32_CLASS, mxREAL);
      checkSupport = (int*) mxGetData(plhs[3]);
    
      plhs[4] = mxCreateNumericMatrix(1,1, mxSINGLE_CLASS, mxREAL);
      convergence_rate = (float*) mxGetData(plhs[4]);
    
      plhs[5] = mxCreateNumericMatrix(n, 1, mxSINGLE_CLASS, mxREAL);
      h_out = (float*) mxGetData(plhs[5]);

    }
    else if (nlhs ==3){
      h_y = (float*)mxGetData(prhs[1]);
      h_rows = (int*)mxGetData(prhs[2]);
      k = (int)mxGetScalar(prhs[3]);
      m = (int)mxGetScalar(prhs[4]);
      n = (int)mxGetScalar(prhs[5]);
    
      plhs[0] = mxCreateNumericMatrix(n, 1, mxSINGLE_CLASS, mxREAL);
      h_out = (float*) mxGetData(plhs[0]);  
    
      plhs[1] = mxCreateNumericMatrix(1,1, mxINT32_CLASS, mxREAL);
      total_iter = (int*) mxGetData(plhs[1]);
    
      plhs[2] = mxCreateNumericMatrix(1,1, mxSINGLE_CLASS, mxREAL);
      convergence_rate = (float*) mxGetData(plhs[2]);
    }
  

// possible options include:
    // tol: specifying convergence rate stopping conditions 
    // 	    (default=0.001)
    // maxiter: maximum number of iterations 
    // 		(default 300 for HTP and CSMPSP, 5000 for others)
    // vecDistribution: distribution of the nonzeros in the sparse vector for the test problem instance 
    // 	   		(default 'binary' indicating random plus or minus 1)
    // seed: seed for random number generator
    //       (default set by clock())
    // numBins: number of bins to use for order statistics
    // 		(default set to max(n/20,1000))
    // threadsPerBlockn: number of threads per block for kernels acting on n vectors
    // 			 (default set to min(n, max_threads_per_block))
    // threadsPerBlockm: number of threads per block for kernels acting on m vectors
    // 			 (default set to min(m, max_threads_per_block))
    // threadsPerBlockBin: number of threads per block for kernels acting on vectors of length numBins
    // 			   (default set to min(num_bins, max_threads_per_block))
    // convRateNum: number of the last iterations to use when calculating average convergence rate
    // 		    (default set to 16)
    // kFixed: flag to force the k used in the problem generate to be that specified
    // 	       (default set to 'off')
    // noise: level of noise as a fraction of the \|Ax\|_2
    // 	      (default set to 0)
    // gpuNumber: which gpu to run the code on
    // 		  (default set to 0)
    // timing: indicates that times per iteration should be recorded
    // 	       (default = 'off')
    // alpha: specifying fraction of k used in early support set identification steps
    // 	      (default set to 0.25)
    // supportFlag: method by which the support set is identified 
    // 		     (default set to 0 for dynamic binning)
    // restartFlag: flag to decide an algorithm should restart some aspect, such as CG in CGIHT
    // 		    (default = "off")
    // projFractol: tolerance for CGIHTprojected

// initialise options at default values
    // some of these may not be used depending on the usage (such as vecDistribution)
    int vecDistribution = 1;  // binary
    int kFixedFlag = 0; // k not fixed
    unsigned int seed = clock();
    int num_bins = max(n/20,1000);
    int gpuNumber = 0;
    int convRateNum = 16;
    float tol = 10^(-4);
    float noise_level = 0.0;
    int timingFlag = 0; // off
    float alpha_start = 0.25;
    int supp_flag = 0;
    int maxiter;
    int restartFlag = 0;
    float projFracTol = 3.0;
    if ( (strcmp(algstr, "HTP")==0) || (strcmp(algstr, "CSMPSP")==0) ) maxiter=300;
    else maxiter=5000;
// unlike other options, the threads_perblock options must be set to default
    // only after the option gpuNumber is determined when checking the options list
    int threads_perblock = 0; // min(n, max_threads_per_block);
    int threads_perblockm = 0; // min(m, max_threads_per_block);
    int threads_perblock_bin = 0; // min(num_bins, max_threads_per_block);
          
// set the gpu device properties in case gpuNumber wasn't an option
    cudaSetDevice(gpuNumber);
    cudaGetDeviceProperties(&dp,gpuNumber);
    max_threads_per_block = dp.maxThreadsPerBlock;

// extract the options if the last input argument is a cell.
    if ( mxIsCell(prhs[nrhs-1]) ){
      // set values for those options that have been specified
      if ( mxGetN(prhs[nrhs-1])==2 ){  // checking that the options list has two columns
        int numOptions = mxGetM(prhs[nrhs-1]);

	int nsubs = 2;
	int index, buflen;
	mxArray  *cell_element_ptr;
        for (int i=0; i< numOptions; i++){
	  // get the index of the i^th row, 1st column of the options cell
          int subs[]={i, 0};
	  index = mxCalcSingleSubscript(prhs[nrhs-1], nsubs, subs);
	  cell_element_ptr = mxGetCell(prhs[nrhs-1], index);
	  buflen = (mxGetM(cell_element_ptr) * 
                    mxGetN(cell_element_ptr)) + 1;
	  char *buf = (char*) malloc(buflen);
      	  algerr = mxGetString(cell_element_ptr, buf, buflen);

	  // go through the list of possible options looking for match with buf
	  subs[1]=1; // move to second column
	  index = mxCalcSingleSubscript(prhs[nrhs-1], nsubs, subs);
    	  cell_element_ptr = mxGetCell(prhs[nrhs-1], index);

	  if (strcmp(buf, "tol")==0){
	    float *p_num = (float*) mxGetData(cell_element_ptr);
	    tol = *p_num; }
	  else if (strcmp(buf, "maxiter")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    maxiter = *p_num; }
	  else if (strcmp(buf, "numBins")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    num_bins = *p_num; }
	  else if (strcmp(buf, "noise")==0){
	    float *p_num = (float*) mxGetData(cell_element_ptr);
	    noise_level = *p_num; }
          else if (strcmp(buf, "projFracTol")==0) {
            float *p_sum = (float*) mxGetData(cell_element_ptr);
            projFracTol = *p_sum; }
	  else if (strcmp(buf, "gpuNumber")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    gpuNumber = *p_num; 
    	    int devCount;
    	    cudaGetDeviceCount(&devCount);
    	    if ((gpuNumber >= devCount) && (gpuNumber != 0)){
      	      cout << "This computer has " << devCount 
      	      	   << " gpus and gpuNumber was" << endl << "selected at "
	   	   << gpuNumber << " which is larger than admissible." 
	   	   << endl << "gpuNumber has been reset to 0." << endl; 
      	      gpuNumber = 0; }
	    cudaSetDevice(gpuNumber);
	    cudaGetDeviceProperties(&dp,gpuNumber);
    	    max_threads_per_block = dp.maxThreadsPerBlock; }
	  else if (strcmp(buf, "threadsPerBlockn")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    threads_perblock = *p_num; }
	  else if (strcmp(buf, "threadsPerBlockm")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    threads_perblockm = *p_num; }
	  else if (strcmp(buf, "threadsPerBlockBin")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    threads_perblock_bin = *p_num; }
	  else if (strcmp(buf, "convRateNum")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    convRateNum = *p_num; }
	  else if (strcmp(buf, "seed")==0){
	    unsigned int *p_num = (unsigned int*) mxGetData(cell_element_ptr);
	    seed = *p_num; }
	  else if (strcmp(buf, "alpha")==0){
	    float *p_num = (float*) mxGetData(cell_element_ptr);
	    alpha_start = *p_num; }
	  else if (strcmp(buf, "supportFlag")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    supp_flag = *p_num; }
	  else if (strcmp(buf, "vecDistribution")==0){
	    int buflen_tmp = (mxGetM(cell_element_ptr) * 
                              mxGetN(cell_element_ptr)) + 1;
	    char *buf_tmp = (char*) malloc(buflen_tmp);
       	    algerr = mxGetString(cell_element_ptr, buf_tmp, buflen_tmp);
	    if (strcmp(buf_tmp, "binary")==0) vecDistribution = 1;
	    else if (strcmp(buf_tmp, "uniform")==0) vecDistribution = 0;
	    else if (strcmp(buf_tmp, "gaussian")==0) vecDistribution = 2;}
	  else if (strcmp(buf, "kFixed")==0){
	    int buflen_tmp = (mxGetM(cell_element_ptr) * 
                              mxGetN(cell_element_ptr)) + 1;
	    char *buf_tmp = (char*) malloc(buflen_tmp);
       	    algerr = mxGetString(cell_element_ptr, buf_tmp, buflen_tmp);
	    if (strcmp(buf_tmp, "on")==0) kFixedFlag = 1;
	    else if (strcmp(buf_tmp, "off")==0) kFixedFlag = 0;}
	  else if (strcmp(buf, "restartFlag")==0){
	    int buflen_tmp = (mxGetM(cell_element_ptr) * 
                              mxGetN(cell_element_ptr)) + 1;
	    char *buf_tmp = (char*) malloc(buflen_tmp);
       	    algerr = mxGetString(cell_element_ptr, buf_tmp, buflen_tmp);
	    if (strcmp(buf_tmp, "on")==0) restartFlag = 1;
	    else if (strcmp(buf_tmp, "off")==0) restartFlag = 0;}
	  else if (strcmp(buf, "timing")==0){
	    int buflen_tmp = (mxGetM(cell_element_ptr) * 
                              mxGetN(cell_element_ptr)) + 1;
	    char *buf_tmp = (char*) malloc(buflen_tmp);
       	    algerr = mxGetString(cell_element_ptr, buf_tmp, buflen_tmp);
	    if (strcmp(buf_tmp, "on")==0) timingFlag = 1;
	    else if (strcmp(buf_tmp, "off")==0) timingFlag = 0;}
	  else{
	    cout << "The following option is not recognised: " << buf << endl;
	  }
	}
      }
    }


// check if any of the threads_perblock variable were not set in the options
    if (threads_perblock == 0) threads_perblock = min(n, max_threads_per_block);
    if (threads_perblockm == 0) threads_perblockm = min(m, max_threads_per_block);
    if (threads_perblock_bin == 0) threads_perblock_bin = min(num_bins, max_threads_per_block);

    float alpha = alpha_start;

// output alert if timing is specified for an algorthm other than NIHT and HTP
   if ( timingFlag == 1 ){
     if ( !((strcmp(algstr, "NIHT")==0) || (strcmp(algstr, "HTP")==0)) )
       cout << "The timing option is only available for NIHT and HTP, using the non-timing variant.\n";
   }

// generate variables for cuda timings
    cudaEvent_t startTest, stopTest;
    cudaEvent_t *p_startTest, *p_stopTest;
    p_startTest = &startTest;
    p_stopTest = &stopTest;
    cudaEventCreate(p_startTest);
    cudaEventCreate(p_stopTest);
    cudaEventRecord(startTest,0);

// Allocate variables on the device
    float * d_vec_input;
    float * d_vec;
    float * d_vec_thres;
    float * grad;
    float * grad_previous;
    float * grad_prev_thres;
    float * d_y;
    float * resid;
    float * resid_update;
    int * d_bin;
    int * d_bin_counters;
    int * d_rows;
    int * d_bin_grad;
    int * d_bin_counters_grad;
//  device variables for FIHT_S_dct and ALPS_S_dct
    float * d_vec_prev;
    float * d_Avec_prev;
    float * d_vec_extra;
    float * d_Avec_extra;

//  device variables for CGIHTprojected
    // grad_prev_thres have been declared
    float * d_vec_diff;
    float * d_p_thres;
    float * d_Ap_thres;

// allocate memory on the device 

    if (nlhs == 6){
      cudaMalloc((void**)&d_vec_input, n * sizeof(float));
      SAFEcudaMalloc("d_vec_input");
    }

    cudaMalloc((void**)&d_vec, n * sizeof(float));
    SAFEcudaMalloc("d_vec");

    cudaMalloc((void**)&d_vec_thres, n * sizeof(float));
    SAFEcudaMalloc("d_vec_thres");

    cudaMalloc((void**)&grad, n * sizeof(float));
    SAFEcudaMalloc("grad");

    cudaMalloc((void**)&d_bin, n * sizeof(int));
    SAFEcudaMalloc("d_bin");
  
    cudaMalloc((void**)&d_y, m * sizeof(float));
    SAFEcudaMalloc("d_y");

    cudaMalloc((void**)&resid, m * sizeof(float));
    SAFEcudaMalloc("resid");

    cudaMalloc((void**)&resid_update, m * sizeof(float));
    SAFEcudaMalloc("resid_update");

    cudaMalloc((void**)&d_rows, m * sizeof(int));
    SAFEcudaMalloc("d_rows");
  
    cudaMalloc((void**)&d_bin_counters, num_bins * sizeof(int));
    SAFEcudaMalloc("d_bin_counters");

    if ((strcmp(algstr, "HTP")==0) || (strcmp(algstr, "ThresholdCG")==0)){
      cudaMalloc((void**)&grad_previous, n * sizeof(float));
      SAFEcudaMalloc("grad_previous");
    }

    if ((strcmp(algstr, "CSMPSP")==0) || (strcmp(algstr, "CGIHT")==0)){
      cudaMalloc((void**)&grad_previous, n * sizeof(float));
      SAFEcudaMalloc("grad_previous");

      cudaMalloc((void**)&d_bin_grad, n * sizeof(int));
      SAFEcudaMalloc("d_bin_grad");

      cudaMalloc((void**)&d_bin_counters_grad, num_bins * sizeof(int));
      SAFEcudaMalloc("d_bin_counters_grad");
    }

    if (strcmp(algstr, "CGIHT")==0 || strcmp(algstr, "CGIHTprojected")==0){
      cudaMalloc((void**)&grad_prev_thres, n * sizeof(float));
      SAFEcudaMalloc("grad_prev_thres");
    }

    if ((strcmp(algstr, "FIHT")==0) || (strcmp(algstr, "ALPS")==0)) {
      cudaMalloc((void**)&d_vec_prev, n * sizeof(float));
      SAFEcudaMalloc("d_vec_prev");

      cudaMalloc((void**)&d_Avec_prev, m * sizeof(float));
      SAFEcudaMalloc("d_Avec_prev");

      cudaMalloc((void**)&d_vec_extra, n * sizeof(float));
      SAFEcudaMalloc("d_vec_extra");
      
      cudaMalloc((void**)&d_Avec_extra, m * sizeof(float));
      SAFEcudaMalloc("d_Avec_extra");      
    }

    if (strcmp(algstr, "CGIHTprojected")==0) {
      cudaMalloc((void**)&d_vec_diff, n * sizeof(float));
      SAFEcudaMalloc("d_vec_diff");

      cudaMalloc((void**)&d_p_thres, n * sizeof(float));
      SAFEcudaMalloc("d_p_thres");

      cudaMalloc((void**)&d_Ap_thres, m * sizeof(float));
      SAFEcudaMalloc("d_Ap_thres");
    }
// allocate memory on the host

    int * h_bin_counters = (int*)malloc( sizeof(int) * num_bins );
    SAFEmalloc_int(h_bin_counters, "h_bin_counters");

    float * residNorm_prev = (float*)malloc( sizeof(float) * convRateNum );
    SAFEmalloc_float(residNorm_prev, "residNorm_prev");

// allocate memory on the host specific for timing set to on

    float *time_per_iteration, *time_supp_set, *cg_per_iteration, *time_for_cg;
    if (timingFlag == 1){
      time_per_iteration = (float*)malloc( sizeof(float) * maxiter );
      SAFEmalloc_float(time_per_iteration, "time_per_iteration");
      time_supp_set = (float*)malloc( sizeof(float) * maxiter );
      SAFEmalloc_float(time_supp_set, "time_supp_set");
      if (strcmp(algstr, "HTP")==0){  // CG info only output for HTP
        cg_per_iteration = (float*)malloc( sizeof(float) * maxiter );
	SAFEmalloc_float(cg_per_iteration, "cg_per_iteration");
	time_for_cg = (float*)malloc( sizeof(float) * maxiter );
	SAFEmalloc_float(time_for_cg, "time_for_cg");
      }
    }




    #ifdef VERBOSE
    if (nlhs == 3) { printf("VERBOSE inactive when passing a problem directly.");}
      float *h_vec_input, *h_vec, *h_vec_thres, *h_grad;
      float *h_resid, *h_resid_update;
    if (nlhs == 6) {
      h_vec_input = (float*)malloc( sizeof(float) * n );
      h_vec = (float*)malloc( sizeof(float) * n );
      h_vec_thres = (float*)malloc( sizeof(float) * n );
      h_grad = (float*)malloc( sizeof(float) * n );
      h_y = (float*)malloc( sizeof(float) * m );
      h_resid = (float*)malloc( sizeof(float) * m );
      h_resid_update = (float*)malloc( sizeof(float) * m );
      h_rows = (int*)malloc( sizeof(int) * m );
    }
    #endif



/*
*****************************************
** Set the kernel execution parameters:                                                  
** For device flexibility, find max threads for current device.                         
** Then use this to determine the different kernel execution configurations you may need.
*****************************************
*/


    dim3 threadsPerBlock(threads_perblock);
    int num_blocks = (int)ceil((float)n/(float)threads_perblock);
    dim3 numBlocks(num_blocks);

    dim3 threadsPerBlockm(threads_perblockm);
    int num_blocksm = (int)ceil((float)m/(float)threads_perblockm);
    dim3 numBlocksm(num_blocksm);

    dim3 threadsPerBlock_bin(threads_perblock_bin);
    int num_blocks_bin = (int)ceil((float)num_bins/(float)threads_perblock_bin);
    dim3 numBlocks_bin(num_blocks_bin);


    dct_gpu_init(n,1);
    SAFEcuda("dct_gpu_init");



/*
*****************************
** CREATE A RANDOM PROBLEM **
*****************************
*/

    #ifdef VERBOSE
    if (verb>3) {printf("Before createProblem, k = %d \n", k);}
    #endif

    if (nlhs ==6){
      if (kFixedFlag==0){
        if (noise_level <= 0){
	  createProblem_dct(&k, m, n, vecDistribution, d_vec_input, d_y, d_rows, &seed);
   	  SAFEcuda("createProblem_dct"); }
	else{ //  when noise_level > 0 
	  createProblem_dct_noise(&k, m, n, vecDistribution, d_vec_input, d_y, d_rows, &seed, noise_level);
 	  SAFEcuda("createProblem_dct_noise"); }
      } else{
        int k_tmp = k;
	do{
	  k = k_tmp;
	  if (noise_level <= 0){
	    createProblem_dct(&k, m, n, vecDistribution, d_vec_input, d_y, d_rows, &seed);
	    if (k != k_tmp)
	      seed = seed +1;
   	    SAFEcuda("createProblem_dct"); }
	  else{ //  when noise_level > 0
	    createProblem_dct_noise(&k, m, n, vecDistribution, d_vec_input, d_y, d_rows, &seed, noise_level);
	    if (k != k_tmp)
	      seed = seed +1;
 	    SAFEcuda("createProblem_dct_noise"); }
	  } while (k_tmp != k);      
      }
    }
    else if (nlhs ==3){
      cudaMemcpy(d_y, h_y, sizeof(float)*m, cudaMemcpyHostToDevice);
      cudaMemcpy(d_rows,h_rows, sizeof(int)*m, cudaMemcpyHostToDevice);
      indexShiftDown<<<numBlocksm,threadsPerBlockm>>>(d_rows,m); 
    }
  


//  ensure initial approximation is the zero vector
    zero_vector_float<<< numBlocks, threadsPerBlock >>>((float*)d_vec, n);
    SAFEcuda("zero_vector_float in random problem");


    #ifdef VERBOSE
    if (nlhs == 6) {
    if (verb>3) {  printf("After createProblem, k = %d \n", k);}
    if (verb>0) {

    printf("The problem size is (k,m,n) = (%d, %d, %d).\n",k, m, n);

    cudaMemcpy(h_rows, d_rows, sizeof(int)*m, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vec_input, d_vec_input, sizeof(float)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vec, d_vec, sizeof(float)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, sizeof(float)*m, cudaMemcpyDeviceToHost);

    printf("Create Problem has provided the following:\n");
    printf("The matrix rows:\n");
    for (int jj=0; jj<min(m,q); jj++){ printf("%d\n ", h_rows[jj]); }
    printf("The initial target vector (x):\n");
    for (int jj=0; jj<min(n,q); jj++){ printf("%f\n", h_vec_input[jj]); }
    printf("The input measurements:\n");
    for (int jj=0; jj<min(m,q); jj++){ printf("%f\n", h_y[jj]); }
    printf("The initial approximation:\n");
    for (int jj=0; jj<min(n,q); jj++){ printf("%f\n", h_vec[jj]); }
    }	
    }
    #endif


/*
*************************************************
** Solve this problem with the input algorithm **
*************************************************
*/


    cudaEvent_t startIHT, stopIHT;
    float timeIHT;
    cudaEventCreate(&startIHT);
    cudaEventCreate(&stopIHT);
    cudaEventRecord(startIHT,0);


// Initialization of parameters and cublas

    int   iter  = 0;
    float mu    = 0;
    float err   = 0;
    int   sum   = 0;

    float time_sum=0.0f;

    if (strcmp(algstr, "NIHT")==0) alg = 0;
    else if (strcmp(algstr, "HTP")==0) alg = 1;
    else if (strcmp(algstr, "IHT")==0) alg = 2;
    else if (strcmp(algstr, "ThresholdSD")==0) alg = 3;
    else if (strcmp(algstr, "ThresholdCG")==0) alg = 4;
    else if (strcmp(algstr, "CSMPSP")==0) alg = 5;
    else if (strcmp(algstr, "CGIHT")==0) alg = 6;
    else if (strcmp(algstr, "FIHT")==0) alg = 7;
    else if (strcmp(algstr, "ALPS")==0) alg = 8;
    else if (strcmp(algstr, "CGIHTprojected")==0) alg = 9;

    switch (alg) {
	case 0:
	   if (timingFlag == 0){
		NIHT_S_dct(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  		SAFEcuda("NIHT_S_dct in gaga_dct");}
	   else {
	   	NIHT_S_timings_dct(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, &iter, mu, err, &sum, alpha, supp_flag, time_per_iteration, time_supp_set, &time_sum,  numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("NIHT_S_timings_dct in gaga_dct");}
	   break;
	case 1:
	   if (timingFlag == 0){
		HTP_S_dct(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  		SAFEcuda("HTP_S_dct in gaga_dct");}
	   else {
	   	HTP_S_timings_dct(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, &iter, mu, err, &sum, alpha, supp_flag, time_per_iteration, time_supp_set, cg_per_iteration, time_for_cg, &time_sum,  numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("HTP_S_timings_dct in gaga_dct");}
	   break;
	case 2:
		mu = 0.65f;
		IHT_S_dct(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  		SAFEcuda("IHT_S_dct in gaga_dct");
  		break;
	case 3:
		HT_SD_S_dct(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  		SAFEcuda("HT_SD_S_dct in gaga_dct");
		break;
	case 4:
		HT_CG_S_dct(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  		SAFEcuda("HT_CG_S_dct in gaga_dct");
		break;
	case 5:
		CSMPSP_S_dct(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, d_bin_grad, d_bin_counters_grad, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  		SAFEcuda("CSMPSP_S_dct in gaga_dct");
		break;
	case 6:
		CGIHT_S_dct(d_vec, d_vec_thres, grad, grad_previous, grad_prev_thres, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, d_bin_grad, d_bin_counters_grad, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, restartFlag, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
  		SAFEcuda("CGIHT_S_dct in gaga_dct");
		break;
        case 7: 
                FIHT_S_dct(d_vec,  d_vec_thres,  grad, d_vec_prev, d_Avec_prev, d_vec_extra, d_Avec_extra, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, 
tol, maxiter, num_bins, k, m, n, &iter, mu, err, &sum, &time_sum, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
                SAFEcuda("FIHT_S_dct in gaga_dct");
                break;
        case 8: 
                ALPS_S_dct(d_vec,  d_vec_thres,  grad, d_vec_prev, d_Avec_prev, d_vec_extra, d_Avec_extra, d_y, resid, resid_update, d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, 
tol, maxiter, num_bins, k, m, n, &iter, mu, err, &sum, &time_sum, numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
                SAFEcuda("ALPS_S_dct in gaga_dct");
                break;
        case 9:
                CGIHTprojected_S_dct(d_vec, d_vec_thres, grad, grad_prev_thres, d_p_thres, d_Ap_thres, d_vec_diff, d_y, resid, resid_update,  d_rows, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, 
tol, maxiter, projFracTol, num_bins,  k, m,  n, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocks_bin, threadsPerBlock_bin);
                SAFEcuda("CGIHTprojected_S_dct in gaga_dct");
                break;
	default:
		printf("[gaga_dct] Error: The possible input strings for algorithms using gaga_dct are:\n NIHT\n IHT\n HTP\n ThresholdSD\n ThresholdCG\n CSMPSP\n CGIHT\n");
		break;
    }

  
    cudaThreadSynchronize();
    cudaEventRecord(stopIHT,0);
    cudaEventSynchronize(stopIHT);
    cudaEventElapsedTime(&timeIHT, startIHT, stopIHT);
    cudaEventDestroy(startIHT);
    cudaEventDestroy(stopIHT);


/*
***********************
** Check the Results **
***********************
*/

    if ( (strcmp(algstr, "CGIHT")==0) && (restartFlag==1) ) {
      strcat(algstr,"restarted");
    }

// some CPU action is needed before the results
// some output is needed before the results.

    if (nlhs == 6){
      if ( ((timingFlag == 0) || ((alg != 0) && (alg != 1)) ) && (noise_level <= 0) ){
        results(d_vec, d_vec_input, vecDistribution, residNorm_prev, h_norms, h_out, h_times, convergence_rate, total_iter, checkSupport, iter, timeIHT, time_sum, &sum, k, m, n, seed, p_startTest, p_stopTest, algstr, numBlocks, threadsPerBlock);
	SAFEcuda("gaga_dct: results_dct"); }
      else if ( ((timingFlag == 0) || ((alg != 0) && (alg != 1)) ) && (noise_level > 0) ) {
        results_dct_noise(d_vec, d_vec_input, vecDistribution, residNorm_prev, h_norms, h_out, h_times, convergence_rate, total_iter, checkSupport, iter, timeIHT, time_sum, &sum, k, m, n, seed, noise_level, p_startTest, p_stopTest, algstr, numBlocks, threadsPerBlock);
 	SAFEcuda("gaga_dct_noise: results_dct_noise"); }
      else if ( (timingFlag == 1) && (alg == 0) ){
        results_timings(d_vec, d_vec_input, vecDistribution, residNorm_prev, h_norms, h_out, h_times, convergence_rate, total_iter, checkSupport, iter, timeIHT, time_sum, time_per_iteration, time_supp_set, &sum, alpha_start, supp_flag, k, m, n, seed, p_startTest, p_stopTest, algstr, numBlocks, threadsPerBlock);
	SAFEcuda("results_timings in gaga_dct"); }
      else if ( (timingFlag == 1) && (alg == 1) ){
        results_timings_HTP(d_vec, d_vec_input, vecDistribution, residNorm_prev, h_norms, h_out, h_times, convergence_rate, total_iter, checkSupport, iter, timeIHT, time_sum, time_per_iteration, time_supp_set, cg_per_iteration, time_for_cg, &sum, alpha_start, supp_flag, k, m, n, seed, p_startTest, p_stopTest, algstr, numBlocks, threadsPerBlock);
	SAFEcuda("results_timings_HTP in gaga_dct_timings"); }
    }
    else if (nlhs == 3){
      cudaMemcpy(h_out, d_vec, sizeof(float)*n, cudaMemcpyDeviceToHost);

      total_iter[0] = iter;

      float convRate, root;
      int temp = min(iter, convRateNum);
      root = 1/(float)temp;
      temp=convRateNum-temp;
      convRate = (residNorm_prev[convRateNum-1]/residNorm_prev[temp]);
      convRate = pow(convRate, root);

      convergence_rate[0]=convRate;
    }


    #ifdef VERBOSE
    if (nlhs == 6){
      printf("Results:\n");
      printf("l2 error = %f\n", h_norms[0]);
      printf("l1 error = %f\n", h_norms[1]);
      printf("l-infinty error = %f\n", h_norms[2]);
      printf("ALG Time = %f ms.\n", h_times[0]);
      printf("Average Iteration Time = %f ms.\n", h_times[1]);
      printf("Total Time (including problem generation) = %f ms.\n", h_times[2]);
      printf("Total iterations = %d\n", iter);
      printf("Support Set identification:\n");
      printf("\t True Positive = %d\n", checkSupport[0]);
      printf("\t False Positive = %d\n", checkSupport[1]);
      printf("\t True Negative = %d\n", checkSupport[2]);
      printf("\t False Negative = %d\n", checkSupport[3]);
      printf("Convergence Rate = %f\n", convergence_rate[0]);
    }
    #endif 




/*
**************
** CLEANUP  **
**************
*/


// free up the allocated memory on the device

    if (nlhs == 6) {cudaFree(d_vec_input);}
      cudaFree(d_vec);
      cudaFree(d_vec_thres);
      cudaFree(d_bin);
      cudaFree(d_y);
      cudaFree(d_rows);
      cudaFree(d_bin_counters);
      cudaFree(resid);
      cudaFree(resid_update);
      cudaFree(grad);
      if ((alg==1) || (alg==4)){
	cudaFree(grad_previous);
      }
      if ( (alg == 5) || (alg == 6) ){
	cudaFree(grad_previous);
  	cudaFree(d_bin_grad);
	cudaFree(d_bin_counters_grad);
      }
      if ((alg == 6) || (alg == 9)){
        cudaFree(grad_prev_thres);
      }
      if ((alg == 7) || (alg == 8)) {
        cudaFree(d_vec_prev);
        cudaFree(d_Avec_prev);
        cudaFree(d_vec_extra);
        cudaFree(d_Avec_extra);
      }

      if (alg == 9) {
        cudaFree(d_vec_diff);
        cudaFree(d_p_thres);
        cudaFree(d_Ap_thres); 
      }
      SAFEcuda("cudaFree in gaga_dct");



      dct_gpu_destroy();
      SAFEcuda("dct_gpu_destroy in gaga_dct");

      cublasShutdown();
      SAFEcublas("cublasShutdown in gaga_dct");


// free the memory on the host

    free(h_bin_counters);
    free(residNorm_prev);
    if (timingFlag == 1){
      free(time_per_iteration);
      free(time_supp_set);
      if (alg == 1){  // CG info only output for HTP
        free(cg_per_iteration);
	free(time_for_cg);
      }
    }



    #ifdef VERBOSE
    if (nlhs == 6) {
      free(h_vec_input);
      free(h_vec);
      free(h_vec_thres);
      free(h_grad);
      free(h_y);
      free(h_resid);
      free(h_resid_update);
      free(h_rows);
    }
    #endif

  }  //closes the else ensuring the algorithm input was valid

  }  //closes the else ensuring a correct number of input and output arguments

  return;
}



