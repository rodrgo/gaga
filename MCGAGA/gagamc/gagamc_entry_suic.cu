/* Copyright 2014 Jeffrey D. Blanchard and Jared Tanner
 *   
 * GPU Accelerated Greedy Algorithms for Matrix Completion
 *
 * Licensed under the GAGAMC License available at gaga4cs.org and included as GAGAMC_license.txt.
 *
 * In  order to use the GAGAMC library, or any of its constituent parts, a user must
 * agree to abide by a set of conditions of use. The library is available at no cost 
 * for ``Internal'' use. ``Internal'' use of the library is defined to be use of the 
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
 * and include a copy of GAGAMC_license.txt.
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
 cout <<"This is gagamc_entry_suic."<<endl;
#endif
//cudaDeviceReset();
//SAFEcuda("DeviceReset");

  if ( (nlhs!=3)  && (nlhs!=7)  )
    printf("[gaga_gen] Error: There are two possible usages for this funtion.\n Five or six (with options) input arguments with six output arguments. \n [norms times iterations convRate MatOut MatInput A] = gagamc_entry(algstring,k,m,n,options).\n Five or six (with options) input arguments with three output arguments.\n [outputMatrix iterations convRate] = gagamc_entry(algstring,y,A,k,options).\n");
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
    if ( (strcmp(algstr, "NIHT")==0) || (strcmp(algstr, "SVT")==0) || (strcmp(algstr, "CGIHT")==0) || (strcmp(algstr, "LMaFit")==0)) valid_alg = 1;


// possible inputs
    int m, n, r, p;    // problem parameters p measurements of a rank r matrix of size m x n.
    float *h_Y, *h_y;
    int *h_A_in;          //  problem information with entry measurements in matrix and vector format, plust the entry sensing operator A 

// possible outputs
    int *h_A_out, *total_iter;
    float *h_norms, *h_times, *convergence_rate, *h_Mat_out, *h_Mat_input;

// control parameter from inputs
    int mn, mr, nr;

// make variables to store properties of the GPU (device)
    unsigned int max_threads_per_block;
    cudaDeviceProp dp;
    


    if (valid_alg == 0){
      printf("[gaga_gen] Error: The possible input strings for algorithms using gaga_gen are:\n NIHT \n CGIHT \n SVT\n LMaFit \n");
    }
    else {

      if (nlhs == 7){
        m = (int)mxGetScalar(prhs[1]);
   	n = (int)mxGetScalar(prhs[2]);
   	r = (int)mxGetScalar(prhs[3]);
   	p = (int)mxGetScalar(prhs[4]);
   
				mn = m * n;
        mr = m * r;
        nr = n * r;

   	plhs[0] = mxCreateNumericMatrix(3, 1, mxSINGLE_CLASS, mxREAL);
   	h_norms = (float*) mxGetData(plhs[0]);

 	plhs[1] = mxCreateNumericMatrix(3, 1, mxSINGLE_CLASS, mxREAL);
   	h_times = (float*) mxGetData(plhs[1]);

 	plhs[2] = mxCreateNumericMatrix(1,1, mxINT32_CLASS, mxREAL);
   	total_iter = (int*) mxGetData(plhs[2]);

	plhs[3] = mxCreateNumericMatrix(1,1, mxSINGLE_CLASS, mxREAL);
   	convergence_rate = (float*) mxGetData(plhs[3]);

  	plhs[4] = mxCreateNumericMatrix(m, n, mxSINGLE_CLASS, mxREAL);
   	h_Mat_out = (float*) mxGetData(plhs[4]);

  	plhs[5] = mxCreateNumericMatrix(m, n, mxSINGLE_CLASS, mxREAL);
   	h_Mat_input = (float*) mxGetData(plhs[5]);

  	plhs[6] = mxCreateNumericMatrix(p, 1, mxINT32_CLASS, mxREAL);
   	h_A_out = (int*) mxGetData(plhs[6]);
      } // ends if (nlhs == 7)
      else if (nlhs == 3){

        h_Y = (float*)mxGetData(prhs[1]);
        h_y = (float*)mxGetData(prhs[2]);
        h_A_in = (int*)mxGetData(prhs[3]);
        r = (int)mxGetScalar(prhs[4]);

        m = (int)mxGetM(prhs[1]);
        n = (int)mxGetN(prhs[1]);
        p = (int)mxGetM(prhs[2]);

        mn = m * n;
        mr = m * r;
        nr = n * r;

    
        plhs[0] = mxCreateNumericMatrix(m, n, mxSINGLE_CLASS, mxREAL);
        h_Mat_out = (float*) mxGetData(plhs[0]);  
    
        plhs[1] = mxCreateNumericMatrix(1,1, mxINT32_CLASS, mxREAL);
        total_iter = (int*) mxGetData(plhs[1]);
    
        plhs[2] = mxCreateNumericMatrix(1,1, mxSINGLE_CLASS, mxREAL);
        convergence_rate = (float*) mxGetData(plhs[2]);
      }  // ends else if (nlhs == 3)


// possible options include:
    // tol: specifying convergence rate stopping conditions 
    // 	    (default=0.001)
    // maxiter: maximum number of iterations 
    // 		(default 300 for HTP and CSMPSP, 5000 for others)
    // vecDistribution: distribution of the nonzeros in the sparse vector for the test problem instance 
    // 	   		(default 'binary' indicating random plus or minus 1)
    // matrixEnsemble: distribution of the nonzeros in the measurement matrix for the test problem
    // 		       (default 'gaussian' indicating normal N(0,1))
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
    // projFracTol: tolerence for CGIHTprojected (default == 3);

// initialise options at default values
    // some of these may not be used depending on the usage (such as vecDistribution)
//    int vecDistribution = 1;  // binary
    int matrixEnsemble = 1;  // gaussian
//    int kFixedFlag = 0; // k not fixed
    unsigned int seed = clock();
//    int num_bins = max(n/20,1000);
    int gpuNumber = 1;
    int convRateNum = 16;
    float tol = 10^(-4);
    float PSVDtol = 0.01;
//    float noise_level = 0.0;
//    int timingFlag = 0; // off
//    float alpha_start = 0.25;
 //   int supp_flag = 0;
    int maxiter=50;
    int PSVDmaxiter = 15; 
    int restartFlag = 0; 
    float projFracTol = 3.0;
    if ( (strcmp(algstr, "HTP")==0) || (strcmp(algstr, "CSMPSP")==0) ) maxiter=300;
    else maxiter=5000;
// unlike other options, the threads_perblock options must be set to default
    // only after the option gpuNumber is determined when checking the options list
    int threads_perblockmn = 0; // min(n, max_threads_per_block);
    int threads_perblockm = 0; // min(m, max_threads_per_block);
    int threads_perblockp = 0; // min(p, max_threads_per_block);
    int threads_perblocknr = 0; // min(nr, max_threads_per_block);
		int threads_perblockmr = 0; //added by Chenshuai Sui
          
// set the gpu device properties in  case gpuNumber wasn't an option
    
    cudaSetDevice(gpuNumber);
    SAFEcuda("cudaSetDevice");
    cudaGetDeviceProperties(&dp,gpuNumber);
    SAFEcuda("cudaGetDeviceProperties");
    
    max_threads_per_block = dp.maxThreadsPerBlock;
		//printf("maxthreadsperblock = %i \n",  max_threads_per_block);
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
	  else if (strcmp(buf, "PSVDtol")==0){
	    float *p_num = (float*) mxGetData(cell_element_ptr);
	    PSVDtol = *p_num; }
	  else if (strcmp(buf, "PSVDmaxiter")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    PSVDmaxiter = *p_num; }
          else if (strcmp(buf, "projFracTol")==0) { 
            float *p_num = (float*) mxGetData(cell_element_ptr);
            projFracTol = *p_num; }
	  else if (strcmp(buf, "gpuNumber")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    gpuNumber = *p_num; 
	    int devCount;
	    cudaGetDeviceCount(&devCount);
            SAFEcuda("DeviceCount");
   	    if ((gpuNumber >= devCount) && (gpuNumber != 0)){
     	      cout << "This computer has " << devCount 
      	      	   << " gpus and gpuNumber was" << endl << "selected at " 
	   	   << gpuNumber << " which is larger than admissible." 
	   	   << endl << "gpuNumber has been reset to 0." << endl; 
     	      gpuNumber = 0; }
	    cudaSetDevice(gpuNumber);
            SAFEcuda("SetDevice");
	    cudaGetDeviceProperties(&dp,gpuNumber);
            SAFEcuda("GetDeviceProperties");
    	    max_threads_per_block = dp.maxThreadsPerBlock; }
	  else if (strcmp(buf, "threadsPerBlockmn")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    threads_perblockmn = *p_num; }
	  else if (strcmp(buf, "threadsPerBlockm")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    threads_perblockm = *p_num; }
	  else if (strcmp(buf, "threadsPerBlockp")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    threads_perblockp = *p_num; }
	  else if (strcmp(buf, "threadsPerBlocknr")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    threads_perblocknr = *p_num; }
	  else if (strcmp(buf, "convRateNum")==0){
	    int *p_num = (int*) mxGetData(cell_element_ptr);
	    convRateNum = *p_num; }
	  else if (strcmp(buf, "seed")==0){
	    unsigned int *p_num = (unsigned int*) mxGetData(cell_element_ptr);
	    seed = *p_num; }
/*
	  else if (strcmp(buf, "noise")==0){
	    float *p_num = (float*) mxGetData(cell_element_ptr);
	    noise_level = *p_num; }
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
	  else if (strcmp(buf, "timing")==0){
	    int buflen_tmp = (mxGetM(cell_element_ptr) * 
                              mxGetN(cell_element_ptr)) + 1;
	    char *buf_tmp = (char*) malloc(buflen_tmp);
       	    algerr = mxGetString(cell_element_ptr, buf_tmp, buflen_tmp);
	    if (strcmp(buf_tmp, "on")==0) timingFlag = 1;
	    else if (strcmp(buf_tmp, "off")==0) timingFlag = 0;}
*/
	  else if (strcmp(buf, "restartFlag")==0){
	    int buflen_tmp = (mxGetM(cell_element_ptr) * 
                              mxGetN(cell_element_ptr)) + 1;
	    char *buf_tmp = (char*) malloc(buflen_tmp);
       	    algerr = mxGetString(cell_element_ptr, buf_tmp, buflen_tmp);
	    if (strcmp(buf_tmp, "on")==0) restartFlag = 1;
	    else if (strcmp(buf_tmp, "off")==0) restartFlag = 0;}
	  else if (strcmp(buf, "matrixEnsemble")==0){
	    int buflen_tmp = (mxGetM(cell_element_ptr) * 
                              mxGetN(cell_element_ptr)) + 1;
	    char *buf_tmp = (char*) malloc(buflen_tmp);
       	    algerr = mxGetString(cell_element_ptr, buf_tmp, buflen_tmp);
	    if (strcmp(buf_tmp, "binary")==0) matrixEnsemble = 2;
	    else if (strcmp(buf_tmp, "gaussian")==0) matrixEnsemble = 1;
	    else cout << "Admissible matrixEnsemble for gen are: binary and gaussian.\n";}
	  else{
	    cout << "The following option is not recognised: " << buf << endl;
	  }
	}
      }
    }



// check if any of the threads_perblock variable were not set in the options
    if (threads_perblockmn == 0) threads_perblockmn = min(mn, max_threads_per_block);
    if (threads_perblockm == 0) threads_perblockm = min(m, max_threads_per_block);
    if (threads_perblockp == 0) threads_perblockp = min(p, max_threads_per_block);
    if (threads_perblocknr == 0) threads_perblocknr = min(nr, max_threads_per_block);
		if (threads_perblockmr == 0) threads_perblockmr = min(mr, max_threads_per_block);  //added by Chenshuai Sui
/*
// output alert if timing is specified for an algorthm other than NIHT and HTP
   if ( timingFlag == 1 ){
     if ( !((strcmp(algstr, "NIHT")==0) || (strcmp(algstr, "HTP")==0)) )
       cout << "The timing option is only available for NIHT and HTP, using the non-timing variant.\n";
   }
*/
// generate variables for cuda timings
    cudaEvent_t startTest, stopTest;
    cudaEvent_t *p_startTest, *p_stopTest;
    p_startTest = &startTest;
    p_stopTest = &stopTest;
    cudaEventCreate(p_startTest);
    SAFEcuda("EventCreate p_startTest");
    cudaEventCreate(p_stopTest);
    SAFEcuda("EventCreate p_stopTest");
    cudaEventRecord(startTest,0);
    SAFEcuda("EventRecord startTest");

// establish a generator and seed for cuRand to use in problem creation and partial SVD
    curandGenerator_t gen;
    curandStatus_t curandCheck;
    curandCheck = curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    SAFEcurand(curandCheck, "curandCreateGenerator in gagamc_entry_suic");
    // note, CURAND_RNG_PSEUDO_DEFAULT selects the random number generator type
    curandCheck = curandSetPseudoRandomGeneratorSeed(gen,seed);
    SAFEcurand(curandCheck, "curandSet...Seed in gagamc_entry_suic"); 



// Allocate variables on the device
    float * d_Mat;
    float * Grad;
    float * Grad_proj;
    float * Grad_prev;
    float * Grad_prev_proj;
    float * d_Y;
    float * d_U;
    float * d_S;
    float * d_V;
    float * d_y;
    float * d_y_work;
    float * d_u;
    float * d_u_prev;
    float * d_v;
    int * d_A;
    float * d_U_prev;		//needed in PartialSVD_SPI
    float * d_MM;				//needed in PartialSVD_SPI
    float *d_R;					//needed in PartialSVD_SPI
		float *d_U_init;  // needed in SVT
		float *d_V_prev;	//needed in LMaFit
		float *d_RR;			//needed in LMaFit
		float *d_SS;				//needed in LMaFit


		float *h_S;				//needed in PartialSVD_SPI
		float *h_S_prev;	//needed in PartialSVD_SPI

		

// allocate memory on the device 


    cudaMalloc((void**)&d_Mat, mn * sizeof(float));
    SAFEcudaMalloc("d_Mat");

    cudaMalloc((void**)&Grad, mn * sizeof(float));
    SAFEcudaMalloc("Grad");

    cudaMalloc((void**)&Grad_proj, mn * sizeof(float));
    SAFEcudaMalloc("Grad_proj");
  
    cudaMalloc((void**)&d_Y, mn * sizeof(float));
    SAFEcudaMalloc("d_Y");
  
    cudaMalloc((void**)&d_y, p * sizeof(float));
    SAFEcudaMalloc("d_y");

    cudaMalloc((void**)&d_u, m * sizeof(float));
    SAFEcudaMalloc("d_u");

    cudaMalloc((void**)&d_u_prev, m * sizeof(float));
    SAFEcudaMalloc("d_u_prev");

    cudaMalloc((void**)&d_v, n * sizeof(float));
    SAFEcudaMalloc("d_v");
  
    cudaMalloc((void**)&d_A, p * sizeof(int));
    SAFEcudaMalloc("d_A");

    cudaMalloc((void**)&d_MM, (m*m) * sizeof(int));
    SAFEcudaMalloc("d_MM"); 

		


    if (strcmp(algstr, "CGIHT")==0) {

      cudaMalloc((void**)&Grad_prev, mn * sizeof(float));
      SAFEcudaMalloc("Grad_prev");

      cudaMalloc((void**)&Grad_prev_proj, mn * sizeof(float));
      SAFEcudaMalloc("Grad_prev_proj");

      cudaMalloc((void**)&d_y_work, p * sizeof(float));
      SAFEcudaMalloc("d_y_work");

    }

		if (strcmp(algstr, "SVT")==0 ){
			//allocate memory on device
			cudaMalloc((void**)&d_U_init, mn * sizeof(float));
      SAFEcudaMalloc("d_U_init");
			curandCheck = curandGenerateNormal(gen, d_U_init, m*n, 0,1);
			SAFEcurand(curandCheck, "curandGenerateNormal in SVT.cu");

			cudaMalloc((void**)&d_U, mn * sizeof(float));
    	SAFEcudaMalloc("d_U");

			cudaMalloc((void**)&d_U_prev, mn * sizeof(float));
    	SAFEcudaMalloc("d_U_prev");
  
    	cudaMalloc((void**)&d_S, std::min(m,n) * sizeof(float));
    	SAFEcudaMalloc("d_S");
  
    	cudaMalloc((void**)&d_V, mn * sizeof(float));
    	SAFEcudaMalloc("d_V");

			cudaMalloc((void**)&d_R,  std::min(m,n)* sizeof(int));
    	SAFEcudaMalloc("d_R"); 

			//allocate memory on host
			h_S = (float*)malloc( sizeof(float) * std::min(m,n));
    	SAFEmalloc_float(h_S, "h_S");

			h_S_prev = (float*)malloc( sizeof(float) * std::min(m,n));
    	SAFEmalloc_float(h_S_prev, "h_S_prev");
		}
		else{

			//allocate memory on device
			cudaMalloc((void**)&d_U, mr * sizeof(float));
    	SAFEcudaMalloc("d_U");
  
    	cudaMalloc((void**)&d_S, r * sizeof(float));
    	SAFEcudaMalloc("d_S");
  
    	cudaMalloc((void**)&d_V, nr * sizeof(float));
    	SAFEcudaMalloc("d_V");

			cudaMalloc((void**)&d_U_prev, mr * sizeof(int));
    	SAFEcudaMalloc("d_U_prev");

			cudaMalloc((void**)&d_R, r * sizeof(int));
    	SAFEcudaMalloc("d_R");

			//allocate memory on host

			h_S = (float*)malloc( sizeof(float) * r );
    	SAFEmalloc_float(h_S, "h_S");

    	h_S_prev = (float*)malloc( sizeof(float) * r );
    	SAFEmalloc_float(h_S_prev, "h_S_prev");

		}

		if (strcmp(algstr, "LMaFit")==0 ){	
			cudaMalloc((void**)&d_V_prev, nr * sizeof(int));
    	SAFEcudaMalloc("d_V_prev"); 
			cudaMalloc((void**)&d_RR, (r*r) * sizeof(int));
    	SAFEcudaMalloc("d_RR");
			cudaMalloc((void**)&d_SS, mn * sizeof(float));
    	SAFEcudaMalloc("d_SS");
		}


			

   
// allocate memory on the host

    float * residNorm_prev = (float*)malloc( sizeof(float) * convRateNum );
    SAFEmalloc_float(residNorm_prev, "residNorm_prev");

		int r_estimate;
		int * p_r = &r_estimate;  // to store rank estimate



// allocate memory on the host specific for timing set to on
/*
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

*/

/*
    #ifdef VERBOSE
    if (nlhs == 3) { printf("VERBOSE inactive when passing a problem directly.");}
      float *h_Mat_input, *h_Mat, *h_y, *h_Grad, *h_Grad_proj;
      float *h_U, *h_S;
    if (nlhs == 6) {
      h_Mat_input = (float*)malloc( sizeof(float) * mn );
      h_Mat = (float*)malloc( sizeof(float) * mn );
      h_Grad_proj = (float*)malloc( sizeof(float) * mn );
      h_Grad = (float*)malloc( sizeof(float) * mn );
      h_y = (float*)malloc( sizeof(float) * p );
      h_U = (float*)malloc( sizeof(float) * mr );
      h_S = (float*)malloc( sizeof(float) * r );
    }
    #endif
*/


/*
*****************************************
** Set the kernel execution parameters:                                                  
** For device flexibility, find max threads for current device.                         
** Then use this to determine the different kernel execution configurations you may need.
*****************************************
*/


    dim3 threadsPerBlockmn(threads_perblockmn);
    int num_blocksmn = (int)ceil((float)mn/(float)threads_perblockmn);
    dim3 numBlocksmn(num_blocksmn);

    dim3 threadsPerBlockm(threads_perblockm);
    int num_blocksm = (int)ceil((float)m/(float)threads_perblockm);
    dim3 numBlocksm(num_blocksm);

    dim3 threadsPerBlockp(threads_perblockp);
    int num_blocksp = (int)ceil((float)p/(float)threads_perblockp);
    dim3 numBlocksp(num_blocksp);

    dim3 threadsPerBlocknr(threads_perblocknr);
    int num_blocksnr = (int)ceil((float)nr/(float)threads_perblocknr);
    dim3 numBlocksnr(num_blocksnr);

		dim3 threadsPerBlockmr(threads_perblockmr);
    int num_blocksmr = (int)ceil((float)mr/(float)threads_perblockmr);
    dim3 numBlocksmr(num_blocksmr);
		//added by Chenshuai Sui

// initialize cublas library
    cublasInit();
		cublasHandle_t handle;
		cublasCreate(&handle);
    SAFEcublas("cublasInit in gagamc_entry.");

/*
*****************************
** CREATE A RANDOM PROBLEM **
*****************************
*/


    if (nlhs == 7){
      createProblem_entry(d_Mat, h_Mat_input, d_U, d_V, d_Y, d_y, d_A, m, n, r, p, matrixEnsemble, &seed, gen, threadsPerBlockp, numBlocksp, threadsPerBlockmn, numBlocksmn, handle);
			curandCheck = curandGenerateNormal(gen, d_U, mr, 0,1);
			SAFEcurand(curandCheck, "curandGenerateNormal d_U");
			curandCheck = curandGenerateNormal(gen, d_V, nr, 0,1);
			SAFEcurand(curandCheck, "curandGenerateNormal d_V");

    }
    else if (nlhs == 3){
      cudaMemcpy(d_Y, h_Y, sizeof(float)*mn, cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, h_y, sizeof(float)*p, cudaMemcpyHostToDevice);
      cudaMemcpy(d_A, h_A_in, sizeof(int)*p, cudaMemcpyHostToDevice);
			curandCheck = curandGenerateNormal(gen, d_U, mr, 0,1);
			SAFEcurand(curandCheck, "curandGenerateNormal d_U");
			curandCheck = curandGenerateNormal(gen, d_V, nr, 0,1);
			SAFEcurand(curandCheck, "curandGenerateNormal d_V");


    }
    

//  ensure initial approximation is the zero vector
    zero_vector_float<<< numBlocksmn, threadsPerBlockmn >>>((float*)d_Mat, mn);
    SAFEcuda("zero_vector_float in random problem in gagamc_entry");

/*
    #ifdef VERBOSE
    if (nlhs == 6) {
    if (verb>3) {  printf("After createProblem, k = %d \n", k);}
    if (verb>0) {

    printf("The problem size is (m, n, r, p) = (%d, %d, %d, %d).\n", m, n, r, p);

    cudaMemcpy(h_A, d_A, sizeof(float)*mn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vec_input, d_vec_input, sizeof(float)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vec, d_vec, sizeof(float)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, sizeof(float)*m, cudaMemcpyDeviceToHost);

    printf("Create Problem has provided the following:\n");
    printf("The matrix entries:\n");
    for (int jj=0; jj<min(mn,q); jj++){ printf("%f\n ", h_A[jj]); }
    printf("The initial target vector (x):\n");
    for (int jj=0; jj<min(n,q); jj++){ printf("%f\n", h_vec_input[jj]); }
    printf("The input measurements:\n");
    for (int jj=0; jj<min(m,q); jj++){ printf("%f\n", h_y[jj]); }
    printf("The initial approximation:\n");
    for (int jj=0; jj<min(n,q); jj++){ printf("%f\n", h_vec[jj]); }
    }
    }
    #endif
*/


/*
*************************************************
** Solve this problem with the input algorithm **
*************************************************
*/


    cudaEvent_t startALG, stopALG;
    float timeALG;
    cudaEventCreate(&startALG);
    cudaEventCreate(&stopALG);
    cudaEventRecord(startALG,0);


// Initialization of parameters and cublas

    int   iter  = 0;
//    float err   = 0;
//    int   sum   = 0;

    float time_sum=0.0f;

    if (strcmp(algstr, "NIHT")==0) alg = 0;
    else if (strcmp(algstr, "SVT")==0) alg = 1;
    else if (strcmp(algstr, "CGIHT")==0) alg = 2;
		else if (strcmp(algstr, "LMaFit")==0) alg = 3;

    switch (alg) {
	case 0:
	   NIHT_MC_S_entry_SPI(d_Mat, Grad, Grad_proj, d_Y, d_U, d_S, d_V, d_A, d_y, d_MM, d_U_prev, d_R, h_S, h_S_prev, residNorm_prev, m, n, r, p, mn, maxiter, tol, PSVDmaxiter, PSVDtol, gen, &iter, &time_sum, threadsPerBlockp, numBlocksp, threadsPerBlocknr, numBlocksnr, threadsPerBlockmn, numBlocksmn, handle);
	   SAFEcuda("NIHT_MC_S_entry_SPI in gagamc_entry_suic");
	   break; 
	case 1:
	   SVT_MC_S_entry_SPI(d_Mat, Grad, Grad_proj, d_Y, d_U, d_S, d_V, d_A, d_MM, d_U_prev, d_U_init, d_R, h_S, h_S_prev, residNorm_prev, m, n, p_r, p, mn, maxiter, tol, PSVDmaxiter, PSVDtol, &iter, &time_sum, threadsPerBlockp, numBlocksp,threadsPerBlockmn, numBlocksmn,max_threads_per_block, handle);
  	   SAFEcuda("SVT_MC_S_entry_SPI in gagamc_entry");
           break;
	case 2:
	   CGIHT_MC_S_entry_SPI(d_Mat, Grad, Grad_proj, Grad_prev, Grad_prev_proj, d_Y, d_U, d_S, d_V, d_A, d_y, d_y_work, d_MM, d_U_prev, d_R, h_S, h_S_prev, residNorm_prev, m, n, r, p, mn, maxiter, tol, PSVDmaxiter, PSVDtol, gen, &iter, &time_sum, threadsPerBlockp, numBlocksp, threadsPerBlocknr, numBlocksnr, threadsPerBlockmn, numBlocksmn, handle);
	   SAFEcuda("CGIHT_MC_S_entry_SPI in gagamc_entry");
	   break; 

	case 3:
	   LMaFit_MC_S_entry(d_Mat, Grad, d_Y, d_U, d_V, d_A, d_RR, d_U_prev, d_V_prev, Grad_proj, d_SS,d_R, residNorm_prev, m, n, r, p, mn, mr, nr, maxiter, tol, gen, &iter, &time_sum, threadsPerBlockp, numBlocksp, threadsPerBlockmr, numBlocksmr,threadsPerBlocknr, numBlocksnr,  threadsPerBlockmn, numBlocksmn, handle);
	   SAFEcuda("LMaFit_MC_S_entry in gagamc_entry");
	   break; 

	default:
	   printf("[gagamc_entry] Error: The possible (case sensitive) input strings for algorithms using gagamc_entry are:\n NIHT\n SVT\n CGIHT\n LMaFit \n");
	   break;
    }


  
    cudaThreadSynchronize();
		SAFEcuda("cudaThreadSynchronize");
    cudaEventRecord(stopALG,0);
		SAFEcuda("cudaEventRecord");
    cudaEventSynchronize(stopALG);
		SAFEcuda("cudaThreadSynchronize: stopALG");
    cudaEventElapsedTime(&timeALG, startALG, stopALG);
		SAFEcuda("cudaEventElapsedTime in gagamc_entry_suic:");
    cudaEventDestroy(startALG);
		SAFEcuda("cudaEventDestroy in gagamc_entry_suic: startALG");
    cudaEventDestroy(stopALG);
		SAFEcuda("cudaEventDestroy in gagamc_entry_suic: stopALG");




/*
***********************
** Check the Results **
***********************
*/

 /*   if ( (strcmp(algstr, "CGIHT")==0) && (restartFlag==1) ) {
      strcat(algstr,"restarted");
    }
*/
// some CPU action is needed before the results

    if (nlhs == 7){

      results_entry(d_Mat, Grad, h_Mat_input, h_Mat_out, residNorm_prev, h_norms, h_times, convergence_rate, total_iter, d_A, h_A_out, iter, timeALG, time_sum, m, n, r, p, mn, convRateNum, seed, p_startTest, p_stopTest, algstr, handle);
			SAFEcuda("results_entry in gagamc_entry_suic");

/*
      cudaMemcpy(h_Mat_out, d_Mat, sizeof(float)*mn, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_A_out, d_A, sizeof(int)*p, cudaMemcpyDeviceToHost);

      total_iter[0] = iter;

      float convRate, root;
      int temp = min(iter, convRateNum);
      root = 1/(float)temp;
      temp=convRateNum-temp;
      convRate = (residNorm_prev[convRateNum-1]/residNorm_prev[temp]);
      convRate = pow(convRate, root);

      convergence_rate[0]=convRate;


      // formulate the norms by computing the matrix Mat_input - Mat which is stored here in Grad
      cudaMemcpy(Grad, h_Mat_input, mn*sizeof(float), cudaMemcpyHostToDevice);
      float norm2_mat_in = cublasSnrm2(mn, Grad, 1); // l2 norm of original matrix
      float norm1_mat_in = cublasSasum(mn, Grad, 1); // l1 norm of original matrix
      cublasSaxpy(mn, -1.0, d_Mat, 1, Grad, 1);
      h_norms[0]=cublasSnrm2(mn, Grad, 1)/norm2_mat_in; // l2 norm
      h_norms[1]=cublasSasum(mn, Grad, 1)/norm1_mat_in; // l1 norm
      // use iter and root as locations for temporary storage in order to compute the l_infity norm    
      iter = cublasIsamax(mn, Grad, 1);
      cudaMemcpy(&root, Grad+iter, sizeof(float), cudaMemcpyDeviceToHost);
      h_norms[2]=abs(root);  // l infinity norm


      //  record the timings
      float timeTest;
      h_times[1] = timeALG;
      h_times[2] = time_sum/(float)iter;
      cudaThreadSynchronize();
      cudaEventRecord(stopTest,0);
      cudaEventSynchronize(stopTest);
      cudaEventElapsedTime(&timeTest, startTest, stopTest);
      cudaEventDestroy(startTest);
      cudaEventDestroy(stopTest);
      h_times[0] = timeTest;
*/
    }  // closes if (nlhs == 7)
    else if (nlhs == 3){
      cudaMemcpy(h_Mat_out, d_Mat, sizeof(float)*mn, cudaMemcpyDeviceToHost);

      total_iter[0] = iter;

      float convRate, root;
      int temp = min(iter, convRateNum);
      root = 1/(float)temp;
      temp=convRateNum-temp;
      convRate = (residNorm_prev[convRateNum-1]/residNorm_prev[temp]);
      convRate = pow(convRate, root);

      convergence_rate[0]=convRate;
    }  // closes else if (nlhs ==3)
/*
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
*/

/*
**************
** CLEANUP  **
**************
*/


// free up the allocated memory on the device


    cudaFree(d_Mat);
		SAFEcuda("cudaFree in gagamc_entry_suic: d_Mat");
    cudaFree(Grad);
		SAFEcuda("cudaFree in gagamc_entry_suic: Grad");
    cudaFree(Grad_proj);
		SAFEcuda("cudaFree in gagamc_entry_suic: Grad_proj");
    cudaFree(d_Y);
		SAFEcuda("cudaFree in gagamc_entry_suic: d_Y");
    cudaFree(d_A);
		SAFEcuda("cudaFree in gagamc_entry_suic: d_A");
    cudaFree(d_U);
		SAFEcuda("cudaFree in gagamc_entry_suic: d_U");
    cudaFree(d_S);
		SAFEcuda("cudaFree in gagamc_entry_suic: d_S");
    cudaFree(d_V);
		SAFEcuda("cudaFree in gagamc_entry_suic: d_V");
    cudaFree(d_y);
		SAFEcuda("cudaFree in gagamc_entry_suic: d_y");
    cudaFree(d_u);
		SAFEcuda("cudaFree in gagamc_entry_suic: d_u");
    cudaFree(d_u_prev);
		SAFEcuda("cudaFree in gagamc_entry_suic: d_u_prev");
    cudaFree(d_v);
		SAFEcuda("cudaFree in gagamc_entry_suic: d_v");
    cudaFree(d_MM);
		SAFEcuda("cudaFree in gagamc_entry_suic: d_MM");
    cudaFree(d_U_prev);
		SAFEcuda("cudaFree in gagamc_entry_suic: d_U_prev");
    cudaFree(d_R);
		SAFEcuda("cudaFree in gagamc_entry_suic: d_R");


    if (alg==2){
      cudaFree(Grad_prev);
			SAFEcuda("cudaFree in gagamc_entry_suic: Grad_prev");
      cudaFree(Grad_prev_proj);
			SAFEcuda("cudaFree in gagamc_entry_suic: Grad_prev_proj");
      cudaFree(d_y_work);
			SAFEcuda("cudaFree in gagamc_entry_suic: d_y_work");
    }
		else if(alg==1){
			cudaFree(d_U_init);
   		SAFEcuda("cudaFree in gagamc_entry_suic: d_U_init");
		}
		else if(alg==3){
			cudaFree(d_V_prev);
			SAFEcuda("cudaFree in gagamc_entry_suic: d_V_prev");
			cudaFree(d_RR);
			SAFEcuda("cudaFree in gagamc_entry_suic: d_RR");
			cudaFree(d_SS);
			SAFEcuda("cudaFree in gagamc_entry_suic: d_SS");
		}
 
	

    curandCheck = curandDestroyGenerator(gen);
    SAFEcurand(curandCheck, "curandDestroyGenerator in gagamc_entry");

   
		cublasDestroy(handle);
		cublasShutdown();
    SAFEcublas("cublasShutdown");


// free the memory on the host

    free(residNorm_prev);
    free(h_S);
    free(h_S_prev);
    

/*
    #ifdef VERBOSE
    if (nlhs == 6) {
      free(h_vec_input);
      free(h_vec);
      free(h_vec_thres);
      free(h_grad);
      free(h_y);
      free(h_resid);
      free(h_resid_update);
      free(h_A);
    }
    #endif
*/
  }  //closes the else ensuring the algorithm input was valid

  }  //closes the else ensuring a correct number of input and output arguments

  return;
}



