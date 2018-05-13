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
	 cout <<"This is gaga_smv."<<endl;
	#endif

	  if ( (nlhs!=5) && (nlhs!=8) && ((nrhs!=5) || (nrhs!=6) || (nrhs!=7)) )
	    printf("[gaga_smv] Error: There are two possible usages for this funtion.\n Five or six (with options) input arguments with eight (including residual and time recordings if algorithm is CCS) output arguments. \n [norms times iterations support convRate vec_out resRecord timeRecord] = gaga_smv(algstring,k,m,n,p,options).\n Six or seven (with options) input arguments with five (including residual and time recordings if algorithm is CCS) output arguments.\n [outputVector iterations convRate resRecord timeRecord] = gaga_smv(algstring,y,smv_rows,smv_cols,smv_vals,k,options).\n");
	  else {

	// reading in the string to determine the algorithm
	    int strlen = mxGetN(prhs[0])+1;
	    char algstr[strlen+100];
	    int algerr = mxGetString(prhs[0], algstr, strlen);

	// creating the switch variable defining the algorithm
	    int alg;

	// check if the algorithms is CCS
	    int ccs_flag = ((strcmp(algstr, "smp") == 0) || \
		(strcmp(algstr, "ssmp") == 0) || \
		(strcmp(algstr, "er") == 0) || \
		(strcmp(algstr, "parallel_lddsr") == 0) || \
		(strcmp(algstr, "parallel_l0") == 0) || \
		(strcmp(algstr, "serial_l0") == 0) || \
		(strcmp(algstr, "er_naive") == 0) || \
		(strcmp(algstr, "ssmp_naive") == 0) || \
		(strcmp(algstr, "parallel_l0_swipe") == 0) || \
		(strcmp(algstr, "rand_robust_l0") == 0) || \
		(strcmp(algstr, "robust_l0") == 0) || \
		(strcmp(algstr, "robust_l0_adaptive") == 0) || \
		(strcmp(algstr, "robust_l0_adaptive_trans") == 0) || \
		(strcmp(algstr, "robust_l0_trans") == 0) || \
		(strcmp(algstr, "smp_robust") == 0) || \
		(strcmp(algstr, "cgiht_robust") == 0) || \
		(strcmp(algstr, "ssmp_robust") == 0)) ? 1 : 0;

	    int ccs_indexed_matrix_flag = ((strcmp(algstr, "ssmp") == 0) || \
		(strcmp(algstr, "er") == 0) || \
		(strcmp(algstr, "er_naive") == 0) || \
		(strcmp(algstr, "ssmp_naive") == 0) || \
		(strcmp(algstr, "ssmp_robust") == 0)) ? 1 : 0;

	    int serial_ccs_flag = strcmp(algstr, "serial_l0") == 0 ? 1 : 0;

	    int robust_ccs_flag = ((strcmp(algstr, "rand_robust_l0") == 0) || \
		(strcmp(algstr, "robust_l0_adaptive") == 0) || \
		(strcmp(algstr, "robust_l0_adaptive_trans") == 0) || \
		(strcmp(algstr, "robust_l0_trans") == 0) || \
		(strcmp(algstr, "robust_l0") == 0) || \
		(strcmp(algstr, "smp_robust") == 0) || \
		(strcmp(algstr, "ssmp_robust") == 0) || \
		(strcmp(algstr, "cgiht_robust") == 0) || \
		(strcmp(algstr, "CGIHT") == 0) || \
		(strcmp(algstr, "CGIHTprojected")) ) ? 1 : 0;

	// check that algstr indicates one of the valid algorithms:
	// NIHT, HTP, IHT, ThresholdSD, ThresholdCG, CSMPSP, CGIHT
	    int valid_alg = ((strcmp(algstr, "NIHT")==0) || \
		(strcmp(algstr, "HTP")==0) || \
		(strcmp(algstr, "IHT")==0) || \
		(strcmp(algstr, "ThresholdSD")==0) || \
		(strcmp(algstr, "ThresholdCG")==0) || \
		(strcmp(algstr, "CSMPSP")==0) || \
		(strcmp(algstr, "CGIHT")==0) || \
		(strcmp(algstr, "FIHT")==0) || \
		(strcmp(algstr, "ALPS")==0) || \
		(strcmp(algstr, "CGIHTprojected")==0) || \
		(ccs_flag == 1)) ? 1 : 0;

	// possible inputs
	    int k, m, n, p;
	    int *h_rows, *h_cols;
	    float *h_y, *h_vals;

	// possible outputs
	    int *total_iter, *checkSupport;
	    float *h_norms, *h_times, *convergence_rate, *h_out;
	    float *resRecord, *timeRecord;

	// control parameter from inputs
	    int nz;

	// make variables to store properties of the GPU (device)
	    unsigned int max_threads_per_block;
	    cudaDeviceProp dp;
	    
	    if (valid_alg == 0){
	      printf("[gaga_smv] Error: The possible (case sensitive) input strings for algorithms using gaga_smv are:\n NIHT\n IHT\n HTP\n ThresholdSD\n ThresholdCG\n CSMPSP\n smp\n ssmp\n er\n parallel_lddsr\n parallel_l0\n serial_l0\n rand_robust_l0\n robust_l0\nFive or eight output arguments only possible if algorithms is CCS.");
	    }
	    else {

	      if (nlhs == 8){
		k = (int)mxGetScalar(prhs[1]);
		m = (int)mxGetScalar(prhs[2]);
		n = (int)mxGetScalar(prhs[3]);
		p = (int)mxGetScalar(prhs[4]);

		nz = n * p;

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
	    else if (nlhs == 5){
	      h_y = (float*)mxGetData(prhs[1]);
	      h_rows = (int*)mxGetData(prhs[2]);
	      h_cols = (int*)mxGetData(prhs[3]);
	      h_vals = (float*)mxGetData(prhs[4]);
	      k = (int)mxGetScalar(prhs[5]);

	      nz = mxGetM(prhs[4]);

	      // get p: For release, need to write function that throws error if vector is not consistent.
	      p = 1;
	      while (h_cols[p-1] == h_cols[p]){
	       p++;
	      }


	// need to calculate m and n.  this can be done on the GPU, but 
	// requires moving quite a bit of of the code out of the standard 
	// order, so we take the efficiency loss and compute them on the CPU.

	      m=h_rows[0];
	      n=h_cols[0];
	      for (int jj=1;jj<nz;jj++){
		if (h_rows[jj]>m)  m = h_rows[jj];
		if (h_cols[jj]>n)  n = h_cols[jj];}

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
	    // threadsPerBlocknp: number of threads per block for kernels acting on vectors of length n*p
	    // 			  (default set to min(nz, max_threads_per_block))
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
	    // projFracTol: tolerance for CGIHTprojected

	// initialise options at default values
	    // some of these may not be used depending on the usage (such as vecDistribution)
	    int vecDistribution = 1;  // binary
	    float band_percentage = 0;
	    int matrixEnsemble = 3;  // expander 
	    int kFixedFlag = 0; // k not fixed
	    unsigned int seed = clock();
	    int num_bins = max(n/20,1000);
	    int gpuNumber = 0;
	    int convRateNum = 16;
	    float tol = 10^(-4);
	    float noise_level = 0.0;
	    int timingFlag = 0; // off
	    float alpha_start = 0.25;
	    int l0_thresh = 1;
	    int supp_flag = 0;
	    int maxiter;
	    int restartFlag = 0;
	    int debug_mode = 0;
	    float projFracTol = 3.0;
	    if ( (strcmp(algstr, "HTP")==0) || (strcmp(algstr, "CSMPSP")==0) ) maxiter=300;
	    else maxiter=5000;
	// unlike other options, the threads_perblock options must be set to default
	    // only after the option gpuNumber is determined when checking the options list
	    int threads_perblock = 0; // min(n, max_threads_per_block);
	    int threads_perblockm = 0; // min(m, max_threads_per_block);
	    int threads_perblocknp = 0; // min(nz, max_threads_per_block);
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
		    float *p_num = (float*) mxGetData(cell_element_ptr);
		    projFracTol = *p_num; }
		  else if (strcmp(buf, "l0_thresh")==0){
		    int *p_num = (int*) mxGetData(cell_element_ptr);
		    l0_thresh = *p_num;
		    l0_thresh = (1 <= l0_thresh && l0_thresh <= p) ? l0_thresh : 1; }
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
		  else if (strcmp(buf, "threadsPerBlocknp")==0){
		    int *p_num = (int*) mxGetData(cell_element_ptr);
		    threads_perblocknp = *p_num; }
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
		  else if (strcmp(buf, "band_percentage")==0){
		    float *p_num = (float*) mxGetData(cell_element_ptr);
		    band_percentage = *p_num; }
		  else if (strcmp(buf, "debug_mode")==0){
		    int *p_num = (int*) mxGetData(cell_element_ptr);
		    debug_mode = *p_num; }
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
		  else if (strcmp(buf, "matrixEnsemble")==0){
		    int buflen_tmp = (mxGetM(cell_element_ptr) * 
				      mxGetN(cell_element_ptr)) + 1;
		    char *buf_tmp = (char*) malloc(buflen_tmp);
		    algerr = mxGetString(cell_element_ptr, buf_tmp, buflen_tmp);
		    if (strcmp(buf_tmp, "binary")==0) matrixEnsemble = 2;
		    else if (strcmp(buf_tmp, "ones")==0) matrixEnsemble = 1;
		    else if (strcmp(buf_tmp, "expander")==0) matrixEnsemble = 3;
		    else cout << "Admissible matrixEnsemble for smv are: binary, ones, expander.\n";}
		  else{
		    cout << "The following option is not recognised: " << buf << endl;
		  }
		}
	      }
	    }

	    if (ccs_flag == 1){
	      matrixEnsemble = 3;
	    }

		if (nlhs == 8){
			plhs[6] = mxCreateNumericMatrix((maxiter + 1), 1, mxSINGLE_CLASS, mxREAL);
			resRecord = (float*) mxGetData(plhs[6]);

			plhs[7] = mxCreateNumericMatrix((maxiter + 1), 1, mxSINGLE_CLASS, mxREAL);
			timeRecord = (float*) mxGetData(plhs[7]);
		}else if(nlhs == 5){
			plhs[3] = mxCreateNumericMatrix((maxiter + 1), 1, mxSINGLE_CLASS, mxREAL);
			resRecord = (float*) mxGetData(plhs[3]);

			plhs[4] = mxCreateNumericMatrix((maxiter + 1), 1, mxSINGLE_CLASS, mxREAL);
			timeRecord = (float*) mxGetData(plhs[4]);
		}

	// check if any of the threads_perblock variable were not set in the options
	    if (threads_perblock == 0) threads_perblock = min(n, max_threads_per_block);
	    if (threads_perblockm == 0) threads_perblockm = min(m, max_threads_per_block);
	    if (threads_perblocknp == 0) threads_perblocknp = min(nz, max_threads_per_block);
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
	    int * d_cols;
	    float * d_vals;
	    int * d_bin_grad;
	    int * d_bin_counters_grad;
	// device variables for FIHT_S_smv and ALPS_S_smv
	    float * d_vec_prev;
	    float * d_Avec_prev;
	    float * d_vec_extra;
	    float * d_Avec_extra;
	// device variables for CGIHTprojected
	    // grad_prev_thres already declared
	    float * d_vec_diff;
	    float * d_p_thres;
	    float * d_Ap_thres;
	// device variables for CCS algorithms 
	    int * d_rm_cols;
	    int * d_rm_rows_index;

	    if (nlhs == 8){
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

	    cudaMalloc((void**)&d_rows, nz * sizeof(int));
	    SAFEcudaMalloc("d_rows");

	    cudaMalloc((void**)&d_cols, nz * sizeof(int));
	    SAFEcudaMalloc("d_cols");

	    cudaMalloc((void**)&d_vals, nz * sizeof(float));
	    SAFEcudaMalloc("d_vals");
	  
	    cudaMalloc((void**)&d_bin_counters, num_bins * sizeof(int));
	    SAFEcudaMalloc("d_bin_counters");

	    if ((strcmp(algstr, "HTP")==0) || (strcmp(algstr, "ThresholdCG")==0)){
	      cudaMalloc((void**)&grad_previous, n * sizeof(float));
	      SAFEcudaMalloc("grad_previous");
	    }

	    if (strcmp(algstr, "CSMPSP")==0 || strcmp(algstr, "CGIHT")==0 || strcmp(algstr, "cgiht_robust")==0){
	      cudaMalloc((void**)&grad_previous, n * sizeof(float));
	      SAFEcudaMalloc("grad_previous");

	      cudaMalloc((void**)&d_bin_grad, n * sizeof(int));
	      SAFEcudaMalloc("d_bin_grad");

	      cudaMalloc((void**)&d_bin_counters_grad, num_bins * sizeof(int));
	      SAFEcudaMalloc("d_bin_counters_grad");
	    }

	    if (strcmp(algstr, "CGIHT")==0 || strcmp(algstr, "CGIHTprojected")==0 || strcmp(algstr, "cgiht_robust")==0){
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

	    if (ccs_indexed_matrix_flag == 1) {
	      cudaMalloc((void**)&d_rm_cols, nz * sizeof(int));
	      SAFEcudaMalloc("d_rm_cols");

	      cudaMalloc((void**)&d_rm_rows_index, 2*m*sizeof(int));
	      SAFEcudaMalloc("d_rm_rows_index");
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
	    if (nlhs == 5) { printf("VERBOSE inactive when passing a problem directly.");}
	      float *h_vec_input, *h_vec, *h_vec_thres, *h_grad;
	      float *h_resid, *h_resid_update;
	    if (nlhs == 8) {
	      h_vec_input = (float*)malloc( sizeof(float) * n );
	      h_vec = (float*)malloc( sizeof(float) * n );
	      h_vec_thres = (float*)malloc( sizeof(float) * n );
	      h_grad = (float*)malloc( sizeof(float) * n );
	      h_y = (float*)malloc( sizeof(float) * m );
	      h_resid = (float*)malloc( sizeof(float) * m );
	      h_resid_update = (float*)malloc( sizeof(float) * m );
	      h_rows = (int*)malloc( sizeof(int) * nz );
	      h_cols = (int*)malloc( sizeof(int) * nz );
	      h_vals = (float*)malloc( sizeof(float) * nz);
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

	  dim3 threadsPerBlocknp(threads_perblocknp);
	  int num_blocksnp = (int)ceil((float)(nz)/(float)threads_perblocknp);
	  dim3 numBlocksnp(num_blocksnp);

	  dim3 threadsPerBlockm(threads_perblockm);
	  int num_blocksm = (int)ceil((float)m/(float)threads_perblockm);
	  dim3 numBlocksm(num_blocksm);

	  dim3 threadsPerBlock_bin(threads_perblock_bin);
	  int num_blocks_bin = (int)ceil((float)num_bins/(float)threads_perblock_bin);
	  dim3 numBlocks_bin(num_blocks_bin);

	/*
	*****************************
	** CREATE A RANDOM PROBLEM **
	*****************************
	*/

	    #ifdef VERBOSE
	    if (verb>3) {printf("Before createProblem, k = %d \n", k);}
	    #endif

	    int error_flagCP = 0;

	    if (nlhs == 8){
	      if (kFixedFlag==0){
		if (noise_level <= 0){
		  error_flagCP = createProblem_smv(&k, m, n, vecDistribution, band_percentage, d_vec_input, d_y, d_rows, d_cols, d_vals, p, matrixEnsemble, &seed);
		  SAFEcuda("createProblem_smv"); }
		else{ //  when noise_level > 0
		  int error_flagCP = createProblem_smv_noise(&k, m, n, vecDistribution, d_vec_input, d_y, d_rows, d_cols, d_vals, p, matrixEnsemble, &seed, noise_level);
		  SAFEcuda("createProblem_smv_noise"); }
	    }  else{
		int k_tmp = k;
		do{
		  k = k_tmp;
		  if (noise_level <= 0){
		    error_flagCP = createProblem_smv(&k, m, n, vecDistribution, band_percentage, d_vec_input, d_y, d_rows, d_cols, d_vals, p, matrixEnsemble, &seed);
		    if (k != k_tmp)
		      seed = seed +1;
		    SAFEcuda("createProblem_smv"); }
		  else{ //  when noise_level > 0
		if (robust_ccs_flag == 1){
		    int error_flagCP = createProblem_smv_noise_ccs(&k, m, n, vecDistribution, d_vec_input, d_y, d_rows, d_cols, d_vals, p, matrixEnsemble, &seed, noise_level);
		} else {
		    int error_flagCP = createProblem_smv_noise(&k, m, n, vecDistribution, d_vec_input, d_y, d_rows, d_cols, d_vals, p, matrixEnsemble, &seed, noise_level);
		}
		    if (k != k_tmp)
		      seed = seed +1;
		    SAFEcuda("createProblem_smv_noise"); }
		} while (k_tmp != k);      
	      }
	    }
	    else if (nlhs == 5){

	      //printf("debug: Before device Variable Copying:\t%s\n", cudaGetErrorString(cudaGetLastError()));
	      cudaMemcpy(d_y, h_y, sizeof(float)*m, cudaMemcpyHostToDevice);
	      cudaDeviceSynchronize();
	      cudaMemcpy(d_rows,h_rows, sizeof(int)*nz, cudaMemcpyHostToDevice);
	      cudaDeviceSynchronize();
	      cudaMemcpy(d_cols,h_cols, sizeof(int)*nz, cudaMemcpyHostToDevice);
	      cudaDeviceSynchronize();
	      cudaMemcpy(d_vals,h_vals, sizeof(float)*nz, cudaMemcpyHostToDevice);
	      cudaDeviceSynchronize();

	      indexShiftDown<<<numBlocksnp,threadsPerBlocknp>>>(d_rows,nz);
	      indexShiftDown<<<numBlocksnp,threadsPerBlocknp>>>(d_cols,nz); 

	      /*
	      int num = 0;
	      printf("debug: checking row consistency\n");
	      for (int i = 0; i <nz; i++){
		cudaMemcpy(&num, d_rows + i, sizeof(int), cudaMemcpyDeviceToHost);
		if (num < 0 || num > m-1){
			printf("d_rows[%d] = %d\n", i, num);
		}
	      }
	      printf("\n");

	      printf("debug: checking column consistency\n");
	      for (int i = 0; i <nz; i++){
		cudaMemcpy(&num, d_cols + i, sizeof(int), cudaMemcpyDeviceToHost);
		if (num < 0 || num > n-1){
			printf("d_cols[%d] = %d\n", i, num);
		}
	      }
	      printf("\n");
	      */

	    }

	  /*
	  if(1 == 1){
	    int *hh_rows;
	    int *hh_cols;
	    int np = n*p;
	    hh_rows = (int*)malloc( np*sizeof(int) );
	    hh_cols = (int*)malloc( np*sizeof(int) );
	    cudaMemcpy(hh_rows, d_rows, np*sizeof(int), cudaMemcpyDeviceToHost); 
	    cudaMemcpy(hh_cols, d_cols, np*sizeof(int), cudaMemcpyDeviceToHost); 

		for (int i = 0; i < np; i++){
			if (hh_rows[i] < 0 || hh_rows[i] > m -1){
				printf("After createProblem: Inconsistent row detected  hh_rows[%d] = %d\n", i, hh_rows[i]);
				printf("Problem: k = %d, m = %d, n = %d, d = %d, seed = %d, seedu = %u\n", k, m, n, p, seed, seed);
			}
			if (hh_cols[i] < 0 || hh_cols[i] > n -1){
				printf("After createProblem: Inconsistent col detected  hh_cols[%d] = %d\n", i, hh_cols[i]);
				printf("Problem: k = %d, m = %d, n = %d, d = %d, seed = %d, seedu = %u\n", k, m, n, p, seed, seed);
			}
		}

	    free(hh_rows);
	    free(hh_cols);
	  }
	  */

	    int h_max_nonzero_rows_count = 0;
	    if (ccs_indexed_matrix_flag == 1){
	     transform_to_row_major_order(m, n, p, d_rows, d_cols, d_rm_rows_index, d_rm_cols, &h_max_nonzero_rows_count); 

	    }

	    int threads_perblockr = min(h_max_nonzero_rows_count*p, max_threads_per_block);
	    dim3 threadsPerBlockr(threads_perblockr);
	    int num_blocksr = (int)ceil((float)(h_max_nonzero_rows_count*p)/(float)threads_perblockr);
	    dim3 numBlocksr(num_blocksr);

	    if(error_flagCP==1){
	      //printf("The sparse matrix was not correctly constructed.\nThis problem has not been passed to the algorithm.\n");
	      h_norms[0] = 1.0f;
	      h_norms[1] = 1.0f;
	      h_norms[2] = 1.0f;
	    }
	    else{

	      //  ensure initial approximation is the zero vector
	      zero_vector_float<<< numBlocks, threadsPerBlock >>>((float*)d_vec, n);
	      SAFEcuda("zero_vector_float in random problem initialization in gaga_smv");

	      #ifdef VERBOSE
	      if (nlhs == 8) {
	      if (verb>3) {  printf("After createProblem, k = %d \n", k);}
	      if (verb>0) {

	      printf("The problem size is (k,m,n) = (%d, %d, %d).\n",k, m, n);

	      cudaMemcpy(h_rows, d_rows, sizeof(int)*nz, cudaMemcpyDeviceToHost);
	      cudaMemcpy(h_cols, d_cols, sizeof(int)*nz, cudaMemcpyDeviceToHost);
	      cudaMemcpy(h_vals, d_vals, sizeof(float)*nz, cudaMemcpyDeviceToHost);
	      cudaMemcpy(h_vec_input, d_vec_input, sizeof(float)*n, cudaMemcpyDeviceToHost);
	      cudaMemcpy(h_vec, d_vec, sizeof(float)*n, cudaMemcpyDeviceToHost);
	      cudaMemcpy(h_y, d_y, sizeof(float)*m, cudaMemcpyDeviceToHost);

	      printf("Create Problem has provided the following:\n");
	      printf("The matrix rows:\n");
	      for (int jj=0; jj<min(n,q); jj++){
		for (int pp=0; pp<p; pp++)
		  printf("%d ", h_rows[jj*p+pp]);
		printf("\n");
	      }
	      printf("The matrix cols:\n");
	      for (int jj=0; jj<min(n,q); jj++){
		for (int pp=0; pp<p; pp++)
		  printf("%d ", h_cols[jj*p+pp]);
		printf("\n");
	      }
	      printf("The matrix vals:\n");
	      for (int jj=0; jj<min(n,q); jj++){
		for (int pp=0; pp<p; pp++)
		  printf("%f ", h_vals[jj*p+pp]);
		printf("\n");
	      }
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
              int fail_update_flag = 0;


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
	    else if (strcmp(algstr, "smp")==0) alg = 10;
	    else if (strcmp(algstr, "ssmp")==0) alg = 11;
	    else if (strcmp(algstr, "er")==0) alg = 12;
	    else if (strcmp(algstr, "parallel_lddsr")==0) alg = 13;
	    else if (strcmp(algstr, "parallel_l0")==0) alg = 14;
	    else if (strcmp(algstr, "serial_l0")==0) alg = 15;
	    else if (strcmp(algstr, "er_naive")==0) alg = 16;
	    else if (strcmp(algstr, "ssmp_naive")==0) alg = 17;
	    else if (strcmp(algstr, "parallel_l0_swipe")==0) alg = 18;
	    else if (strcmp(algstr, "rand_robust_l0")==0) alg = 19;
	    else if (strcmp(algstr, "robust_l0")==0) alg = 20;
	    else if (strcmp(algstr, "ssmp_robust")==0) alg = 21;
	    else if (strcmp(algstr, "robust_l0_adaptive")==0) alg = 22;
	    else if (strcmp(algstr, "robust_l0_adaptive_trans")==0) alg = 23;
	    else if (strcmp(algstr, "robust_l0_trans")==0) alg = 24;
	    else if (strcmp(algstr, "smp_robust")==0) alg = 25;
	    else if (strcmp(algstr, "cgiht_robust")==0) alg = 26;

      
      switch (alg) {
	case 0:
	   if (timingFlag == 0){
		NIHT_S_smv(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, nz, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("NIHT_S_smv in gaga_smv");}
	   else {
	   	NIHT_S_timings_smv(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, nz, &iter, mu, err, &sum, alpha, supp_flag, time_per_iteration, time_supp_set, &time_sum,  numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("NIHT_S_timings_smv in gaga_smv");}
	   break;
	case 1:
	   if (timingFlag == 0){
		HTP_S_smv(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, nz, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("HTP_S_smv in gaga_smv");}
	   else {
	   	HTP_S_timings_smv(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, nz, &iter, mu, err, &sum, alpha, supp_flag, time_per_iteration, time_supp_set, cg_per_iteration, time_for_cg, &time_sum,  numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("HTP_S_timings_smv in gaga_smv");}
	   break;
	case 2:
		mu = 0.65f;
		IHT_S_smv(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, nz, &iter, mu, err, &sum, &time_sum, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("IHT_S_smv in gaga_smv");
  		break;
	case 3:
		HT_SD_S_smv(d_vec, d_vec_thres, grad, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, nz, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("HT_SD_S_smv in gaga_smv");
		break;
	case 4:
		HT_CG_S_smv(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, nz, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("HT_CG_S_smv in gaga_smv");
		break;
	case 5:
		CSMPSP_S_smv(d_vec, d_vec_thres, grad, grad_previous, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, d_bin_grad, d_bin_counters_grad, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, nz, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("CSMPSP_S_smv in gaga_smv");
		break;
	case 6:
		CGIHT_S_smv(d_vec, d_vec_thres, grad, grad_previous, grad_prev_thres, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, d_bin_grad, d_bin_counters_grad, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, nz, restartFlag, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
  		SAFEcuda("CGIHT_S_smv in gaga_smv");
		break;
        case 7: 
                FIHT_S_smv(d_vec, d_vec_thres,  grad, d_vec_prev, d_Avec_prev, d_vec_extra, d_Avec_extra, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, 
tol, maxiter, num_bins, k, m, n, nz, &iter, mu, err, &sum, &time_sum, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
                SAFEcuda("FIHT_S_smv in gaga_smv");
                break;
        case 8: 
                ALPS_S_smv(d_vec, d_vec_thres,  grad, d_vec_prev, d_Avec_prev, d_vec_extra, d_Avec_extra, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, 
tol, maxiter, num_bins, k, m, n, nz, &iter, mu, err, &sum, &time_sum, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
                SAFEcuda("ALPS_S_smv in gaga_smv");
                break;
        case 9:
                CGIHTprojected_S_smv(d_vec, d_vec_thres, grad, grad_prev_thres, d_p_thres, d_Ap_thres,  d_vec_diff, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters,residNorm_prev, tol, maxiter, projFracTol, num_bins, k, m, n,nz, &iter, mu, err, &sum, &time_sum, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp,numBlocksm, threadsPerBlockm,  numBlocks_bin, threadsPerBlock_bin);
                SAFEcuda("ALPS_S_smv in gaga_smv");
                break;
	case 10:
		smp(d_vec, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, p, nz, &iter, err, &sum, &time_sum, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin, timeRecord, resRecord);
		SAFEcuda("SMP_S_smv in gaga_smv");
		break;
	case 11:
		ssmp(d_vec, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_rm_rows_index, d_rm_cols, h_max_nonzero_rows_count, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, p, nz, &iter, err, &sum, &time_sum, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin, numBlocksr, threadsPerBlockr, timeRecord, resRecord);
		SAFEcuda("SSMP_S_smv in gaga_smv");
		break;
	case 12:
		er(d_vec, d_y, resid, d_rows, d_cols, d_vals, d_rm_rows_index, d_rm_cols, h_max_nonzero_rows_count, tol, maxiter, k, m, n, p, nz, resRecord, timeRecord, &iter, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocksr, threadsPerBlockr);
		SAFEcuda("ER_S_smv in gaga_smv");
		break;
	case 13:
		parallel_lddsr(d_vec, d_y, resid, d_rows, d_cols, d_vals, tol, maxiter, k, m, n, p, nz, resRecord, timeRecord, &iter, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm);
		SAFEcuda("PARALLEL_LDDSR_S_smv in gaga_smv");
		break;
	case 14:
		parallel_l0(d_vec, d_y, resid, d_rows, d_cols, d_vals, tol, maxiter, k, m, n, p, l0_thresh, nz, resRecord, timeRecord, &iter, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm);
		SAFEcuda("PARALLEL_L0_S_smv in gaga_smv");
		break;
	case 15:
		int *hs_rows;
		int *hs_cols;
		float *hs_y;
		float *hs_vec;
		float *hs_resid;

		hs_rows = (int*)malloc( sizeof(int) * nz );
		hs_cols = (int*)malloc( sizeof(int) * nz );
		hs_y = (float*)malloc( sizeof(float) * m );
		hs_vec = (float*)malloc( sizeof(float) * n );
		hs_resid = (float*)malloc( sizeof(float) * m );

		cudaMemcpy(hs_rows, d_rows, sizeof(int)*nz, cudaMemcpyDeviceToHost);
		cudaMemcpy(hs_cols, d_cols, sizeof(int)*nz, cudaMemcpyDeviceToHost);
		cudaMemcpy(hs_y, d_y, sizeof(float)*m, cudaMemcpyDeviceToHost);
		cudaMemcpy(hs_vec, d_vec, sizeof(float)*n, cudaMemcpyDeviceToHost);
		cudaMemcpy(hs_resid, d_y, sizeof(float)*m, cudaMemcpyDeviceToHost);

		serial_l0(hs_vec, hs_y, hs_resid, hs_rows, hs_cols, n, m, k, p, l0_thresh, tol, maxiter, &iter, resRecord, timeRecord);
		SAFEcuda("SERIAL_L0_S_smv in gaga_smv");

		cudaMemcpy(resid, hs_resid, sizeof(float)*m, cudaMemcpyHostToDevice);
		cudaMemcpy(d_vec, hs_vec, sizeof(float)*n, cudaMemcpyHostToDevice);

		free(hs_rows);
		free(hs_cols);
		free(hs_y);
		free(hs_resid);
		free(hs_vec);
		break;
	case 16:
		er_naive(d_vec, d_y, resid, d_rows, d_cols, d_vals, tol, maxiter, k, m, n, p, nz, resRecord, timeRecord, &iter, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm);
		SAFEcuda("ER_NAIVE_S_smv in gaga_smv");
		break;
	case 17:
		ssmp_naive(d_vec, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, p, nz, &iter, err, &sum, &time_sum, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin, timeRecord, resRecord);
		SAFEcuda("SSMP_NAIVE_S_smv in gaga_smv");
		break;
	case 18:
		parallel_l0_swipe(d_vec, d_y, resid, d_rows, d_cols, d_vals, tol, maxiter, k, m, n, p, nz, resRecord, timeRecord, &iter, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm);
		SAFEcuda("PARALLEL_L0_SWIPE_S_smv in gaga_smv");
		break;
	case 19:
		robust_l0(d_vec, d_y, resid, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, num_bins, &sum, tol, maxiter, k, m, n, p, l0_thresh, nz, noise_level, resRecord, timeRecord, &iter, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("RAND_ROBUST_L0_S_smv in gaga_smv");
		break;
	case 20:
		adaptive_robust_l0(d_vec, d_y, resid, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, num_bins, &sum, tol, maxiter, k, m, n, p, l0_thresh, nz, noise_level, 0, 0, resRecord, timeRecord, &iter, debug_mode, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin, &fail_update_flag);
		//deterministic_robust_l0(d_vec, d_y, resid, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, num_bins, &sum, tol, maxiter, k, m, n, p, l0_thresh, nz, noise_level, resRecord, timeRecord, &iter, debug_mode, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("ROBUST_L0_S_smv in gaga_smv");
		break;
	case 21:
		ssmp_robust(d_vec, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_rm_rows_index, d_rm_cols, h_max_nonzero_rows_count, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, p, nz, noise_level, &iter, err, &sum, &time_sum, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin, numBlocksr, threadsPerBlockr, timeRecord, resRecord);
		SAFEcuda("SSMP_ROBUST_S_smv in gaga_smv");
		break;
	case 22:
		adaptive_robust_l0(d_vec, d_y, resid, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, num_bins, &sum, tol, maxiter, k, m, n, p, l0_thresh, nz, noise_level, 0, 1, resRecord, timeRecord, &iter, debug_mode, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin, &fail_update_flag);
		SAFEcuda("ROBUST_L0_ADAPTIVE_S_smv in gaga_smv");
		break;
	case 23:
		adaptive_robust_l0(d_vec, d_y, resid, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, num_bins, &sum, tol, maxiter, k, m, n, p, l0_thresh, nz, noise_level, 1, 1, resRecord, timeRecord, &iter, debug_mode, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin, &fail_update_flag);
		SAFEcuda("ROBUST_L0_ADAPTIVE_TRANS_S_smv in gaga_smv");
		break;
	case 24:
		adaptive_robust_l0(d_vec, d_y, resid, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, num_bins, &sum, tol, maxiter, k, m, n, p, l0_thresh, nz, noise_level, 1, 0, resRecord, timeRecord, &iter, debug_mode, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin, &fail_update_flag);
		SAFEcuda("ROBUST_L0_TRANS_S_smv in gaga_smv");
		break;
	case 25:
		smp_robust(d_vec, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, p, nz, noise_level, &iter, err, &sum, &time_sum, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin, timeRecord, resRecord);
		SAFEcuda("SMP_S_smv in gaga_smv");
		break;
	case 26:
		CGIHT_robust(d_vec, d_vec_thres, grad, grad_previous, grad_prev_thres, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_bin, d_bin_counters, d_bin_grad, d_bin_counters_grad, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, nz, noise_level, restartFlag, &iter, mu, err, &sum, &time_sum,  numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
  		SAFEcuda("CGIHT_robust in gaga_smv");
		break;
	default:
		printf("[gaga_smv] Error: The possible (case sensitive) input strings for algorithms using gaga_smv are:\n NIHT\n IHT\n HTP\n ThresholdSD\n ThresholdCG\n CSMPSP\n CGIHT\n");
		break;
  }

  if (ccs_flag == 1) {
    for (int j=0; j<16; j++) {
      residNorm_prev[j]=0;
    }
    for (int j = 0; j < min(iter, 16); j++){
      residNorm_prev[15 - j] = resRecord[iter - j];
    }
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

    if ( (strcmp(algstr, "CGIHT")==0 || strcmp(algstr, "cgiht_robust")==0) && (restartFlag==1) ) {
      strcat(algstr,"restarted");
    }

// some CPU action is needed before the results
// some output is needed before the results.
//cout << " ";


      if (nlhs == 8){
        if ( ((timingFlag == 0) || ((alg != 0) && (alg != 1)) ) && (noise_level <= 0) ){
	  results_smv(d_vec, d_vec_input, vecDistribution, residNorm_prev, h_norms, h_out, h_times, convergence_rate, total_iter, checkSupport, iter, timeIHT, time_sum, &sum, k, m, n, p, matrixEnsemble, seed,  p_startTest, p_stopTest, algstr, numBlocks, threadsPerBlock, band_percentage);
   	  SAFEcuda("results_smv in gaga_smv"); }
	else if ( ((timingFlag == 0) || ((alg != 0) && (alg != 1)) ) && (noise_level > 0) ){
	  results_smv_noise(d_vec, d_vec_input, vecDistribution, residNorm_prev, h_norms, h_out, h_times, convergence_rate, total_iter, checkSupport, iter, timeIHT, time_sum, &sum, k, m, n, p, matrixEnsemble, seed, noise_level, p_startTest, p_stopTest, algstr, numBlocks, threadsPerBlock, band_percentage, fail_update_flag);
 	  SAFEcuda("results_smv_noise in gaga_smv"); }
	else if ( (timingFlag == 1) && (alg == 0) ){
	  results_timings_smv(d_vec, d_vec_input, vecDistribution, residNorm_prev, h_norms, h_out, h_times, convergence_rate, total_iter, checkSupport, iter, timeIHT, time_sum, time_per_iteration, time_supp_set, &sum, alpha_start, supp_flag, k, m, n, p, matrixEnsemble, seed,  p_startTest, p_stopTest, algstr, numBlocks, threadsPerBlock);
	  SAFEcuda("results_timings_smv in gaga_smv"); }
	else if ( (timingFlag == 1) && (alg == 1) ){
	  results_timings_HTP_smv(d_vec, d_vec_input, vecDistribution, residNorm_prev, h_norms, h_out, h_times, convergence_rate, total_iter, checkSupport, iter, timeIHT, time_sum, time_per_iteration, time_supp_set, cg_per_iteration, time_for_cg, &sum, alpha_start, supp_flag, k, m, n, p, matrixEnsemble, seed,  p_startTest, p_stopTest, algstr, numBlocks, threadsPerBlock);
	  SAFEcuda("results_timings_HTP_smv in gaga_smv"); }
      }
      else if (nlhs == 5){
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
      if (nlhs == 8){
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

    } // this ends the if/else which checks if the sparse matrix could be constucted


/*
**************
** CLEANUP  **
**************
*/


// free up the allocated memory on the device

      if (nlhs == 8) {cudaFree(d_vec_input);}
      cudaFree(d_vec);
      cudaFree(d_vec_thres);
      cudaFree(d_bin);
      cudaFree(d_y);
      cudaFree(d_rows);
      cudaFree(d_cols);
      cudaFree(d_vals);
      cudaFree(d_bin_counters);
      cudaFree(resid);
      cudaFree(resid_update);
      cudaFree(grad);

      SAFEcuda("1 cudaFree in gaga_smv");
      if ((alg==1) || (alg==4)){
	cudaFree(grad_previous);
      }

      SAFEcuda("2 cudaFree in gaga_smv");
      if ( (alg == 5) || (alg == 6) || (alg == 26)){
	cudaFree(grad_previous);
      SAFEcuda("3.1 cudaFree in gaga_smv");
  	cudaFree(d_bin_grad);
      SAFEcuda("3.2 cudaFree in gaga_smv");
	cudaFree(d_bin_counters_grad);
      }


      SAFEcuda("3 cudaFree in gaga_smv");
      if (alg == 6 || alg == 9 || alg == 26){
        cudaFree(grad_prev_thres);
      }

      SAFEcuda("4 cudaFree in gaga_smv");
      if ((alg == 7) || (alg == 8) ) {
        cudaFree(d_vec_prev);
        cudaFree(d_Avec_prev);
        cudaFree(d_vec_extra);
        cudaFree(d_Avec_extra);
      }
      SAFEcuda("5 cudaFree in gaga_smv");

      if (alg == 9) {
        cudaFree(d_vec_diff);
        cudaFree(d_p_thres);
        cudaFree(d_Ap_thres); 
      }
      SAFEcuda("6 cudaFree in gaga_smv");

      if (ccs_indexed_matrix_flag == 1) {
        cudaFree(d_rm_cols);
        cudaFree(d_rm_rows_index);
      }
      SAFEcuda("7 cudaFree in gaga_smv");

      SAFEcuda("8 cudaFree in gaga_smv");

    cublasShutdown();
    SAFEcublas("cublasShutdown");

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
    if (nlhs == 8) {
      free(h_vec_input);
      free(h_vec);
      free(h_vec_thres);
      free(h_grad);
      free(h_y);
      free(h_resid);
      free(h_resid_update);
      free(h_rows);
      free(h_cols);
      free(h_vals);
    }
    #endif
    
  }  //closes the else ensuring the algorithm input was valid

  }  //closes the else ensuring a correct number of input and output arguments

  return;
}

