
	#include "greedyheader_debug.cu"


	// Host function
	int main()
	{

	// Synthetic nlhs variable
	int nlhs = 8;

	// Define all parameters 
	char algstr[] = "adaptive_robust_l0";
	float delta = 0.5;
	float rho = 0.03;
	int n = 16348;
	int m = ((int)delta*n);
	int k = ((int)rho*m);
	int p = 7;

	// define options
	unsigned int seed = 294262+111;
	int vecDistribution = 2;  // 0=uniform, 1=binary, 2=gaussian
	int matrixEnsemble = 3;  // 1=ones, 2=binary, 3=expander
	float noise_level = 0.01;
	int kFixedFlag = 0;

	// These options shouldn't matter to replicate bug
	float band_percentage = 0;
	int num_bins = max(n/20,1000);
	int convRateNum = 16;
	float tol = 1;
	int timingFlag = 0; // off
	float alpha_start = 0.25;
	int l0_thresh = 1;
	int supp_flag = 0;
	int maxiter = 1000;
	int restartFlag = 0;
	int debug_mode = 0;
	float projFracTol = 3.0;

	int threads_perblock = 0;
	int threads_perblockm = 0;
	int threads_perblocknp = 0;
	int threads_perblock_bin = 0;

	// set the gpu device properties
	int gpuNumber = 0;

	unsigned int max_threads_per_block;
	cudaDeviceProp dp;
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
	max_threads_per_block = dp.maxThreadsPerBlock;

	  if ( (nlhs!=8) )
	    printf("Error: we must have nlhs==8");
	  else {

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
		(strcmp(algstr, "robust_l0") == 0) || \
		(strcmp(algstr, "deterministic_robust_l0") == 0) || \
		(strcmp(algstr, "adaptive_robust_l0") == 0) || \
		(strcmp(algstr, "ssmp_robust") == 0)) ? 1 : 0;

	    int ccs_indexed_matrix_flag = ((strcmp(algstr, "ssmp") == 0) || \
		(strcmp(algstr, "er") == 0) || \
		(strcmp(algstr, "er_naive") == 0) || \
		(strcmp(algstr, "ssmp_naive") == 0) || \
		(strcmp(algstr, "ssmp_robust") == 0)) ? 1 : 0;

	    int serial_ccs_flag = strcmp(algstr, "serial_l0") == 0 ? 1 : 0;

	    int robust_ccs_flag = ( (strcmp(algstr, "robust_l0") == 0) || \
		(strcmp(algstr, "deterministic_robust_l0") == 0) || \
		(strcmp(algstr, "adaptive_robust_l0") == 0) || \
		(strcmp(algstr, "ssmp_robust") == 0) || \
		(strcmp(algstr, "CGIHT") == 0) || \
		(strcmp(algstr, "CGIHTprojected")) ) ? 1 : 0;

	// check that algstr indicates one of the valid algorithms:
	// NIHT, HTP, IHT, ThresholdSD, ThresholdCG, CSMPSP, CGIHT
	    int valid_alg = ( (strcmp(algstr, "NIHT")==0) || \
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

	// possible outputs
	    int *total_iter, *checkSupport;
	    float *h_norms, *h_times, *convergence_rate, *h_out;
	    float *resRecord, *timeRecord;

	// control parameter from inputs
	    int nz;
	    
	    if (valid_alg == 0){
	      printf("[gaga_smv] Error: Algorithm name is nnot valid");
	    }
	    else {

	      if (nlhs == 8){

		nz = n * p;

		float* h_norms;
		h_norms = (float*)malloc( sizeof(float) * 3 );

		float* h_times;
		h_times = (float*)malloc( sizeof(float) * 3);

		int* total_iter;
		total_iter = (int*)malloc( sizeof(int) * 1);

		int* checkSupport;
		checkSupport = (int*)malloc( sizeof(int) * 4);

		float* convergence_rate;
		convergence_rate = (float*)malloc( sizeof(float) * 1);

		float* h_out;
		h_out = (float*)malloc( sizeof(float) * n);

	    }

	    if (ccs_flag == 1){
	      matrixEnsemble = 3;
	    }

		if (nlhs == 8){
			float* resRecord;
			resRecord = (float*)malloc( sizeof(float) * (maxiter + 1));

			float* timeRecord;
			timeRecord = (float*)malloc( sizeof(float) * (maxiter + 1));
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
		 if (robust_ccs_flag == 1){
		    int error_flagCP = createProblem_smv_noise_ccs(&k, m, n, vecDistribution, d_vec_input, d_y, d_rows, d_cols, d_vals, p, matrixEnsemble, &seed, noise_level);
		}else{
		  int error_flagCP = createProblem_smv_noise(&k, m, n, vecDistribution, d_vec_input, d_y, d_rows, d_cols, d_vals, p, matrixEnsemble, &seed, noise_level);
		}
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
	    else if (strcmp(algstr, "smp")==0) alg = 10;
	    else if (strcmp(algstr, "ssmp")==0) alg = 11;
	    else if (strcmp(algstr, "er")==0) alg = 12;
	    else if (strcmp(algstr, "parallel_lddsr")==0) alg = 13;
	    else if (strcmp(algstr, "parallel_l0")==0) alg = 14;
	    else if (strcmp(algstr, "serial_l0")==0) alg = 15;
	    else if (strcmp(algstr, "er_naive")==0) alg = 16;
	    else if (strcmp(algstr, "ssmp_naive")==0) alg = 17;
	    else if (strcmp(algstr, "parallel_l0_swipe")==0) alg = 18;
	    else if (strcmp(algstr, "robust_l0")==0) alg = 19;
	    else if (strcmp(algstr, "deterministic_robust_l0")==0) alg = 20;
	    else if (strcmp(algstr, "ssmp_robust")==0) alg = 21;
	    else if (strcmp(algstr, "adaptive_robust_l0")==0) alg = 22;

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
		SAFEcuda("ROBUST_L0_S_smv in gaga_smv");
		break;
	case 20:
		deterministic_robust_l0(d_vec, d_y, resid, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, num_bins, &sum, tol, maxiter, k, m, n, p, l0_thresh, nz, noise_level, resRecord, timeRecord, &iter, debug_mode, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("DETERMINISTIC_ROBUST_L0_S_smv in gaga_smv");
		break;
	case 21:
		ssmp_robust(d_vec, d_y, resid, resid_update, d_rows, d_cols, d_vals, d_rm_rows_index, d_rm_cols, h_max_nonzero_rows_count, d_bin, d_bin_counters, h_bin_counters, residNorm_prev, tol, maxiter, num_bins, k, m, n, p, nz, noise_level, &iter, err, &sum, &time_sum, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin, numBlocksr, threadsPerBlockr, timeRecord, resRecord);
		SAFEcuda("SSMP_ROBUST_S_smv in gaga_smv");
		break;
	case 22:
		adaptive_robust_l0(d_vec, d_y, resid, d_rows, d_cols, d_vals, d_bin, d_bin_counters, h_bin_counters, num_bins, &sum, tol, maxiter, k, m, n, p, l0_thresh, nz, noise_level, resRecord, timeRecord, &iter, debug_mode, numBlocks, threadsPerBlock, numBlocksnp, threadsPerBlocknp, numBlocksm, threadsPerBlockm, numBlocks_bin, threadsPerBlock_bin);
		SAFEcuda("ADAPTIVE_ROBUST_L0_S_smv in gaga_smv");
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

    if ( (strcmp(algstr, "CGIHT")==0) && (restartFlag==1) ) {
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
	  results_smv_noise(d_vec, d_vec_input, vecDistribution, residNorm_prev, h_norms, h_out, h_times, convergence_rate, total_iter, checkSupport, iter, timeIHT, time_sum, &sum, k, m, n, p, matrixEnsemble, seed, noise_level, p_startTest, p_stopTest, algstr, numBlocks, threadsPerBlock, band_percentage);
 	  SAFEcuda("results_smv_noise in gaga_smv"); }
	else if ( (timingFlag == 1) && (alg == 0) ){
	  results_timings_smv(d_vec, d_vec_input, vecDistribution, residNorm_prev, h_norms, h_out, h_times, convergence_rate, total_iter, checkSupport, iter, timeIHT, time_sum, time_per_iteration, time_supp_set, &sum, alpha_start, supp_flag, k, m, n, p, matrixEnsemble, seed,  p_startTest, p_stopTest, algstr, numBlocks, threadsPerBlock);
	  SAFEcuda("results_timings_smv in gaga_smv"); }
	else if ( (timingFlag == 1) && (alg == 1) ){
	  results_timings_HTP_smv(d_vec, d_vec_input, vecDistribution, residNorm_prev, h_norms, h_out, h_times, convergence_rate, total_iter, checkSupport, iter, timeIHT, time_sum, time_per_iteration, time_supp_set, cg_per_iteration, time_for_cg, &sum, alpha_start, supp_flag, k, m, n, p, matrixEnsemble, seed,  p_startTest, p_stopTest, algstr, numBlocks, threadsPerBlock);
	  SAFEcuda("results_timings_HTP_smv in gaga_smv"); }
      }

    } // this ends the if/else which checks if the sparse matrix could be constucted


/*
**************
** CLEANUP  **
**************
*/

	if (nlhs == 8){

		cudaFree(h_norms);
		cudaFree(h_times);
		cudaFree(total_iter);
		cudaFree(checkSupport);
		cudaFree(convergence_rate);
		cudaFree(h_out);

		cudaFree(resRecord);
		cudaFree(timeRecord);

	}

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
      if ( (alg == 5) || (alg == 6) ){
	cudaFree(grad_previous);
      SAFEcuda("3.1 cudaFree in gaga_smv");
  	cudaFree(d_bin_grad);
      SAFEcuda("3.2 cudaFree in gaga_smv");
	cudaFree(d_bin_counters_grad);
      }


      SAFEcuda("3 cudaFree in gaga_smv");
      if (alg == 6 || alg == 9){
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

  }  //closes the else ensuring the algorithm input was valid

  }  //closes the else ensuring a correct number of input and output arguments
  return 0;

}

