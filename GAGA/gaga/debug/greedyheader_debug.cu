
//#define SAFE


#include <iostream>
using namespace std;

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include <curand.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include "cublas.h"

#include <math.h>
#include "kernels.cu"
#include "matrices.cu"
#include "functions.cu"

#include "expander/kernels.cu"
#include "expander/functions.cu"
#include "expander/smp.cu"
#include "expander/ssmp.cu"
#include "expander/er.cu"
#include "expander/parallel_l0.cu"
#include "expander/parallel_lddsr.cu"
#include "expander/serial_l0.cu"
#include "expander/parallel_l0_swipe.cu"
#include "expander/robust_l0.cu"
#include "expander/deterministic_robust_l0.cu"
#include "expander/ssmp_robust.cu"


#include "algorithms.cu"
#include "algorithms_timings.cu"
#include "createProblem.cu"
#include "results.cu"



