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
** This is a single header file for all algorithms.
** The order of the #include is important based on dependencies 
** some of the files.
*/


//#define VERBOSE

/* Using VERBOSE will cause warnings when compiling
 * but the functionality is intact.
 * verb = 2,3 are no longer used and are free for the user
 * to use to define new print statements.
 */

#ifdef VERBOSE
  int verb = 0; // verbosity level:
		// 0: just print the results
		// 1: also print the initial problem
		// 2: depricated, same as 1
		// 3: depricated, same as 1
		// 4: also print value of k given and value of k used by createProblem
		// 5: activate all print commands
  int q = 16;   // the maximum number of entries to print out for each vector


  float * h_vec_input;
  float * h_vec;
  float * h_vec_thres;
  float * h_grad;
  float * h_y;
  float * h_resid;
  float * h_resid_update;
  int * h_rows;

#endif


//#define SAFE


#include <iostream>
using namespace std;

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include <curand.h>
#include <mex.h>
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


#include "algorithms.cu"
#include "algorithms_timings.cu"
#include "createProblem.cu"
#include "results.cu"



