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
***************************************************************
** This is a file to be included in the header of any main   **
** function in the package.  It is simply a list of problem  **
** generation routines by matrix ensemble.                   **
***************************************************************
*/

#include "createProblem_kernels.cu"
#include "createVector.cu"

#include "createMeasurements_dct.cu"
#include "createMeasurements_smv.cu"
#include "createMeasurements_gen.cu"

#include "createProblem_dct.cu"
#include "createProblem_smv.cu"
#include "createProblem_gen.cu"




