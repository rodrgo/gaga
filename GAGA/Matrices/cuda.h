/*
 * Basic CUDA environment
 *
 * Includes cuda headers, define types & useful macros.
 *
 * Author: Sangkyun Lee
 * Last Modified: 2008/7/31
 */

#ifndef __CUDA_H__
#define __CUDA_H__

// CUDA headers
#include <math_constants.h>
#include <math_functions.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cufft.h>

// C headers
#include <stdio.h>
#include <stdlib.h>
//#include <time.h>
#include <sys/time.h>

// Types
#define FLOAT float
#define INT int
#define UINT unsigned int
#define Complex float2

// BLOCK_SIZE should be defined outside, before including other .cu files like cudaMath.cu
//#define BLOCK_SIZE 16

// Output function for debug.
// Define "MATLAB" to output the message to the Matlab window.
#ifdef MATLAB
    #define debug(str, ...)	mexPrintf(str, ## __VA_ARGS__ )
#else
    #define debug(str, ...)	fprintf(stderr, str, ## __VA_ARGS__ )
#endif

// CUDA function call error reporting helpers.
// Define MATLAB to output the message to the matlab window.
//#define SAFE_MODE

#ifdef SAFE_MODE
#define CUDA_CALL(call) {\
    cudaError err = call;\
    if( cudaSuccess != err ) {\
	debug("Cuda error in %s:%d: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err) );\
    }}
#define CUBLAS_CALL(call) {\
    cublasStatus err = call;\
    if( CUBLAS_STATUS_SUCCESS != err ) {\
	debug("Cublas error in %s:%d: %s.\n", __FILE__, __LINE__, cublasGetError());\
    }}
#define CUFFT_CALL(call) {\
    cufftResult err = call;\
    if( CUFFT_SUCCESS != err ) {\
	debug("Cufft error in %s:%d: %d.\n", __FILE__, __LINE__, err );\
    }}
#define CUDA_CHECK_ERROR(errorMessage) do {                                 \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        debug("Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
    }                                                                        \
    err = cudaThreadSynchronize();                                           \
    if( cudaSuccess != err) {                                                \
        debug("Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
    } } while (0)

#else
#define CUDA_CALL(call) call;
#define CUBLAS_CALL(call) call;
#define CUFFT_CALL(call) call;
#define CUDA_CHECK_ERROR(errorMessage)
#endif

// For index calculation in device code
#define __imul(a,b)	__mul24((a),(b))
#define __fsqr_rn(a)	__fmul_rn((a),(a))

// Linear index calculation helper
#define X_INDEX (__imul(blockDim.x,blockIdx.x) + threadIdx.x)
#define Y_INDEX (__imul(blockDim.y,blockIdx.y) + threadIdx.y)

// Time mesaurement
//#define cputime() ((double)clock()/CLOCKS_PER_SEC)
inline double cputime() 
{ 
    struct timeval tm; 
    gettimeofday(&tm, 0); 
    //return (1000.0*tm.tv_sec + 0.001*tm.tv_usec); 	// returns time in miliseconds
    return (1.0*tm.tv_sec + 0.000001*tm.tv_usec); 	// returns time in seconds
}

// Some other useful macros
#define swapi(a, b) { int tmp = a; a = b; b = tmp; }
#define swapf(a, b) { float tmp = a; a = b; b =tmp; }
#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )
// Align a to the nearest higher multiple of b
//#define ialignup(a, b) ( ((a)%(b) != 0) ? (a) - (a)/(b) + (b) : (a) )


#endif// __CUDA_H__

