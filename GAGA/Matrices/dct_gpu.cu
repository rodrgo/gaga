/*
 * 1-D and 2-D Discrete Consine Transform using CUFFT
 *
 * All input signal dimension lengths have to be even numbers.
 *
 * Author: Sangkyun Lee
 * Date: 2008/11/7
----------------------------------
Copyright (c) 2008 Sangkyun Lee and Stephen J. Wright.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
------------------------------
 *
 *
 * MODIFIED BY Jeff Blanchard and Jared Tanner, October 2010
 *
 *
 */



#include "cuda.h"

#define PIO2 CUDART_PIO2
#define BLK_DCT_1D 32
#define BLK_DCT_2D 16

int dct_gpu_init_1D = 0;
int dct_gpu_init_2D = 0;

//#define DCT_SAFE_MODE

#ifdef DCT_SAFE_MODE
    #define CHECK_INIT_1D()	{ if(!dct_gpu_init_1D){ debug("Error: dct_gpu is not initialized for 1-D transforms!\n"); exit(-1);}}
    #define CHECK_INIT_2D()	{ if(!dct_gpu_init_2D){ debug("Error: dct_gpu is not initialized for 2-D transforms!\n"); exit(-1);}}
#else
    #define CHECK_INIT_1D()	{}
    #define CHECK_INIT_2D()	{}
#endif

// Global variables
Complex *DCT_y, *DCT_fft_y;
FLOAT *DCT_whole_x, *DCT_whole_out, *tmp;
cufftHandle DCT_plan_row, DCT_plan_col;

// Initialization
void dct_gpu_init(int M, int N)
{
    int MN = M*N;

    CUDA_CALL( cudaMalloc((void**)&DCT_y, sizeof(Complex)*MN) );
    CUDA_CALL( cudaMalloc((void**)&DCT_fft_y, sizeof(Complex)*MN) );
    CUDA_CALL( cudaMalloc((void**)&DCT_whole_x, sizeof(FLOAT)*MN) );
    CUDA_CALL( cudaMalloc((void**)&DCT_whole_out, sizeof(FLOAT)*MN) );


    // fft plan for column-wise transform
    CUFFT_CALL( cufftPlan1d(&DCT_plan_col, M, CUFFT_C2C, N) );
    dct_gpu_init_1D = 1;

    if(N > 1) {
	// fft plan for row-wise transform
	CUFFT_CALL( cufftPlan1d(&DCT_plan_row, N, CUFFT_C2C, M) );
	CUDA_CALL( cudaMalloc((void**)&tmp, sizeof(FLOAT)*MN) );
	dct_gpu_init_2D = 1;
    }
}

// Deallocation
void dct_gpu_destroy()
{

    CUDA_CALL( cudaFree(DCT_y) );
    CUDA_CALL( cudaFree(DCT_fft_y) );
    CUDA_CALL( cudaFree(DCT_whole_x) );
    CUDA_CALL( cudaFree(DCT_whole_out) );
    CUFFT_CALL( cufftDestroy(DCT_plan_col) );
    dct_gpu_init_1D = 0;

    if(dct_gpu_init_2D) {
        CUFFT_CALL( cufftDestroy(DCT_plan_row) );
        CUDA_CALL( cudaFree(tmp) );
	dct_gpu_init_2D = 0;
    }
}


/*
 * Utility functions to manipulate signals
 *
 */

// 1-D
__global__ void __split(Complex* y, const FLOAT* x, INT N)
{
    int m = X_INDEX;
    Complex c;

    if(m < N/2)
    {
	c.x = x[2*m];
	c.y = 0.f;
	y[m] = c;
    }
    else if(m < N)
    {
	m -= N/2;
	c.x = x[2*m+1];
	c.y = 0.f;
	y[N-m-1] = c;
    }
}

// 1-D
__global__ void __gather(FLOAT* x, const Complex* y, INT N)
{
    int m = X_INDEX;

    if(m < N/2)
	x[2*m] = y[m].x;
    else if(m < N)
    {
	m -= N/2;
	x[2*m+1] = y[N-m-1].x;
    }
}

// 1-D
__global__ void __fft2dct(FLOAT* out, Complex* in, INT N)
{
    int n = X_INDEX;
    FLOAT w;
    FLOAT f = n*PIO2/N;

    /*
    if(n < N/2+1)
	c = in[n];
    else if(n < N)
    {
	c = in[N-n];
	c.y *= -1.f;
    }
    */

    if(n < N) {
	Complex c = in[n];

	if(n==0)
	    w = sqrtf(1.f/N);
	else
	    w = sqrtf(2.f/N);

	out[n] = w*(c.x*__cosf(f) + c.y*__sinf(f));
    }
}


// 1-D
__global__ void __dct2fft(Complex* out, const FLOAT* in, INT N)
{
    int n = X_INDEX;

    if(n < N) {
        FLOAT f = n*PIO2/N;
	FLOAT d = in[n];
        FLOAT w;
        Complex c;

        if(n==0)
	    w = sqrtf(1.f/N);
	else
	    w = sqrtf(2.f/N);

	c.x = w*d*__cosf(f);
	c.y = w*d*__sinf(f);

	out[n] = c;
    }
}


/*
 * Main routines.
 *
 */

//
// FORWARD TRANSFORMS
//

/*
 * Perform 1-D DCT on the signal x of length N
 */




void dct_gpu(FLOAT* out, const FLOAT* x, INT N)
{
    CHECK_INIT_1D();
//    Complex* y;
//    Complex* fft_y;

//    CUDA_CALL( cudaMalloc((void**)&y, sizeof(Complex)*N) );
//    CUDA_CALL( cudaMalloc((void**)&fft_y, sizeof(Complex)*N) );

    dim3 dimBlock(BLK_DCT_1D, 1);
    dim3 dimGrid((int)ceilf(1.f*N/BLK_DCT_1D), 1);

    __split<<<dimGrid, dimBlock>>> (DCT_y, x, N);

    // do fft on y
//    cufftHandle plan;
//    CUFFT_CALL( cufftPlan1d(&plan, N, CUFFT_C2C, 1) );
    CUFFT_CALL( cufftExecC2C(DCT_plan_col, (cufftComplex*)DCT_y, (cufftComplex*)DCT_fft_y, CUFFT_FORWARD) );	// CUFFT_FORWARD

    __fft2dct<<<dimGrid, dimBlock>>> (out, DCT_fft_y, N);

    // cleanup
//    CUFFT_CALL( cufftDestroy(plan) );
//    CUDA_CALL( cudaFree(y) );
//    CUDA_CALL( cudaFree(fft_y) );
///    cudaThreadSynchronize();
}




/*
 * Functions for dealing with sampled rows of DCT(x)
 */
__global__ void __select_rows(FLOAT* out, const FLOAT* in, INT* rows, INT k)
{
    int X = X_INDEX;

    if(X < k)
	out[X] = in[rows[X]];			
}
__global__ void __fill_rows(FLOAT* out, const FLOAT* in, INT *rows, INT k)
{
    int X = X_INDEX;

    if(X < k)
	out[rows[X]] = in[X];
}



//
// INVERSE TRANSFORMS
//

/*
 * Perform 1-D inverse DCT transform
 */




void idct_gpu(FLOAT* out, const FLOAT* x, INT N)
{
    CHECK_INIT_1D();

//    Complex* y;
//    Complex* fft_y;

//    CUDA_CALL( cudaMalloc((void**)&y, sizeof(Complex)*N) );
//    CUDA_CALL( cudaMalloc((void**)&fft_y, sizeof(Complex)*N) );
    dim3 dimBlock(BLK_DCT_1D, 1);
    dim3 dimGrid((int)fmaxf(1.f, (int)ceilf(1.f*N/BLK_DCT_1D)), 1);

    __dct2fft<<<dimGrid, dimBlock>>> (DCT_fft_y, x, N);

    // do ifft on fft_y
//    cufftHandle plan;
//    CUFFT_CALL( cufftPlan1d(&plan, N, CUFFT_C2C, 1) );
    CUFFT_CALL( cufftExecC2C(DCT_plan_col, (cufftComplex*)DCT_fft_y, (cufftComplex*)DCT_y, CUFFT_INVERSE) ); // CUFFT_INVERSE

    __gather<<<dimGrid, dimBlock>>> (out, DCT_y, N);

    // cleanup
//    CUFFT_CALL( cufftDestroy(plan) );
//    CUDA_CALL( cudaFree(y) );
//    CUDA_CALL( cudaFree(fft_y) );

///    cudaThreadSynchronize();
}




/*
*********************************
** Formatted Call for A and AT **
**        using the DCT        **
*********************************
*/



void A_dct(FLOAT* out, const FLOAT* x, INT N, INT k, INT* rows)
{
    CHECK_INIT_1D();

    dct_gpu(DCT_whole_out, x, N);

    dim3 dimBlock(BLK_DCT_1D, 1);
    dim3 dimGrid((int)fmaxf(1.f, ceilf(1.f*k/BLK_DCT_1D)), 1);
    __select_rows<<<dimGrid, dimBlock>>>(out, DCT_whole_out, rows, k);

}


void AT_dct(float* out, const float* x, int N, int k, int* rows)
{
    CHECK_INIT_1D();

    CUDA_CALL( cudaMemset(DCT_whole_x, 0, sizeof(FLOAT)*N) );

    dim3 dimBlock(BLK_DCT_1D, 1);
    dim3 dimGrid((int)fmaxf(1.f, ceilf(1.f*k/BLK_DCT_1D)), 1);
    __fill_rows<<<dimGrid, dimBlock>>>(DCT_whole_x, x, rows, k);

    idct_gpu(out, DCT_whole_x, N);

}






