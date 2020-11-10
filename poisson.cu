/*
*   The Lattice Boltzmann Method with EHD convection
*   Yifei Guan
*   NRG lab, University of Washington
*   Oct/17/2018
*
*/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include "LBM.h"
#include <device_functions.h>

__global__ void gpu_efield(double*, double*, double*, double*);
__global__ void odd_extension(double*, double*, cufftDoubleComplex*);
__global__ void gpu_derivative(double*, double*, double*, cufftDoubleComplex*);
__global__ void odd_extract(double*, cufftDoubleComplex*);
__global__ void gpu_bc(double*);

// =========================================================================
// Electric field solver domain extension
// =========================================================================
__host__
void efield(double *phi_gpu, double *Ex_gpu, double *Ey_gpu, double *Ez_gpu) {
	// blocks in grid
	dim3  grid(NX / nThreads, NY, NZ);
	// threads in block
	dim3  threads(nThreads, 1, 1);

	gpu_efield << < grid, threads >> > (phi_gpu, Ex_gpu, Ey_gpu, Ez_gpu);
	gpu_bc << <grid, threads >> > (Ez_gpu);
	getLastCudaError("Efield kernel error");
}

__global__ void gpu_efield(double *fi, double *ex, double *ey, double *ez){

	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int xp1 = (x + 1) % NX;
	unsigned int yp1 = (y + 1) % NY;
	unsigned int zp1 = (z + 1) % NZ;

	unsigned int xm1 = (NX + x - 1) % NX;
	unsigned int ym1 = (NY + y - 1) % NY;
	unsigned int zm1 = (NZ + z - 1) % NZ;

	ex[gpu_scalar_index(x, y, z)] = 0.5*(fi[gpu_scalar_index(xm1,y,z)] - fi[gpu_scalar_index(xp1, y, z)]) / dx;
	ey[gpu_scalar_index(x, y, z)] = 0.5*(fi[gpu_scalar_index(x, ym1, z)] - fi[gpu_scalar_index(x, yp1, z)]) / dy;
	ez[gpu_scalar_index(x, y, z)] = 0.5*(fi[gpu_scalar_index(x, y, zm1)] - fi[gpu_scalar_index(x, y, zp1)]) / dz;
}
__global__ void gpu_bc(double *ez) {
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (z == 0) {
		ez[gpu_scalar_index(x, y, 0)] = ez[gpu_scalar_index(x, y, 1)];
		return;
	}
	if (z == NZ - 1) {
		ez[gpu_scalar_index(x, y, NZ - 1)] = ez[gpu_scalar_index(x, y, NZ - 2)];
		return;
	}
}

// =========================================================================
// Fast poisson solver domain extension
// =========================================================================

__host__ void fast_Poisson(double *charge_gpu, double *chargen_gpu, double *kx, double *ky, double *kz, cufftHandle plan) {

	checkCudaErrors(cudaMalloc((void**)&freq_gpu_ext, sizeof(cufftDoubleComplex)*mem_size_ext_scalar));
	checkCudaErrors(cudaMalloc((void**)&phi_gpu_ext, sizeof(cufftDoubleComplex)*mem_size_ext_scalar));
	checkCudaErrors(cudaMalloc((void**)&charge_gpu_ext, sizeof(cufftDoubleComplex)*mem_size_ext_scalar));


	// Extend the domain
	extension(charge_gpu, chargen_gpu, charge_gpu_ext);

	// Execute a real-to-complex 2D FFT
	CHECK_CUFFT(cufftExecZ2Z(plan, charge_gpu_ext, freq_gpu_ext, CUFFT_FORWARD));

	// Execute the derivatives in frequency domain
	derivative(kx, ky, kz, freq_gpu_ext);

	// Execute a complex-to-complex 2D IFFT
	CHECK_CUFFT(cufftExecZ2Z(plan, freq_gpu_ext, phi_gpu_ext, CUFFT_INVERSE));

	// Extraction of phi from extended domain phi_gpu_ext
	extract(phi_gpu, phi_gpu_ext);

	// Calculate electric field strength
	efield(phi_gpu, Ex_gpu, Ey_gpu, Ez_gpu);

	checkCudaErrors(cudaFree(charge_gpu_ext));
	checkCudaErrors(cudaFree(phi_gpu_ext));
	checkCudaErrors(cudaFree(freq_gpu_ext));
}

__host__ void extension(double *c, double *cn, cufftDoubleComplex *c_ext) {
	// blocks in grid
	dim3  grid(NX / nThreads, NY, NE);
	// threads in block
	dim3  threads(nThreads, 1, 1);
	odd_extension << < grid, threads >> > (c, cn, c_ext);
	getLastCudaError("Odd Extension error");
}

__global__ void odd_extension(double *charge, double *chargen, cufftDoubleComplex *charge_ext) {
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (z == 0) {
		charge_ext[gpu_scalar_index(x, y, z)].x = 0.0;
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z == 1) {
		charge_ext[gpu_scalar_index(x, y, z)].x = -convertCtoCharge*(charge[gpu_scalar_index(x, y, z)] - chargen[gpu_scalar_index(x, y, z)])/ eps - voltage / dz / dz;
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z > 1 && z < NZ - 2) {
		charge_ext[gpu_scalar_index(x, y, z)].x = -convertCtoCharge*(charge[gpu_scalar_index(x, y, z)] - chargen[gpu_scalar_index(x, y, z)]) / eps;
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z == NZ - 2) {
		charge_ext[gpu_scalar_index(x, y, z)].x = -convertCtoCharge*(charge[gpu_scalar_index(x, y, z)] - chargen[gpu_scalar_index(x, y, z)]) / eps - voltage2 / dz / dz;
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z == NZ - 1) {
		charge_ext[gpu_scalar_index(x, y, z)].x = 0.0;
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z == NZ) {
		charge_ext[gpu_scalar_index(x, y, z)].x = convertCtoCharge*(charge[gpu_scalar_index(x, y, NE-z)] - chargen[gpu_scalar_index(x, y, NE-z)]) / eps + voltage2 / dz / dz;;
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z > NZ && z<NE-1) {
		charge_ext[gpu_scalar_index(x, y, z)].x = convertCtoCharge*(charge[gpu_scalar_index(x, y, NE-z)] - chargen[gpu_scalar_index(x, y, NE-z)]) / eps;
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z == NE - 1) {
		charge_ext[gpu_scalar_index(x, y, z)].x = convertCtoCharge*(charge[gpu_scalar_index(x, y, 1)] - chargen[gpu_scalar_index(x, y, 1)]) / eps + voltage / dz / dz;
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
}

__host__ void derivative(double *kx, double *ky, double *kz, cufftDoubleComplex *source) {
	// blocks in grid
	dim3  grid(NX / nThreads, NY, NE);
	// threads in block
	dim3  threads(nThreads, 1, 1);
	gpu_derivative << < grid, threads >> > (kx, ky, kz, source);
	getLastCudaError("Gpu derivative error");
}
 
__global__ void gpu_derivative(double *kx, double *ky, double *kz, cufftDoubleComplex *source) {
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	double I = kx[x];
	double J = ky[y];
	double K = kz[z];
	double mu = (4.0 / dz / dz)*(sin(K*dz*0.5)*sin(K*dz*0.5)) + I*I + J*J;
	if (y == 0 && x == 0 && z == 0) mu = 1.0;
	source[gpu_scalar_index(x, y, z)].x = -source[gpu_scalar_index(x, y, z)].x / mu;
	source[gpu_scalar_index(x, y, z)].y = -source[gpu_scalar_index(x, y, z)].y / mu;
}

__host__ void extract(double *fi, cufftDoubleComplex *fi_ext) {
	// blocks in grid
	dim3  grid(NX / nThreads, NY, NZ);
	// threads in block
	dim3  threads(nThreads, 1, 1);
	odd_extract << < grid, threads >> > (fi, fi_ext);
	getLastCudaError("Extraction error");
}

__global__ void odd_extract(double *phi, cufftDoubleComplex *phi_ext) {
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (z == 0) {
		phi[gpu_scalar_index(x, y, z)] = voltage;
		return;
	}
	if (z == NZ-1) {
		phi[gpu_scalar_index(x, y, z)] = voltage2;
		return;
	}
	phi[gpu_scalar_index(x, y, z)] = phi_ext[gpu_scalar_index(x, y, z)].x/size;
}