/**
 * Lyra2 (v2) CUDA Implementation
 *
 * Based on djm34/VTC sources and incredible 2x boost by Nanashi Meiyo-Meijin (May 2016)
 */
#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 500
#endif

#define TPB 32

#include "cuda_lyra2_vectors.h"

static uint32_t *d_gnounce[MAX_GPUS];
static uint32_t *d_GNonce[MAX_GPUS];
__constant__ uint2 c_data[10];

__device__ uint2 *DMatrix;
__device__ uint2 *DState;

#define timeCost 2
#define nRows 330
#define nCols 256

static __device__ __forceinline__ uint2 __ldL1(const uint2 *ptr)
{
	uint2 ret;
	asm("ld.global.ca.v2.u32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR(ptr));
	return ret;

}

static __device__ __forceinline__ void __stL1(const uint2 *ptr, const uint2 value)
{
	asm("st.global.wb.v2.u32 [%0], {%1, %2};" ::__LDG_PTR(ptr), "r"(value.x), "r"(value.y));
}

static __device__ __forceinline__ void prefetch(const uint2 *ptr)
{
	asm("prefetch.global.L1 [%0+0];" ::__LDG_PTR(ptr));
	asm("prefetch.global.L1 [%0+4];" ::__LDG_PTR(ptr));
}

__device__ __forceinline__ uint2 shuffle2(const uint32_t mask, uint2 a, uint32_t b, const uint32_t c)
{
	return make_uint2(__shfl_sync(mask, a.x, b, c), __shfl_sync(mask, a.y, b, c));
}

__device__ __forceinline__
void G(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
{
	uint32_t tmp;
	a += b; d ^= a; tmp = d.x; d.x = d.y; d.y = tmp;
	c += d; b ^= c; b = ROR2(b, 24);
	a += b; d ^= a; d = ROR2(d, 16);
	c += d; b ^= c; b = ROR2(b, 63);
}

__device__ __forceinline__
void round_lyra(uint2x4 s[4], const uint32_t count)
{
	G(s[0].x, s[1].x, s[2].x, s[3].x);
	G(s[0].y, s[1].y, s[2].y, s[3].y);
	G(s[0].z, s[1].z, s[2].z, s[3].z);
	G(s[0].w, s[1].w, s[2].w, s[3].w);

	G(s[0].x, s[1].y, s[2].z, s[3].w);
	G(s[0].y, s[1].z, s[2].w, s[3].x);
	G(s[0].z, s[1].w, s[2].x, s[3].y);
	G(s[0].w, s[1].x, s[2].y, s[3].z);
}

__device__ __forceinline__ void round_lyra(uint2 &s)
{
	uint2 s0, s1, s2, s3;
	s0 = shuffle2(0xFFFFFFFF, s, threadIdx.x + 0, 16);
	s1 = shuffle2(0xFFFFFFFF, s, threadIdx.x + 4, 16);
	s2 = shuffle2(0xFFFFFFFF, s, threadIdx.x + 8, 16);
	s3 = shuffle2(0xFFFFFFFF, s, threadIdx.x + 12, 16);
	G(s0, s1, s2, s3);
	s1 = shuffle2(0xFFFFFFFF, s1, threadIdx.x + 1, 4);
	s2 = shuffle2(0xFFFFFFFF, s2, threadIdx.x + 2, 4);
	s3 = shuffle2(0xFFFFFFFF, s3, threadIdx.x + 3, 4);
	G(s0, s1, s2, s3);
	s1 = shuffle2(0xFFFFFFFF, s1, threadIdx.x + 3, 4);
	s2 = shuffle2(0xFFFFFFFF, s2, threadIdx.x + 2, 4);
	s3 = shuffle2(0xFFFFFFFF, s3, threadIdx.x + 1, 4);
	if (threadIdx.y == 0) s = s0;
	else if (threadIdx.y == 1) s = s1;
	else if (threadIdx.y == 2) s = s2;
	else if (threadIdx.y == 3) s = s3;
}

__device__ __forceinline__ void blake2bLyra(uint2x4 v[4]) {
	round_lyra(v, 0);
	round_lyra(v, 1);
	round_lyra(v, 2);
	round_lyra(v, 3);
	round_lyra(v, 4);
	round_lyra(v, 5);
	round_lyra(v, 6);
	round_lyra(v, 7);
	round_lyra(v, 8);
	round_lyra(v, 9);
	round_lyra(v, 10);
	round_lyra(v, 11);
}

__device__ __forceinline__ void reducedBlake2bLyra(uint2 &v) {
	round_lyra(v);
}

#define Mdev(r,c) DMatrix[((thread * nRows + (r)) * nCols + (c)) * 12 + threadIdx.y * 4 + threadIdx.x]
//#define Mdev(r,c) shared_mem[(((r) * nCols + (c)) * blockDim.z + threadIdx.z) * 12 + threadIdx.y * 4 + threadIdx.x]
#define Sdev(n, c) shared_mem[(((n) * 16 + (c)) * 2 + threadIdx.z) * 12 + threadIdx.y * 4 + threadIdx.x]

__device__ __forceinline__ void reducedSqueezeRow0(uint2 &state, uint32_t rowOut, const uint32_t thread)
{
	extern __shared__ uint2 shared_mem[];

	for (uint32_t i = 0; i < nCols; i++) {
		if (threadIdx.y < 3) __stL1(&Mdev(rowOut, nCols - i - 1), state);

		reducedBlake2bLyra(state);
	}
}

__device__ __forceinline__ void reducedDuplexRow1(uint2 &state, uint32_t rowIn, uint32_t rowOut, const uint32_t thread)
{
	extern __shared__ uint2 shared_mem[];
	uint2 bufIn;

	for (uint32_t i = 0; i < nCols; i++) {
		if (threadIdx.y < 3) {
			bufIn = __ldL1(&Mdev(rowIn, i));
			if (i < nCols - 1) {
				prefetch(&Mdev(rowIn, i + 1));
			}
			state ^= bufIn;
		}

		reducedBlake2bLyra(state);

		if (threadIdx.y < 3) __stL1(&Mdev(rowOut, nCols - i - 1), bufIn ^ state);
	}
}

__device__ __forceinline__ void reducedDuplexRowSetup(uint2 &state, uint32_t rowIn, uint32_t rowInOut, uint32_t rowOut, const uint32_t thread)
{
	extern __shared__ uint2 shared_mem[];
	uint32_t subthread = threadIdx.y * 4 + threadIdx.x;
	uint2 bufIn, bufInOut, bufOut, bufstate;

	if (subthread == 0) subthread = 11;
	else subthread -= 1;

	for (uint32_t i = 0; i < nCols; i++) {
		if (threadIdx.y < 3) {
			bufIn = __ldL1(&Mdev(rowIn, i));
			bufInOut = __ldL1(&Mdev(rowInOut, i));
			if (i < nCols - 1) {
				prefetch(&Mdev(rowIn, i + 1));
				prefetch(&Mdev(rowInOut, i + 1));
			}
			state ^= bufIn + bufInOut;
		}

		reducedBlake2bLyra(state);

		bufstate = shuffle2(0xFFFFFFFF, state, subthread, 16);

		if (threadIdx.y < 3) {
			bufOut = bufIn ^ state;
			__stL1(&Mdev(rowOut, nCols - i - 1), bufOut);
			__stL1(&Mdev(rowInOut, i), bufInOut ^ bufstate);
		}
	}
}

__device__ __forceinline__ void reducedDuplexRow(uint2 &state, uint32_t rowIn, uint32_t rowInOut, uint32_t rowOut, const uint32_t thread)
{
	extern __shared__ uint2 shared_mem[];
	uint32_t subthread = threadIdx.y * 4 + threadIdx.x;
	uint2 bufIn, bufInOut, bufOut, bufstate;

	if (subthread == 0) subthread = 11;
	else subthread -= 1;

	for (uint32_t i = 0; i < nCols; i++) {
		if (threadIdx.y < 3) {
			bufIn = __ldL1(&Mdev(rowIn, i));
			bufInOut = __ldL1(&Mdev(rowInOut, i));
			bufOut = __ldL1(&Mdev(rowOut, i));
			if (i < nCols - 1) {
				prefetch(&Mdev(rowIn, i + 1));
				prefetch(&Mdev(rowInOut, i + 1));
				prefetch(&Mdev(rowOut, i + 1));
			}
			state ^= bufIn + bufInOut;
		}

		reducedBlake2bLyra(state);

		bufstate = shuffle2(0xFFFFFFFF, state, subthread, 16);

		if (threadIdx.y < 3) {
			bufOut ^= state;
			if (rowOut == rowInOut) bufInOut = bufOut;
			else __stL1(&Mdev(rowOut, i), bufOut);
			__stL1(&Mdev(rowInOut, i), bufInOut ^ bufstate);
		}
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2v2_gpu_hash_32_1(uint32_t threads, const uint32_t startNonce)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	const uint2x4 blake2b_IV[2] = {
		0xf3bcc908UL, 0x6a09e667UL, 0x84caa73bUL, 0xbb67ae85UL,
		0xfe94f82bUL, 0x3c6ef372UL, 0x5f1d36f1UL, 0xa54ff53aUL,
		0xade682d1UL, 0x510e527fUL, 0x2b3e6c1fUL, 0x9b05688cUL,
		0xfb41bd6bUL, 0x1f83d9abUL, 0x137e2179UL, 0x5be0cd19UL
	};

	const uint2x4 Mask[3] = {
		0x00000020UL, 0x00000000UL, 0x00000050UL, 0x00000000UL,
		0x00000050UL, 0x00000000UL, 0x00000002UL, 0x00000000UL,
		0x0000014AUL, 0x00000000UL, 0x00000100UL, 0x00000000UL,
		0x00000080UL, 0x00000000UL, 0x00000000UL, 0x00000000UL,
		0x00000000UL, 0x00000000UL, 0x00000000UL, 0x00000000UL,
		0x00000000UL, 0x00000000UL, 0x00000000UL, 0x01000000UL
	};

	uint2x4 state[4];

	if (thread < threads)
	{
		state[0].x = c_data[0];
		state[0].y = c_data[1];
		state[0].z = c_data[2];
		state[0].w = c_data[3];
		state[1].x = c_data[4];
		state[1].y = c_data[5];
		state[1].z = c_data[6];
		state[1].w = c_data[7];
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		blake2bLyra(state);

		state[0].x ^= c_data[8];
		state[0].y.x ^= c_data[9].x;
		state[0].y.y ^= cuda_swab32(startNonce + thread);
		state[0].z ^= c_data[0];
		state[0].w ^= c_data[1];
		state[1].x ^= c_data[2];
		state[1].y ^= c_data[3];
		state[1].z ^= c_data[4];
		state[1].w ^= c_data[5];

		blake2bLyra(state);

		state[0].x ^= c_data[6];
		state[0].y ^= c_data[7];
		state[0].z ^= c_data[8];
		state[0].w.x ^= c_data[9].x;
		state[0].w.y ^= cuda_swab32(startNonce + thread);
		state[1] ^= Mask[0];
		
		blake2bLyra(state);

		state[0] ^= Mask[1];
		state[1] ^= Mask[2];

		blake2bLyra(state);

		((uint2x4*)DState)[thread * 4 + 0] = state[0];
		((uint2x4*)DState)[thread * 4 + 1] = state[1];
		((uint2x4*)DState)[thread * 4 + 2] = state[2];
		((uint2x4*)DState)[thread * 4 + 3] = state[3];
	}
}


__global__
__launch_bounds__(TPB, 1)
void lyra2v2_gpu_hash_32_2a(uint32_t threads, uint32_t offset)
{
	extern __shared__ uint2 shared_mem[];
	const uint32_t thread = blockIdx.x * blockDim.z + threadIdx.z;
	const uint32_t subthread = threadIdx.y * 4 + threadIdx.x;

	uint32_t row = 2;
	uint32_t prev = 1;
	uint32_t rowa = 0;
	uint32_t step = 1;
	uint32_t window = 2;
	uint32_t gap = 1;
	uint32_t tau = 1;

	if ((thread + offset) < threads)
	{
		uint2 state = __ldg(&DState[(thread + offset) * 16 + subthread]);

		reducedSqueezeRow0(state, 0, thread);

		reducedDuplexRow1(state, 0, 1, thread);

		for (row = 2, prev = 1, rowa = 0; row < nRows; row++) {
			reducedDuplexRowSetup(state, prev, rowa, row, thread);

			rowa = (rowa + step) & (window - 1);
			prev = row;

			if (rowa == 0) {
				step = window + gap;
				window <<= 1;
				gap = -gap;
			}
		}

		DState[(thread + offset) * 16 + subthread] = state;
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2v2_gpu_hash_32_2b(uint32_t threads, uint32_t offset)
{
	const uint32_t thread = blockIdx.x * blockDim.z + threadIdx.z;
	const uint32_t subthread = threadIdx.y * 4 + threadIdx.x;

	if ((thread + offset) < threads)
	{
		uint2 state = __ldg(&DState[(thread + offset) * 16 + subthread]);

		uint32_t row = 0;
		uint32_t prev = nRows - 1;
		uint32_t rowa;

		do {

			//rowa = __shfl_sync(0xFFFFFFFF, state.x, 0, 16) & (uint32_t)(nRows - 1);
			rowa = devectorize(shuffle2(0xFFFFFFFF, state, 0, 16)) % nRows;

			reducedDuplexRow(state, prev, rowa, row, thread);

			prev = row;

			//row = (row + step) & (uint32_t)(nRows - 1);
			row = (row + (nRows >> 1) - 1) % nRows;

		} while (row != 0);

		DState[(thread + offset) * 16 + subthread] = state;
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2v2_gpu_hash_32_2c(uint32_t threads, uint32_t offset)
{
	const uint32_t thread = blockIdx.x * blockDim.z + threadIdx.z;
	const uint32_t subthread = threadIdx.y * 4 + threadIdx.x;

	if ((thread + offset) < threads)
	{
		uint2 state = __ldg(&DState[(thread + offset) * 16 + subthread]);

		uint32_t row = 0;
		uint32_t prev = (nRows >> 1) + 1;
		uint32_t rowa;

		do {

			//rowa = __shfl_sync(0xFFFFFFFF, state.x, 0, 16) & (uint32_t)(nRows - 1);
			rowa = devectorize(shuffle2(0xFFFFFFFF, state, 0, 16)) % nRows;

			reducedDuplexRow(state, prev, rowa, row, thread);

			prev = row;

			//row = (row + step) & (uint32_t)(nRows - 1);
			row = (row + 15) & 15;

		} while (row != 0);

		if (threadIdx.y < 3) state ^= Mdev(rowa, 0);

		DState[(thread + offset) * 16 + subthread] = state;
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2v2_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint32_t target, uint32_t *const __restrict__ nonceVector)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint2x4 state[4];

	if (thread < threads)
	{
		state[0] = __ldg4(&((uint2x4*)DState)[thread * 4 + 0]);
		state[1] = __ldg4(&((uint2x4*)DState)[thread * 4 + 1]);
		state[2] = __ldg4(&((uint2x4*)DState)[thread * 4 + 2]);
		state[3] = __ldg4(&((uint2x4*)DState)[thread * 4 + 3]);

		blake2bLyra(state);

		if (state[0].w.y <= target)
		{
			uint32_t tmp = atomicExch(&nonceVector[0], startNounce + thread);
			if (tmp != 0)
				nonceVector[1] = tmp;
		}
	}
}

__host__
void lyra2z330_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix, uint64_t *d_state)
{
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DMatrix, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(DState, &d_state, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);

	cudaMalloc(&d_GNonce[thr_id], 2 * sizeof(uint32_t));
	cudaMallocHost(&d_gnounce[thr_id], 2 * sizeof(uint32_t));
}

__host__
void lyra2z330_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resultnonces, uint32_t target)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	if (device_sm[dev_id] < 500) {
		cudaFuncSetCacheConfig(lyra2v2_gpu_hash_32_2a, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(lyra2v2_gpu_hash_32_2b, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(lyra2v2_gpu_hash_32_2c, cudaFuncCachePreferShared);
	}
	else if (device_sm[dev_id] >= 700) {
		cudaFuncSetAttribute(lyra2v2_gpu_hash_32_2a, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
		cudaFuncSetAttribute(lyra2v2_gpu_hash_32_2b, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
		cudaFuncSetAttribute(lyra2v2_gpu_hash_32_2c, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
	}

	const uint32_t tpb = TPB;

	dim3 grid((threads + tpb - 1) / tpb);	// 4096
	dim3 block1(tpb);
	dim3 block2(4, 4, tpb / 16);

	cudaMemset(d_GNonce[thr_id], 0, 2 * sizeof(uint32_t));

	lyra2v2_gpu_hash_32_1 << < grid, block1 >> > (threads, startNounce);
	for (uint32_t i = 0; i < 16; i++)
	{
		lyra2v2_gpu_hash_32_2a << < grid, block2 >> > (threads, grid.x * 2 * i);
		lyra2v2_gpu_hash_32_2b << < grid, block2 >> > (threads, grid.x * 2 * i);
		lyra2v2_gpu_hash_32_2c << < grid, block2 >> > (threads, grid.x * 2 * i);
	}
	lyra2v2_gpu_hash_32_3 << < grid, block1 >> > (threads, startNounce, target, d_GNonce[thr_id]);


	cudaMemcpy(d_gnounce[thr_id], d_GNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	resultnonces[0] = *(d_gnounce[thr_id]);
	resultnonces[1] = *(d_gnounce[thr_id] + 1);
}

__host__
void lyra2z330_cpu_free(int thr_id)
{
	cudaFree(d_GNonce[thr_id]);
	cudaFreeHost(d_gnounce[thr_id]);
}

__host__
void lyra2z330_setData(const void *data)
{
	cudaMemcpyToSymbol(c_data, data, 80, 0, cudaMemcpyHostToDevice);
}
