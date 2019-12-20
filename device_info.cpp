///////////////////////////////////////////////////////////////////////////////
// File:		  device_info.cpp
// Revision:	  1
// Date Creation: 28.10.2018
// Last Change:	  28.10.2018
// Author:		  Christian Steinbrecher
// Descrition:	  Function to print info of the gpu
///////////////////////////////////////////////////////////////////////////////

#include "cuda_check.h"
#include "cuda_runtime.h"
#include "pfc/pfc_cuda_device_info.h"

#include <iostream>

using namespace std;


/// <summary>
/// Calculates the number of cuda cores
/// </summary>
/// <param name="devProp">Device properties</param>
int getSPcores(cudaDeviceProp const devProp)
{
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major) {
	case 2: // Fermi
		if (devProp.minor == 1) cores = mp * 48;
		else cores = mp * 32;
		break;
	case 3: // Kepler
		cores = mp * 192;
		break;
	case 5: // Maxwell
		cores = mp * 128;
		break;
	case 6: // Pascal
		if (devProp.minor == 1) cores = mp * 128;
		else if (devProp.minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	case 7: // Volta
		if (devProp.minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	return cores;
}

inline auto smem_bank_width(pfc::cuda::device_info const & info) noexcept {
	return (info.smem_banks > 0) ? info.width_cl1 / info.smem_banks / 4 : 0;
}


void printDeviceInfo()
{
	// get number of devices
	int count{};
	CUDA_CHECK( cudaGetDeviceCount(&count) );

	if (count <= 0)
	{
		cout << "No device found" << endl;
	}

	// print info of device 0
	cudaDeviceProp props{};
	CUDA_CHECK( cudaGetDeviceProperties(&props, 0) );

	auto const info = pfc::cuda::get_device_info(props);

	cout << "##########################################################" << endl;
	cout << "Device (Type)" << endl;
	cout << "  Name: " << props.name << endl;
	cout << "  Compute Capability: " << props.major << "." << props.minor << endl;
	cout << endl;
	cout << "Device (Processors)" << endl;
	cout << "  Cuda Cores:             " << info.cores_sm * props.multiProcessorCount << endl;
	cout << "  SMs:                    " << props.multiProcessorCount << endl;
	cout << "  FP32 cores/SM:          " << info.cores_sm << endl;
	cout << "  clock rate:             " << props.clockRate / 1000 << " MHz" << endl;
	cout << endl;
	cout << "Device (Memory)" << endl;
	cout << "  gmem size:              " << props.totalGlobalMem / 1024 / 1024 / 1024 << " GiB" << endl;
	cout << "  cmem size:              " << props.totalConstMem / 1024 << " KiB" << endl;
	cout << "  max. smem/SM:           " << props.sharedMemPerMultiprocessor / 1024 << " KiB" << endl;
	cout << "  max. smem/block:        " << props.sharedMemPerBlock / 1024 << " KiB" << endl;
	cout << "  max. regs/SM:           " << props.regsPerMultiprocessor / 1024 << " Ki" << endl;
	cout << "  max. regs/block:        " << props.regsPerBlock / 1024 << " Ki" << endl;
	cout << "  max. regs/thread*:      " << std::min(props.regsPerBlock / props.maxThreadsPerBlock, props.regsPerMultiprocessor / props.maxThreadsPerMultiProcessor) << endl;
	cout << "  L2$ size:               " << props.l2CacheSize / 1024 << " KiB" << endl;
	cout << "  bandwidth gmem:         " << props.memoryClockRate * props.memoryBusWidth / 8 / 1024 / 1024 << " GiB/s" << endl;
	cout << "  bus width gmem:         " << props.memoryBusWidth / 8 << " Bytes" << endl;

	cout << endl;
	cout << "Blocks and Threads" << endl;
	cout << "  max. block size:        " << props.maxThreadsDim[0] << " " << props.maxThreadsDim[1] << " " << props.maxThreadsDim[2] << " " << " threads" << endl;
	cout << "  max. threads/block:     " << props.maxThreadsPerBlock << endl;
	cout << "  max. threads/SM:        " << props.maxThreadsPerMultiProcessor << endl;
	cout << "  warp size:              " << props.warpSize << " threads" << endl;
	cout << "  max. warps/SM:          " << props.maxThreadsPerMultiProcessor / props.warpSize << endl;
	cout << endl;
	cout << "Execution Timeout Enabled: " << props.kernelExecTimeoutEnabled << endl;
	cout << "##########################################################" << endl;
	cout << endl;
}
