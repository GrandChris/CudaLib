///////////////////////////////////////////////////////////////////////////////
// File:		  pinned_memeory.h
// Revision:	  1
// Date Creation: 28.10.2018
// Last Change:	  28.10.2018
// Author:		  Christian Steinbrecher
// Descrition:	  Functions for encapsuation creation and deletion of 
//				  pinned memory on the host
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "cuda_check.h"

#include <memory>
#include <iostream>


/// <summary>
/// Used as deleter for std::unique_ptr to delete memory of the gpu
/// </summary>
class cudaFreeHost_deleter
{
public:
	/// <summary>
	/// Deletes memory on the gpu
	/// </summary>
	/// <param name="p">Pointer to allocated memory</param>
	template <typename T>
	void operator()(T * p) noexcept;
};


/// <summary>
/// Typedef for unique_ptr with memory of the gpu
/// </summary>
template<typename T>
typename std::unique_ptr<T, cudaFreeHost_deleter> pinned_unique_ptr;


/// <summary>
/// Creates memory on the gpu
/// </summary>
/// <param name="size">Array size</param>
/// <return>unique_ptr to allocated memory</return>
template<typename T>
std::unique_ptr<T, cudaFreeHost_deleter> pinned_make_unique(size_t const size);








// #######+++++++ Implementation +++++++#######

template<typename T>
inline void cudaFreeHost_deleter::operator()(T * p) noexcept
{
	// will be called in D-Tor of unique-ptr
	// exceptions are not allowed in D-Tor
	// however, we can print the error on the console, if free fails
	cudaError_t const error = cudaFreeHost(p);

	if (error != cudaSuccess)
	{
		std::cerr << cudaGetErrorString(error);
	}
}


template<typename T>
inline std::unique_ptr<T, cudaFreeHost_deleter> pinned_make_unique(size_t const size)
{
	using remove_extent_t = typename std::remove_extent<T>::type;

	remove_extent_t * dp = nullptr;
	CUDA_CHECK(cudaMallocHost(&dp, size * sizeof(remove_extent_t)));
	return std::unique_ptr<T, cudaFreeHost_deleter>(dp);
}
