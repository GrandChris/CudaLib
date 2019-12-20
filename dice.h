///////////////////////////////////////////////////////////////////////////////
// File:		  dice.h
// Revision:	  1
// Date Creation: 28.10.2018
// Last Change:	  28.10.2018
// Author:		  Christian Steinbrecher
// Descrition:	  Random number generator
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "random.h"

/// <summary>
/// Creates a random number
/// </summary>
/// <param name="range">Range of random numbers (from 0 to range)</param>
/// <return>The generated random number</return>
template<typename T>
T dice(size_t const range)
{
	return get_random_uniform<T>(0, static_cast<T>(range));
}