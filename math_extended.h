///////////////////////////////////////////////////////////////////////////////
// File:		  math_extemded.h
// Revision:	  1
// Date Creation: 28.10.2018
// Last Change:	  28.10.2018
// Author:		  Christian Steinbrecher
// Descrition:	  Defines some math-functions
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <math.h>


/// <summary>
/// Returns a upward runded division
/// </summary>
/// <param name="points">Array of points on the device</param>
/// <param name="indices">Array for the calculated indices on the device</param>
/// <param name="size">Size of the arrays</param>
inline size_t ceil_div(size_t const a, size_t const b)
{
	double da = static_cast<double>(a);
	double ba = static_cast<double>(b);

	return static_cast<size_t>(ceil(da / ba));
}