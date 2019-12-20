//           $Id: random.cpp 37984 2018-10-27 15:50:30Z p20068 $
//          $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/ESD/SPS3/2018-WS/ILV/src/Snippets/random.cpp $
//     $Revision: 37984 $
//         $Date: 2018-10-27 17:50:30 +0200 (Sa., 27 Okt 2018) $
//       $Author: p20068 $
//
//       Creator: Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
// Creation Date:
//     Copyright: (c) 2018 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//
//       License: This document contains proprietary information belonging to
//                University of Applied Sciences Upper Austria, Campus
//                Hagenberg. It is distributed under the Boost Software License,
//                Version 1.0 (see http://www.boost.org/LICENSE_1_0.txt).
//
//    Annotation: This file is part of the code snippets handed out during one
//                of my HPC lessons held at the University of Applied Sciences
//                Upper Austria, Campus Hagenberg.


#pragma once

#include <random>
#include <ctime>


template <typename T> T get_random_uniform(T const l, T const u) noexcept {
	static_assert (
		std::is_integral_v <T> || std::is_floating_point_v <T>,
		"get_random_uniform<T>: T must be an integral or floating-point type"
		);

	static std::mt19937_64 generator{
	   static_cast <unsigned> (std::time(nullptr))
	};

	if constexpr (std::is_integral_v <T>) {
		return std::uniform_int_distribution  <T> {l, u} (generator);
	}
	else {
		return std::uniform_real_distribution <T> {l, u} (generator);
	}
}
