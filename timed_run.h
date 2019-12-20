//           $Id: chrono.cpp 37984 2018-10-27 15:50:30Z p20068 $
//          $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/ESD/SPS3/2018-WS/ILV/src/Snippets/chrono.cpp $
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

#include <chrono>
#include <ostream>

template <typename clock_t = std::chrono::steady_clock, typename size_t, typename fun_t, typename ...args_t> 
auto timed_run (size_t const n, fun_t && fun, args_t && ...args) noexcept (std::is_nothrow_invocable_v <fun_t, args_t...>) {
   static_assert (clock_t::is_steady);
   static_assert (std::is_integral_v <size_t>);

   typename clock_t::duration elapsed {};

   if (0 < n) {
      auto const start {clock_t::now ()};

      for (int i {0}; i < n; ++i) {
         std::invoke (std::forward <fun_t> (fun), std::forward <args_t> (args)...);
      }

      elapsed = (clock_t::now () - start) / n;
   }

   return elapsed;
}


std::ostream & operator<<(std::ostream & ost, std::chrono::steady_clock::duration const & time)
{
	ost << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms";
	return ost;
}