//           $Id: exception.cpp 37984 2018-10-27 15:50:30Z p20068 $
//          $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/ESD/SPS3/2018-WS/ILV/src/Snippets/exception.cpp $
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


#include "cuda_check.h"

#include <exception>
#include <stdexcept>

using namespace std;

#undef  CUDA_CHECK
#define CUDA_CHECK(error) \
   check (error, __FILE__, __LINE__)

auto make_error_message (cudaError_t const error, std::string const & file = {}, int const line = {}) {
   auto message {"CUDA error #"s};

   message += std::to_string (error);
   message += " '";
   message += cudaGetErrorString (error);
   message += "' occurred";

   if (!file.empty () && (line > 0)) {
      message += " in file '";
      message += file;
      message += "' on line ";
      message += std::to_string (line);
   }

   return message;
}

class cuda_exception final : public std::runtime_error {
   using inherited = std::runtime_error;

   public:
      explicit cuda_exception(cudaError_t const error, std::string const & file, int const line)
         : inherited {make_error_message (error, file, line)} {
      }
};

void check (cudaError_t const error, std::string const & file, int const line) {
   if (error != cudaSuccess) {
      throw cuda_exception(error, file, line);
   }
}
