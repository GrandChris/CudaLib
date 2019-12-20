//           $Id: main.cpp 37984 2018-10-27 15:50:30Z p20068 $
//          $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/ESD/SPS3/2018-WS/ILV/src/Snippets/bitmap-gsl/main.cpp $
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

#include "./pfc_bitmap_3.h"

void test_1 (pfc::bitmap & bmp) {
   for (auto & pixel : bmp.pixel_span ()) {   // iterate over the pixels
      pixel = {128, 123, 64};
   }

   bmp.to_file ("./bitmap-1.bmp");
}

void test_2 (pfc::bitmap & bmp) {
   auto const height {bmp.height ()};
   auto const width  {bmp.width ()};

   auto & span {bmp.pixel_span ()};

   auto * const p_buffer {std::data (span)};   // get pointer to first pixel in pixel buffer
// auto const   size     {std::size (span)};   // get size of pixel buffer

   for (int y {0}; y < height; ++y) {
      for (int x {0}; x < width; ++x) {
         p_buffer[y * width + x] = {
            pfc::byte_t (255 * y / height), 123, 64
         };
      }
   }

   bmp.to_file ("./bitmap-2.bmp");
}

int main3 () {
   pfc::bitmap bmp {1000, 750};

   test_1 (bmp);
   test_2 (bmp);

   return 0;
}
