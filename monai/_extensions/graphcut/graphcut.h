/*
Copyright 2020 - 2021 MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#if !defined(CHANNEL_COUNT) || !defined(DIMENSION_COUNT) || !defined(USE_DIAGONALS)
    #error Definition of CHANNEL_COUNT, DIMENSION_COUNT, and USE_DIAGONALS required
#endif

#if CHANNEL_COUNT < 1 
    #error CHANNEL_COUNT must be greater than 0
#endif

#if DIMENSION_COUNT < 1 || DIMENSION_COUNT > 3
    #error DIMENSION_COUNT must be 1, 2, or 3
#endif

#if USE_DIAGONALS != true && USE_DIAGONALS != false
    #error USE_DIAGONALS must be true or false
#endif

#if USE_DIAGONALS
    #if DIMENSION_COUNT == 1
        #define CONNECTION_COUNT 2
    #elif DIMENSION_COUNT == 2
        #define CONNECTION_COUNT 8
    #elif DIMENSION_COUNT == 3
        #define CONNECTION_COUNT 26
    #endif
#else
    #define CONNECTION_COUNT (2 * DIMENSION_COUNT)
#endif

//defining dummy values for ease of programming
#ifndef CHANNEL_COUNT 
    #define CHANNEL_COUNT 3
    #define DIMENSION_COUNT 2
    #define USE_DIAGONALS false
#endif

void graphcut_cuda(const float* image, const float* weights, float* output, const uint batch_count, const uint element_count, const uint sizes[DIMENSION_COUNT], const uint strides[DIMENSION_COUNT]);
void graphcut_cpu(const float* image, const float* weights, float* output, const uint batch_count, const uint element_count, const uint sizes[DIMENSION_COUNT], const uint strides[DIMENSION_COUNT]);
