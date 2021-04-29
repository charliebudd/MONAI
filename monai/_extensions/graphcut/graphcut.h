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

#if USE_DIGONALS
#define CONNECTION_COUNT 8
#else
#define CONNECTION_COUNT 4
#endif

void graphcut_cuda(const float* inputs, const float* weights, float* output, const uint sizes[DIMENSION_COUNT], const uint element_count);
void graphcut_cpu(const float* inputs, const float* weights, float* output, const uint sizes[DIMENSION_COUNT], const uint element_count);

//defining dummy values for ease of programming
#ifndef CHANNEL_COUNT 
#define CHANNEL_COUNT 3
#endif
#ifndef DIMENSION_COUNT 
#define DIMENSION_COUNT 2
#endif
#ifndef USE_DIAGONALS 
#define USE_DIAGONALS false
#endif