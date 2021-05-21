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

#if !defined(CHANNEL_COUNT) || !defined(FEATURE_COUNT)
#error Definition of CHANNEL_COUNT, and FEATURE_COUNT required
#define CHANNEL_COUNT 3
#define FEATURE_COUNT 5
#endif

#if CHANNEL_COUNT < 1 || FEATURE_COUNT < 1 
#error CHANNEL_COUNT, MIXTURE_COUNT, and MIXTURE_SIZE must be positive
#endif

#define ELEVATED_COUNT (FEATURE_COUNT + 1)

void phl_cpu(const float* inputs, const float* features, float* output, const uint batch_count, const uint element_count);
void phl_cuda(const float* inputs, const float* features, float* output, const uint batch_count, const uint element_count);
