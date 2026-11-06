// Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
// Modifications Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_DEBUG_H_
#define MSCCLPP_DEBUG_H_

// Deprecated compatibility declarations for the existing C logging ABI.
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  mscclppSuccess = 0,
  mscclppUnhandledCudaError = 1,
  mscclppSystemError = 2,
  mscclppInternalError = 3,
  mscclppInvalidArgument = 4,
  mscclppInvalidUsage = 5,
  mscclppRemoteError = 6,
  mscclppInProgress = 7,
  mscclppNumResults = 8
} mscclppResult_t;

typedef void (*mscclppLogHandler_t)(const char* msg);

const char* mscclppGetErrorString(mscclppResult_t result);
void mscclppDefaultLogHandler(const char* msg);
mscclppResult_t mscclppSetLogHandler(mscclppLogHandler_t handler);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MSCCLPP_DEBUG_H_
