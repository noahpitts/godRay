
#pragma once

#include <optixu/optixu_vector_types.h>

struct PerRayData_radiance
{
  
  unsigned int seed;

  // shading state
  bool done;
  float3 beta;
  float3 radiance;

  // next ray for path tracing
  int depth;
  float3 origin;
  float3 direction;

  int in_media;
};

struct PerRayData_shadow
{
    float3 isect;
    float3 beta;

    bool done;
    bool blocked;

    int in_media;
};


