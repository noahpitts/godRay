
#pragma once

#include <optixu/optixu_vector_types.h>

struct PerRayData_radiance
{
  unsigned int seed;

  float3 beta;
  float3 tr;
  float3 radiance;

  float3 origin;
  float3 direction;

  int depth;

  bool done;
  bool in_media;

  // shading state
  // next ray for path tracing

};

struct PerRayData_shadow
{
    float3 isect;
    float3 beta;

    bool done;
    bool blocked;
    bool in_media;
};


