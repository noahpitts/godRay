#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "prd.h"
#include "random.h"

using namespace optix;

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(float3, cutoff_color, , );
rtDeclareVariable(int, max_depth, , );
rtBuffer<uchar4, 2> output_buffer;
rtBuffer<float4, 2> accum_buffer;
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, frame, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

// ------------------------
// PIXEL SAMPLING FUNCTIONS
// ------------------------

// Subpixel jitter: send the ray through a different position inside the pixel each time
__inline__ __device__ float2 samplePixel_jitter(unsigned int *seed)
{
  size_t2 screen = output_buffer.size();
  *seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);
  float2 subpixel_jitter = frame == 0 ? make_float2(0.0f) : make_float2(rnd(*seed) - 0.5f, rnd(*seed) - 0.5f);

  return (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
}

// -----------------------------
// CAMERA RAY FUNCTIONS
// -----------------------------

// Pinhole Camera
__inline__ __device__ void genPinholeCameraRay(float3* o, float3* d, float2 sample_pt)
{
    *o = eye;
    *d = normalize(sample_pt.x * U + sample_pt.y * V + W);
}

// ----------------------
// TONE MAPPING FUNCTIONS
// ----------------------
__inline__ __device__ float3 tonemap(const float3 in)
{
  // hard coded exposure for sun/sky
  const float exposure = 1.0f / 30.0f;
  float3 x = exposure * in;

  // "filmic" map from a GDC talk by John Hable.  This includes 1/gamma.
  x = fmaxf(x - make_float3(0.004f), make_float3(0.0f));
  float3 ret = (x * (6.2f * x + make_float3(.5f))) / (x * (6.2f * x + make_float3(1.7f)) + make_float3(0.06f));

  return ret;
}

RT_PROGRAM void pinhole_camera()
{
  // seed for random num generator
  unsigned int seed;

  // sample the pixel
  float2 xy = samplePixel_jitter(&seed);

  // generate a pinhole camera ray
  float3 ray_origin; 
  float3 ray_dir;
  genPinholeCameraRay(&ray_origin, &ray_dir, xy);




  PerRayData_radiance prd;
  prd.depth = 0;
  prd.seed = seed;
  prd.done = false;

  // These represent the current shading state and will be set by the closest-hit or miss program

  // brdf attenuation from surface interaction
  prd.attenuation = make_float3(1.0f);

  // light from a light source or miss program
  prd.radiance = make_float3(0.0f);

  // next ray to be traced
  prd.origin = make_float3(0.0f);
  prd.direction = make_float3(0.0f);

  float3 result = make_float3(0.0f);

  // Main render loop. This is not recursive, and for high ray depths
  // will generally perform better than tracing radiance rays recursively
  // in closest hit programs.
  for (;;)
  {
    optix::Ray ray(ray_origin, ray_dir, /*ray type*/ 0, scene_epsilon);
    rtTrace(top_object, ray, prd);

    result += prd.attenuation * prd.radiance;

    if (prd.done)
    {
      break;
    }
    else if (prd.depth >= max_depth)
    {
      result += prd.attenuation * cutoff_color;
      break;
    }

    prd.depth++;

    // Update ray data for the next path segment
    ray_origin = prd.origin;
    ray_dir = prd.direction;
  }

  float4 acc_val = accum_buffer[launch_index];
  if (frame > 0)
  {
    acc_val = lerp(acc_val, make_float4(result, 0.f), 1.0f / static_cast<float>(frame + 1));
  }
  else
  {
    acc_val = make_float4(result, 0.f);
  }
  // output_buffer[launch_index] = make_color( tonemap( make_float3( acc_val ) ) );
  output_buffer[launch_index] = make_color(make_float3(acc_val));
  accum_buffer[launch_index] = acc_val;
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf("Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y);
  output_buffer[launch_index] = make_color(bad_color);
}
