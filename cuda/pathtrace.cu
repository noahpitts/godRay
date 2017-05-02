#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "../inc/helpers.h"
#include "../inc/random.h"
#include "../inc/prd.h"

using namespace optix;

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(float, scene_epsilon, , );
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
__inline__ __device__ float genPinholeCameraRay(float3* o, float3* d, float2 sample_pt)
{
  *o = eye;
  *d = normalize(sample_pt.x * U + sample_pt.y * V + W);
  return 1.0f;
}

// ThinLens Camera - TODO

// ---------------------------
// INTEGRATOR FUNCTIONS
// ---------------------------

// Iterative Pathtracer
__inline__ __device__ float3 Li_pathtrace(float3 ray_origin, float3 ray_dir, unsigned int seed)
{
  // Initialize per ray data structure
  PerRayData_radiance prd;
  prd.depth = 0;
  prd.seed = seed;
  //prd.done = false;

  int min_depth = 3;

  //prd.in_media = 0;
  prd.beta = make_float3(1.0f);
  prd.radiance = make_float3(0.0f); // light from a light source or miss program

  // next ray to be traced
  prd.origin = make_float3(0.0f);
  prd.direction = make_float3(0.0f);

  float3 L = make_float3(0.0f);

  // pathtrace loop. 
  for (;;)
  {
    // intersect ray with scene and store intersection radiance and attenuation(beta)
    optix::Ray ray(ray_origin, ray_dir, /*ray type*/ 0, scene_epsilon);
    rtTrace(top_object, ray, prd);

    L += prd.beta * prd.radiance;

    // terminate path if no more contribution
    //if (prd.beta.x <= 0.001f && prd.beta.y <= 0.001f && prd.beta.z <= 0.001f) prd.done = true;
    //if (prd.done)
    //{
    //  break;
    //}
    // terminate path if max depth was reached
    if (prd.depth >= max_depth)
    {
      //L += prd.beta * cutoff_color;
      break;
    }
    // russian roulette termination | pbrt 879
    if (prd.depth > min_depth)
    {
      float q = 1.0f - prd.beta.y;
      if (q < 0.05f) q = 0.05f;
      if (rnd(prd.seed) < q)
        break;
      prd.beta /= 1.0f - q;
    }

    prd.depth++;

    // Update ray data for the next path segment
    ray_origin = prd.origin;
    ray_dir = prd.direction;
  }

  return L;
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

RT_PROGRAM void render_pixel()
{
  // seed for random num generator
  unsigned int seed;

  // Initialize camera sample for current sample | pbrt 30
  float2 cameraSample = samplePixel_jitter(&seed);

  // Generate Camera Ray for current Sample | pbrt 31
  float3 ray_origin; 
  float3 ray_dir;
  float ray_weight = genPinholeCameraRay(&ray_origin, &ray_dir, cameraSample); // ray_weight is used for vignetting

  // Evaluate Radiance along Camera Ray | pbrt 31
  float3 L = make_float3(0.0f);
  if (ray_weight > 0.0f) 
    L = Li_pathtrace(ray_origin, ray_dir, seed) * ray_weight;


  // ACCUMULATE AND OUTPUT TO IMAGE

  float4 acc_val = accum_buffer[launch_index];
  if (frame > 0)
  {
    acc_val = lerp(acc_val, make_float4(L, 0.f), 1.0f / static_cast<float>(frame + 1));
  }
  else
  {
    acc_val = make_float4(L, 0.f);
  }
  output_buffer[launch_index] = make_color( tonemap( make_float3( acc_val ) ) );
  // output_buffer[launch_index] = make_color(make_float3(acc_val));
  accum_buffer[launch_index] = acc_val;
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  switch(code) {
    case RT_EXCEPTION_TEXTURE_ID_INVALID: rtPrintf("TEXTURE_ID\n"); break;
    case RT_EXCEPTION_BUFFER_ID_INVALID: rtPrintf("BUFFER_ID\n"); break;
    case RT_EXCEPTION_INDEX_OUT_OF_BOUNDS: rtPrintf("INDEX_OUT_OF_BOUNDS\n"); break;
    case RT_EXCEPTION_STACK_OVERFLOW: rtPrintf("STACK_OVERFLOW\n"); break;
    case RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS: rtPrintf("BUFFER_INDEX_OUT_OF_BOUNDS\n"); break;
    case RT_EXCEPTION_INVALID_RAY: rtPrintf("INVALID_RAY\n"); break;
    case RT_EXCEPTION_INTERNAL_ERROR: rtPrintf("INTERNAL_ERROR\n"); break;
    case RT_EXCEPTION_USER: rtPrintf("USER\n"); break;
    case RT_EXCEPTION_ALL: rtPrintf("ALL\n"); break;
    default: rtPrintf("UNKNOWN Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y);
  }
  output_buffer[launch_index] = make_color(bad_color);
}


