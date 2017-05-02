#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "../inc/helpers.h"
#include "../inc/random.h"
#include "../inc/commonStructs.h"
#include "../inc/prd.h"

using namespace optix;

rtDeclareVariable(float3, object_geometric_normal, attribute object_geometric_normal, ); 
rtDeclareVariable(float3, object_shading_normal, attribute object_shading_normal, ); 

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

//rtDeclareVariable(float3, front_hit_point, , );
//rtDeclareVariable(float, t_entry, rtIntersectionDistance, );
//rtDeclareVariable(float, t_exit, rtIntersectionDistance, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type, , );
rtDeclareVariable(float, scene_epsilon, , );

rtDeclareVariable(rtObject, top_object, , );
//rtDeclareVariable(rtObject, top_geometry, , );
//rtDeclareVariable(rtObject, top_media, , );

rtDeclareVariable(float3,   Kd, , );
rtBuffer<DirectionalLight> light_buffer;

rtDeclareVariable(float3, atmos_sigma_t, , );
rtDeclareVariable(float3, atmos_sigma_s, , );
rtDeclareVariable(float, atmos_g, , );
rtDeclareVariable(float, atmos_dist, , );

rtDeclareVariable(float,    overcast, , );
rtDeclareVariable(optix::float3,   sun_direction, , );
rtDeclareVariable(optix::float3,   sun_color, , );
rtDeclareVariable(optix::float3,   sky_up, , );

rtDeclareVariable(optix::float3, inv_divisor_Yxy, ,);
rtDeclareVariable(optix::float3, c0, ,);
rtDeclareVariable(optix::float3, c1, ,);
rtDeclareVariable(optix::float3, c2, ,);
rtDeclareVariable(optix::float3, c3, ,);
rtDeclareVariable(optix::float3, c4, ,);

// --------------------------------------------------
// HELPER FUNCTIONS
// --------------------------------------------------

__inline__ __device__ bool isBlack(float3 v)
{
  return (v.x <= 0.0f && v.y <= 0.0f && v.z <= 0.0f);
}


// --------------------------------------------------
// SCATTER FUNCTIONS
// --------------------------------------------------

// TODO: FIX THIS GIVEN THE NEW INTERFACE - t_entry and t_exit
// Sample for probability of scattering in a homogeneous atmosphere before intersection 
__inline__ __device__ bool atmos_scatter(unsigned int &seed, float &isect_dist)
{
  // Sample an extinction channel | pbrt 894
  int channel = min((int)rnd(seed) * 3, 2);
  float channel_sigma_t = 1.0f;
  //TODO: clean this up so there are not so many conditional tests
  if (channel == 0) channel_sigma_t = atmos_sigma_t.x; 
  else if (channel == 1) channel_sigma_t = atmos_sigma_t.y;
  else if (channel == 2) channel_sigma_t = atmos_sigma_t.z;

  // Sample a distance along the ray | pbrt 894
  float sample_dist = -logf(rnd(seed)) / channel_sigma_t - scene_epsilon;

  // If a scattering event occured
  if (sample_dist < isect_dist) 
  {
    // Update the intersection
    isect_dist = sample_dist;
    return true;
  }
  // Other use the original intersection
  else return false;
}

// Sample for probability of scattering in a heterogeneous media before intersection
__inline__ __device__ bool media_scatter(unsigned int *seed, float *isect_dist)
{
  // 

}

// --------------------------------------------------
// INDIRECT LIGHTING PROGRAMS
// --------------------------------------------------

// Intersection Types
enum ISECT_TYPE {
  ATMOS,
  MEDIA,
  DIFFUSE
};

// Solid angle sample from Heyney-Greenstein phase function | pbrt 899
__inline__ __device__ float hg_sample_phase(float u, float v, float g, float3 w_out, float3 &w_in)
{
  // Compute cosine theta for Heyney-Greenstein sample | pbrt 899
  float cosTheta;
  if (abs(g) < 0.001f)
    cosTheta = 1.0f - 2.0f * u;
  else
  {
    float s1 = (1.0f - g * g) / (1.0f - g + 2.0f * g * u);
    cosTheta = (1.0f + g * g - s1 * s1) / (2.0f * g);
  }

  // Compute direction w_in for Heyney-Greenstein sample
  float s2 = 1.0f - cosTheta * cosTheta;
  float sinTheta = 0.0f;
  if (s2 >= sinTheta)
    sinTheta = sqrtf(s2);

  float phi = 2.0f * M_PIf * v;

  float3 v1;
  if (abs(w_out.x) > abs(w_out.y))
    v1 = make_float3(-w_out.z, 0.0f, w_out.x) / sqrtf(w_out.x * w_out.x + w_out.z * w_out.z);
  else
    v1 = make_float3(0.0f, w_out.z, -w_out.y) / sqrtf(w_out.y * w_out.y + w_out.z * w_out.z);
  float3 v2 = cross(v1, v2);

  w_in = normalize((sinTheta * cosf(phi) * v1) + (sinTheta * sinf(phi) * v2) + (cosTheta * -w_out));

  // TODO: maybe we don't need the p(w) since pdf the and function are the same by definition
  float d = 1.0f + g * g + 2.0f * g * -cosTheta;
  return (1.0f / (4.0f * M_PIf)) * (1.0f - g * g) / (d * sqrtf(d));
}

// Solid angle sample from Heyney-Greenstein phase function | pbrt 899
__inline__ __device__ float hg_phase(float g, float3 w_out, float3 w_in)
{
  // returns the phase function for henyney-Greenstein
  float d = 1.0f + g * g + 2.0f * g * dot(w_out, w_in);
  return (1.0f / (4.0f * M_PIf)) * (1.0f - g * g) / (d * sqrtf(d));
}

// Sample the SunLight
__inline__ __device__ float3 sample_sunLight(const DirectionalLight &sunlight, float3 isect)
{
  const float3 sun_center = isect + sunlight.direction;
  const float2 disk_sample = square_to_disk(make_float2(rnd(prd_radiance.seed), rnd(prd_radiance.seed)));
  const float3 jittered_pos = sun_center + sunlight.radius * disk_sample.x * sunlight.v0 + sunlight.radius * disk_sample.y * sunlight.v1;
  return normalize(jittered_pos - isect);
}

// Estimate of direct Lighting contribution
__inline__ __device__ float3 estimate_direct_light(float3 isect, int isect_type, float3 w_o, float3 ff_normal)
{
  // Sunlight will always be the first light in the buffer
  const DirectionalLight &sunlight = light_buffer[0];
  float3 Ld = make_float3(0.0f);

  // Sample Light Source with multiple importance sampling | pbrt 858
  float3 w_i = sample_sunLight(sunlight, isect);
  float light_pdf = 1.0f / (sunlight.radius * sunlight.radius * M_PIf);
  float scattering_pdf;
  float3 Li = sunlight.color;
  float3 f = make_float3(0.0f);

  if (light_pdf > 0.0f && !isBlack(Li))
  {
    // Compute BSDF or phase function value for light sample | pbrt 859
    switch (isect_type)
    {
      // Evaluate phase function for light smapling | pbrt 900
      case ATMOS: // Atmospheric scatter

        float p = hg_phase(atmos_g, w_o, w_i);
        f = make_float3(p);
        scattering_pdf = p;
        break;

        // Evaluate BSDF for light sampling strategy | pbrt 859 // TODO: CHECK THIS
      case DIFFUSE: // Lambertian scatter
        const float NdotL = dot(ff_normal, w_i);                        // wo = ffnormal ???? check this
        f = (Kd / M_PIf) * abs(NdotL);                                  // pbrt 532 & 575 
        scattering_pdf = NdotL > 0.0f ? abs(NdotL) / M_PIf : 0.0f;      // pbrt 807
        break;
    }

    if (!isBlack(f))
    {
      // Compute effect of visibility for light source sample | prbt 859
      PerRayData_shadow shadow_prd;
      shadow_prd.done = false;
      shadow_prd.blocked = false;
      shadow_prd.isect = isect;
      shadow_prd.beta = make_float3(1.0f);
      shadow_prd.in_media = prd_radiance.in_media;
      
      optix::Ray shadow_ray(isect, w_i, shadow_ray_type, scene_epsilon);
      rtTrace(top_object, shadow_ray, shadow_prd); // TODO top_geometry

      // Test if occluded
      if (shadow_prd.blocked)
        Li = make_float3(0.0f);

      // Compute beam transmittance | 718
      else
      {
        // set temp isect to the isect
        while (true)
        {
          optix::Ray trans_ray(shadow_prd.isect, w_i, shadow_ray_type, scene_epsilon);
          rtTrace(top_object, trans_ray, shadow_prd); // TODO top_media
          if (shadow_prd.done)
            break;

        }
        Li *= shadow_prd.beta * expf(-atmos_sigma_t * atmos_dist);
      }

      // Add light contribution to reflected radiance | pbrt 860
      // weight light sample for importance
      if (!isBlack(Li))
      {
        float weight = (light_pdf * light_pdf) / (light_pdf * light_pdf + scattering_pdf * scattering_pdf);
        Ld += f * Li * weight / light_pdf;
      }
    }
  }

  // TODO -----------------------------------------------------------------------------------------------

  // Sample BSDF with multiple importance sampling | 860
  // TODO: finish sampling the

  return Ld;
}



// TODO
// Closest Hit Program for media materials 
RT_PROGRAM void media_hit_radiance() // closest hit
{
  // Find the distance to the closest intersection
  //const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);
  const float3 fhp = ray.origin + t_hit * ray.direction;
  float isect_dist = length(fhp - ray.origin);


  // Testing for Media Bounding interact
  prd_radiance.radiance = make_float3(1.0f, 0.0f, 0.0f);

}

// Closest Hit Program for diffuse materials
RT_PROGRAM void diffuse_hit_radiance() // closest hit
{
  // Find the distance to the closest intersection
  //const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);
  const float3 fhp = ray.origin + t_hit * ray.direction;
  float isect_dist = length(fhp - ray.origin);

  // Determine if scatter occured in atmosphere
  if (atmos_scatter(prd_radiance.seed, isect_dist))
  {
    // Handle Atmosphere scatter

    // Compute the transmittance and sampleing density | pbrt 894
    float3 transmittance = expf(-atmos_sigma_t * isect_dist);
    float3 density = atmos_sigma_t * transmittance;
    float atmos_pdf = (density.x + density.y + density.z) / 3.0f;

    // Set weighting factor for scattering from atmosphere | pbrt 894
    prd_radiance.beta *= transmittance * atmos_sigma_s / atmos_pdf;
    float3 isect = ray.origin + isect_dist * ray.direction;
  
    // Esitmate Direct Lighting // update for surface type
    prd_radiance.radiance = estimate_direct_light(isect, ATMOS, -prd_radiance.direction, -prd_radiance.direction) / (2.0f * M_PIf); // TODO: check this

    // Sample solid angle of atmos scatter bounce
    float3 w_in = make_float3(0.0f);
    float p = hg_sample_phase(rnd(prd_radiance.seed), rnd(prd_radiance.seed), atmos_g, -ray.direction, w_in);

    // Set next ray bounce
    prd_radiance.origin = isect;
    prd_radiance.direction = w_in;
  }
  // Scatter occured on surface
  else
  {
    // Handle scatting at point on diffuse surface
    // -------------------------------------------

    // Compute the transmittance and sampleing density | pbrt 894
    float3 transmittance = expf(-atmos_sigma_t * isect_dist);
    float atmos_pdf = (transmittance.x + transmittance.y + transmittance.z) / 3.0f;

    // Return weighting factor for scattering from surface and atmosphere | pbrt 894
    if(atmos_pdf < 0.0001f) rtPrintf("PDF IS CLOSE TO 0: %f\n", atmos_pdf);
    prd_radiance.beta *= Kd * transmittance / atmos_pdf; // TODO: check this

    // ------->>>

    // Calculate world surface normal
    const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, object_shading_normal));
    const float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, object_geometric_normal));
    const float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

    // Sample surface solid angle // TODO: check this
    float3 w_in;
    optix::cosine_sample_hemisphere(rnd(prd_radiance.seed), rnd(prd_radiance.seed), w_in);
    const optix::Onb onb(ffnormal);
    onb.inverse_transform(w_in);

    // Set next ray bounce
    prd_radiance.origin = fhp;
    prd_radiance.direction = w_in;

    // Esitmate Direct Lighting // update for surface type
    prd_radiance.radiance = estimate_direct_light(fhp, DIFFUSE, -prd_radiance.direction, ffnormal) / (2.0f * M_PIf); // TODO: check this
  }
}





// --------------------------------------------------
// DIRECT LIGHTING PROGRAMS
// --------------------------------------------------

// TODO
RT_PROGRAM void diffuse_hit_shadow() // any hit
{
  prd_shadow.blocked = true;
  prd_shadow.done = true;
  prd_shadow.beta = make_float3(0.0f);
  rtTerminateRay();
}

// TODO
RT_PROGRAM void media_hit_shadow() // closest hit
{
  prd_shadow.done = true;
  rtTerminateRay();
}

//
// SUN SKY
// 

static __host__ __device__ __inline__ optix::float3 querySkyModel( bool CEL, const optix::float3& direction )
{
  using namespace optix;

  float3 overcast_sky_color = make_float3( 0.0f );
  float3 sunlit_sky_color   = make_float3( 0.0f );

  // Preetham skylight model
  if( overcast < 1.0f ) {
    float3 ray_direction = direction;
    if( CEL && dot( ray_direction, sun_direction ) > 94.0f / sqrtf( 94.0f*94.0f + 0.45f*0.45f) ) {
      sunlit_sky_color = sun_color;
    } else {
      float inv_dir_dot_up = 1.f / dot( ray_direction, sky_up ); 
      if(inv_dir_dot_up < 0.f) {
        ray_direction = reflect(ray_direction, sky_up );
        inv_dir_dot_up = -inv_dir_dot_up;
      }

      float gamma = dot(sun_direction, ray_direction);
      float acos_gamma = acosf(gamma);
      float3 A =  c1 * inv_dir_dot_up;
      float3 B =  c3 * acos_gamma;
      float3 color_Yxy = ( make_float3( 1.0f ) + c0*make_float3( expf( A.x ),expf( A.y ),expf( A.z ) ) ) *
        ( make_float3( 1.0f ) + c2*make_float3( expf( B.x ),expf( B.y ),expf( B.z ) ) + c4*gamma*gamma );
      color_Yxy *= inv_divisor_Yxy;

      color_Yxy.y = 0.33f + 1.2f * ( color_Yxy.y - 0.33f ); // Pump up chromaticity a bit
      color_Yxy.z = 0.33f + 1.2f * ( color_Yxy.z - 0.33f ); //
      float3 color_XYZ = Yxy2XYZ( color_Yxy );
      sunlit_sky_color = XYZ2rgb( color_XYZ ); 
      sunlit_sky_color /= 1000.0f; // We are choosing to return kilo-candellas / meter^2
    }
  }

  // CIE standard overcast sky model
  float Y =  15.0f;
  overcast_sky_color = make_float3( ( 1.0f + 2.0f * fabsf( direction.y ) ) / 3.0f * Y );

  // return linear combo of the two
  return lerp( sunlit_sky_color, overcast_sky_color, overcast );
}
RT_PROGRAM void radiance_miss()
{
  const bool show_sun = (prd_radiance.depth == 0);
  prd_radiance.radiance = ray.direction.y <= 0.0f ? make_float3( 0.0f ) : querySkyModel( show_sun, ray.direction );
  prd_radiance.done = true;

  // beta from the atmosphere
  prd_radiance.beta *= expf(-atmos_sigma_t * atmos_dist);
}

RT_PROGRAM void shadow_miss()
{
  prd_shadow.done = true;
}

