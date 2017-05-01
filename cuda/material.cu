#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "../inc/helpers.h"
#include "../inc/random.h"
#include "../inc/commonStructs.h"
#include "../inc/prd.h"

using namespace optix;

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

rtDeclareVariable(RayData, rd_result, rtPayload, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type, , );
rtDeclareVariable(float, scene_epsilon, , );

rtDeclareVariable(rtObject, top_object, , );

rtDeclareVariable(float3,   Kd, , ); 
rtBuffer<DirectionalLight> lightBuffer;

rtDeclareVariable(float, atmos_sigma_t, , );
rtDeclareVariable(float, atmos_sigma_s, , );
rtDeclareVariable(float, atmos_g, , );
rtDeclareVariable(float, atmos_dist, , );


RT_PROGRAM void diffuse_hit_radiance() // closest hit
{
}

RT_PROGRAM void diffuse_hit_shadow() // any hit
{
}

RT_PROGRAM void media_hit_radiance() // closest hit
{
}

RT_PROGRAM void media_hit_shadow() // closest hit
{
}