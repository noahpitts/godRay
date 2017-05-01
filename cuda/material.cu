#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "../inc/helpers.h"
#include "../inc/random.h"
#include "../inc/commonStructs.h"
#include "../inc/prd.h"

using namespace optix;

rtDeclareVariable(float3, object_geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, object_shading_normal, attribute shading_normal, ); 

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

rtDeclareVariable(float, t_entry, rtIntersectionDistance, );
rtDeclareVariable(float, t_exit, rtIntersectionDistance, );


rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

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


RT_PROGRAM void media_hit_radiance() // closest hit
{
}

RT_PROGRAM void media_hit_shadow() // closest hit
{
}




RT_PROGRAM void diffuse_hit_radiance() // closest hit
{
}

RT_PROGRAM void diffuse_hit_shadow() // any hit
{
}





