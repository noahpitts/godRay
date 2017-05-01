#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );

rtDeclareVariable(float4, plane, , );
rtDeclareVariable(float3, v1, , );
rtDeclareVariable(float3, v2, , );
rtDeclareVariable(float3, anchor, , );

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

rtDeclareVariable(int, lgt_instance, , ) = {0};
rtDeclareVariable(int, lgt_idx, attribute lgt_idx, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

static __device__ float3 boxnormal(float t)
{
  float3 t0 = (boxmin - ray.origin)/ray.direction;
  float3 t1 = (boxmax - ray.origin)/ray.direction;
  float3 neg = make_float3(t==t0.x?1:0, t==t0.y?1:0, t==t0.z?1:0);
  float3 pos = make_float3(t==t1.x?1:0, t==t1.y?1:0, t==t1.z?1:0);
  return pos-neg;
}

RT_PROGRAM void box_intersect(int)
{
  float3 t0 = (boxmin - ray.origin)/ray.direction;
  float3 t1 = (boxmax - ray.origin)/ray.direction;
  float3 near = fminf(t0, t1);
  float3 far = fmaxf(t0, t1);
  float tmin = fmaxf( near );
  float tmax = fminf( far );

  if(tmin <= tmax) {
    bool check_second = true;
    if( rtPotentialIntersection( tmin ) ) {
       texcoord = make_float3( 0.0f );
       shading_normal = geometric_normal = boxnormal( tmin );
       if(rtReportIntersection(0))
         check_second = false;
    } 
    if(check_second) {
      if( rtPotentialIntersection( tmax ) ) {
        texcoord = make_float3( 0.0f );
        shading_normal = geometric_normal = boxnormal( tmax );
        rtReportIntersection(0);
      }
    }
  }
}

RT_PROGRAM void box_bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->set(boxmin, boxmax);
}

RT_PROGRAM void par_intersect(int primIdx)
{
  float3 n = make_float3( plane );
  float dt = dot(ray.direction, n );
  float t = (plane.w - dot(n, ray.origin))/dt;
  if( t > ray.tmin && t < ray.tmax ) {
    float3 p = ray.origin + ray.direction * t;
    float3 vi = p - anchor;
    float a1 = dot(v1, vi);
    if(a1 >= 0 && a1 <= 1){
      float a2 = dot(v2, vi);
      if(a2 >= 0 && a2 <= 1){
        if( rtPotentialIntersection( t ) ) {
          shading_normal = geometric_normal = n;
          texcoord = make_float3(a1,a2,0);
          lgt_idx = lgt_instance;
          rtReportIntersection( 0 );
        }
      }
    }
  }
}

RT_PROGRAM void par_bounds (int, float result[6])
{
  // v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
  const float3 tv1  = v1 / dot( v1, v1 );
  const float3 tv2  = v2 / dot( v2, v2 );
  const float3 p00  = anchor;
  const float3 p01  = anchor + tv1;
  const float3 p10  = anchor + tv2;
  const float3 p11  = anchor + tv1 + tv2;
  const float  area = length(cross(tv1, tv2));
  
  optix::Aabb* aabb = (optix::Aabb*)result;
  
  if(area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf( fminf( p00, p01 ), fminf( p10, p11 ) );
    aabb->m_max = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ) );
  } else {
    aabb->invalidate();
  }
}

