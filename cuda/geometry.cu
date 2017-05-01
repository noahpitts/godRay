#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "../inc/intersection_refinement.h"

using namespace optix;

rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );

rtBuffer<float3> vertex_buffer;     
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3>   index_buffer;
rtBuffer<int>    material_buffer;

rtDeclareVariable(float3, back_hit_point,   attribute back_hit_point, ); 
rtDeclareVariable(float3, front_hit_point,  attribute front_hit_point, ); 

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, object_geometric_normal, attribute object_geometric_normal, ); 
rtDeclareVariable(float3, object_shading_normal, attribute object_shading_normal, ); 

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
       object_shading_normal = object_geometric_normal = boxnormal( tmin );
       if(rtReportIntersection(0))
         check_second = false;
    } 
    if(check_second) {
      if( rtPotentialIntersection( tmax ) ) {
        texcoord = make_float3( 0.0f );
        object_shading_normal = object_geometric_normal = boxnormal( tmax );
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

template<bool DO_REFINE>
static __device__
void meshIntersect( int primIdx )
{
  const int3 v_idx = index_buffer[primIdx];

  const float3 p0 = vertex_buffer[ v_idx.x ];
  const float3 p1 = vertex_buffer[ v_idx.y ];
  const float3 p2 = vertex_buffer[ v_idx.z ];

  // Intersect ray with triangle
  float3 n;
  float  t, beta, gamma;
  if( intersect_triangle( ray, p0, p1, p2, n, t, beta, gamma ) ) {

    if(  rtPotentialIntersection( t ) ) {

      object_geometric_normal = normalize( n );
      if( normal_buffer.size() == 0 ) {
        object_shading_normal = object_geometric_normal; 
      } else {
        float3 n0 = normal_buffer[ v_idx.x ];
        float3 n1 = normal_buffer[ v_idx.y ];
        float3 n2 = normal_buffer[ v_idx.z ];
        object_shading_normal = normalize( n1*beta + n2*gamma + n0*(1.0f-beta-gamma) );
      }

      if( texcoord_buffer.size() == 0 ) {
        texcoord = make_float3( 0.0f, 0.0f, 0.0f );
      } else {
        float2 t0 = texcoord_buffer[ v_idx.x ];
        float2 t1 = texcoord_buffer[ v_idx.y ];
        float2 t2 = texcoord_buffer[ v_idx.z ];
        texcoord = make_float3( t1*beta + t2*gamma + t0*(1.0f-beta-gamma) );
      }

      if( DO_REFINE ) {
          refine_and_offset_hitpoint(
                  ray.origin + t*ray.direction,
                  ray.direction,
                  object_geometric_normal,
                  p0,
                  back_hit_point,
                  front_hit_point );
      }

      rtReportIntersection(material_buffer[primIdx]);
    }
  }
}


RT_PROGRAM void mesh_intersect( int primIdx )
{
    meshIntersect<false>( primIdx );
}


RT_PROGRAM void mesh_intersect_refine( int primIdx )
{
    meshIntersect<true>( primIdx );
}


RT_PROGRAM void mesh_bounds (int primIdx, float result[6])
{
  const int3 v_idx = index_buffer[primIdx];

  const float3 v0   = vertex_buffer[ v_idx.x ];
  const float3 v1   = vertex_buffer[ v_idx.y ];
  const float3 v2   = vertex_buffer[ v_idx.z ];
  const float  area = length(cross(v1-v0, v2-v0));

  optix::Aabb* aabb = (optix::Aabb*)result;
  
  if(area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf( fminf( v0, v1), v2 );
    aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
  } else {
    aabb->invalidate();
  }
}

