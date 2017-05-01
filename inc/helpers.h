#pragma once

#include <optix.h>
#include <optix_math.h>

#include "commonStructs.h"
#include "random.h"

using namespace optix;

#define FLT_MAX         1e30;


static __device__ __inline__ float3 exp( const float3& x )
{
  return make_float3(exp(x.x), exp(x.y), exp(x.z));
}

static __device__ __inline__ float beers_law(float sigma, float dist) { return exp(-sigma * dist); }
static __device__ __inline__ float3 uniform_sample_sphere(unsigned int &seed) 
{
  float u = -1.0f + 2.0f * rnd(seed);
  float v = 2.0f * M_PI * rnd(seed);
  
  float r = sqrt(1.0f - u*u);
  float3 pt = make_float3( r * cos(v), r * sin(v), u );
  return pt / optix::length(pt);
}

static __device__ __inline__ float step( float min, float value )
{
  return value<min?0:1;
}

static __device__ __inline__ float3 mix( float3 a, float3 b, float x )
{
  return a*(1-x) + b*x;
}

static __device__ __inline__ float3 schlick( float nDi, const float3& rgb )
{
  float r = fresnel_schlick(nDi, 5, rgb.x, 1);
  float g = fresnel_schlick(nDi, 5, rgb.y, 1);
  float b = fresnel_schlick(nDi, 5, rgb.z, 1);
  return make_float3(r, g, b);
}

static __device__ __inline__ uchar4 make_color(const float3& c)
{
    return make_uchar4( static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
                        static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
                        static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
                        255u);                                                 /* A */
}

// Sample Phong lobe relative to U, V, W frame
static
__host__ __device__ __inline__ optix::float3 sample_phong_lobe( optix::float2 sample, float exponent, 
                                                                optix::float3 U, optix::float3 V, optix::float3 W )
{
  const float power = expf( logf(sample.y)/(exponent+1.0f) );
  const float phi = sample.x * 2.0f * (float)M_PIf;
  const float scale = sqrtf(1.0f - power*power);
  
  const float x = cosf(phi)*scale;
  const float y = sinf(phi)*scale;
  const float z = power;

  return x*U + y*V + z*W;
}

// Sample Phong lobe relative to U, V, W frame
static
__host__ __device__ __inline__ optix::float3 sample_phong_lobe( const optix::float2 &sample, float exponent, 
                                                                const optix::float3 &U, const optix::float3 &V, const optix::float3 &W, 
                                                                float &pdf, float &bdf_val )
{
  const float cos_theta = powf(sample.y, 1.0f/(exponent+1.0f) );

  const float phi = sample.x * 2.0f * M_PIf;
  const float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
  
  const float x = cosf(phi)*sin_theta;
  const float y = sinf(phi)*sin_theta;
  const float z = cos_theta;

  const float powered_cos = powf( cos_theta, exponent );
  pdf = (exponent+1.0f) / (2.0f*M_PIf) * powered_cos;
  bdf_val = (exponent+2.0f) / (2.0f*M_PIf) * powered_cos;  

  return x*U + y*V + z*W;
}

// Get Phong lobe PDF for local frame
static
__host__ __device__ __inline__ float get_phong_lobe_pdf( float exponent, const optix::float3 &normal, const optix::float3 &dir_out, 
                                                         const optix::float3 &dir_in, float &bdf_val)
{  
  optix::float3 r = -optix::reflect(dir_out, normal);
  const float cos_theta = fabs(optix::dot(r, dir_in));
  const float powered_cos = powf(cos_theta, exponent );

  bdf_val = (exponent+2.0f) / (2.0f*M_PIf) * powered_cos;  
  return (exponent+1.0f) / (2.0f*M_PIf) * powered_cos;
}

// Create ONB from normal.  Resulting W is parallel to normal
static
__host__ __device__ __inline__ void create_onb( const optix::float3& n, optix::float3& U, optix::float3& V, optix::float3& W )
{
  W = optix::normalize( n );
  U = optix::cross( W, optix::make_float3( 0.0f, 1.0f, 0.0f ) );

  if ( fabs( U.x ) < 0.001f && fabs( U.y ) < 0.001f && fabs( U.z ) < 0.001f  )
    U = optix::cross( W, optix::make_float3( 1.0f, 0.0f, 0.0f ) );

  U = optix::normalize( U );
  V = optix::cross( W, U );
}

// Create ONB from normalized vector
static
__device__ __inline__ void create_onb( const optix::float3& n, optix::float3& U, optix::float3& V)
{

  U = optix::cross( n, optix::make_float3( 0.0f, 1.0f, 0.0f ) );

  if ( optix::dot( U, U ) < 1e-3f )
    U = optix::cross( n, optix::make_float3( 1.0f, 0.0f, 0.0f ) );

  U = optix::normalize( U );
  V = optix::cross( n, U );
}

// Compute the origin ray differential for transfer
static
__host__ __device__ __inline__ optix::float3 differential_transfer_origin(optix::float3 dPdx, optix::float3 dDdx, float t, optix::float3 direction, optix::float3 normal)
{
  float dtdx = -optix::dot((dPdx + t*dDdx), normal)/optix::dot(direction, normal);
  return (dPdx + t*dDdx)+dtdx*direction;
}

// Compute the direction ray differential for a pinhole camera
static
__host__ __device__ __inline__ optix::float3 differential_generation_direction(optix::float3 d, optix::float3 basis)
{
  float dd = optix::dot(d,d);
  return (dd*basis-optix::dot(d,basis)*d)/(dd*sqrtf(dd));
}

// Compute the direction ray differential for reflection
static
__host__ __device__ __inline__
optix::float3 differential_reflect_direction(optix::float3 dPdx, optix::float3 dDdx, optix::float3 dNdP, 
                                             optix::float3 D, optix::float3 N)
{
  optix::float3 dNdx = dNdP*dPdx;
  float dDNdx = optix::dot(dDdx,N) + optix::dot(D,dNdx);
  return dDdx - 2*(optix::dot(D,N)*dNdx + dDNdx*N);
}

// Compute the direction ray differential for refraction
static __host__ __device__ __inline__ 
optix::float3 differential_refract_direction(optix::float3 dPdx, optix::float3 dDdx, optix::float3 dNdP, 
                                             optix::float3 D, optix::float3 N, float ior, optix::float3 T)
{
  float eta;
  if(optix::dot(D,N) > 0.f) {
    eta = ior;
    N = -N;
  } else {
    eta = 1.f / ior;
  }

  optix::float3 dNdx = dNdP*dPdx;
  float mu = eta*optix::dot(D,N)-optix::dot(T,N);
  float TN = -sqrtf(1-eta*eta*(1-optix::dot(D,N)*optix::dot(D,N)));
  float dDNdx = optix::dot(dDdx,N) + optix::dot(D,dNdx);
  float dmudx = (eta - (eta*eta*optix::dot(D,N))/TN)*dDNdx;
  return eta*dDdx - (mu*dNdx+dmudx*N);
}

// Color space conversions
static __host__ __device__ __inline__ optix::float3 Yxy2XYZ( const optix::float3& Yxy )
{
  // avoid division by zero
  if( Yxy.z < 1e-4 ) 
    return optix::make_float3( 0.0f, 0.0f, 0.0f );

  return optix::make_float3(  Yxy.y * ( Yxy.x / Yxy.z ),
                              Yxy.x,
                              ( 1.0f - Yxy.y - Yxy.z ) * ( Yxy.x / Yxy.z ) );
}

static __host__ __device__ __inline__ optix::float3 XYZ2rgb( const optix::float3& xyz)
{
  const float R = optix::dot( xyz, optix::make_float3(  3.2410f, -1.5374f, -0.4986f ) );
  const float G = optix::dot( xyz, optix::make_float3( -0.9692f,  1.8760f,  0.0416f ) );
  const float B = optix::dot( xyz, optix::make_float3(  0.0556f, -0.2040f,  1.0570f ) );
  return optix::make_float3( R, G, B );
}

static __host__ __device__ __inline__ optix::float3 Yxy2rgb( optix::float3 Yxy )
{
  // avoid division by zero
  if( Yxy.z < 1e-4 ) 
    return optix::make_float3( 0.0f, 0.0f, 0.0f );

  // First convert to xyz
  float3 xyz = optix::make_float3( Yxy.y * ( Yxy.x / Yxy.z ),
                                   Yxy.x,
                                   ( 1.0f - Yxy.y - Yxy.z ) * ( Yxy.x / Yxy.z ) );

  const float R = optix::dot( xyz, optix::make_float3(  3.2410f, -1.5374f, -0.4986f ) );
  const float G = optix::dot( xyz, optix::make_float3( -0.9692f,  1.8760f,  0.0416f ) );
  const float B = optix::dot( xyz, optix::make_float3(  0.0556f, -0.2040f,  1.0570f ) );
  return optix::make_float3( R, G, B );
}

static __host__ __device__ __inline__ optix::float3 rgb2Yxy( optix::float3 rgb)
{
  // convert to xyz
  const float X = optix::dot( rgb, optix::make_float3( 0.4124f, 0.3576f, 0.1805f ) );
  const float Y = optix::dot( rgb, optix::make_float3( 0.2126f, 0.7152f, 0.0722f ) );
  const float Z = optix::dot( rgb, optix::make_float3( 0.0193f, 0.1192f, 0.9505f ) );
  
  // avoid division by zero
  // here we make the simplifying assumption that X, Y, Z are positive
  float denominator = X + Y + Z;
  if ( denominator < 1e-4 )
    return optix::make_float3( 0.0f, 0.0f, 0.0f );

  // convert xyz to Yxy
  return optix::make_float3( Y, 
                             X / ( denominator ),
                             Y / ( denominator ) );
}

static __host__ __device__ __inline__ optix::float3 tonemap( const optix::float3 &hdr_value, float Y_log_av, float Y_max)
{
  optix::float3 val_Yxy = rgb2Yxy( hdr_value );
  
  float Y        = val_Yxy.x; // Y channel is luminance
  const float a = 0.04f;
  float Y_rel = a * Y / Y_log_av;
  float mapped_Y = Y_rel * (1.0f + Y_rel / (Y_max * Y_max)) / (1.0f + Y_rel);

  optix::float3 mapped_Yxy = optix::make_float3( mapped_Y, val_Yxy.y, val_Yxy.z ); 
  optix::float3 mapped_rgb = Yxy2rgb( mapped_Yxy ); 

  return mapped_rgb;
}

enum HitType {
  HT_MISS = 0,
  HT_DIFFUSE_RADIANCE,
  HT_DIFFUSE_SHADOW,
  HT_MEDIA
};

struct RayData
{
  unsigned int seed;
  int in_media;
  float distance;
  float3 attenuation;
  float3 color;
  float3 hit_point;
  HitType hit_type;
};
