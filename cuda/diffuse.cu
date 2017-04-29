#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "prd.h"
#include "random.h"
#include "commonStructs.h"

using namespace optix;

rtDeclareVariable( float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable( float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable( float4, geometry_color, attribute geometry_color, );

rtDeclareVariable(optix::Ray, ray,   rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow, rtPayload, );

rtDeclareVariable( float3, Kd, , );
rtDeclareVariable(rtObject,      top_object, , );


rtDeclareVariable(float3, atmos_sigma_t, , );
rtDeclareVariable(float3, atmos_sigma_s, , );
rtDeclareVariable(float, atmos_g, , );



rtDeclareVariable(float, atmos_dist, , );

rtBuffer<DirectionalLight> light_buffer;

__inline__ __device__ bool isBlack(float3 v)
{
    return (v.x <= 0.0f && v.y <= 0.0f && v.z <= 0.0f);
}

__inline__ __device__ bool isNotBlack(float3 v)
{
    return !(v.x <= 0.0f && v.y <= 0.0f && v.z <= 0.0f);
}

// Sample for probability of scattering in atmosphere before surface intersection
__inline__ __device__ bool atmosphereic_scatter(unsigned int seed, float3 spectrum_sigma_t, float isect_dist, float* sample_dist)
{
    // Sample an extinction channel | pbrt 894
    int channel = min((int)rnd(seed) * 3, 2);
    float channel_sigma_t = 0.0f;
    if (channel == 0) channel_sigma_t = spectrum_sigma_t.x; //TODO: clean this up so there are not so many conditional tests
    else if (channel == 1) channel_sigma_t = spectrum_sigma_t.y;
    else if (channel == 2) channel_sigma_t = spectrum_sigma_t.z;

    // Sample a distance along the ray | pbrt 894
    *sample_dist = -logf(1.0f - rnd(seed)) / channel_sigma_t;
    return *sample_dist < isect_dist;
}

// Solid angle sample from Heyney-Greenstein phase function | pbrt 899
__inline__ __device__ float hg_sample_phase(float u, float v, float g, float3 w_out, float3* w_in)
{
    // Compute cosine theta for Heyney-Greenstein sample | pbrt 899
    float cosTheta;
    if (abs(g) < 0.001f) cosTheta = 1.0f - 2.0f * u;
    else
    {
        float s1 = (1.0f - g * g) / (1.0f - g + 2.0f * g * u);
        cosTheta = (1.0f + g * g - s1 * s1) / (2.0f * g);
    }

    // Compute direction w_in for Heyney-Greenstein sample
    float s2 = 1.0f - cosTheta * cosTheta;
    float sinTheta = 0.0f;
    if (s2 >= sinTheta) sinTheta = sqrtf(s2);

    float phi = 2.0f * M_PIf * v;

    float3 v1;
    if (abs(w_out.x) > abs(w_out.y))
        v1 = make_float3(-w_out.z, 0.0f, w_out.x) / sqrtf(w_out.x * w_out.x + w_out.z * w_out.z);
    else
        v1 = make_float3(0.0f, w_out.z, -w_out.y) / sqrtf(w_out.y * w_out.y + w_out.z * w_out.z);
    float3 v2 = cross(v1, v2);

    *w_in = normalize((sinTheta * cosf(phi) * v1) + (sinTheta * sinf(phi) * v2) + (cosTheta * -w_out));

    // TODO: maybe we don't need the p(w) since pdf the and function are the same by definition
    float d = 1.0f + g * g + 2.0f * g * -cosTheta;
    return (1.0f / (4.0f * M_PIf)) * (1.0f - g * g) / (d * sqrtf(d));
}

// Sample the SunLight
__inline__ __device__ float3 sample_sunLight(const DirectionalLight& sunlight, float3 isect)
{
    const float3 sun_center = isect + sunlight.direction;
    const float2 disk_sample = square_to_disk(make_float2(rnd(prd_radiance.seed), rnd(prd_radiance.seed)));
    const float3 jittered_pos = sun_center + sunlight.radius * disk_sample.x * sunlight.v0 + sunlight.radius * disk_sample.y * sunlight.v1;
    return normalize(jittered_pos - isect);
}

// Test media visibility
__inline__ __device__ float3 visible_transmittance(float3 isect)
{
    float3 Tr = make_float3(1.0f);
    return Tr * expf(-atmos_sigma_t * atmos_dist); // TODO: Fix This | pbrt 718
}

// Estimate of direct Lighting contribution
__inline__ __device__ float3 estimate_direct_sunlight(float3 isect, bool surface, float3 ff_normal)
{
    // Sunlight will always be the first light in the buffer
    const DirectionalLight& sunlight = light_buffer[0]; 
    float3 Ld = make_float3(0.0f);

    // Sample Light Source with multiple importance sampling | pbrt 858
    float3 wi = sample_sunLight(sunlight, isect);
    float light_pdf = 1.0f / (sunlight.radius * sunlight.radius * M_PIf);
    float scattering_pdf;
    float3 Li = sunlight.color;

    if (light_pdf > 0.0f && isNotBlack(Li))
    {
        // Compute BSDF or phase function value for light sample
        float3 f;
        if (surface) 
        {
            // BSDF - Lambertian // EDIT THIE FOR OTHER SURFACE BSDFs
            const float NdotL = dot(ff_normal, wi); // wo = ffnormal
            f = (Kd * make_float3(geometry_color) / M_PIf) * abs(NdotL); //  pbrt 532 & 575
            scattering_pdf = NdotL > 0.0f ? abs(NdotL) / M_PIf : 0.0f; // pbrt 807
        }
        else
        {
            // phase function
            f = make_float3(1.0f); // TODO 
            scattering_pdf = 1.0f;
        }

        if (isNotBlack(f))
        {
            // Test for visibility
            PerRayData_shadow shadow_prd;
            shadow_prd.blocked = false;
            optix::Ray shadow_ray(isect, wi, /*shadow ray type*/ 1, 0.0f);
            rtTrace(top_object, shadow_ray, shadow_prd);

            // Test if occluded
            if (shadow_prd.blocked)
                Li = make_float3(0.0f);
            else
                Li *= visible_transmittance(isect);

            // weight light sample for importance
            if (!(Li.x <= 0.0f && Li.y <= 0.0f && Li.z <= 0.0f))
            {
                float weight = (light_pdf * light_pdf) / (light_pdf * light_pdf + scattering_pdf * scattering_pdf);
                Ld += f * Li * weight / light_pdf;
            }
        }
    }
    // Sample BSDF with multiple importance sampling | 860

    return Ld;
}

RT_PROGRAM void any_hit_shadow()
{
    prd_shadow.blocked = true;
    prd_shadow.beta = make_float3( 0.0f );
    rtTerminateRay();
}

// Note: both the hemisphere and direct light sampling below use pure random numbers to avoid any patent issues.
// Stratified sampling or QMC would improve convergence.  Please keep this in mind when judging noise levels.

RT_PROGRAM void closest_hit_radiance()
{

    const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);

    float intersection_dist = length(fhp - ray.origin);
    float atmos_sample_dist = intersection_dist;

    // Sample the atmosphere medium and handle interaction with either medium or surface
    if (atmosphereic_scatter(prd_radiance.seed, atmos_sigma_t, intersection_dist, &atmos_sample_dist))
    {
        // Handle scattering at point in atmospheric medium
        // ------------------------------------------------

        // Compute the transmittance and sampleing density | pbrt 894
        float3 transmittance = expf(-atmos_sigma_t * atmos_sample_dist);
        float3 density = atmos_sigma_t * transmittance;
        float atmos_pdf = (density.x + density.y + density.z) / 3.0f;

        // Return weighting factor for scattering from atmosphere | pbrt 894
        prd_radiance.beta *= transmittance * atmos_sigma_s / atmos_pdf; // TODO: check this

        // Sample solid angle of atmos scatter
        float3 w_in;
        float p = hg_sample_phase(rnd(prd_radiance.seed), rnd(prd_radiance.seed), atmos_g, -ray.direction, &w_in);

        // Set next ray bounce
        prd_radiance.origin = ray.origin + atmos_sample_dist * ray.direction;
        prd_radiance.direction = w_in;

        // Sample Direct Lighting
        prd_radiance.radiance += estimate_direct_sunlight(prd_radiance.origin, false, prd_radiance.direction) / (2.0f * M_PIf);

    }
    else {
        // Handle scatting at point on diffuse surface
        // -------------------------------------------

        // Compute the transmittance and sampleing density | pbrt 894
        float3 transmittance = expf(-atmos_sigma_t * intersection_dist);
        float atmos_pdf = (transmittance.x + transmittance.y + transmittance.z) / 3.0f;

        // Return weighting factor for scattering from surface and atmosphere | pbrt 894
        prd_radiance.beta *= Kd * make_float3(geometry_color) * transmittance / atmos_pdf; // TODO: check this

        // ------->>>

        // Calculate world surface normal
        const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
        const float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
        const float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

        // Sample surface solid angle
        float3 w_in;
        optix::cosine_sample_hemisphere(rnd(prd_radiance.seed), rnd(prd_radiance.seed), w_in);
        const optix::Onb onb(ffnormal);
        onb.inverse_transform(w_in);

        // Set next ray bounce
        prd_radiance.origin = front_hit_point;
        prd_radiance.direction = w_in;

        // Sample Direct Lighting


    }

    // -----------------

    // prd_radiance.beta *= Kd * make_float3( geometry_color );

    // Add direct light sample weighted by shadow term and 1/probability.
    // The pdf for a directional area light is 1/solid_angle.

    //const DirectionalLight& light = light_buffer[0];
    //const float3 light_center = fhp + light.direction;
    //const float r1 = rnd( prd_radiance.seed );
    //const float r2 = rnd( prd_radiance.seed );
    //const float2 disk_sample = square_to_disk( make_float2( r1, r2 ) );
    //const float3 jittered_pos = light_center + light.radius*disk_sample.x*light.v0 + light.radius*disk_sample.y*light.v1;
    //const float3 L = normalize( jittered_pos - fhp );

    //const float NdotL = dot( ffnormal, L);
    //if(NdotL > 0.0f) {
    //    PerRayData_shadow shadow_prd;
    //    shadow_prd.beta = make_float3( 1.0f );
    //    optix::Ray shadow_ray ( fhp, L, /*shadow ray type*/ 1, 0.0f );
    //    rtTrace(top_object, shadow_ray, shadow_prd);

    //    const float solid_angle = light.radius*light.radius*M_PIf;
    //   
    //    float3 contribution = NdotL * light.color * solid_angle * shadow_prd.beta;
    //    prd_radiance.radiance += contribution * exp(-sigma_a * atmosphere);
    //}
    

}

