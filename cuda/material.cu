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


rtDeclareVariable(float3, front_hit_point, , );
rtDeclareVariable(float, t_entry, rtIntersectionDistance, );
rtDeclareVariable(float, t_exit, rtIntersectionDistance, );


rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type, , );
rtDeclareVariable(float, scene_epsilon, , );

rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_geometry, , );
rtDeclareVariable(rtObject, top_media, , );

rtDeclareVariable(float3,   Kd, , );
rtBuffer<DirectionalLight> lightBuffer;

rtDeclareVariable(float, atmos_sigma_t, , );
rtDeclareVariable(float, atmos_sigma_s, , );
rtDeclareVariable(float, atmos_g, , );
rtDeclareVariable(float, atmos_dist, , );

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
__inline__ __device__ bool atmos_scatter(unsigned int *seed, float *isect_dist)
{
    // Sample an extinction channel | pbrt 894
    int channel = min((int)rnd(*seed) * 3, 2);
    float channel_sigma_t = 1.0f;
    //TODO: clean this up so there are not so many conditional tests
    if (channel == 0) channel_sigma_t = atmos_sigma_t.x; 
    else if (channel == 1) channel_sigma_t = atmos_sigma_t.y;
    else if (channel == 2) channel_sigma_t = atmos_sigma_t.z;

    // Sample a distance along the ray | pbrt 894
    float sample_dist = -logf(rnd(*seed)) / channel_sigma_t - scene_epsilon;

    // If a scattering event occured
    if (sample_dist < *isect_dist) 
    {
        // Update the intersection
        *isect_dist = sample_dist;
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
__inline__ __device__ float hg_sample_phase(float u, float v, float g, float3 w_out, float3 *w_in)
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

    *w_in = normalize((sinTheta * cosf(phi) * v1) + (sinTheta * sinf(phi) * v2) + (cosTheta * -w_out));

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
__inline__ __device__ float3 estimate_direct_sunlight(float3 isect, int isect_type, float3 w_o, float3 ff_normal)
{
    // Sunlight will always be the first light in the buffer
    const DirectionalLight &sunlight = light_buffer[0];
    float3 Ld = make_float3(0.0f);

    // Sample Light Source with multiple importance sampling | pbrt 858
    float3 w_i = sample_sunLight(sunlight, isect);
    float light_pdf = 1.0f / (sunlight.radius * sunlight.radius * M_PIf);
    float scattering_pdf;
    float3 Li = sunlight.color;

    if (light_pdf > 0.0f && !isBlack(Li))
    {
        // Compute BSDF or phase function value for light sample | pbrt 859
        switch (isect_type)
        {
        // Evaluate phase function for light smapling | pbrt 900
        case ISECT_TYPE.ATMOS: // Atmospheric scatter
        
            float p = hg_phase(atmos_g, w_o, w_i);
            f = make_float3(p);
            scattering_pdf = p;
            break;

        // Evaluate BSDF for light sampling strategy | pbrt 859 // TODO: CHECK THIS
        case ISECT_TYPE.DIFFUSE: // Lambertian scatter
            const float NdotL = dot(ff_normal, w_i);                        // wo = ffnormal ???? check this
            f = (Kd / M_PIf) * abs(NdotL);                                  // pbrt 532 & 575 
            scattering_pdf = NdotL > 0.0f ? abs(NdotL) / M_PIf : 0.0f;      // pbrt 807
            break;
        }

        if (!isBlack(f))
        {
            // Compute effect of visibility for light source sample | prbt 859
            PerRayData_shadow prd_shadow;
            prd_shadow.blocked = false;
            prd_shadow.isect = isect;
            prd_shadow.beta = make_float3(1.0f);
            prd_shadow.in_media = prd_radiance.in_media;

            optix::Ray shadow_ray(prd_shadow.isect, w_i, /*shadow ray type*/ 1, 0.0f);
            rtTrace(top_geometry, shadow_ray, prd_shadow);

            // Test if occluded
            if (prd_shadow.blocked)
                Li = make_float3(0.0f);

            // Compute beam transmittance | 718
            else
            {
                // set temp isect to the isect
                while (true)
                {
                    optix::Ray trans_ray(prd_shadow.isect, w_i, /*shadow ray type*/ 1, 0.0f);
                    rtTrace(top_media, trans_ray, prd_shadow);
                    if (prd_shadow.done)
                        break;

                }
                Li *= prd_shadow.beta * expf(-atmos_sigma_t * atmos_dist);
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
    const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);
    float isect_dist = length(fhp - ray.origin);


    // Testing for Media Bounding interact
    prd_radiance.radiance = make_float3(1.0f, 0.0f, 0.0f);

}

// Closest Hit Program for diffuse materials
RT_PROGRAM void diffuse_hit_radiance() // closest hit
{
    // Find the distance to the closest intersection
    const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);
    float isect_dist = length(fhp - ray.origin);

    // Determine if scatter occured in atmosphere
    if (atmos_scatter(&prd_radiance.seed, &isect_dist))
    {
        // Handle Atmosphere scatter

        // Compute the transmittance and sampleing density | pbrt 894
        float3 transmittance = expf(-atmos_sigma_t * isect_dist);
        float3 density = atmos_sigma_t * transmittance;
        float atmos_pdf = (density.x + density.y + density.z) / 3.0f;

        // Set weighting factor for scattering from atmosphere | pbrt 894
        prd_radiance.beta *= transmittance * atmos_sigma_s / atmos_pdf;
        float3 isect = ray.origin + atmos_sample_dist * ray.direction;

        // Esitmate Direct Lighting // update for surface type
        prd_radiance.radiance = estimate_direct_light(isect, ISECT_TYPE.ATMOS, -prd_radiance.direction, -prd_radiance.direction) / (2.0f * M_PIf); // TODO: check this

        // Sample solid angle of atmos scatter bounce
        float3 w_in;
        float p = hg_sample_phase(rnd(prd_radiance.seed), rnd(prd_radiance.seed), atmos_g, -ray.direction, &w_in);

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
        prd_radiance.origin = front_hit_point;
        prd_radiance.direction = w_in;

        // Esitmate Direct Lighting // update for surface type
        prd_radiance.radiance = estimate_direct_light(isect, ISECT_TYPE.DIFFUSE, -prd_radiance.direction, ffnormal) / (2.0f * M_PIf); // TODO: check this
    }
}





// --------------------------------------------------
// DIRECT LIGHTING PROGRAMS
// --------------------------------------------------

// TODO
RT_PROGRAM void diffuse_hit_shadow() // any hit
{
    prd_shadow.blocked = true;
    prd_shadow.beta = make_float3(0.0f);
    rtTerminateRay();
}

// TODO
RT_PROGRAM void media_hit_shadow() // closest hit
{
    prd_shadow.done = true;
}





