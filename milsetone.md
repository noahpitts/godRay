# godRay
## GPU Acceleration of Volumetric Rendering | Isak Karlsson | Noah Pitts

## Summary
For our First Milestone we were primarily focused on transitioning over to the Nvidia OptiX ray tracing platform. This required setting up our environment in Windows with Visual Studio and leaning the necessary CUDA and OptiX programs. We set up a basic iterative pathtracing program with which we can build the volume rendering algorithms into.

## Milestones
### GPU Pathtracer Skeleton in OptiX
The pathtraching program in Optix was primarily built from referance sample programs. No single sample program contained the full set of features neccessary to build off of. However between a set 4 or 5 key samples we were able to piece together the neccessary functions to have a program that is similar in capabilities to project 3 in the course.


### Emission/Absorbtion Model
The first Volumetric Rendering model that we implemented is the emission absorption model for homogeneous mediums which is a partial solution to the full volume rendering equation.

$$L(x,\omega) = T_r(x_0\rightarrow x)L_0(x_0, -\omega) + \int_{0}^{t}T_r(x'\rightarrow x)L_e(x_0, -\omega) dt'$$

where the transmittance is simplified using beers law

$$T_r(x_0\rightarrow x_1) = e^{-\sigma_t d}$$



### Single Scattering Model
The single scattering model then adapts the emission/absorbtion model by including an in-scattering contribution making the volume rendering equation as follows:

$$L(x,\omega) = T_r(x_0\rightarrow x)L_0(x_0, -\omega) + \int_{0}^{t}T_r(x'\rightarrow x)\left[L_e(x_0, -\omega) + \sigma_s(x, \omega)\int_{s_2}\rho_p(x, \omega, \omega')L_d(x, \omega')d\omega' \right] dt'$$

### Whats Next and Review of Schedule
The next two weeks of the project we will be primarily focused on implementing the volumetric rendering portion of our project. Now that we have our basic application in place here is the order in which we plan to implement the rest of the project:
- implement the volumetric path tracing model where a scattering event may occur along the ray before a surface intersection

$$


## Video
[LINK VIDEO HERE](http://www.linkToOurVideo.com)

## ACM Template
We downloaded and began to look at the ACM template for formatting of the final report (see link below). Most of the report is still from the template but will be updated as we continue to move forward.
[godRay-acm PDF](./acm/godRay-acm.pdf)
