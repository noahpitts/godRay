# godRay
Demonstration of volumetric atmospheric scattering running on a GPU via NVidia's Optix SDK.

Final Project Report: https://noahpitts.github.io/godRay/

## Authors
Noah Pitts and Isak Karlsson

## Command Line Args
    -h  || --help
    -f  || --file
    -n  || --nopbo
    -nf || --num_frames
    -md || --max_depth
    -wh || --width_height
    -as || --atmos_in_scatter
    -ae || --atmos_extinction
    -ah || --atmos_helios
    -ad || --atmos_dist
    -ag || --atmos_g
    -ce || --cam_exposure
    -cp || --cam_position
    -ct || --cam_target
    -sr || --sun_radius
    -st || --sun_theta
    -sp || --sun_phi

## Main Program Overview

Launches an OpenGL window and a cuda context.
Parses a mesh file, path hardcoded to "model/obj/temple_highres_video.obj"
Sets up geometry, intersection, bounding box, and materials for each respective mesh.
Executes OptiX for raytracing given the above configuration / context.

## INSTALLATION / SETUP

- Download and install Optix SDK 4.0.2
  * https://developer.nvidia.com/optix

- Clone the advanced sdk samples from git
  * https://github.com/nvpro-samples/optix_advanced_samples

- Clone this repo into the following folder:
  * <PATH_TO>/optix_advanced_samples/src/godRay

- Edit the following file to include godRay as a sample project.
  * <PATH>/optix_advanced_samples/src/CMakeLists.txt
  * Above add_subdirectory(optixHello) add the following:
    - add_subdirectory(godRay)

- Install CMake
- Install Visual Studios 2015

- run cmake-gui
  * set src/build
  * run configure
  * run generate
  * open project

- Compile, run, and enjoy!
