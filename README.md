# godRay
for making a godRay

## Command Line Args
    -h
    -n
    -s <INT>                Maximum number of camera rays per pixel if output to file (default = 256)
    -l <INT>                Number of samples per area light (default = 32)
    -m <INT>                Maximum ray depth (default = 10)
    -r <INT> <INT>          Width and height in pixels of the image
    -f <FILENAME>           Image (.png) to save output to in a windowless mode
    -g <FLOAT> <FLOAT>      Geolocation in Latitude-Longitude (default = 41.90322, 12.49564)
    -t <INT> <FLOAT>        Day of year(0-365) and hour of day(0.0 - 23.99) (default = 78, 12.0)

    <mesh path>


## Main Program Overview
- read cmd line args
    - `-h` print help and exit - Fx
    - `-f <path>` ste `output_file_path` to `<path>`
    - `-n` set pbo
    - `-` unknown cmd option
    - All other options are added to `mesh_file_paths`

- initialize a GLFW window - Fx
- create context (set pbo)
- if no mesh files specified load default scene
- create materials
    - create glass material - Fx
    - create diffuse material - Fx
- validate context





## TODO:
- What is pbo?
- import general obj mesh file

