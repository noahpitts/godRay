/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//#ifndef __APPLE__
#include <GL/glew.h>
#if defined(_WIN32)
#include <GL/wglew.h>
#endif
//#endif

#include <GLFW/glfw3.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <Camera.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>

#include "commonStructs.h"
#include "random.h"
#include "perlin.h"

#include <sutil.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdint.h>

using namespace optix;

const char* const SAMPLE_NAME = "testingGround";
enum RAY_TYPE {
  RADIANCE = 0,
  SHADOW,
  NUM_RAYS
};

static float rand_range(float min, float max)
{
  static unsigned int seed = 0u;
  return min + (max - min) * rnd(seed);
}


//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context      context;
uint32_t     width  = 1080u;
uint32_t     height = 720;
bool         use_pbo = true;

std::string  texture_path;
std::string  pathtrace_ptx;
std::string  geometry_ptx;
std::string  material_ptx;

// Camera state
float3       camera_up;
float3       camera_lookat;
float3       camera_eye;
Matrix4x4    camera_rotate;

// Mouse state
int2       mouse_prev_pos;
int        mouse_button;


//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

std::string ptxPath( const std::string& cuda_file );
Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void createGeometry();
void setupCamera();
void setupLights();
void updateCamera();


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

std::string ptxPath( const std::string& cuda_file )
{
  return
    std::string(sutil::samplesPTXDir()) +
    "/" + std::string(SAMPLE_NAME) + "_generated_" +
    cuda_file +
    ".ptx";
}


Buffer getOutputBuffer()
{
  return context[ "output_buffer" ]->getBuffer();
}


void destroyContext()
{
  if( context )
  {
    context->destroy();
    context = 0;
  }
}


void registerExitHandler()
{
  // register shutdown handler
#ifdef _WIN32
  glutCloseFunc( destroyContext );  // this function is freeglut-only
#else
  atexit( destroyContext );
#endif
}


void createContext()
{
  // Set up context
  context = Context::create();
  context->setRayTypeCount( NUM_RAYS );
  context->setEntryPointCount( 1 );
  context->setStackSize( 4640 );
  context->setPrintEnabled( true );

  // Note: high max depth for reflection and refraction through glass
  context["max_depth"]->setInt( 100 );
  context["radiance_ray_type"]->setUint( RADIANCE );
  context["shadow_ray_type"]->setUint( SHADOW );
  context["scene_epsilon"]->setFloat( 1.e-4f );
  context["importance_cutoff"]->setFloat( 0.01f );
  context["ambient_light_color"]->setFloat( 0.31f, 0.33f, 0.28f );

  // Output buffer
  // First allocate the memory for the GL buffer, then attach it to OptiX.
  GLuint vbo = 0;
  glGenBuffers( 1, &vbo );
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  glBufferData( GL_ARRAY_BUFFER, 4 * width * height, 0, GL_STREAM_DRAW);
  glBindBuffer( GL_ARRAY_BUFFER, 0 );

  Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo );
  context["output_buffer"]->set( buffer );


  // Ray generation program
  {
    Program ray_gen_program = context->createProgramFromPTXFile( pathtrace_ptx, "camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );
  }

  // Exception program
  Program exception_program = context->createProgramFromPTXFile( pathtrace_ptx, "exception" );
  context->setExceptionProgram( RADIANCE, exception_program );
  context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

  // Miss program
  {
    const std::string miss_name = "miss";
    context->setMissProgram( RADIANCE, context->createProgramFromPTXFile( pathtrace_ptx, miss_name ) );
    const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
    const std::string texpath = texture_path + "/" + std::string( "CedarCity.hdr" );
    context["envmap"]->setTextureSampler( sutil::loadTexture( context, texpath, default_color) );
    context["bg_color"]->setFloat( make_float3(0.0f) ); //make_float3( 0.34f, 0.55f, 0.85f ) );
  }

  // 3D solid noise buffer, 1 float channel, all entries in the range [0.0, 1.0].

  const int tex_width  = 64;
  const int tex_height = 64;
  const int tex_depth  = 64;
  Buffer noiseBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, tex_width, tex_height, tex_depth);
  float *tex_data = (float *) noiseBuffer->map();

  // Random noise in range [0, 1]
  for (int i = tex_width * tex_height * tex_depth;  i > 0; i--) {
    // One channel 3D noise in [0.0, 1.0] range.
    *tex_data++ = rand_range(0.0f, 1.0f);
  }
  noiseBuffer->unmap(); 


  // Noise texture sampler
  TextureSampler noiseSampler = context->createTextureSampler();

  noiseSampler->setWrapMode(0, RT_WRAP_REPEAT);
  noiseSampler->setWrapMode(1, RT_WRAP_REPEAT);
  noiseSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
  noiseSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
  noiseSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
  noiseSampler->setMaxAnisotropy(1.0f);
  noiseSampler->setMipLevelCount(1);
  noiseSampler->setArraySize(1);
  noiseSampler->setBuffer(0, 0, noiseBuffer);

  context["noise_texture"]->setTextureSampler(noiseSampler);
}

float4 make_plane( float3 n, float3 p )
{
  n = normalize(n);
  float d = -dot(n, p);
  return make_float4( n, d );
}

Geometry createParallelogram(float3 anchor, float3 v1, float3 v2) {
  Geometry parallelogram = context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setBoundingBoxProgram( context->createProgramFromPTXFile( geometry_ptx, "par_bounds" ) );
  parallelogram->setIntersectionProgram( context->createProgramFromPTXFile( geometry_ptx, "par_intersect" ) );
  float3 normal = cross( v2, v1 );
  normal = normalize( normal );
  float d = dot( normal, anchor );
  v1 *= 1.0f/dot( v1, v1 );
  v2 *= 1.0f/dot( v2, v2 );
  float4 plane = make_float4( normal, d );
  parallelogram["plane"]->setFloat( plane );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );
  parallelogram["anchor"]->setFloat( anchor );
  return parallelogram;
}

Material createMediaMaterial(float sigma_a) {
  Material matl = context->createMaterial();
  Program ch = context->createProgramFromPTXFile( material_ptx, "media_hit" );
  matl->setClosestHitProgram( RADIANCE, ch );
  matl->setClosestHitProgram( SHADOW, ch );

  matl["sigma_a"]->setFloat(sigma_a);
  return matl;
}

Material createPhongMaterial(float3 Ka, float3 Kd, float3 Ks, int phong_exp) {
  Material matl = context->createMaterial();
  Program ch = context->createProgramFromPTXFile( material_ptx, "diffuse_hit_radiance" );
  matl->setClosestHitProgram( RADIANCE, ch );

  Program ah = context->createProgramFromPTXFile( material_ptx, "diffuse_hit_shadow" );
  matl->setAnyHitProgram( SHADOW, ah );

  matl["Ka"]->setFloat( Ka );
  matl["Kd"]->setFloat( Kd );
  matl["Ks"]->setFloat( Ks );
  matl["phong_exp"]->setFloat( phong_exp );
  return matl;
}

Geometry createBox(float3 boxMin, float3 boxMax) {
  Program box_bounds    = context->createProgramFromPTXFile( geometry_ptx, "box_bounds" );
  Program box_intersect = context->createProgramFromPTXFile( geometry_ptx, "box_intersect" );

  Geometry box = context->createGeometry();
  box->setPrimitiveCount( 1u );
  box->setBoundingBoxProgram( box_bounds );
  box->setIntersectionProgram( box_intersect );
  box["boxmin"]->setFloat( boxMin );
  box["boxmax"]->setFloat( boxMax );

  return box;
}

void createGeometry()
{
  std::vector<GeometryInstance> gis;

  Geometry box = createBox(
      make_float3(-1.0f, 2.0f, -1.0f),
      make_float3( 1.0f, 4.0f,  1.0f)
      );
  Material box_matl = createPhongMaterial(
      make_float3(0.2f),
      make_float3(0.5f),
      make_float3(0.2f),
      88
      );
  //gis.push_back( context->createGeometryInstance( box, &box_matl, &box_matl+1 ) );

  //Geometry med = createSphere(make_float3(0.0f, 5.0f, 0.0f), 5.0f);
  //Material med_matl = createMediaMaterial(0.01f);


  Geometry med = createBox(
      make_float3(-5.0f,  1.0f, -5.0f),
      make_float3( 5.0f, 11.0f,  5.0f)
      );
  Material med_matl = createMediaMaterial(0.01f);
  gis.push_back( context->createGeometryInstance( med, &med_matl, &med_matl+1 ) );

  // Voxel media
  //float len = 5.0f;
  //int dim = 3;
  //for(int x = -dim; x <= dim; ++x) {
  //  for(int y = 0; y <= dim; ++y) {
  //    for(int z = -dim; z <= dim; ++z) {
  //      float fx = (x+dim) * ((float)Perlin::SIZE) / (2.0f * dim);
  //      float fy = (y    ) * ((float)Perlin::SIZE) / (       dim);
  //      float fz = (z+dim) * ((float)Perlin::SIZE) / (2.0f * dim);

  //      float perlin_val = (1.0f + Perlin::sample(fx, fy, fz));
  //      perlin_val *= perlin_val;
  //      perlin_val *= 0.01f;

  //      Geometry med = createBox(
  //          make_float3((x+0)*len, 2.0f + (y+0)*len, (z+0)*len),
  //          make_float3((x+1)*len, 2.0f + (y+1)*len, (z+1)*len)
  //      );
  //      Material med_matl = createMediaMaterial(perlin_val);
  //      gis.push_back( context->createGeometryInstance( med, &med_matl, &med_matl+1 ) );
  //    }
  //  }
  //}

  Geometry floor = createParallelogram(
      make_float3(-64.0f, 0.01f, -64.0f),
      make_float3(128.0f, 0.0f, 0.0f),
      make_float3(0.0f, 0.0f, 128.0f)
      );
  Material floor_matl = createPhongMaterial(
      make_float3(0.2f),
      make_float3(0.8f),
      make_float3(0.2f),
      88
      );
  gis.push_back( context->createGeometryInstance( floor, &floor_matl, &floor_matl+1 ) );

  Geometry wall1 = createParallelogram(
      make_float3(64.0f, 0.01f, 64.0f),
      make_float3(-128.0f, 0.0f, 0.0f),
      make_float3(0.0f, 32.0f, 0.0f)
      );
  Material wall1_matl = createPhongMaterial(
      make_float3(0.2f),
      make_float3(1.0f),
      make_float3(0.2f),
      88
      );
  gis.push_back( context->createGeometryInstance( wall1, &wall1_matl, &wall1_matl+1 ) );

  Geometry wall2 = createParallelogram(
      make_float3(64.0f, 0.01f, -64.0f),
      make_float3(0.0f, 0.0f, 128.0f),
      make_float3(0.0f, 32.0f, 0.0f)
      );
  Material wall2_matl = createPhongMaterial(
      make_float3(0.2f),
      make_float3(1.0f,0.5f,0.5f),
      make_float3(0.2f),
      88
      );
  gis.push_back( context->createGeometryInstance( wall2, &wall2_matl, &wall2_matl+1 ) );

  Geometry wall3 = createParallelogram(
      make_float3(-64.0f, 0.01f, -64.0f),
      make_float3(0.0f, 0.0f, 128.0f),
      make_float3(0.0f, 32.0f, 0.0f)
      );
  Material wall3_matl = createPhongMaterial(
      make_float3(0.2f),
      make_float3(0.5f,0.5f,1.0f),
      make_float3(0.2f),
      88
      );
  gis.push_back( context->createGeometryInstance( wall3, &wall3_matl, &wall3_matl+1 ) );

  Geometry roof = createParallelogram(
      make_float3(-64.0f, 32.01f, -64.0f),
      make_float3(128.0f, 0.0f, 0.0f),
      make_float3(0.0f, 0.0f, 128.0f)
      );
  Material roof_matl = createPhongMaterial(
      make_float3(0.0f),
      make_float3(0.4f),
      make_float3(0.0f),
      22
      );
  //gis.push_back( context->createGeometryInstance( roof, &roof_matl, &roof_matl+1 ) );

  // Place all in group
  GeometryGroup geometrygroup = context->createGeometryGroup();
  geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
  for(int i = 0; i < gis.size(); ++i) geometrygroup->setChild( i, gis[i] );
  geometrygroup->setAcceleration( context->createAcceleration("Trbvh") );

  context["top_object"]->set( geometrygroup );
  context["top_shadower"]->set( geometrygroup );

}


void setupCamera()
{
  camera_eye    = make_float3( 1.0f, 16.0f, -64.0f );
  camera_lookat = make_float3( 0.0f, 4.0f,  0.0f );
  camera_up     = make_float3( 0.0f, 1.0f,  0.0f );

  camera_rotate  = Matrix4x4::identity();
}


void setupLights()
{
  BasicLight lights[] = { 
    { make_float3( 0.0f, 30.0f, 0.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 }
  };

  Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
  light_buffer->setFormat( RT_FORMAT_USER );
  light_buffer->setElementSize( sizeof( BasicLight ) );
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  context[ "lights" ]->set( light_buffer );
}


void updateCamera()
{
  const float vfov = 60.0f;
  const float aspect_ratio = static_cast<float>(width) /
    static_cast<float>(height);

  float3 camera_u, camera_v, camera_w;
  sutil::calculateCameraVariables(
      camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
      camera_u, camera_v, camera_w, true );

  const Matrix4x4 frame = Matrix4x4::fromBasis(
      normalize( camera_u ),
      normalize( camera_v ),
      normalize( -camera_w ),
      camera_lookat);
  const Matrix4x4 frame_inv = frame.inverse();
  // Apply camera rotation twice to match old SDK behavior
  const Matrix4x4 trans   = frame*camera_rotate*camera_rotate*frame_inv;

  camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
  camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
  camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

  sutil::calculateCameraVariables(
      camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
      camera_u, camera_v, camera_w, true );

  camera_rotate = Matrix4x4::identity();

  context["eye"]->setFloat( camera_eye );
  context["U"  ]->setFloat( camera_u );
  context["V"  ]->setFloat( camera_v );
  context["W"  ]->setFloat( camera_w );
}

/*
   void glutInitialize( int* argc, char** argv )
   {
   glutInit( argc, argv );
   glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
   glutInitWindowSize( width, height );
   glutInitWindowPosition( 100, 100 );
   glutCreateWindow( SAMPLE_NAME );
   glutHideWindow();
   }

   void glutRun()
   {
// Initialize GL state
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
glOrtho(0, 1, 0, 1, -1, 1 );

glMatrixMode(GL_MODELVIEW);
glLoadIdentity();

glViewport(0, 0, width, height);

glutShowWindow();
glutReshapeWindow( width, height);

// register glut callbacks
glutDisplayFunc( glutDisplay );
glutIdleFunc( glutDisplay );
glutReshapeFunc( glutResize );
glutKeyboardFunc( glutKeyboardPress );
glutMouseFunc( glutMousePress );
glutMotionFunc( glutMouseMotion );

registerExitHandler();

glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
updateCamera();

context->launch( 0, width, height );

Buffer buffer = getOutputBuffer();
sutil::displayBufferGL( getOutputBuffer() );

{
static unsigned frame_count = 0;
sutil::displayFps( frame_count++ );
}

glutSwapBuffers();
}


void glutKeyboardPress( unsigned char k, int x, int y )
{

switch( k )
{
case( 'q' ):
case( 27 ): // ESC
{
  destroyContext();
  exit(0);
}
case( 's' ):
{
  const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
  std::cerr << "Saving current frame to '" << outputImage << "'\n";
  sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer() );
  break;
}
}
}


void glutMousePress( int button, int state, int x, int y )
{
  if( state == GLUT_DOWN )
  {
    mouse_button = button;
    mouse_prev_pos = make_int2( x, y );
  }
  else
  {
    // nothing
  }
}


void glutMouseMotion( int x, int y)
{
  if( mouse_button == GLUT_RIGHT_BUTTON )
  {
    const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
      static_cast<float>( width );
    const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
      static_cast<float>( height );
    const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
    const float scale = fminf( dmax, 0.9f );
    camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
  }
  else if( mouse_button == GLUT_LEFT_BUTTON )
  {
    const float2 from = { static_cast<float>(mouse_prev_pos.x),
      static_cast<float>(mouse_prev_pos.y) };
    const float2 to   = { static_cast<float>(x),
      static_cast<float>(y) };

    const float2 a = { from.x / width, from.y / height };
    const float2 b = { to.x   / width, to.y   / height };

    camera_rotate = arcball.rotate( b, a );
  }

  mouse_prev_pos = make_int2( x, y );
}


void glutResize( int w, int h )
{
  if ( w == (int)width && h == (int)height ) return;

  width  = w;
  height = h;

  sutil::resizeBuffer( getOutputBuffer(), width, height );

  glViewport(0, 0, width, height);

  glutPostRedisplay();
}
*/

//------------------------------------------------------------------------------
//
//  GLFW callbacks
//
//------------------------------------------------------------------------------

struct CallbackData
{
  sutil::Camera &camera;
  unsigned int &accumulation_frame;
};

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  bool handled = false;

  if (action == GLFW_PRESS)
  {
    switch (key)
    {
      case GLFW_KEY_Q:
      case GLFW_KEY_ESCAPE:
        if (context)
          context->destroy();
        if (window)
          glfwDestroyWindow(window);
        glfwTerminate();
        exit(EXIT_SUCCESS);

      case (GLFW_KEY_S):
        {
          const std::string outputImage = std::string(SAMPLE_NAME) + ".png";
          std::cerr << "Saving current frame to '" << outputImage << "'\n";
          sutil::writeBufferToFile(outputImage.c_str(), getOutputBuffer());
          handled = true;
          break;
        }
      case (GLFW_KEY_F):
        {
          CallbackData *cb = static_cast<CallbackData *>(glfwGetWindowUserPointer(window));
          cb->camera.reset_lookat();
          cb->accumulation_frame = 0;
          handled = true;
          break;
        }
    }
  }

  if (!handled)
  {
    // forward key event to imgui
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
  }
}

void windowSizeCallback(GLFWwindow *window, int w, int h)
{
  if (w < 0 || h < 0)
    return;

  const unsigned width = (unsigned)w;
  const unsigned height = (unsigned)h;

  CallbackData *cb = static_cast<CallbackData *>(glfwGetWindowUserPointer(window));
  if (cb->camera.resize(width, height))
  {
    cb->accumulation_frame = 0;
  }

  sutil::resizeBuffer(getOutputBuffer(), width, height);
  sutil::resizeBuffer(context["accum_buffer"]->getBuffer(), width, height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1, 0, 1, -1, 1);
  glViewport(0, 0, width, height);
}

//------------------------------------------------------------------------------
//
// GLFW setup and run
//
//------------------------------------------------------------------------------

GLFWwindow *glfwInitialize()
{
  GLFWwindow *window = sutil::initGLFW();

  // Note: this overrides imgui key callback with our own.  We'll chain this.
  glfwSetKeyCallback(window, keyCallback);

  glfwSetWindowSize(window, (int)WIDTH, (int)HEIGHT);
  glfwSetWindowSizeCallback(window, windowSizeCallback);

  return window;
}

void glfwRun(GLFWwindow *window, sutil::Camera &camera, sutil::PreethamSunSky &sky, DirectionalLight &sun, Buffer light_buffer)
{
  // Initialize GL state
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1, 0, 1, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glViewport(0, 0, WIDTH, HEIGHT);

  unsigned int frame_count = 0;
  unsigned int accumulation_frame = 0;

  // Sun and Sky
  float sun_phi = sky.getSunPhi();
  float sun_theta = 0.5f * M_PIf - sky.getSunTheta();
  float sun_radius = DEFAULT_SUN_RADIUS;
  float overcast = DEFAULT_OVERCAST;

  // Atmosphere
  float sigma_t = DEFAULT_SIGMA_A;
  float atmos = DEFAULT_ATMOSPHERE;

  // Camera
  float aper = DEFAULT_APERTURE;

  // Renderer
  int max_depth = DEFAULT_MAXDEPTH;

  // Expose user data for access in GLFW callback functions when the window is resized, etc.
  // This avoids having to make it global.
  CallbackData cb = {camera, accumulation_frame};
  glfwSetWindowUserPointer(window, &cb);

  while (!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    ImGui_ImplGlfw_NewFrame();
    ImGuiIO &io = ImGui::GetIO();
    // Let imgui process the mouse first
    if (!io.WantCaptureMouse)
    {
      double x, y;
      glfwGetCursorPos(window, &x, &y);
      if (camera.process_mouse((float)x, (float)y, ImGui::IsMouseDown(0), ImGui::IsMouseDown(1), ImGui::IsMouseDown(2)))
      {
        accumulation_frame = 0;
      }
    }

    // imgui pushes
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.8f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 2.0f);

    // Title Bar Colors
    ImGui::PushStyleColor(ImGuiCol_TitleBg, ImColor(30, 150, 250, 200));
    ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, ImColor(60, 60, 60, 150));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImColor(30, 150, 250, 200));

    // Title Bar Colors
    ImGui::PushStyleColor(ImGuiCol_Header, ImColor(60, 60, 60, 150));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImColor(150, 150, 150, 150));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImColor(30, 150, 250, 200));

    sutil::displayFps(frame_count++);

    {
      static const ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar;

      static const ImGuiWindowFlags header_flags =
        ImGuiTreeNodeFlags_OpenOnDoubleClick |
        ImGuiTreeNodeFlags_OpenOnArrow;

      ImGui::SetNextWindowPos(ImVec2(2.0f, 40.0f));
      ImGui::Begin("Controls", 0, window_flags);

      // Sun and Sky Control
      if (ImGui::CollapsingHeader(" Sun & Sky", header_flags)) {
        bool sun_changed = false;
        // Sun Rotation Control
        if (ImGui::SliderAngle("sun rotation", &sun_phi, 0.0f, 360.0f))
        {
          sky.setSunPhi(sun_phi);
          sky.setVariables(context);
          sun.direction = sky.getSunDir();
          sun_changed = true;
        }
        // Sun Elevation Control
        if (ImGui::SliderAngle("sun elevation", &sun_theta, 0.0f, 90.0f))
        {
          sky.setSunTheta(0.5f * M_PIf - sun_theta);
          sky.setVariables(context);
          sun.direction = sky.getSunDir();
          sun_changed = true;
        }
        // Sun Radius Control
        if (ImGui::SliderFloat("sun radius", &sun_radius, PHYSICAL_SUN_RADIUS, 0.4f))
        {
          sun.radius = sun_radius;
          sun_changed = true;
        }
        // Overcast Sky Control
        if (ImGui::SliderFloat("overcast", &overcast, 0.0f, 1.0f))
        {
          sky.setOvercast(overcast);
          sky.setVariables(context);
          sun_changed = true;
        }
        if (sun_changed)
        {
          // recalculate frame for area sampling
          optix::Onb onb(sun.direction);
          sun.v0 = onb.m_tangent;
          sun.v1 = onb.m_binormal;
          // keep total sun energy constant and realistic if we increase area.
          const float sqrt_sun_scale = PHYSICAL_SUN_RADIUS / sun_radius;
          sun.color = sky.sunColor() * sqrt_sun_scale * sqrt_sun_scale;
          memcpy(light_buffer->map(), &sun, sizeof(DirectionalLight));
          light_buffer->unmap();
          accumulation_frame = 0;
        }
      }
      // Atmosphere Control
      if (ImGui::CollapsingHeader(" Atmosphere", header_flags)) {
        if (ImGui::SliderFloat("attenuation", &sigma_t, 0.0f, 100.0f))
        {
          context["atmos_sigma_t"]->setFloat(make_float3(sigma_t));
          accumulation_frame = 0;
        }
        if (ImGui::SliderFloat("depth", &atmos, 0.0f, 100.0f))
        {
          context["atmos_dist"]->setFloat(atmos);
          accumulation_frame = 0;
        }
      }
      // Camera Control
      if (ImGui::CollapsingHeader(" Camera", header_flags)) {
        if (ImGui::SliderFloat("aperature", &aper, 0.0f, 0.5f)) //TODO: find a better representation for the aperature scale
        {
          context["aper"]->setFloat(aper);
          accumulation_frame = 0;
        }
      }
      // Renderer Control
      if (ImGui::CollapsingHeader(" Renderer", header_flags)) {
        if (ImGui::SliderInt("max depth", &max_depth, 1, 25)) {
          context["max_depth"]->setInt(max_depth);
          accumulation_frame = 0;
        }
      }
      ImGui::End();
    }

    // imgui pops
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(6);

    // Render main window
    context["frame"]->setUint(accumulation_frame++);
    context->launch(0, camera.width(), camera.height());
    sutil::displayBufferGL(getOutputBuffer());

    // Render gui over it
    ImGui::Render();

    glfwSwapBuffers(window);
  }

  destroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
}
//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
  std::cerr << "\nUsage: " << argv0 << " [options]\n";
  std::cerr <<
    "App Options:\n"
    "  -h | --help         Print this usage message and exit.\n"
    "  -f | --file         Save single frame to file and exit.\n"
    "  -n | --nopbo        Disable GL interop for display buffer.\n"
    "App Keystrokes:\n"
    "  q  Quit\n"
    "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
    << std::endl;

  exit(1);
}

int main( int argc, char** argv )
{
  Perlin::initialize_grid();

  std::string out_file;
  for( int i=1; i<argc; ++i )
  {
    const std::string arg( argv[i] );

    if( arg == "-h" || arg == "--help" )
    {
      printUsageAndExit( argv[0] );
    }
    else if ( arg == "-f" || arg == "--file" )
    {
      if( i == argc-1 )
      {
        std::cerr << "Option '" << arg << "' requires additional argument.\n";
        printUsageAndExit( argv[0] );
      }
      out_file = argv[++i];
    } 
    else if( arg == "-n" || arg == "--nopbo"  )
    {
      use_pbo = false;
    }
    else
    {
      std::cerr << "Unknown option '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( texture_path.empty() ) {
    texture_path = std::string( sutil::samplesDir() ) + "/data";
  }

  try
  {
    /*
       glutInitialize( &argc, argv );

#ifndef __APPLE__
glewInit();
#endif
*/

    GLFWwindow *window = glfwInitialize();

    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
      std::cerr << "GLEW init failed: " << glewGetErrorString(err) << std::endl;
      exit(EXIT_FAILURE);
    }

    pathtrace_ptx = ptxPath( "pathtrace.cu" );
    geometry_ptx = ptxPath( "geometry.cu" );
    material_ptx = ptxPath( "material.cu" );

    createContext();
    createGeometry();
    setupCamera();
    setupLights();

    context->validate();

    if ( out_file.empty() )
    {
      //glutRun();
      glfwRun(window, camera, sky, sun, light_buffer);
    }
    else
    {
      updateCamera();
      context->launch( 0, width, height );
      sutil::displayBufferPPM( out_file.c_str(), getOutputBuffer() );
      destroyContext();
    }
    return 0;
  }
  SUTIL_CATCH( context->get() )
}

