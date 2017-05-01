
//-----------------------------------------------------------------------------
//
// godRay
//
//-----------------------------------------------------------------------------

//#ifndef __APPLE__
#include <GL/glew.h>
#if defined(_WIN32)
#include <GL/wglew.h>
#endif
//#endif

#include <GLFW/glfw3.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <OptiXMesh.h>

#include <sutil.h>
#include <Camera.h>
#include <SunSky.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <stdint.h>

#include "../inc/commonStructs.h"

using namespace optix;

const char *const SAMPLE_NAME = "godRay";
const unsigned int WIDTH = 1024u;
const unsigned int HEIGHT = 768u;

// SUN/SKY
const float PHYSICAL_SUN_RADIUS = 0.004675f; // from Wikipedia
const float DEFAULT_SUN_RADIUS = 0.05f;      // Softer default to show off soft shadows
const float DEFAULT_SUN_THETA = 1.1f;
const float DEFAULT_SUN_PHI = 300.0f * M_PIf / 180.0f;
const float DEFAULT_OVERCAST = 0.3f;

// ATMOSPHERE
const float DEFAULT_ATMOS_SIGMA_S = 0.05f; // In scattering parameter
const float DEFAULT_ATMOS_SIGMA_T = 0.05f; // Extinction parameter
const float DEFAULT_ATMOS_G       = 0.05f; // G parameter
const float DEFAULT_ATMOS_DIST    = 0.05f; // Atmosphere distance parameter
const float DEFAULT_ATMOSPHERE    = 50.0f; // Atmosphere parameter

// CAMERA
const float DEFAULT_APERTURE = 1 / 8.0f;

// RENDERER
const int DEFAULT_MAXDEPTH = 10;

// Ray Types
enum RAY_TYPE {
  RADIANCE = 0,
  SHADOW,
  NUM_RAYS
};

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context context = 0;

//------------------------------------------------------------------------------
//
//  Helper functions - from Adv Samples
//
//------------------------------------------------------------------------------

static std::string ptxPath(const std::string &cuda_file)
{
  return std::string(sutil::samplesPTXDir()) +
    "/" + std::string(SAMPLE_NAME) + "_generated_" +
    cuda_file +
    ".ptx";
}

// PATHS
const std::string geometry_ptx = ptxPath("geometry.cu");
const std::string material_ptx = ptxPath("material.cu");
const std::string pathtrace_ptx = ptxPath("pathtrace.cu");

static Buffer getOutputBuffer()
{
  return context["output_buffer"]->getBuffer();
}

void destroyContext()
{
  if (context)
  {
    context->destroy();
    context = 0;
  }
}

void createContext(bool use_pbo)
{
  // Set up context
  context = Context::create();
  context->setRayTypeCount(NUM_RAYS);
  context->setEntryPointCount(1);
  context->setStackSize(600);     // TODO: debug for optimal stack size

  context["frame"]->setUint(0u);
  context["scene_epsilon"]->setFloat(1.e-3f);
  context->setPrintEnabled(true);
  context->setPrintBufferSize(1024);

  // Set Gloabel Renderer Parameters
  context["max_depth"]->setInt(DEFAULT_MAXDEPTH);

  // Set Gloabal Atmosphere Parameters
  context["atmos_sigma_s"]->setFloat(make_float3(DEFAULT_ATMOS_SIGMA_S));
  context["atmos_sigma_t"]->setFloat(make_float3(DEFAULT_ATMOS_SIGMA_T));
  context["atmos_g"]->setFloat(DEFAULT_ATMOS_G);
  context["atmos_dist"]->setFloat(DEFAULT_ATMOSPHERE);

  // Set Global Camera Paramters
  context["aper"]->setFloat(DEFAULT_APERTURE);

  Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT, use_pbo);
  context["output_buffer"]->set(buffer);

  // Accumulation buffer
  Buffer accum_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
      RT_FORMAT_FLOAT4, WIDTH, HEIGHT);
  context["accum_buffer"]->set(accum_buffer);

  // Ray generation program
  Program ray_gen_program = context->createProgramFromPTXFile(pathtrace_ptx, "render_pixel");
  context->setRayGenerationProgram(0, ray_gen_program);

  // Exception program
  Program exception_program = context->createProgramFromPTXFile(pathtrace_ptx, "exception");
  context->setExceptionProgram(0, exception_program);
  context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);
}

void createLights(sutil::PreethamSunSky &sky, DirectionalLight &sun, Buffer &light_buffer)
{
  //
  // Sun and sky model
  //
  context->setMissProgram(0, context->createProgramFromPTXFile(pathtrace_ptx, "miss"));

  sky.setSunTheta(DEFAULT_SUN_THETA); // 0: noon, pi/2: sunset
  sky.setSunPhi(DEFAULT_SUN_PHI);
  sky.setTurbidity(2.2f);
  sky.setOvercast(DEFAULT_OVERCAST);
  sky.setVariables(context);

  // Split out sun for direct sampling
  sun.direction = sky.getSunDir();
  optix::Onb onb(sun.direction);
  sun.radius = DEFAULT_SUN_RADIUS;
  sun.v0 = onb.m_tangent;
  sun.v1 = onb.m_binormal;
  const float sqrt_sun_scale = PHYSICAL_SUN_RADIUS / sun.radius;
  sun.color = sky.sunColor() * sqrt_sun_scale * sqrt_sun_scale;
  sun.casts_shadow = 1;

  light_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
  light_buffer->setElementSize(sizeof(DirectionalLight));
  memcpy(light_buffer->map(), &sun, sizeof(DirectionalLight));
  light_buffer->unmap();

  context["light_buffer"]->set(light_buffer);
}

Geometry createBox(float3 boxmin, float3 boxmax)
{
  Geometry box = context->createGeometry();
  box->setPrimitiveCount( 1u );

  Program box_bounds = context->createProgramFromPTXFile( geometry_ptx, "box_bounds" );
  Program box_intersect = context->createProgramFromPTXFile( geometry_ptx, "box_intersect" );

  box->setBoundingBoxProgram( box_bounds );
  box->setIntersectionProgram( box_intersect );

  box["boxmin"]->setFloat( boxmin );
  box["boxmax"]->setFloat( boxmax );
  
  return box;
}

//Material createMediaMaterial(float sigma_a) {
//  Material matl = context->createMaterial();
//  Program ch = context->createProgramFromPTXFile( material_ptx, "media_hit_radiance" );
//  matl->setClosestHitProgram( RADIANCE, ch );
//
//  Program ah = context->createProgramFromPTXFile( material_ptx, "media_hit_shadow" );
//  matl->setClosestHitProgram( SHADOW, ah );
//
//  matl["sigma_a"]->setFloat(sigma_a);
//  return matl;
//}

Material createPhongMaterial( float3 Kd )
{
  Material matl = context->createMaterial();

  Program ch = context->createProgramFromPTXFile( material_ptx, "diffuse_hit_radiance" );
  matl->setClosestHitProgram( RADIANCE, ch );

  Program ah = context->createProgramFromPTXFile( material_ptx, "diffuse_hit_shadow" );
  matl->setAnyHitProgram( SHADOW, ah );

  matl["Kd"]->setFloat( Kd );
  return matl;
}

void createGeometry()
{
  std::vector<GeometryInstance> gis;

  GeometryGroup geometry_group = context->createGeometryGroup();
  geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
  
  Geometry med = createBox( make_float3(-2.0f), make_float3(2.0f) );
  Material med_matl = createPhongMaterial( make_float3(0.99f) );
  gis.push_back( context->createGeometryInstance( med, &med_matl, &med_matl+1 ) );

  //GeometryGroup media_group = context->createGeometryGroup();
  //media_group->setAcceleration(context->createAcceleration("None"));

  // Load mesh
  //OptiXMesh mesh;
  //mesh.context = context;
  //mesh.intersection = context->createProgramFromPTXFile( geometry_ptx, "mesh_intersect" );
  //mesh.bounds = context->createProgramFromPTXFile( geometry_ptx, "mesh_bounds" );
  //mesh.material = createPhongMaterial( make_float3(0.6f, 0.3f, 0.3f) );
  //Matrix4x4 xform = Matrix4x4::identity();
 
  //loadMesh( std::string( sutil::samplesDir() ) + "/godRay/model/obj/dome_simple.obj", mesh, xform );
  //gis.push_back(mesh.geom_instance);

  geometry_group->setChildCount( static_cast<unsigned int>(gis.size()) );
  for(int i = 0; i < gis.size(); ++i) geometry_group->setChild( i, gis[i] );

  //context["top_media"]->set(geometry_group);
  //context["top_geometry"]->set(geometry_group);
  context["top_object"]->set(geometry_group);
}

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
  float atmos_sigma_s = DEFAULT_ATMOS_SIGMA_S;
  float atmos_sigma_t = DEFAULT_ATMOS_SIGMA_T;
  float atmos_dist = DEFAULT_ATMOS_DIST;
  float atmos_g = DEFAULT_ATMOS_G;
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
        if (ImGui::SliderFloat("in scattering", &atmos_sigma_t, 0.0f, 1.0f))
        {
          context["atmos_sigma_s"]->setFloat(make_float3(atmos_sigma_s));
          accumulation_frame = 0;
        }
        if (ImGui::SliderFloat("extinction", &atmos_sigma_t, 0.0f, 100.0f))
        {
          context["atmos_sigma_t"]->setFloat(make_float3(atmos_sigma_t));
          accumulation_frame = 0;
        }
        if (ImGui::SliderFloat("dist", &atmos, 0.0f, 100.0f))
        {
          context["atmos_dist"]->setFloat(atmos_dist);
          accumulation_frame = 0;
        }
        if (ImGui::SliderFloat("g", &atmos, 0.0f, 100.0f))
        {
          context["atmos_g"]->setFloat(atmos_g);
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

void printUsageAndExit(const std::string &argv0)
{
  std::cerr << "\nUsage: " << argv0 << " [options] [file0.vox] [file1.vox] ...\n";
  std::cerr << "App Options:\n"
    "  -h | --help                  Print this usage message and exit.\n"
    "  -f | --file <output_file>    Save image to file and exit.\n"
    "  -n | --nopbo                 Disable GL interop for display buffer.\n"
    "App Keystrokes:\n"
    "  q  Quit\n"
    "  s  Save image to '"
    << SAMPLE_NAME << ".png'\n"
    "  f  Re-center camera\n"
    "\n"
    << std::endl;

  exit(1);
}

int main(int argc, char **argv)
{
  bool use_pbo = true;
  std::string out_file;
  std::vector<std::string> vox_files; //TODO get rid of this
  std::vector<std::string> mesh_files;
  std::vector<optix::Matrix4x4> mesh_xforms;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg(argv[i]);

    if (arg == "-h" || arg == "--help")
    {
      printUsageAndExit(argv[0]);
    }
    else if (arg == "-f" || arg == "--file")
    {
      if (i == argc - 1)
      {
        std::cerr << "Option '" << arg << "' requires additional argument.\n";
        printUsageAndExit(argv[0]);
      }
      out_file = argv[++i];
    }
    else if (arg == "-n" || arg == "--nopbo")
    {
      use_pbo = false;
    }
    else if (arg[0] == '-')
    {
      std::cerr << "Unknown option '" << arg << "'\n";
      printUsageAndExit(argv[0]);
    }
    else
    {
      // Interpret argument as a mesh file. TODO: get rid of this
      vox_files.push_back(std::string(argv[i]));

      // Interpret argument as a mesh file.
      mesh_files.push_back(argv[i]);
      mesh_xforms.push_back(optix::Matrix4x4::identity());
    }
  }







  try
  {
    GLFWwindow *window = glfwInitialize();

    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
      std::cerr << "GLEW init failed: " << glewGetErrorString(err) << std::endl;
      exit(EXIT_FAILURE);
    }

    createContext(use_pbo);

    if (vox_files.empty()) // TODO: get rid of this
    {
      // Default scene
      vox_files.push_back(std::string(sutil::samplesDir()) + "/data/scene_parade.vox");
    }

    if (mesh_files.empty()) {

      // Default scene

      const optix::Matrix4x4 xform = optix::Matrix4x4::rotate(-M_PIf / 2.0f, make_float3(0.0f, 1.0f, 0.0f));
      mesh_files.push_back(std::string(sutil::samplesDir()) + "/data/teapot_lid.ply");
      mesh_xforms.push_back(xform);
      mesh_files.push_back(std::string(sutil::samplesDir()) + "/data/teapot_body.ply");
      mesh_xforms.push_back(xform);
    }

    sutil::PreethamSunSky sky;
    DirectionalLight sun;
    Buffer light_buffer;
    createLights(sky, sun, light_buffer);

    createGeometry();

    // Note: lighting comes from miss program

    context->validate();

    const optix::float3 camera_eye(optix::make_float3(0.0f, 16.0f, -64.0f));
    const optix::float3 camera_lookat(optix::make_float3(0.0f, 0.0f, 0.0f));
    const optix::float3 camera_up(optix::make_float3(0.0f, 1.0f, 0.0f));
    sutil::Camera camera(WIDTH, HEIGHT,
        &camera_eye.x, &camera_lookat.x, &camera_up.x,
        context["eye"], context["U"], context["V"], context["W"]);


    // RUN IN WINDOW
    if (out_file.empty())
    {
      glfwRun(window, camera, sky, sun, light_buffer);
    }
    // WRITE TO FILE
    else
    {
      // Accumulate frames for anti-aliasing
      const unsigned int numframes = 800;
      std::cerr << "Accumulating " << numframes << " frames ..." << std::endl;
      for (unsigned int frame = 0; frame < numframes; ++frame)
      {
        context["frame"]->setUint(frame);
        context->launch(0, WIDTH, HEIGHT);
      }
      sutil::writeBufferToFile(out_file.c_str(), getOutputBuffer());
      std::cerr << "Wrote " << out_file << std::endl;
      destroyContext();
    }
    return 0;
  }
  SUTIL_CATCH(context->get())
}
