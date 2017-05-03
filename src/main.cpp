
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

const char *const PROGRAM_NAME = "godRay";
const unsigned int DEF_WIDTH = 1280u;
const unsigned int DEF_HEIGHT = 720u;
unsigned int width = DEF_WIDTH;
unsigned int height = DEF_HEIGHT;

// SUN/SKY
const float PHYSICAL_SUN_RADIUS = 0.004675f; // from Wikipedia
const float DEF_SUN_RADIUS = 0.20f;          // Softer default to show off soft shadows
const float DEF_SUN_THETA = 70.0f * M_PIf / 180.0f;
const float DEF_SUN_PHI = 90.0f * M_PIf / 180.0f;
const float DEF_OVERCAST = 0.60f;
float sun_radius = DEF_SUN_RADIUS;
float sun_theta = DEF_SUN_THETA;
float sun_phi = DEF_SUN_PHI;
float sun_overcast = DEF_OVERCAST;

// OBJ matl diffuse Kd
const float DIFFUSE_CONST = 0.85f;

// ATMOSPHERE
// In scattering parameter
const float MIN_ATMOS_SIGMA_S = 0.0001f;
const float DEF_ATMOS_SIGMA_S = 0.0010f;
const float MAX_ATMOS_SIGMA_S = 0.0100f;
float atmos_sigma_s = DEF_ATMOS_SIGMA_S;

// Extinction parameter
const float MIN_ATMOS_SIGMA_T = 0.0001f;
const float DEF_ATMOS_SIGMA_T = 0.0010f;
const float MAX_ATMOS_SIGMA_T = 0.0100f;
float atmos_sigma_t = DEF_ATMOS_SIGMA_T;

// G parameter
const float MIN_ATMOS_G = -1.00f;
const float DEF_ATMOS_G = 0.10f;
const float MAX_ATMOS_G = 1.00f;
float atmos_g = DEF_ATMOS_G;

// Atmosphere distance parameter
const float MIN_ATMOS_DIST = 0.10f;
const float DEF_ATMOS_DIST = 1.00f;
const float MAX_ATMOS_DIST = 100.00f;
float atmos_dist = DEF_ATMOS_DIST;

// CAMERA
//const float MIN_APERATURE = 1 / 32.0f;
//const float DEF_APERATURE = 1 / 8.0f;
//const float MAX_APERATURE = 1 / 2.0f;
//float cam_aperature = DEF_APERATURE;

const float MIN_EXPOSURE = 1.0f;
const float DEF_EXPOSURE = 50.0f;
const float MAX_EXPOSURE = 1000.0f;
float cam_exposure = DEF_EXPOSURE;

const float MIN_ZEUS = 1.0f;
const float DEF_ZEUS = 50.0f;
const float MAX_ZEUS = 100.0f;
float cam_zeus = DEF_ZEUS;

const float3 DEF_CAM_POSITION = make_float3(40.0f, 110.0f, 0.0f);
const float3 DEF_CAM_TARGET = make_float3(0.0f, 40.0f, -400.0f);
float3 cam_position = DEF_CAM_POSITION;
float3 cam_target = DEF_CAM_TARGET;

// RENDERER
const int MIN_MAXDEPTH = 1;
const int DEF_MAXDEPTH = 6;
const int MAX_MAXDEPTH = 10;
int maxdepth = DEF_MAXDEPTH;

const unsigned int DEF_NUMFRAMES = 800u;
unsigned int numframes = DEF_NUMFRAMES;

// Ray Types
enum RAY_TYPE
{
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
         "/" + std::string(PROGRAM_NAME) + "_generated_" +
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
  context->setStackSize(1200); // TODO: debug for optimal stack size

  context["frame"]->setUint(0u);
  context["scene_epsilon"]->setFloat(1.e-3f);
  context->setPrintEnabled(true);
  context->setPrintBufferSize(1024);

  context["radiance_ray_type"]->setUint(RADIANCE);
  context["shadow_ray_type"]->setUint(SHADOW);

  // Set Gloabel Renderer Parameters
  context["max_depth"]->setInt(DEF_MAXDEPTH);

  // Set Gloabal Atmosphere Parameters
  context["atmos_sigma_s"]->setFloat(make_float3(atmos_sigma_s));
  context["atmos_sigma_t"]->setFloat(make_float3(atmos_sigma_t));
  context["atmos_g"]->setFloat(atmos_g);
  context["atmos_dist"]->setFloat(atmos_dist);

  // Set Global Camera Paramters
  //context["aper"]->setFloat(cam_aperature);
  context["exposure"]->setFloat(cam_exposure);
  context["zeus"]->setFloat(make_float3(cam_zeus));

  Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo);
  context["output_buffer"]->set(buffer);

  // Accumulation buffer
  Buffer accum_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                              RT_FORMAT_FLOAT4, width, height);
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
  context->setMissProgram(RADIANCE, context->createProgramFromPTXFile(material_ptx, "radiance_miss"));
  context->setMissProgram(SHADOW, context->createProgramFromPTXFile(material_ptx, "shadow_miss"));

  sky.setSunTheta(sun_theta);
  sky.setSunPhi(sun_phi);
  sky.setTurbidity(2.2f);
  sky.setOvercast(sun_overcast);
  sky.setVariables(context);

  // Split out sun for direct sampling
  sun.direction = sky.getSunDir();
  optix::Onb onb(sun.direction);
  sun.radius = sun_radius;
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
  box->setPrimitiveCount(1u);

  Program box_bounds = context->createProgramFromPTXFile(geometry_ptx, "box_bounds");
  Program box_intersect = context->createProgramFromPTXFile(geometry_ptx, "box_intersect");

  box->setBoundingBoxProgram(box_bounds);
  box->setIntersectionProgram(box_intersect);

  box["boxmin"]->setFloat(boxmin);
  box["boxmax"]->setFloat(boxmax);

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

Material createPhongMaterial(float3 Kd)
{
  Material matl = context->createMaterial();

  Program ch = context->createProgramFromPTXFile(material_ptx, "diffuse_hit_radiance");
  matl->setClosestHitProgram(RADIANCE, ch);

  Program ah = context->createProgramFromPTXFile(material_ptx, "diffuse_hit_shadow");
  matl->setAnyHitProgram(SHADOW, ah);

  matl["Kd"]->setFloat(Kd);
  return matl;
}

void createGeometry()
{
  std::vector<GeometryInstance> gis;

  GeometryGroup geometry_group = context->createGeometryGroup();
  geometry_group->setAcceleration(context->createAcceleration("Trbvh"));

  // Load mesh
  OptiXMesh mesh;
  mesh.context = context;
  mesh.intersection = context->createProgramFromPTXFile(geometry_ptx, "mesh_intersect");
  mesh.bounds = context->createProgramFromPTXFile(geometry_ptx, "mesh_bounds");
  mesh.material = createPhongMaterial(make_float3(DIFFUSE_CONST));
  Matrix4x4 xform = Matrix4x4::identity();

  loadMesh(std::string(sutil::samplesDir()) + "/godRay/model/obj/temple_highres_video.obj", mesh, xform);
  gis.push_back(mesh.geom_instance);

  geometry_group->setChildCount(static_cast<unsigned int>(gis.size()));
  for (int i = 0; i < gis.size(); ++i)
    geometry_group->setChild(i, gis[i]);

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
      const std::string outputImage = std::string(PROGRAM_NAME) + ".png";
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

  const unsigned nw = (unsigned)w;
  const unsigned nh = (unsigned)h;

  CallbackData *cb = static_cast<CallbackData *>(glfwGetWindowUserPointer(window));
  if (cb->camera.resize(nw, nh))
  {
    cb->accumulation_frame = 0;
  }

  sutil::resizeBuffer(getOutputBuffer(), nw, nh);
  sutil::resizeBuffer(context["accum_buffer"]->getBuffer(), nw, nh);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1, 0, 1, -1, 1);
  glViewport(0, 0, nw, nh);
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

  glfwSetWindowSize(window, (int)width, (int)height);
  glfwSetWindowSizeCallback(window, windowSizeCallback);

  return window;
}

void frameCounter(unsigned int frame_count, unsigned int accumulation_frame)
{
  static double fps = -1.0;
  static unsigned last_frame_count = 0;
  static double last_update_time = sutil::currentTime();
  static double current_time = 0.0;
  current_time = sutil::currentTime();
  if (current_time - last_update_time > 0.5f)
  {
    fps = (frame_count - last_frame_count) / (current_time - last_update_time);
    last_frame_count = frame_count;
    last_update_time = current_time;
  }
  if (frame_count > 0 && fps >= 0.0)
  {

    ImGui::SetNextWindowPos(ImVec2(2.0f, 2.0f));
    ImGui::Begin("fps", 0,
                 ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoInputs);
    ImGui::Text("fps: %7.2f", fps);
    ImGui::Text("accum: %d", accumulation_frame);
    ImGui::End();
  }
}

void glfwRun(GLFWwindow *window, sutil::Camera &camera, sutil::PreethamSunSky &sky, DirectionalLight &sun, Buffer light_buffer)
{
  // Initialize GL state
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1, 0, 1, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glViewport(0, 0, width, height);

  unsigned int frame_count = 0;
  unsigned int accumulation_frame = 0;

  // Sun and Sky
  //float sun_phi = sky.getSunPhi();
  //float sun_theta = 0.5f * M_PIf - sky.getSunTheta();
  //float sun_radius = DEF_SUN_RADIUS;
  //float overcast = DEF_OVERCAST;
  //float exposure = DEF_EXPOSURE;

  // Atmosphere
  //float atmos_sigma_s = DEF_ATMOS_SIGMA_S;
  //float atmos_sigma_t = DEF_ATMOS_SIGMA_T;
  //float atmos_dist = DEF_ATMOS_DIST;
  //float atmos_g = DEF_ATMOS_G;

  // Camera
  //float aper = DEF_APERTURE;

  // Renderer
  //int max_depth = DEF_MAXDEPTH;

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

    frameCounter(frame_count++, accumulation_frame);

    {
      static const ImGuiWindowFlags window_flags =
          ImGuiWindowFlags_AlwaysAutoResize |
          ImGuiWindowFlags_NoMove |
          ImGuiWindowFlags_NoScrollbar;

      static const ImGuiWindowFlags header_flags =
          ImGuiTreeNodeFlags_OpenOnDoubleClick |
          ImGuiTreeNodeFlags_OpenOnArrow;

      ImGui::SetNextWindowPos(ImVec2(2.0f, 60.0f));
      ImGui::Begin("Controls", 0, window_flags);

      // Sun and Sky Control
      if (ImGui::CollapsingHeader(" Sun & Sky", header_flags))
      {
        bool sun_changed = false;
        // Sun Rotation Control
        if (ImGui::SliderFloat("sun rotation", &sun_phi, 0.0f, 2.0f * M_PIf))
        {
          sky.setSunPhi(sun_phi);
          sky.setVariables(context);
          sun.direction = sky.getSunDir();
          sun_changed = true;
        }
        // Sun Elevation Control
        if (ImGui::SliderFloat("sun elevation", &sun_theta, 0.0f, 0.5f * M_PIf))
        {
          sky.setSunTheta(sun_theta);
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
        if (ImGui::SliderFloat("overcast", &sun_overcast, 0.0f, 1.0f))
        {
          sky.setOvercast(sun_overcast);
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
      if (ImGui::CollapsingHeader(" Atmosphere", header_flags))
      {
        if (ImGui::SliderFloat("in scattering", &atmos_sigma_s, MIN_ATMOS_SIGMA_S, MAX_ATMOS_SIGMA_S))
        {
          context["atmos_sigma_s"]->setFloat(make_float3(atmos_sigma_s));
          accumulation_frame = 0;
        }
        if (ImGui::SliderFloat("extinction", &atmos_sigma_t, MIN_ATMOS_SIGMA_T, MAX_ATMOS_SIGMA_T))
        {
          context["atmos_sigma_t"]->setFloat(make_float3(atmos_sigma_t));
          accumulation_frame = 0;
        }
        if (ImGui::SliderFloat("dist", &atmos_dist, MIN_ATMOS_DIST, MAX_ATMOS_DIST))
        {
          context["atmos_dist"]->setFloat(atmos_dist);
          accumulation_frame = 0;
        }
        if (ImGui::SliderFloat("g", &atmos_g, MIN_ATMOS_G, MAX_ATMOS_G))
        {
          context["atmos_g"]->setFloat(atmos_g);
          accumulation_frame = 0;
        }
      }
      // Camera Control
      if (ImGui::CollapsingHeader(" Tonemap", header_flags))
      {
        //TODO: find a better representation for the aperature scale
        //if (ImGui::SliderFloat("aperature", &cam_aperature, MIN_APERATURE, MAX_APERATURE))
        //{
        //  context["aper"]->setFloat(cam_aperature);
        //  accumulation_frame = 0;
        //}
        // Tonemap control
        if (ImGui::SliderFloat("exposure", &cam_exposure, MIN_EXPOSURE, MAX_EXPOSURE))
        {
          context["exposure"]->setFloat(cam_exposure);
          accumulation_frame = 0;
        }
        if (ImGui::SliderFloat("zeus", &cam_zeus, MIN_ZEUS, MAX_ZEUS))
        {
          context["zeus"]->setFloat(make_float3(cam_zeus));
          accumulation_frame = 0;
        }
      }
      // Renderer Control
      if (ImGui::CollapsingHeader(" Renderer", header_flags))
      {
        if (ImGui::SliderInt("max depth", &maxdepth, MIN_MAXDEPTH, MAX_MAXDEPTH))
        {
          context["max_depth"]->setInt(maxdepth);
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

               "  -nf | --num_frames           \n"
               "  -md | --max_depth            \n"
               "  -nf | --num_frames           \n"
               "  -wh | --width_height         \n"

               "  -as | --atmos_in_scatter     \n"
               "  -ae | --atmos_extinction     \n"
               "  -ad | --atmos_dist           \n"

               "  -ce | --cam_exposure         \n"
               "  -cp | --cam_position         \n"
               "  -ct | --cam_target           \n"

               "  -sr | --sun_radius           \n"
               "  -st | --sun_theta            \n"
               "  -sp | --sun_phi              \n"

               "App Keystrokes:\n"
               "  q  Quit\n"
               "  s  Save image to '"
            << PROGRAM_NAME << ".png'\n"
                               "  f  Re-center camera\n"
                               "\n"
            << std::endl;

  exit(1);
}

void missing_arg(int i, int argc, int n, const std::string arg, const std::string prog)
{
  if (i == argc - n)
  {
    std::cerr << "Option '" << arg << "' requires " << n << " additional argument.\n";
    printUsageAndExit(prog);
  }
}

int main(int argc, char **argv)
{
  bool use_pbo = true;
  std::string out_file;

  const std::string prog(argv[0]);
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg(argv[i]);

    if (arg == "-h" || arg == "--help")
    {
      printUsageAndExit(prog);
    }
    else if (arg == "-f" || arg == "--file")
    {
      missing_arg(i, argc, 1, arg, prog);
      out_file = argv[++i];
    }
    else if (arg == "-n" || arg == "--nopbo")
    {
      use_pbo = false;
    }
    else if (arg == "-nf" || arg == "--num_frames")
    {
      missing_arg(i, argc, 1, arg, prog);
      numframes = atoi(argv[++i]);
    }
    else if (arg == "-md" || arg == "--max_depth")
    {
      missing_arg(i, argc, 1, arg, prog);
      maxdepth = atoi(argv[++i]);
    }
    else if (arg == "-wh" || arg == "--width_height")
    {
      missing_arg(i, argc, 2, arg, prog);
      width = atoi(argv[++i]);
      height = atoi(argv[++i]);
    }
    else if (arg == "-as" || arg == "--atmos_in_scatter")
    {
      missing_arg(i, argc, 1, arg, prog);
      atmos_sigma_s = atof(argv[++i]);
    }
    else if (arg == "-ae" || arg == "--atmos_extinction")
    {
      missing_arg(i, argc, 1, arg, prog);
      atmos_sigma_t = atof(argv[++i]);
    }
    else if (arg == "-ad" || arg == "--atmos_dist")
    {
      missing_arg(i, argc, 1, arg, prog);
      atmos_dist = atof(argv[++i]);
    }
    else if (arg == "-ce" || arg == "--cam_exposure")
    {
      missing_arg(i, argc, 1, arg, prog);
      cam_exposure = atof(argv[++i]);
    }
    else if (arg == "-cp" || arg == "--cam_position")
    {
      missing_arg(i, argc, 3, arg, prog);
      float cx = (float)atof(argv[++i]);
      float cy = (float)atof(argv[++i]);
      float cz = (float)atof(argv[++i]);
      cam_position = make_float3(cx, cy, cz);
    }
    else if (arg == "-ct" || arg == "--cam_target")
    {
      missing_arg(i, argc, 3, arg, prog);
      cam_target = make_float3(atof(argv[++i]), atof(argv[++i]), atof(argv[++i]));
    }
    else if (arg == "-sr" || arg == "--sun_radius")
    {
      missing_arg(i, argc, 1, arg, prog);
      sun_radius = atof(argv[++i]);
    }
    else if (arg == "-st" || arg == "--sun_theta")
    {
      missing_arg(i, argc, 1, arg, prog);
      sun_theta = atof(argv[++i]);
    }
    else if (arg == "-sp" || arg == "--sun_phi")
    {
      missing_arg(i, argc, 1, arg, prog);
      sun_phi = atof(argv[++i]);
    }
    else if (arg[0] == '-')
    {
      std::cerr << "Unknown option '" << arg << "'\n";
      printUsageAndExit(argv[0]);
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

    sutil::PreethamSunSky sky;
    DirectionalLight sun;
    Buffer light_buffer;
    createLights(sky, sun, light_buffer);

    createGeometry();

    // Note: lighting comes from miss program

    context->validate();

    const optix::float3 camera_eye(cam_position);  //optix::make_float3(0.0f, 0.0f, -150.0f));
    const optix::float3 camera_lookat(cam_target); //optix::make_float3(0.0f, 48.0f, -400.0f));
    const optix::float3 camera_up(optix::make_float3(0.0f, 1.0f, 0.0f));

    sutil::Camera camera(width, height,
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
      //const unsigned int numframes = 800;
      std::cerr << "Accumulating " << numframes << " frames ..." << std::endl;
      for (unsigned int frame = 0; frame < numframes; ++frame)
      {
        if (frame % 100 == 0)
          std::cerr << "Frame: " << frame << std::endl;
        context["frame"]->setUint(frame);
        context->launch(0, width, height);
      }
      sutil::writeBufferToFile(out_file.c_str(), getOutputBuffer());
      std::cerr << "Wrote " << out_file << std::endl;
      destroyContext();
    }
    return 0;
  }
  SUTIL_CATCH(context->get())
}
