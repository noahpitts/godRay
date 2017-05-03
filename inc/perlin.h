/*** Make some Perlin noise! ***/
#include <random>
#include <math.h>

namespace Perlin {
  const unsigned int SIZE = 3;

  float G[SIZE+1][SIZE+1][SIZE+1][3];

  inline float rand01() { return ((float)rand()) / RAND_MAX; }

  inline void sample_sphere(float &x, float &y, float &z) {
    float r;
    do {
      x = rand01();
      y = rand01();
      z = rand01();
      r = x*x + y*y + z*z;
    } while(r > 1.0f);

    r = sqrt(r);
    x /= r;
    y /= r;
    z /= r;
  }

  float p_lerp(float u, float v, float t) { return u + (v - u) * t; }

  float dot_grid_grad(float fx, float fy, float fz, int ix, int iy, int iz) {
    float dx = fx - (float) ix;
    float dy = fy - (float) iy;
    float dz = fz - (float) iz;
    float *v = G[ix][iy][iz];
    return dx * v[0] + dy * v[1] + dz * v[2];
  }

  float sample(float x, float y, float z) {
    int x0 = (int) x; int x1 = x0 + 1; float sx = x - (float) x0;
    int y0 = (int) y; int y1 = y0 + 1; float sy = y - (float) y0;
    int z0 = (int) z; int z1 = z0 + 1; float sz = z - (float) z0;

    float c000 = dot_grid_grad(x, y, z, x0, y0, z0);
    float c001 = dot_grid_grad(x, y, z, x0, y0, z1);
    float c010 = dot_grid_grad(x, y, z, x0, y1, z0);
    float c011 = dot_grid_grad(x, y, z, x0, y1, z1);
    float c100 = dot_grid_grad(x, y, z, x1, y0, z0);
    float c101 = dot_grid_grad(x, y, z, x1, y0, z1);
    float c110 = dot_grid_grad(x, y, z, x1, y1, z0);
    float c111 = dot_grid_grad(x, y, z, x1, y1, z1);

    float c00 = p_lerp(c000, c001, sz);
    float c01 = p_lerp(c010, c011, sz);
    float c10 = p_lerp(c100, c101, sz);
    float c11 = p_lerp(c110, c111, sz);

    float c0 = p_lerp(c00, c01, sy);
    float c1 = p_lerp(c10, c11, sy);

    float c = p_lerp(c0, c1, sx);

    return c;
  }

  void initialize_grid() {
    for(int x = 0; x <= SIZE; ++x) {
      for(int y = 0; y <= SIZE; ++y) {
        for(int z = 0; z <= SIZE; ++z) {
          sample_sphere(G[x][y][z][0], G[x][y][z][1], G[x][y][z][2]);
        }
      }
    }
  }
}
