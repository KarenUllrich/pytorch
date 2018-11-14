#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)
#define SAFE_ADD(input, x, y, n, c, H, W, value)    \
  do {    \
    if (WITHIN_BOUNDS(x, y, H, W)) {    \
      atomicAdd(&input[n][c][y][x], value);   \
    }   \
  } while(0)

#define WITHIN_BOUNDS3(x, y,z, H, W,D) (x >= 0 && x < W && y >= 0 && y < H&& z >= 0 && z < D)
#define SAFE_ADD3(input, x, y, z, n, c, H, W, D, value)    \
  do {    \
    if (WITHIN_BOUNDS3(x, y, z, H, W, D)) {    \
      atomicAdd(&input[n][c][y][x][z], value);   \
    }   \
  } while(0)

template <typename Dtype>
__launch_bounds__(1024)
__global__ void SpatialSliceExtractorTrilinear_updateOutput_kernel(
    const int nthreads,
    THCDeviceTensor<Dtype, 5> input,
    THCDeviceTensor<Dtype, 4> grid,
    THCDeviceTensor<Dtype, 4> output) {

  int N = input.getSize(0);
  int C = input.getSize(1);
  int IH = input.getSize(2);
  int IW = input.getSize(3);
  int ID = input.getSize(4);
  int H = grid.getSize(1);
  int W = grid.getSize(2);

  CUDA_KERNEL_LOOP(index, nthreads) {

    const int n = index % N;
    const int h = (index / N) % H;
    const int w = (index / (N * H)) % W;
    int c;

    // get the corresponding input x, y co-ordinates from grid
    Dtype ix = grid[n][h][w][0];
    Dtype iy = grid[n][h][w][1];
    Dtype iz = grid[n][h][w][2];

    // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
    //ix = ScalarConvert<float,Dtype>::to(((ix + 1) / 2) * (IW-1));
    //iy = ScalarConvert<float,Dtype>::to(((iy + 1) / 2) * (IH-1));
    //iz = ScalarConvert<float,Dtype>::to(((iz + 1) / 2) * (ID-1));

    // get pixel coord for all 8 surounding pixels and pixel-value
    int px_0 = floor(ScalarConvert<Dtype,float>::to(ix));
    int py_0 = floor(ScalarConvert<Dtype,float>::to(iy));
    int pz_0 = floor(ScalarConvert<Dtype,float>::to(iz));

    int px_1 = ceil(ScalarConvert<Dtype,float>::to(ix));
    int py_1 = ceil(ScalarConvert<Dtype,float>::to(iy));
    int pz_1 = ceil(ScalarConvert<Dtype,float>::to(iz));

    // get surfaces to each neighbor:
    Dtype dx = ix - px_0;
    Dtype dy = iy - py_0;
    Dtype dz = iz - pz_0;

    // create holder for pixel values
    Dtype c_00; Dtype c_01; Dtype c_10; Dtype c_11;
    Dtype c_0; Dtype c_1;

    for (c = 0; c < C; ++c) {
      c_00 = ScalarConvert<int,Dtype>::to(0);
      c_01 = ScalarConvert<int,Dtype>::to(0);
      c_10 = ScalarConvert<int,Dtype>::to(0);
      c_11 = ScalarConvert<int,Dtype>::to(0);

      if (WITHIN_BOUNDS3(px_0, py_0, pz_0, IH, IW, ID) ) {
        c_00 += input[n][c][py_0][px_0][pz_0] * (ScalarConvert<int,Dtype>::to(1) - dx) ;
      }
      if (WITHIN_BOUNDS3(px_1, py_0, pz_0, IH, IW, ID)) {
        c_00 += input[n][c][py_0][px_1][pz_0] * dx;
      }
      if (WITHIN_BOUNDS3(px_0, py_0, pz_1, IH, IW, ID) ) {
        c_01 += input[n][c][py_0][px_0][pz_1] * (ScalarConvert<int,Dtype>::to(1) - dx) ;
      }
      if (WITHIN_BOUNDS3(px_1, py_0, pz_1, IH, IW, ID)) {
        c_01 += input[n][c][py_0][px_1][pz_1] * dx;
      }
      if (WITHIN_BOUNDS3(px_0, py_1, pz_0, IH, IW, ID) ) {
        c_10 += input[n][c][py_1][px_0][pz_0] * (ScalarConvert<int,Dtype>::to(1) - dx) ;
      }
      if (WITHIN_BOUNDS3(px_1, py_1, pz_0, IH, IW, ID)) {
        c_10 += input[n][c][py_1][px_1][pz_0] * dx;
      }
      if (WITHIN_BOUNDS3(px_0, py_1, pz_1, IH, IW, ID) ) {
        c_11 += input[n][c][py_1][px_0][pz_1] * (ScalarConvert<int,Dtype>::to(1) - dx) ;
      }
      if (WITHIN_BOUNDS3(px_1, py_1, pz_1, IH, IW, ID)) {
        c_11 += input[n][c][py_1][px_1][pz_1] * dx;
      }
      c_0 = c_00 * (ScalarConvert<int,Dtype>::to(1) - dy) + c_10 * dy;
      c_1 = c_01 * (ScalarConvert<int,Dtype>::to(1) - dy) + c_11 * dy;

      output[n][c][h][w] = c_0 * (ScalarConvert<int,Dtype>::to(1) - dz) + c_1 * dz;
    }
  }
}

template <typename Dtype>
__launch_bounds__(1024)
__global__ void SpatialSliceExtractorTrilinear_updateGradInput_kernel(
    const int nthreads,
    THCDeviceTensor<Dtype, 5> input, THCDeviceTensor<Dtype, 5> gradInput,
    THCDeviceTensor<Dtype, 4> grid, THCDeviceTensor<Dtype, 4> gradGrid,
    THCDeviceTensor<Dtype, 4> gradOutput) {

  int N = input.getSize(0);
  int C = input.getSize(1);
  int IH = input.getSize(2);
  int IW = input.getSize(3);
  int ID = input.getSize(4);
  int H = grid.getSize(1);
  int W = grid.getSize(2);

  CUDA_KERNEL_LOOP(index, nthreads) {

    const int n = index % N;
    const int h = (index / N) % H;
    const int w = (index / (N * H)) % W;

    // get the corresponding input x, y co-ordinates from grid
    Dtype ix = grid[n][h][w][0];
    Dtype iy = grid[n][h][w][1];
    Dtype iz = grid[n][h][w][2];

    Dtype gix = ScalarConvert<int,Dtype>::to(0);
    Dtype giy = ScalarConvert<int,Dtype>::to(0);
    Dtype giz = ScalarConvert<int,Dtype>::to(0);

    // normalize ix, iy from [-1, 1] to [0, H-1] & [0, W-1]
    //ix = ScalarConvert<float,Dtype>::to(((ix + 1) / 2) * (IW-1));
    //iy = ScalarConvert<float,Dtype>::to(((iy + 1) / 2) * (IH-1));
    //iz = ScalarConvert<float,Dtype>::to(((iz + 1) / 2) * (ID-1));;

    // get pixel coord for all 8 surounding pixels and pixel-value
    int px_0 = floor(ScalarConvert<Dtype,float>::to(ix));
    int py_0 = floor(ScalarConvert<Dtype,float>::to(iy));
    int pz_0 = floor(ScalarConvert<Dtype,float>::to(iz));
    int px_1 = ceil(ScalarConvert<Dtype,float>::to(ix));
    int py_1 = ceil(ScalarConvert<Dtype,float>::to(iy));
    int pz_1 = ceil(ScalarConvert<Dtype,float>::to(iz));

    // get surfaces to each neighbor:
    Dtype dx = ix - px_0;
    Dtype dy = iy - py_0;
    Dtype dz = iz - pz_0;

    Dtype one = ScalarConvert<int,Dtype>::to(1);
    Dtype negone = ScalarConvert<int,Dtype>::to(-1);

    // compute surfaces
    Dtype s_000 = (one - dx) * (one - dy) * (one - dz);
    Dtype s_100 = (dx) * (one - dy) * (one - dz);
    Dtype s_010 = (one - dx) * (dy) * (one - dz);
    Dtype s_110 = (dx) * (dy) * (one - dz);
    Dtype s_001 = (one - dx) * (one - dy) * (dz);
    Dtype s_101 = (dx) * (one - dy) * (dz);
    Dtype s_011 = (one - dx) * (dy) * (dz);
    Dtype s_111 = (dx) * (dy) * (dz);

    // calculate derivatives
    Dtype gradout;

    for (int c = 0; c < C; ++c) {
      gradout = gradOutput[n][c][h][w];

      // calculate and set gradInput eq.(6) in Spatial Transformer paper
      SAFE_ADD3(gradInput, px_0, py_0, pz_0, n, c, IH, IW, ID, s_000 * gradout);
      SAFE_ADD3(gradInput, px_1, py_0, pz_0, n, c, IH, IW, ID, s_100 * gradout);
      SAFE_ADD3(gradInput, px_0, py_1, pz_0, n, c, IH, IW, ID, s_010 * gradout);
      SAFE_ADD3(gradInput, px_1, py_1, pz_0, n, c, IH, IW, ID, s_110 * gradout);

      SAFE_ADD3(gradInput, px_0, py_0, pz_1, n, c, IH, IW, ID, s_001 * gradout);
      SAFE_ADD3(gradInput, px_1, py_0, pz_1, n, c, IH, IW, ID, s_101 * gradout);
      SAFE_ADD3(gradInput, px_0, py_1, pz_1, n, c, IH, IW, ID, s_011 * gradout);
      SAFE_ADD3(gradInput, px_1, py_1, pz_1, n, c, IH, IW, ID, s_111 * gradout);

      // calculate gradGrid eq.(7) in Spatial Transformer paper
      Dtype c_000 = ScalarConvert<int,Dtype>::to(0);
      Dtype c_100 = ScalarConvert<int,Dtype>::to(0);
      Dtype c_010 = ScalarConvert<int,Dtype>::to(0);
      Dtype c_110 = ScalarConvert<int,Dtype>::to(0);

      Dtype c_001 = ScalarConvert<int,Dtype>::to(0);
      Dtype c_101 = ScalarConvert<int,Dtype>::to(0);
      Dtype c_011 = ScalarConvert<int,Dtype>::to(0);
      Dtype c_111 = ScalarConvert<int,Dtype>::to(0);

      if (WITHIN_BOUNDS3(px_0, py_0, pz_0, IH, IW, ID)) {
        c_000 = input[n][c][py_0][px_0][pz_0];
      }
      if (WITHIN_BOUNDS3(px_0, py_1, pz_0, IH, IW, ID)) {
        c_010 = input[n][c][py_1][px_0][pz_0];
      }
      if (WITHIN_BOUNDS3(px_1, py_1, pz_0, IH, IW, ID)) {
        c_110 = input[n][c][py_1][px_1][pz_0];
      }
      if (WITHIN_BOUNDS3(px_1, py_0, pz_0, IH, IW, ID)) {
        c_100 = input[n][c][py_0][px_1][pz_0];
      }

      if (WITHIN_BOUNDS3(px_0, py_0, pz_1, IH, IW, ID)) {
        c_001 = input[n][c][py_0][px_0][pz_1];
      }
      if (WITHIN_BOUNDS3(px_1, py_0, pz_1, IH, IW, ID)) {
        c_101 = input[n][c][py_0][px_1][pz_1];
      }
      if (WITHIN_BOUNDS3(px_0, py_1, pz_1, IH, IW, ID)) {
        c_011 = input[n][c][py_1][px_0][pz_1];
      }
      if (WITHIN_BOUNDS3(px_1, py_1, pz_1, IH, IW, ID)) {
        c_111 = input[n][c][py_1][px_1][pz_1];
      }

  	  gix += negone *  (c_000 * (one - dy) * (one - dz) * gradout);
  	  gix += c_100 * (one - dy) * (one - dz) * gradout;
  	  gix += negone *  (c_010 * (dy) * (one - dz) * gradout);
  	  gix += c_110 * (dy) * (one - dz) * gradout;
  	  gix += negone *  (c_001 * (one - dy) * (dz) * gradout);
  	  gix += c_101 * (one - dy) * (dz) * gradout;
  	  gix += negone *  (c_011 * (dy) * (dz) * gradout);
  	  gix += c_111 * (dy) * (dz) * gradout;

  	  giy += negone *  (c_000 * (one - dx) * (one - dz) * gradout);
  	  giy += negone *  (c_100 * (dx) * (one - dz) * gradout);
  	  giy += c_010 * (one - dx) * (one - dz) * gradout;
  	  giy += c_110 * (dx) * (one - dz) * gradout;
  	  giy += negone *  (c_001 * (one - dx) * (dz) * gradout);
  	  giy += negone *  (c_101 * (dx) * (dz) * gradout);
  	  giy += c_011 * (one - dx) * (dz) * gradout;
  	  giy += c_111 * (dx) * (dz) * gradout;

  	  giz += negone *  (c_000 * (one - dx) * (one - dy) * gradout);
  	  giz += negone *  (c_100 * (dx) * (one - dy) * gradout);
  	  giz += negone *  (c_010 * (one - dx) * (dy) * gradout);
  	  giz += negone *  (c_110 * (dx) * (dy)* gradout);
  	  giz += c_001 * (one - dx) * (one - dy) * gradout;
  	  giz += c_101 * (dx) * (one - dy) * gradout;
  	  giz += c_011 * (one - dx) * (dy) * gradout;
  	  giz += c_111 * (dx) * (dy) * gradout;
  	}

    // un-normalize gradGrid values back to [-1, 1] constraints
    //gix = gix * (IW - 1) / 2;
    //giy = giy * (IH - 1) / 2;
    //giz = giz * (ID - 1) / 2;

    Dtype gix_old = gradGrid[n][h][w][0];
    Dtype giy_old = gradGrid[n][h][w][1];
    Dtype giz_old = gradGrid[n][h][w][2];

    gradGrid[n][h][w][0] = gix_old + gix;
    gradGrid[n][h][w][1] = giy_old + giy;
    gradGrid[n][h][w][2] = giz_old + giz;
  }
}

#undef WITHIN_BOUNDS
#undef SAFE_ADD
#undef WITHIN_BOUNDS3
#undef SAFE_ADD3

#include "generic/SpatialSliceExtractorTrilinear.cu"
#include "THCGenerateFloatTypes.h"
