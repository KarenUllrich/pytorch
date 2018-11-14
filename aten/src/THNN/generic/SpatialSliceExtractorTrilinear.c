#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialSliceExtractorTrilinear.c"
#else

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )

static inline void THNN_(SpatialSliceExtractorTrilinear_shapeCheck)
     (THTensor *input, THTensor *grid, THTensor *gradOutput) {
  THNN_ARGCHECK(input->nDimension == 5, 2, input,
		"5D input tensor expected but got: %s");
  THNN_ARGCHECK(grid->nDimension == 4, 2, grid,
		"4D grid tensor expected but got: %s");

  int nbatch   = THTensor_(size)(input, 0);
  int channels = THTensor_(size)(input, 1);
  int iheight   = THTensor_(size)(input, 2);
  int iwidth    = THTensor_(size)(input, 3);
  int idepth    = THTensor_(size)(input, 4);
  int oheight   = THTensor_(size)(grid, 1);
  int owidth    = THTensor_(size)(grid, 2);

  THNN_CHECK_DIM_SIZE(grid, 4, 0, nbatch);
  THNN_CHECK_DIM_SIZE(grid, 4, 3, 3);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 0, nbatch);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 1, channels);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 2, oheight);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 3, owidth);
  }
}

#define SAFE_GET(input, x, y, z, n, c, H, W, D) x >= 0 && x < W && y >=0 \
    && y < H && z >= 0 && z < D ? THTensor_fastGet5d(input, n, c, y, x, z) : 0

TH_API void THNN_(SpatialSliceExtractorTrilinear_updateOutput)(
	  THNNState *state,
	  THTensor *input,
	  THTensor *grid,
	  THTensor *output) {

  THNN_(SpatialSliceExtractorTrilinear_shapeCheck)(input, grid, NULL);
  int N = THTensor_(size)(input, 0);
  int C = THTensor_(size)(input, 1);
  int IH = THTensor_(size)(input, 2);
  int IW = THTensor_(size)(input, 3);
  int ID = THTensor_(size)(input, 4);
  int H = THTensor_(size)(grid, 1);
  int W = THTensor_(size)(grid, 2);

  // resize output to the same shape as input
  THTensor_(resize4d)(output, N, C, H, W);

  // loop over each output pixel
  int n, h, w, c;
#pragma omp parallel for private(n, h, w, c)
  for (n = 0; n < N; ++n) {
    for (h = 0; h < H; ++h) {
      for (w = 0; w < W; ++w) {
	// get the corresponding input x, y, z co-ordinates from grid
	real ix = THTensor_fastGet4d(grid, n, h, w, 0);
	real iy = THTensor_fastGet4d(grid, n, h, w, 1);
  real iz = THTensor_fastGet4d(grid, n, h, w, 2);

	// normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
	//ix = ((ix + 1) / 2) * (IW-1);
	//iy = ((iy + 1) / 2) * (IH-1);
  //iz = ((iz + 1) / 2) * (ID-1);

	// get pixel coord for all 8 surounding pixels and pixel-value
  int px_0 = floor(ix);
	int py_0 = floor(iy);
  int pz_0 = floor(iz);
  int px_1 = ceil(ix);
	int py_1 = ceil(iy);
  int pz_1 = ceil(iz);
  real dx = ix - px_0;
  real dy = iy - py_0;
  real dz = iz - pz_0;

	// calculate trilinear weighted pixel value and set output pixel
	for (c = 0; c < C; ++c) {
    	real c_000 = SAFE_GET(input, px_0, py_0, pz_0, n, c, IH, IW, ID);
    	real c_100 = SAFE_GET(input, px_1, py_0, pz_0, n, c, IH, IW, ID);
    	real c_010 = SAFE_GET(input, px_0, py_1, pz_0, n, c, IH, IW, ID);
    	real c_110 = SAFE_GET(input, px_1, py_1, pz_0, n, c, IH, IW, ID);

    	real c_001 = SAFE_GET(input, px_0, py_0, pz_1, n, c, IH, IW, ID);
    	real c_101 = SAFE_GET(input, px_1, py_0, pz_1, n, c, IH, IW, ID);
    	real c_011 = SAFE_GET(input, px_0, py_1, pz_1, n, c, IH, IW, ID);
    	real c_111 = SAFE_GET(input, px_1, py_1, pz_1, n, c, IH, IW, ID);

      real c_00 = c_000 * (1. - dx) + c_100 * (dx);
      real c_01 = c_001 * (1. - dx) + c_101 * (dx);
      real c_10 = c_010 * (1. - dx) + c_110 * (dx);
      real c_11 = c_011 * (1. - dx) + c_111 * (dx);

      real c_0 = c_00 * (1. - dy) + c_10 * (dy);
      real c_1 = c_01 * (1. - dy) + c_11 * (dy);

      real out_val = c_0 * (1. - dz) + c_1 * (dz);

	    THTensor_fastSet4d(output, n, c, h, w, out_val);
	}
      }
    }
  }
}

#define SAFE_ADD(input, x, y, z, n, c, H, W, D, value)		\
  do {								\
    if (x >= 0 && x < W && y >=0 && y < H && z >=0 && z < D) {			\
      real old_value = THTensor_fastGet5d(input, n, c, y, x, z);	\
      THTensor_fastSet5d(input, n, c, y, x, z, value + old_value);	\
    }								\
  } while(0)

TH_API void THNN_(SpatialSliceExtractorTrilinear_updateGradInput)(
	  THNNState *state,
	  THTensor *input, THTensor *gradInput,
	  THTensor *grid, THTensor *gradGrid,
	  THTensor *gradOutput) {

  THNN_(SpatialSliceExtractorTrilinear_shapeCheck)(input, grid, gradOutput);
  int N = THTensor_(size)(input, 0);
  int C = THTensor_(size)(input, 1);
  int IH = THTensor_(size)(input, 2);
  int IW = THTensor_(size)(input, 3);
  int ID = THTensor_(size)(input, 4);
  int H = THTensor_(size)(grid, 1);
  int W = THTensor_(size)(grid, 2);

  THTensor_(resize5d)(gradInput, N, C, IH, IW, ID);
  THTensor_(resize4d)(gradGrid, N, H, W, 3);
  THTensor_(zero)(gradInput);
  THTensor_(zero)(gradGrid);

  // loop over each output pixel
  int n, h, w;
#pragma omp parallel for private(n, h, w)
  for (n = 0; n < N; ++n) {
    for (h = 0; h < H; ++h) {
      for (w = 0; w < W; ++w) {
	// get the corresponding input x, y co-ordinates from grid
	real ix = THTensor_fastGet4d(grid, n, h, w, 0);
	real iy = THTensor_fastGet4d(grid, n, h, w, 1);
	real iz = THTensor_fastGet4d(grid, n, h, w, 2);

	real gix = 0;
	real giy = 0;
	real giz = 0;

	// normalize ix, iy from [-1, 1] to [0, H-1] & [0, W-1]
	//ix = ((ix + 1) / 2) * (IW-1);
	//iy = ((iy + 1) / 2) * (IH-1);
	//iz = ((iz + 1) / 2) * (ID-1);

  // get pixel coord for all 8 surounding pixels and pixel-value
  int px_0 = floor(ix);
	int py_0 = floor(iy);
  int pz_0 = floor(iz);
  int px_1 = ceil(ix);
	int py_1 = ceil(iy);
  int pz_1 = ceil(iz);

  // get surfaces to each neighbor:
  real dx = ix - px_0;
  real dy = iy - py_0;
  real dz = iz - pz_0;

  // compute surfaces
  real s_000 = (1. - dx) * (1. - dy) * (1. - dz);
  real s_100 = (dx) * (1. - dy) * (1. - dz);
  real s_010 = (1. - dx) * (dy) * (1. - dz);
  real s_110 = (dx) * (dy) * (1. - dz);
  real s_001 = (1. - dx) * (1. - dy) * (dz);
  real s_101 = (dx) * (1. - dy) * (dz);
  real s_011 = (1. - dx) * (dy) * (dz);
  real s_111 = (dx) * (dy) * (dz);

	for (int c = 0; c < C; ++c) {
	  real gradout = THTensor_fastGet4d(gradOutput, n, c, h, w);

    // calculate and set gradInput eq.(6) in Spatial Transformer paper
    SAFE_ADD(gradInput, px_0, py_0, pz_0, n, c, IH, IW, ID, s_000 * gradout);
    SAFE_ADD(gradInput, px_1, py_0, pz_0, n, c, IH, IW, ID, s_100 * gradout);
    SAFE_ADD(gradInput, px_0, py_1, pz_0, n, c, IH, IW, ID, s_010 * gradout);
    SAFE_ADD(gradInput, px_1, py_1, pz_0, n, c, IH, IW, ID, s_110 * gradout);

    SAFE_ADD(gradInput, px_0, py_0, pz_1, n, c, IH, IW, ID, s_001 * gradout);
    SAFE_ADD(gradInput, px_1, py_0, pz_1, n, c, IH, IW, ID, s_101 * gradout);
    SAFE_ADD(gradInput, px_0, py_1, pz_1, n, c, IH, IW, ID, s_011 * gradout);
    SAFE_ADD(gradInput, px_1, py_1, pz_1, n, c, IH, IW, ID, s_111 * gradout);

    // calculate gradGrid eq.(7) in Spatial Transformer paper
    real c_000 = SAFE_GET(input, px_0, py_0, pz_0, n, c, IH, IW, ID);
    real c_100 = SAFE_GET(input, px_1, py_0, pz_0, n, c, IH, IW, ID);
    real c_010 = SAFE_GET(input, px_0, py_1, pz_0, n, c, IH, IW, ID);
    real c_110 = SAFE_GET(input, px_1, py_1, pz_0, n, c, IH, IW, ID);

    real c_001 = SAFE_GET(input, px_0, py_0, pz_1, n, c, IH, IW, ID);
    real c_101 = SAFE_GET(input, px_1, py_0, pz_1, n, c, IH, IW, ID);
    real c_011 = SAFE_GET(input, px_0, py_1, pz_1, n, c, IH, IW, ID);
    real c_111 = SAFE_GET(input, px_1, py_1, pz_1, n, c, IH, IW, ID);

    gix -= c_000 * (1. - dy) * (1. - dz) * gradout;
    gix += c_100 * (1. - dy) * (1. - dz) * gradout;
    gix -= c_010 * (dy) * (1. - dz) * gradout;
    gix += c_110 * (dy) * (1. - dz) * gradout;
    gix -= c_001 * (1. - dy) * (dz) * gradout;
    gix += c_101 * (1. - dy) * (dz) * gradout;
    gix -= c_011 * (dy) * (dz) * gradout;
    gix += c_111 * (dy) * (dz) * gradout;

    giy -= c_000 * (1. - dx) * (1. - dz) * gradout;
    giy -= c_100 * (dx) * (1. - dz) * gradout;
    giy += c_010 * (1. - dx) * (1. - dz) * gradout;
    giy += c_110 * (dx) * (1. - dz) * gradout;
    giy -= c_001 * (1. - dx) * (dz) * gradout;
    giy -= c_101 * (dx) * (dz) * gradout;
    giy += c_011 * (1. - dx) * (dz) * gradout;
    giy += c_111 * (dx) * (dz) * gradout;

    giz -= c_000 * (1. - dx) * (1. - dy) * gradout;
    giz -= c_100 * (dx) * (1. - dy) * gradout;
    giz -= c_010 * (1. - dx) * (dy) * gradout;
    giz -= c_110 * (dx) * (dy)* gradout;
    giz += c_001 * (1. - dx) * (1. - dy) * gradout;
    giz += c_101 * (dx) * (1. - dy) * gradout;
    giz += c_011 * (1. - dx) * (dy) * gradout;
    giz += c_111 * (dx) * (dy) * gradout;
	}

	// un-normalize gradGrid values back to [-1, 1] constraints
	//gix = gix * (IW - 1) / 2;
	//giy = giy * (IH - 1) / 2;
  //giz = giz * (ID - 1) / 2;

	real gix_old = THTensor_fastGet4d(gradGrid, n, h, w, 0);
	real giy_old = THTensor_fastGet4d(gradGrid, n, h, w, 1);
  real giz_old = THTensor_fastGet4d(gradGrid, n, h, w, 2);

	THTensor_fastSet4d(gradGrid, n, h, w, 0, gix_old + gix);
	THTensor_fastSet4d(gradGrid, n, h, w, 1, giy_old + giy);
  THTensor_fastSet4d(gradGrid, n, h, w, 2, giz_old + giz);
      }
    }
  }
}

#undef MIN
#undef SAFE_GET
#undef SAFE_ADD

#endif
