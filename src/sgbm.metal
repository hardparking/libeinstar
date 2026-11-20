// OpenCV StereoSGBM, MODE_SGBM_3WAY, on Apple GPUs.
//
// Reproduces cv::StereoSGBM::compute (minDisparity 0, disp12MaxDiff 1,
// preFilterCap 0, any block size) up to, but not including, its 3x3 median
// and speckle filters, which the host runs with OpenCV. Every step uses the
// same integer arithmetic as OpenCV's scalar code, so the result is
// bit-identical, not just close:
//
//   bt_prep     Birchfield-Tomasi inputs per pixel: the clipped x-Sobel
//               ("prefiltered") channel and the raw channel, each with the
//               min/max of its half-pixel neighbours.
//   top_pass    matching cost C (the per-pixel costs summed over the block)
//               and the top-to-bottom aggregation path, one simdgroup per
//               column.
//   horiz_pass  left-to-right and right-to-left paths, winner-take-all,
//               uniqueness, subpixel fit and the left-right check, one
//               simdgroup per row.
//
// OpenCV splits the image into 4 horizontal stripes and restarts the top
// path in each, after a warm-up overlap; that is reproduced here too. Rows
// are processed in bands of Rs rows per stripe to bound memory.
//
// Within a simdgroup, lane l owns disparities l + 32 * j, j < CHUNK, so
// every load and store of a disparity vector is coalesced, and the minimum
// over all disparities is one simd_min. CHUNK (ceil(D / 32)) and HALF (half
// the block size) are set per pipeline through preprocessor macros.
#include <metal_stdlib>
using namespace metal;

#ifndef CHUNK
#define CHUNK 1
#endif
#ifndef HALF
#define HALF 2  // block_size / 2
#endif

constant int kStripes = 4;
constant int kInf = 0x7fff;  // SHRT_MAX, OpenCV's "no neighbour" cost
constant int kCap = 15;      // OpenCV's ftzero for preFilterCap 0

struct Params {
  int W, H;      // padded image size
  int D;         // number of disparities, searched over [0, D)
  int width1;    // W - D: columns that get a disparity, starting at column D
  int SW2, SH2;  // half block width and height
  int P1, P2;
  int uniqueness;
  int disp12_max_diff;
  int stripe_sz;
  int Rs;       // band rows per stripe
  int band;     // current band
  uint plane;  // P's path cost plane follows its C plane at this offset
  int overlap;  // warm-up rows before each stripe after the first
  int pad0;
  int src[kStripes];   // first row of each stripe, warm-up included
  int rows[kStripes];  // rows each stripe processes
};

// OpenCV's calcPixelCostBT row buffer value at column x: channel 0 is the
// clipped Sobel response, channel 1 the raw intensity; both are kCap at the
// first and last column.
inline int prow(const device uchar* img, int W, int H, int x, int y, int ch) {
  if (x <= 0 || x >= W - 1) return kCap;
  const device uchar* r = img + y * W;
  if (ch == 1) return r[x];
  const device uchar* rn = img + max(y - 1, 0) * W;
  const device uchar* rs = img + min(y + 1, H - 1) * W;
  const int v = (int(r[x + 1]) - int(r[x - 1])) * 2 + int(rn[x + 1]) - int(rn[x - 1]) + int(rs[x + 1]) -
                int(rs[x - 1]);
  return clamp(v, -kCap, kCap) + kCap;
}

inline uchar4 bt_value(const device uchar* img, int W, int H, int x, int y, int ch) {
  const int u = prow(img, W, H, x, y, ch);
  const int ul = x > 0 ? (u + prow(img, W, H, x - 1, y, ch)) / 2 : u;
  const int ur = x < W - 1 ? (u + prow(img, W, H, x + 1, y, ch)) / 2 : u;
  return uchar4(u, min(min(ul, ur), u), max(max(ul, ur), u), 0);
}

// PL/PR: [H][W][2] of (value, lo, hi, 0) for channels 0 and 1.
kernel void bt_prep(const device uchar* L [[buffer(0)]], const device uchar* R [[buffer(1)]],
                    device uchar4* PL [[buffer(2)]], device uchar4* PR [[buffer(3)]],
                    constant Params& p [[buffer(4)]], device short* OUT [[buffer(5)]],
                    uint2 gid [[thread_position_in_grid]]) {
  const int x = gid.x, y = gid.y;
  if (x >= p.W || y >= p.H) return;
  OUT[uint(y) * p.W + x] = -16;
  const uint i = (uint(y) * p.W + x) * 2;
  for (int ch = 0; ch < 2; ++ch) {
    PL[i + ch] = bt_value(L, p.W, p.H, x, y, ch);
    PR[i + ch] = bt_value(R, p.W, p.H, x, y, ch);
  }
}

inline int bt_cost(uchar4 a, uchar4 b) {
  const int u = a.x, u0 = a.y, u1 = a.z, v = b.x, v0 = b.y, v1 = b.z;
  const int c0 = max(0, max(u - v1, v0 - u));
  const int c1 = max(0, max(v - u1, u0 - v));
  return min(c0, c1);
}

// Matching cost of left column D + x against right column D + x - d.
inline int pixel_cost(const device uchar4* PL, const device uchar4* PR, uint row, int D, int x, int d) {
  const uint il = (row + D + x) * 2, ir = (row + D + x - d) * 2;
  return bt_cost(PL[il], PR[ir]) + (bt_cost(PL[il + 1], PR[ir + 1]) >> 2);
}

// One step of an SGM path: Lr(d) = C(d) + min(Lp(d), Lp(d+-1) + P1,
// minLp + P2) - (minLp + P2), with C already holding the +P2 offset as in
// OpenCV. Lane l holds d = l + 32 * j in slot j, so d - 1 is in the lane
// below (lane 31 of slot j - 1 for lane 0) and d + 1 in the lane above.
// Returns this lane's minimum over its valid disparities.
inline int sgm_step(thread const short* C, thread const short* Lp, thread short* Ln, int min_prev, uint lane,
                    constant Params& p) {
  const int mp2 = min_prev + p.P2;
  int lane_min = kInf;
  // Rolling shuffles (not arrays) to keep register pressure down.
  int below_prev = kInf;
  int below = simd_shuffle_rotate_up(Lp[0], ushort(1));    // lane l - 1 (lane 0: lane 31)
  int above = simd_shuffle_rotate_down(Lp[0], ushort(1));  // lane l + 1 (lane 31: lane 0)
  for (int j = 0; j < CHUNK; ++j) {
    const int above_next = j + 1 < CHUNK ? simd_shuffle_rotate_down(Lp[j + 1], ushort(1)) : kInf;
    const int below_next = j + 1 < CHUNK ? simd_shuffle_rotate_up(Lp[j + 1], ushort(1)) : kInf;
    const int d = int(lane) + 32 * j;
    if (d < p.D) {
      const int lm = lane > 0 ? below : below_prev;
      const int lp = d + 1 >= p.D ? kInf : (lane < 31 ? above : above_next);
      const int v = C[j] + min(min(lm + p.P1, lp + p.P1), min(int(Lp[j]), mp2)) - mp2;
      Ln[j] = short(v);
      lane_min = min(lane_min, v);
    } else {
      Ln[j] = kInf;
    }
    below_prev = below;
    below = below_next;
    above = above_next;
  }
  return lane_min;
}

// The disparity OpenCV's SIMD winner-take-all picks among those costing
// min_cost. It keeps, in each of its 8 int16 lanes (disparities d with the
// same d % 8), the last d reaching that lane's minimum, then takes the
// smallest d over the lanes holding the overall minimum. With a single
// minimum that is simply its disparity.
constant int kSimdLanes = 8;  // v_int16 lanes of 128-bit SIMD (NEON, SSE)

inline int best_disparity(thread const short* T, int min_cost, int d0, int D) {
  int count = 0, first = 0x7fffffff;
  for (int j = 0; j < CHUNK; ++j)
    if (d0 + 32 * j < D && int(T[j]) == min_cost) {
      ++count;
      first = min(first, d0 + 32 * j);
    }
  if (simd_sum(count) == 1) return simd_min(first);
  int last[kSimdLanes];
  for (int c = 0; c < kSimdLanes; ++c) last[c] = -1;
  for (int j = 0; j < CHUNK; ++j)
    if (d0 + 32 * j < D && int(T[j]) == min_cost) last[(d0 + 32 * j) % kSimdLanes] = d0 + 32 * j;
  int best = 0x7fffffff;
  for (int c = 0; c < kSimdLanes; ++c) {
    const int v = simd_max(last[c]);
    if (v >= 0) best = min(best, v);
  }
  return best;
}

constexpr constant int kTileColumns = 8;  // columns (one simdgroup each) per top_pass threadgroup
constexpr constant int kTileWidth = kTileColumns + 2 * HALF;
constexpr constant int kFifo = 2 * HALF + 1;  // block rows
constexpr constant int kMaxD = 32 * CHUNK;

// Pixel costs of image row g for the threadgroup's columns plus HALF on
// either side (clamped to the image), into t[column][d].
inline void cost_tile(threadgroup uchar* t, const device uchar4* PL, const device uchar4* PR, constant Params& p,
                      int g, int x0, uint tid) {
  const uint row = uint(g) * p.W;
  for (int c = 0; c < kTileWidth; ++c) {
    const int x = clamp(x0 - HALF + c, 0, p.width1 - 1);
    for (int d = int(tid); d < p.D; d += 32 * kTileColumns)
      t[c * kMaxD + d] = uchar(pixel_cost(PL, PR, row, p.D, x, d));
  }
}

// Block-row sum (hsum) of this simdgroup's column from a cost tile.
inline void tile_hsum(threadgroup const uchar* t, uint sg, int d0, thread short* hs) {
  for (int j = 0; j < CHUNK; ++j) {
    int v = 0;
    for (int k = 0; k <= 2 * HALF; ++k) v += t[(int(sg) + k) * kMaxD + d0 + 32 * j];
    hs[j] = short(v);
  }
}

// Matching cost C (block sums, +P2 as in OpenCV) and the top-to-bottom path,
// one simdgroup per column. The block's column sums (hsum) come from a cost
// tile shared by the threadgroup's 8 columns; the last 2 * HALF + 1 of them
// stay in registers (F, oldest first) so the vertical sum slides.
//
// P: two planes [kStripes][Rs][width1][D], C then the top path cost. TS carries each
// column's C, top path cost and F from one band to the next:
// [kStripes][width1][2 + kFifo][D].
kernel void top_pass(const device uchar4* PL [[buffer(0)]], const device uchar4* PR [[buffer(1)]],
                     device ushort* P [[buffer(2)]], device short* TS [[buffer(3)]],
                     constant Params& p [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                     uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup uchar tile[2][kTileWidth * kMaxD];
  const int s = tg.y, x0 = tg.x * kTileColumns, x = x0 + int(sg);
  const int r0 = p.band * p.Rs;
  if (r0 >= p.rows[s]) return;  // whole threadgroup
  const bool active = x < p.width1;
  const int nr = min(p.Rs, p.rows[s] - r0);
  const int src = p.src[s];
  const int d0 = int(lane);  // this lane's disparities: d0 + 32 * j
  device short* ts = TS + uint(s * p.width1 + min(x, p.width1 - 1)) * (2 + kFifo) * p.D;

  short C[CHUNK], L[CHUNK], Ln[CHUNK], F[kFifo][CHUNK], hs[CHUNK];
  int min_prev = 0, buf = 0;
  if (r0 > 0) {
    int m = kInf;
    for (int j = 0; j < CHUNK; ++j) {
      const int d = min(d0 + 32 * j, p.D - 1);
      C[j] = ts[d];
      L[j] = d0 + 32 * j < p.D ? ts[p.D + d] : short(kInf);
      for (int k = 0; k < kFifo; ++k) F[k][j] = ts[(2 + k) * p.D + d];
      m = min(m, int(L[j]));
    }
    min_prev = simd_min(m);
  } else {
    // First row of the stripe: the window holds row src SH2 + 1 times (rows
    // above it replicate it), then rows src + 1 .. src + SH2.
    for (int k = 0; k <= HALF; ++k) {
      cost_tile(tile[buf], PL, PR, p, min(src + k, p.H - 1), x0, tid);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      tile_hsum(tile[buf], sg, d0, hs);
      buf ^= 1;
      for (int j = 0; j < CHUNK; ++j) {
        if (k == 0)
          for (int q = 0; q <= HALF; ++q) F[q][j] = hs[j];
        else
          F[HALF + k][j] = hs[j];
      }
    }
    for (int j = 0; j < CHUNK; ++j) {
      int c = p.P2;
      for (int q = 0; q < kFifo; ++q) c += F[q][j];
      C[j] = short(c);
      L[j] = 0;
    }
  }
  for (int r = 0; r < nr; ++r) {
    const int y = src + r0 + r;
    if (y != src) {
      cost_tile(tile[buf], PL, PR, p, min(y + HALF, p.H - 1), x0, tid);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      tile_hsum(tile[buf], sg, d0, hs);
      buf ^= 1;
      for (int j = 0; j < CHUNK; ++j) {
        C[j] = short(int(C[j]) + int(hs[j]) - int(F[0][j]));
        for (int q = 0; q < kFifo - 1; ++q) F[q][j] = F[q + 1][j];
        F[kFifo - 1][j] = hs[j];
      }
    }
    min_prev = simd_min(sgm_step(C, L, Ln, min_prev, lane, p));
    device ushort* out = P + (uint(s * p.Rs + r) * p.width1 + x) * p.D;  // C plane; path cost plane follows
    for (int j = 0; j < CHUNK; ++j) {
      L[j] = Ln[j];
      if (active && d0 + 32 * j < p.D) {
        out[d0 + 32 * j] = ushort(C[j]);
        out[p.plane + d0 + 32 * j] = ushort(L[j]);
      }
    }
  }
  if (active)
    for (int j = 0; j < CHUNK; ++j) {
      const int d = d0 + 32 * j;
      if (d >= p.D) continue;
      ts[d] = C[j];
      ts[p.D + d] = L[j];
      for (int k = 0; k < kFifo; ++k) ts[(2 + k) * p.D + d] = F[k][j];
    }
}

// OUT, D2, D2C: [H][W] disparity (16x fixed point), and the pseudo right
// disparity and its cost for the left-right check.
kernel void horiz_pass(device ushort* P [[buffer(0)]], device short* OUT [[buffer(1)]],
                       device short* D2 [[buffer(2)]], device short* D2C [[buffer(3)]],
                       constant Params& p [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint sgs [[simdgroups_per_threadgroup]],
                       uint2 tg [[threadgroup_position_in_grid]]) {
  const int r = tg.x * sgs + sg, s = tg.y;
  const int r0 = p.band * p.Rs;
  if (r >= p.Rs || r0 + r >= p.rows[s]) return;
  // OpenCV assembles output row y of stripe s > 0 from the stripe's local
  // row overlap + (y - s * stripe_sz). That is image row y itself, except
  // on images under ~16 rows, where the stripe start is clamped to row 0 and
  // OpenCV outputs a shifted (or never computed, so invalid) row instead.
  const int yp = p.src[s] + r0 + r;  // image row processed here
  const int y = s == 0 ? yp : yp - p.src[s] - p.overlap + s * p.stripe_sz;
  if (y < s * p.stripe_sz || y >= min((s + 1) * p.stripe_sz, p.H)) return;  // warm-up row
  const int W = p.W, D = p.D;
  constexpr short kInvalid = -16;  // (minDisparity - 1) * 16
  device short* out = OUT + y * W;  // already kInvalid (bt_prep)
  device short* d2 = D2 + y * W;
  device short* d2c = D2C + y * W;
  for (int x = lane; x < W; x += 32) {
    d2[x] = kInvalid;
    d2c[x] = kInf;
  }
  simdgroup_barrier(mem_flags::mem_device);

  device ushort* row = P + uint(s * p.Rs + r) * p.width1 * D;
  const int d0 = int(lane);  // this lane's disparities: d0 + 32 * j
  short C[CHUNK], T[CHUNK], Lp[CHUNK], Ln[CHUNK];
  auto load = [&](int x) {
    const device ushort* e = row + uint(x) * D + d0;
    for (int j = 0; j < CHUNK; ++j) {
      const bool ok = d0 + 32 * j < D;
      C[j] = ok ? short(e[32 * j]) : short(0);
      T[j] = ok ? short(e[p.plane + 32 * j]) : short(0);
    }
  };

  // Left to right: replace the top path cost by top + left.
  int min_prev = 0;
  for (int j = 0; j < CHUNK; ++j) Lp[j] = 0;
  for (int x = 0; x < p.width1; ++x) {
    load(x);
    min_prev = simd_min(sgm_step(C, Lp, Ln, min_prev, lane, p));
    device ushort* e = row + p.plane + uint(x) * D + d0;
    for (int j = 0; j < CHUNK; ++j) {
      Lp[j] = Ln[j];
      if (d0 + 32 * j < D) e[32 * j] = ushort(T[j] + Ln[j]);
    }
  }

  // Right to left: total cost, best disparity, uniqueness, subpixel fit.
  min_prev = 0;
  for (int j = 0; j < CHUNK; ++j) Lp[j] = 0;
  for (int x = p.width1 - 1; x >= 0; --x) {
    load(x);
    min_prev = simd_min(sgm_step(C, Lp, Ln, min_prev, lane, p));
    int lane_min = kInf;
    for (int j = 0; j < CHUNK; ++j) {
      Lp[j] = Ln[j];
      if (d0 + 32 * j >= D) continue;
      T[j] = short(T[j] + Ln[j]);  // total cost; at most ~14000, no saturation
      lane_min = min(lane_min, int(T[j]));
    }
    const int min_cost = simd_min(lane_min);
    const int best = best_disparity(T, min_cost, d0, D);
    if (p.uniqueness > 0) {
      // OpenCV's SIMD test (what runs on NEON/SSE): ambiguous when another
      // disparity costs at most (100 * min_cost) / (100 - uniqueness),
      // rounded down. Its scalar fallback uses a strict '<' instead.
      const int thresh = (100 * min_cost) / (100 - p.uniqueness);
      bool ambiguous = false;
      for (int j = 0; j < CHUNK; ++j) {
        const int d = d0 + 32 * j;
        if (d < D && abs(d - best) > 1 && int(T[j]) <= thresh) ambiguous = true;
      }
      if (simd_any(ambiguous)) continue;
    }
    uint nb = 0;
    for (int j = 0; j < CHUNK; ++j) {
      const int d = d0 + 32 * j;
      if (d == best - 1) nb |= uint(ushort(T[j]));
      if (d == best + 1) nb |= uint(ushort(T[j])) << 16;
    }
    nb = simd_or(nb);
    if (lane == 0) {
      const int xc = x + D;
      const int x2 = xc - best;
      if (x2 >= 0 && x2 < W && int(d2c[x2]) > min_cost) {
        d2c[x2] = short(min_cost);
        d2[x2] = short(best);
      }
      int disp = best * 16;
      if (0 < best && best < D - 1) {
        const int sm = int(nb & 0xffffu), sp = int(nb >> 16);
        const int denom2 = max(sm + sp - 2 * min_cost, 1);
        disp += ((sm - sp) * 16 + denom2) / (denom2 * 2);
      }
      out[xc] = short(disp);
    }
  }
  simdgroup_barrier(mem_flags::mem_device);

  // Pseudo left-right check against the right disparities gathered above.
  for (int x = D + int(lane); x < W; x += 32) {
    const int d1 = out[x];
    if (d1 == kInvalid) continue;
    const int dl = d1 >> 4, dh = (d1 + 15) >> 4;
    const int xl = x - dl, xh = x - dh;
    if (0 <= xl && xl < W && d2[xl] >= 0 && abs(d2[xl] - dl) > p.disp12_max_diff && 0 <= xh && xh < W &&
        d2[xh] >= 0 && abs(d2[xh] - dh) > p.disp12_max_diff)
      out[x] = kInvalid;
  }
}
