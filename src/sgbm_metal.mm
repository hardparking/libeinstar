// Metal host side of the GPU SGBM (see sgbm.metal).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "sgbm.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#include "sgbm_metal_source.h"  // generated: kSgbmMetalSource

namespace einstar::detail {
namespace {

constexpr int kStripes = 4;
constexpr int kTileColumns = 8;  // as in sgbm.metal
// GPU memory for the per-band cost volumes; sets the band height.
// (At scale 0.5, 128 rows in flight already saturate an M5 Pro; more did not help.)
constexpr size_t kBandBudget = size_t(256) << 20;

// Must match struct Params in sgbm.metal.
struct Params {
  int W, H, D, width1;
  int SW2, SH2, P1, P2;
  int uniqueness, disp12_max_diff;
  int stripe_sz, Rs, band;
  unsigned plane;
  int overlap, pad0;
  int src[kStripes], rows[kStripes];
};

struct Pipelines {
  id<MTLComputePipelineState> prep, top, horiz;
};

class Context {
 public:
  static Context* get() {
    static Context* ctx = [] {
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      return dev ? new Context(dev) : nullptr;
    }();
    return ctx;
  }

  cv::Mat run(const cv::Mat& left, const cv::Mat& right, const SgbmConfig& c) {
    std::lock_guard<std::mutex> lock(mu_);
    @autoreleasepool {
      return run_locked(left, right, c);
    }
  }

 private:
  explicit Context(id<MTLDevice> dev) : dev_(dev), queue_([dev newCommandQueue]) {}

  const Pipelines& pipelines(int chunk, int half) {
    const auto key = std::make_pair(chunk, half);
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second;
    MTLCompileOptions* opt = [MTLCompileOptions new];
    opt.preprocessorMacros = @{@"CHUNK" : @(chunk), @"HALF" : @(half)};
    NSError* err = nil;
    id<MTLLibrary> lib = [dev_ newLibraryWithSource:@(kSgbmMetalSource) options:opt error:&err];
    if (!lib) throw std::runtime_error("einstar Metal SGBM: shader build failed: " + describe(err));
    auto pso = [&](const char* name) {
      id<MTLFunction> f = [lib newFunctionWithName:@(name)];
      NSError* e = nil;
      id<MTLComputePipelineState> p = f ? [dev_ newComputePipelineStateWithFunction:f error:&e] : nil;
      if (!p) throw std::runtime_error(std::string("einstar Metal SGBM: pipeline ") + name + ": " + describe(e));
      return p;
    };
    return pipelines_[key] = Pipelines{pso("bt_prep"), pso("top_pass"), pso("horiz_pass")};
  }

  static std::string describe(NSError* e) { return e ? e.localizedDescription.UTF8String : "unknown error"; }

  // Grow-only shared buffer.
  id<MTLBuffer> buffer(const char* name, size_t bytes) {
    __strong id<MTLBuffer>& b = buffers_[name];
    if (!b || b.length < bytes) {
      b = [dev_ newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
      if (!b) throw std::runtime_error(std::string("einstar Metal SGBM: cannot allocate ") + name);
    }
    return b;
  }

  cv::Mat run_locked(const cv::Mat& left, const cv::Mat& right, const SgbmConfig& c) {
    CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1 && left.size() == right.size());
    if (!sgbm_metal_supports(c)) throw std::invalid_argument("einstar Metal SGBM: unsupported configuration");
    Params p{};
    p.W = left.cols;
    p.H = left.rows;
    p.D = c.num_disparities;
    p.width1 = p.W - p.D;
    if (p.width1 <= 0) throw std::invalid_argument("einstar Metal SGBM: image narrower than the disparity range");
    const int chunk = (p.D + 31) / 32;  // <= kMaxChunk (sgbm_metal_supports)
    p.SW2 = p.SH2 = c.block_size / 2;
    p.P1 = c.p1 > 0 ? c.p1 : 2;
    p.P2 = std::max(c.p2 > 0 ? c.p2 : 5, p.P1 + 1);
    p.uniqueness = c.uniqueness_ratio >= 0 ? c.uniqueness_ratio : 10;
    p.disp12_max_diff = 1;

    // OpenCV's fixed stripes (computeDisparity3WaySGBM<4>).
    p.stripe_sz = (p.H + kStripes - 1) / kStripes;
    p.overlap = p.SH2 + 1 + (p.stripe_sz + 9) / 10;  // (SADWindowSize/2+1) + ceil(0.1*stripe_sz)
    int max_rows = 0;
    for (int s = 0; s < kStripes; ++s) {
      const int start = std::max(std::min(s * p.stripe_sz - p.overlap, p.H), 0);
      const int end = std::min((s + 1) * p.stripe_sz, p.H);
      p.src[s] = start;
      p.rows[s] = std::max(0, end - start);
      max_rows = std::max(max_rows, p.rows[s]);
    }
    const size_t col_bytes = size_t(p.width1) * p.D;
    const size_t per_band_row = kStripes * col_bytes * 4;
    p.Rs = static_cast<int>(std::clamp<size_t>(kBandBudget / per_band_row, 4, 64));
    p.Rs = std::min(p.Rs, max_rows);
    const int bands = (max_rows + p.Rs - 1) / p.Rs;
    p.plane = static_cast<unsigned>(kStripes * size_t(p.Rs) * col_bytes);

    const size_t img = size_t(p.W) * p.H;
    id<MTLBuffer> L = buffer("L", img), R = buffer("R", img);
    id<MTLBuffer> PL = buffer("PL", img * 8), PR = buffer("PR", img * 8);
    id<MTLBuffer> P = buffer("P", kStripes * size_t(p.Rs) * col_bytes * 4);
    id<MTLBuffer> TS = buffer("TS", kStripes * col_bytes * 2 * (2 + 2 * p.SH2 + 1));
    id<MTLBuffer> OUT = buffer("OUT", img * 2), D2 = buffer("D2", img * 2), D2C = buffer("D2C", img * 2);
    if (kStripes * size_t(p.Rs) * col_bytes >= (size_t(1) << 31))
      throw std::invalid_argument("einstar Metal SGBM: image too large");
    for (int y = 0; y < p.H; ++y) {
      std::memcpy(static_cast<uint8_t*>(L.contents) + size_t(y) * p.W, left.ptr(y), p.W);
      std::memcpy(static_cast<uint8_t*>(R.contents) + size_t(y) * p.W, right.ptr(y), p.W);
    }

    const Pipelines& k = pipelines(chunk, p.SW2);
    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];  // serial: dispatches run in order
    [enc setComputePipelineState:k.prep];
    [enc setBuffer:L offset:0 atIndex:0];
    [enc setBuffer:R offset:0 atIndex:1];
    [enc setBuffer:PL offset:0 atIndex:2];
    [enc setBuffer:PR offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof p atIndex:4];
    [enc setBuffer:OUT offset:0 atIndex:5];
    [enc dispatchThreads:MTLSizeMake(p.W, p.H, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    for (int b = 0; b < bands; ++b) {
      p.band = b;
      [enc setComputePipelineState:k.top];
      [enc setBuffer:PL offset:0 atIndex:0];
      [enc setBuffer:PR offset:0 atIndex:1];
      [enc setBuffer:P offset:0 atIndex:2];
      [enc setBuffer:TS offset:0 atIndex:3];
      [enc setBytes:&p length:sizeof p atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake((p.width1 + kTileColumns - 1) / kTileColumns, kStripes, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * kTileColumns, 1, 1)];
      const NSUInteger sgs = 4;  // rows per threadgroup
      [enc setComputePipelineState:k.horiz];
      [enc setBuffer:P offset:0 atIndex:0];
      [enc setBuffer:OUT offset:0 atIndex:1];
      [enc setBuffer:D2 offset:0 atIndex:2];
      [enc setBuffer:D2C offset:0 atIndex:3];
      [enc setBytes:&p length:sizeof p atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake((p.Rs + sgs - 1) / sgs, kStripes, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * sgs, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted)
      throw std::runtime_error("einstar Metal SGBM: GPU error: " + describe(cb.error));
    cv::Mat out(p.H, p.W, CV_16S);
    std::memcpy(out.data, OUT.contents, img * 2);
    return out;
  }

  std::mutex mu_;
  id<MTLDevice> dev_;
  id<MTLCommandQueue> queue_;
  std::map<std::pair<int, int>, Pipelines> pipelines_;  // by (CHUNK, HALF)
  std::map<std::string, id<MTLBuffer>> buffers_;
};

}  // namespace

bool sgbm_metal_available() { return Context::get() != nullptr; }

cv::Mat sgbm_metal_raw(const cv::Mat& left, const cv::Mat& right, const SgbmConfig& c) {
  Context* ctx = Context::get();
  if (!ctx) throw std::runtime_error("einstar Metal SGBM: no Metal device");
  return ctx->run(left, right, c);
}

cv::Mat sgbm_metal(const cv::Mat& left, const cv::Mat& right, const SgbmConfig& c) {
  cv::Mat d = sgbm_metal_raw(left, right, c);
  sgbm_postprocess(d, c);
  return d;
}

}  // namespace einstar::detail
