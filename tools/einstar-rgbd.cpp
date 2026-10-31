// Writes RGB-D frames for reconstruction tools such as tools/einstar_fusion.py.
//
//   einstar-rgbd OUT_DIR [--calib FILE] [--seconds N]        live from the scanner
//   einstar-rgbd OUT_DIR --calib FILE --replay FRAME_DIR     einstar-probe / EINSTAR_RECORD_DIR captures
//
// Each frame set becomes OUT_DIR/c_NNNNN.png (colour, BGR) and
// OUT_DIR/d_NNNNN.png (depth, uint16 in 0.1 mm, 0 = none), both at half
// resolution (640x512) in the undistorted colour camera, upright.
// OUT_DIR/intrinsics.txt holds "fx fy cx cy width height" for those images
// and is written first; OUT_DIR/done is written last. Live frames appear
// while scanning (the depth image is renamed into place last), so a reader
// can follow the directory. Live mode reads the calibration from the scanner
// unless --calib is given, keeps the IR exposure steady with AutoExposure and
// runs until Ctrl-C or --seconds.
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <string>

#include "einstar/calibration.hpp"
#include "einstar/depth.hpp"
#include "einstar/device.hpp"
#include "einstar/exposure.hpp"

using namespace einstar;
namespace fs = std::filesystem;

namespace {

constexpr int kWidth = 1280, kHeight = 1024;
std::atomic<bool> g_stop{false};

// IR pair (indexed by stream camera) and the latest raw Bayer colour frame.
struct FrameSet {
  cv::Mat ir[2];
  cv::Mat color;
};

class Writer {
 public:
  Writer(fs::path dir, const StereoDepth& depth) : dir_(std::move(dir)), depth_(depth) {
    // Output is upright: the calibration frame is upside down relative to
    // the scanner's natural orientation, so the principal point mirrors.
    const cv::Mat& K = depth.color_K();
    const double cx = kWidth - 1 - K.at<double>(0, 2), cy = kHeight - 1 - K.at<double>(1, 2);
    std::ofstream f(dir_ / "intrinsics.txt");
    f.precision(9);
    f << K.at<double>(0, 0) / 2 << ' ' << K.at<double>(1, 1) / 2 << ' ' << (cx + 0.5) / 2 - 0.5 << ' '
      << (cy + 0.5) / 2 - 0.5 << ' ' << kWidth / 2 << ' ' << kHeight / 2 << '\n';
  }

  // Returns the share of the frame with depth.
  double write(const FrameSet& s) {
    cv::Mat left, right;
    for (unsigned stream = 0; stream < 2; ++stream)
      (calibration_index_for_stream(stream) == kIrLeftCamera ? left : right) =
          to_calibration_orientation(s.ir[stream], stream);
    const cv::Mat depth_mm = depth_.depth_in_color(left, right);
    cv::Mat d16, d, rgb, c;
    depth_mm.convertTo(d16, CV_16U, 10.0);
    cv::resize(d16, d, {kWidth / 2, kHeight / 2}, 0, 0, cv::INTER_NEAREST);
    cv::rotate(d, d, cv::ROTATE_180);
    rgb = depth_.undistort_color(color_in_calibration_orientation(s.color));
    cv::resize(rgb, c, {kWidth / 2, kHeight / 2}, 0, 0, cv::INTER_AREA);
    cv::rotate(c, c, cv::ROTATE_180);
    char name[32];
    std::snprintf(name, sizeof name, "c_%05d.png", n_);
    cv::imwrite((dir_ / name).string(), c);
    std::snprintf(name, sizeof name, "d_%05d.png", n_);
    const fs::path tmp = dir_ / ".d_tmp.png";
    cv::imwrite(tmp.string(), d);
    fs::rename(tmp, dir_ / name);
    ++n_;
    return static_cast<double>(cv::countNonZero(depth_mm)) / static_cast<double>(depth_mm.total());
  }

  int count() const { return n_; }
  void finish() { std::ofstream(dir_ / "done") << n_ << '\n'; }

 private:
  fs::path dir_;
  const StereoDepth& depth_;
  int n_ = 0;
};

int replay(Writer& writer, const std::string& dir) {
  std::set<int> ids;
  for (const auto& e : fs::directory_iterator(dir)) {
    int id, cam;
    if (std::sscanf(e.path().filename().c_str(), "frame_%d_cam%d.pgm", &id, &cam) == 2) ids.insert(id);
  }
  cv::Mat last_color;
  for (int id : ids) {
    if (g_stop) break;
    FrameSet s;
    for (int cam = 0; cam < 3; ++cam) {
      char name[64];
      std::snprintf(name, sizeof name, "/frame_%06d_cam%d.pgm", id, cam);
      cv::Mat m = cv::imread(dir + name, cv::IMREAD_GRAYSCALE);
      if (cam < 2) s.ir[cam] = m;
      else if (!m.empty()) last_color = m;
    }
    s.color = last_color;
    if (s.ir[0].empty() || s.ir[1].empty() || s.color.empty()) continue;
    writer.write(s);
  }
  return 0;
}

int live(Device& dev, Writer& writer, int seconds) {
  std::mutex m;
  std::condition_variable ready_cv;
  std::map<std::uint32_t, FrameSet> pending;
  std::deque<FrameSet> ready;
  cv::Mat last_color;
  int discard = 2;  // the first sets after the trigger starts are overexposed (as in CameraEinstar)
  auto s = dev.start_stream([&](Frame&& f) {
    const auto cam = f.info.camera_index();
    if (!cam) return;
    cv::Mat img = cv::Mat(kHeight, kWidth, CV_8UC1, f.data.data()).clone();
    std::lock_guard<std::mutex> lock(m);
    if (*cam == 2) {
      last_color = img;
      return;
    }
    FrameSet& g = pending[f.info.frame_id];
    g.ir[*cam] = img;
    if (!g.ir[0].empty() && !g.ir[1].empty()) {
      g.color = last_color;
      if (discard > 0) --discard;
      else if (!g.color.empty()) {
        ready.push_back(std::move(g));
        if (ready.size() > 2) ready.pop_front();  // keep latency low
        ready_cv.notify_one();
      }
      pending.erase(f.info.frame_id);
    }
    while (pending.size() > 8) pending.erase(pending.begin());
  }, static_cast<std::size_t>(kWidth) * kHeight);
  if (!s) {
    std::fprintf(stderr, "stream failed: %s\n", s.error().message.c_str());
    return 1;
  }
  if (auto st = dev.apply(ScanSettings{}); !st) {
    std::fprintf(stderr, "scan settings failed: %s\n", st.error().message.c_str());
    dev.stop_stream();
    return 1;
  }
  if (auto st = dev.start_scanning(); !st) {
    std::fprintf(stderr, "start scanning failed: %s\n", st.error().message.c_str());
    dev.stop_stream();
    return 1;
  }
  const AutoExposure ae;
  std::uint32_t exposure = ScanSettings{}.ir_exposure_us;
  const auto start = std::chrono::steady_clock::now();
  while (!g_stop && (seconds <= 0 || std::chrono::steady_clock::now() - start < std::chrono::seconds(seconds))) {
    FrameSet set;
    {
      std::unique_lock<std::mutex> lock(m);
      if (!ready_cv.wait_for(lock, std::chrono::milliseconds(500), [&] { return !ready.empty(); })) continue;
      set = std::move(ready.front());
      ready.pop_front();
    }
    if (auto next = ae.update(ByteView(set.ir[0].data, set.ir[0].total()), ByteView(set.ir[1].data, set.ir[1].total()),
                              exposure)) {
      if (dev.set_exposure_us(0, *next) && dev.set_exposure_us(1, *next)) exposure = *next;
    }
    const double coverage = writer.write(set);
    if (writer.count() % 20 == 0)
      std::printf("frames %d, depth %.0f%%, IR exposure %u us\n", writer.count(), coverage * 100, exposure);
  }
  dev.stop_scanning();
  dev.lights_off();
  dev.stop_stream();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string out, calib_path, replay_dir;
  int seconds = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--calib" && i + 1 < argc) calib_path = argv[++i];
    else if (a == "--replay" && i + 1 < argc) replay_dir = argv[++i];
    else if (a == "--seconds" && i + 1 < argc) seconds = std::atoi(argv[++i]);
    else if (out.empty() && a[0] != '-') out = a;
    else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return 2;
    }
  }
  if (out.empty() || (!replay_dir.empty() && calib_path.empty())) {
    std::fprintf(stderr, "usage: %s OUT_DIR [--calib FILE] [--seconds N] | OUT_DIR --calib FILE --replay FRAME_DIR\n",
                 argv[0]);
    return 2;
  }
  std::signal(SIGINT, [](int) { g_stop = true; });
  std::signal(SIGTERM, [](int) { g_stop = true; });

  std::unique_ptr<Device> dev;
  if (replay_dir.empty()) {
    OpenOptions options;
    options.allow_writes = true;  // scan trigger and lighting (volatile settings only)
    auto opened = Device::open(std::nullopt, options);
    if (!opened) {
      std::fprintf(stderr, "open failed: %s\n", opened.error().message.c_str());
      return 1;
    }
    dev = std::move(opened.value());
  }
  std::optional<CameraModel> model;
  if (calib_path.empty()) {
    auto from_device = dev->read_calibration();
    if (!from_device) {
      std::fprintf(stderr, "cannot read calibration from the scanner: %s\n", from_device.error().message.c_str());
      return 1;
    }
    model = from_device.value();
  } else {
    auto loaded = load_calibration(calib_path);
    if (!loaded.model) {
      std::fprintf(stderr, "cannot load calibration %s: %s\n", calib_path.c_str(), to_string(*loaded.error));
      return 1;
    }
    model = loaded.model;
  }
  std::error_code ec;
  fs::create_directories(out, ec);
  for (const auto& e : fs::directory_iterator(out)) {
    const std::string n = e.path().filename().string();
    if (n == "done" || ((n.rfind("c_", 0) == 0 || n.rfind("d_", 0) == 0) && e.path().extension() == ".png")) {
      std::fprintf(stderr, "%s already holds frames; use an empty directory\n", out.c_str());
      return 1;
    }
  }
  const StereoDepth depth(*model, {kWidth, kHeight});
  Writer writer(out, depth);
  const int rc = dev ? live(*dev, writer, seconds) : replay(writer, replay_dir);
  writer.finish();
  std::printf("wrote %d frames to %s\n", writer.count(), out.c_str());
  return rc;
}
