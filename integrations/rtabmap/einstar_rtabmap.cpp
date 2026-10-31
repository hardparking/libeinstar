// Scans with an Einstar (or a replay directory) into an RTAB-Map database.
//
//   einstar-rtabmap [--calib PATH] [--replay DIR] [--db out.db] [--frames N]
//                   [--poses odom.txt] [--icp] [--scan-max M] [--param Key=Value ...]
//
// --icp adds a depth-derived point cloud to each frame and registers with
// visual features plus point-to-plane ICP. --scan-max limits that cloud to
// M metres (default 0.8), e.g. to the scanning volume on a turntable so the
// static surroundings do not hold ICP still.
//
// Open the result with `rtabmap-databaseViewer` or export a mesh with
// `rtabmap-export --mesh --texture out.db`.
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/core/util3d_filtering.h>
#include <rtabmap/core/util3d_surface.h>
#include <pcl/common/io.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UStl.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <rtabmap/core/camera/CameraEinstar.h>

using namespace rtabmap;

namespace {
volatile std::sig_atomic_t g_stop = 0;
void on_sigint(int) { g_stop = 1; }

// Builds the point cloud ICP needs from the registered depth image.
void add_scan_from_depth(SensorData& data, float maxRange) {
  std::vector<int> valid;
  auto cloud = util3d::cloudFromSensorData(data, 4, maxRange, 0.15f, &valid);
  pcl::IndicesPtr indices(new std::vector<int>(std::move(valid)));
  cloud = util3d::voxelize(cloud, indices, 0.002f);
  if (cloud->size() < 100) return;
  auto normals = util3d::computeNormals(cloud, 20, 0.0f);
  pcl::PointCloud<pcl::PointNormal> withNormals;
  pcl::concatenateFields(*cloud, *normals, withNormals);
  data.setLaserScan(util3d::laserScanFromPointCloud(withNormals), false);
}
}  // namespace

int main(int argc, char** argv) {
  std::string calib, replay, db = "einstar.db", posesPath;
  int maxFrames = 0;
  bool icp = false;
  float scanMax = 0.8f;
  std::vector<std::pair<std::string, std::string>> overrides;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--calib" && i + 1 < argc) calib = argv[++i];
    else if (a == "--replay" && i + 1 < argc) replay = argv[++i];
    else if (a == "--db" && i + 1 < argc) db = argv[++i];
    else if (a == "--frames" && i + 1 < argc) maxFrames = std::atoi(argv[++i]);
    else if (a == "--poses" && i + 1 < argc) posesPath = argv[++i];
    else if (a == "--icp") icp = true;
    else if (a == "--scan-max" && i + 1 < argc) scanMax = static_cast<float>(std::atof(argv[++i]));
    else if (a == "--param" && i + 1 < argc) {
      const std::string kv = argv[++i];
      const auto eq = kv.find('=');
      if (eq == std::string::npos) {
        std::fprintf(stderr, "--param expects Key=Value\n");
        return 2;
      }
      overrides.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
    }
    else {
      std::fprintf(stderr, "usage: %s [--calib PATH] [--replay DIR] [--db out.db] [--frames N] [--poses odom.txt] [--icp] [--scan-max M] [--param Key=Value]\n", argv[0]);
      return 2;
    }
  }
  if (calib.empty() && !replay.empty()) {
    std::fprintf(stderr, "--calib is required with --replay\n");
    return 2;
  }
  ULogger::setType(ULogger::kTypeConsole);
  ULogger::setLevel(ULogger::kWarning);
  std::signal(SIGINT, on_sigint);

  CameraEinstar camera(calib, replay);
  if (!camera.init()) return 1;

  // Close-range handheld scanning: tighten depth limits and use F2M odometry
  // with geometry+visual registration.
  ParametersMap params;
  params.insert(ParametersPair(Parameters::kRGBDLinearUpdate(), "0.01"));
  params.insert(ParametersPair(Parameters::kRGBDAngularUpdate(), "0.05"));
  params.insert(ParametersPair(Parameters::kRegStrategy(), icp ? "2" : "0"));  // visual (+ ICP)
  // Frame-to-frame odometry with optical-flow correspondences tracked a real
  // handheld capture far better than frame-to-map with descriptors (72 vs ~10
  // of 200 frames); re-initialise after a loss so one bad frame does not end
  // the scan.
  params.insert(ParametersPair(Parameters::kOdomStrategy(), "1"));
  params.insert(ParametersPair(Parameters::kVisCorType(), "1"));
  params.insert(ParametersPair(Parameters::kOdomResetCountdown(), "1"));
  params.insert(ParametersPair(Parameters::kVisMaxDepth(), "0.8"));
  params.insert(ParametersPair(Parameters::kIcpVoxelSize(), "0.002"));
  params.insert(ParametersPair(Parameters::kIcpMaxCorrespondenceDistance(), "0.01"));
  params.insert(ParametersPair(Parameters::kIcpPointToPlane(), "true"));
  params.insert(ParametersPair(Parameters::kGridRangeMax(), "0.8"));

  for (const auto& [key, value] : overrides) uInsert(params, ParametersPair(key, value));

  std::unique_ptr<Odometry> odom(Odometry::create(params));
  Rtabmap rtabmap;
  rtabmap.init(params, db);

  std::ofstream poses;
  if (!posesPath.empty()) poses.open(posesPath);
  int processed = 0, lost = 0;
  while (!g_stop && (maxFrames <= 0 || processed < maxFrames)) {
    SensorData data = camera.takeImage();
    if (!data.isValid()) {
      if (!replay.empty()) break;  // end of replay
      continue;
    }
    if (icp) add_scan_from_depth(data, scanMax);
    if (processed == 0 && std::getenv("EINSTAR_DEBUG"))
      std::printf("camera local transform: %s\n", data.cameraModels()[0].localTransform().prettyPrint().c_str());
    OdometryInfo info;
    Transform pose = odom->process(data, &info);
    if (std::getenv("EINSTAR_DEBUG"))
      std::printf("frame %d: %s inliers %d matches %d features %d icp %.3f\n", processed,
                  pose.isNull() ? "LOST" : "ok  ", info.reg.inliers, info.reg.matches, info.features,
                  info.reg.icpInliersRatio);
    if (pose.isNull()) {
      ++lost;
    } else {
      rtabmap.process(data, pose);
      if (poses.is_open()) {
        // Odometry pose of the colour camera's base frame, in metres.
        const Eigen::Matrix4f M = pose.toEigen4f();
        poses << processed;
        for (int c = 0; c < 3; ++c) poses << ' ' << M(c, 3);
        for (int rr = 0; rr < 3; ++rr)
          for (int c = 0; c < 3; ++c) poses << ' ' << M(rr, c);
        poses << '\n';
      }
    }
    ++processed;
    if (processed % 10 == 0)
      std::printf("frames %d, odometry lost %d, map nodes %zu\n", processed, lost,
                  rtabmap.getLocalOptimizedPoses().size());
  }
  rtabmap.close();
  std::printf("done: %d frames (%d lost) -> %s\n", processed, lost, db.c_str());
  return 0;
}
