#!/usr/bin/env python3
"""Fuses einstar-rgbd frames into a mesh, KinectFusion-style.

    einstar_fusion.py FRAME_DIR [--follow] [--out PREFIX] [--max-depth-mm 300] [--voxel-mm 1.5]

Each frame is aligned by point-to-plane ICP, then fused into a TSDF volume.
By default (--track keyframe) it is aligned to the last keyframe, an
unsmoothed earlier frame, and a frame that overlaps the keyframe less than
--keyframe-fitness becomes the next keyframe. Aligning to a raycast of the
fused model instead (--track model, KinectFusion-style) slides sideways
on smooth surfaces, because the TSDF smooths away the fine geometry: a
still scanner drifted 27 mm in 8 s that way and about 1 mm frame to frame.
EXStar adds texture and markers against that. In keyframe mode the model
raycast is the fallback when the keyframe does not align. At the end,
keyframes that see the same surface are aligned directly and a pose graph
spreads the accumulated error over all keyframes, then the frames are fused
again at the corrected poses (--no-loop-closure skips this). Only depth closer
than --max-depth-mm is used, so a still background does not hold the
tracking when the object turns on a turntable. Frames that fail to align are
not fused.

With --follow the script processes frames as `einstar-rgbd FRAME_DIR` writes
them and shows a preview: the colour view with the distance to the surface
and the tracking state, next to the model so far seen from the scanner.
Press q in the preview to finish. The mesh is saved to PREFIX_mesh.ply
(default FRAME_DIR/scan_mesh.ply) and the poses (4x4, camera to model) to
PREFIX_poses.txt.

Needs numpy, opencv-python and open3d (tested with Open3D 0.20 on the CPU;
Open3D's own frame-to-model tracker needs CUDA, so ICP is done here).
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c

DEPTH_SCALE = 10000.0  # einstar-rgbd depth is in 0.1 mm
DEPTH_MIN = 0.13
MIN_FITNESS = 0.3      # share of frame points with a model point within --max-corr-mm


def frame_paths(frame_dir, start):
    """Depth images from index `start` on, in order, that are complete."""
    out = []
    i = start
    while os.path.exists(os.path.join(frame_dir, f'd_{i:05d}.png')):
        out.append(os.path.join(frame_dir, f'd_{i:05d}.png'))
        i += 1
    return out


def shade(depth, fx, fy, cx, cy):
    """Grey shading of a raycast depth image (headlight from the camera)."""
    z = depth.astype(np.float32)
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    p = np.dstack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    n = np.cross(np.gradient(p, axis=1), np.gradient(p, axis=0))
    norm = np.linalg.norm(n, axis=2)
    ok = (z > 0) & (norm > 0)
    lit = np.zeros((h, w), np.uint8)
    view = p / np.maximum(np.linalg.norm(p, axis=2, keepdims=True), 1e-9)
    lit[ok] = (np.abs((n[ok] / norm[ok, None] * view[ok]).sum(1)) * 220 + 25).astype(np.uint8)
    return cv2.cvtColor(lit, cv2.COLOR_GRAY2BGR)


def overlay(color, depth_m, max_depth, status, n_frames, n_lost):
    """Colour view with the distance to the surface in the centre of the view and the tracking state."""
    img = color.copy()
    h, w = depth_m.shape
    centre = depth_m[h // 3:2 * h // 3, w // 3:2 * w // 3]
    valid = centre[centre > 0]
    if valid.size < 50:
        dist, tint = 'no depth in the centre', (0, 0, 255)
    else:
        mm = float(np.median(valid)) * 1000
        good = DEPTH_MIN * 1000 + 30 <= mm <= max_depth * 1000 - 30
        dist = f'distance {mm:.0f} mm' + ('' if good else ' (too close)' if mm < 200 else ' (too far)')
        tint = (0, 200, 0) if good else (0, 165, 255)
    cv2.rectangle(img, (w // 3, h // 3), (2 * w // 3, 2 * h // 3), tint, 2)
    for k, (text, col) in enumerate([(dist, tint),
                                     (status, (0, 200, 0) if status == 'tracking' else (0, 0, 255)),
                                     (f'{n_frames} frames, {n_lost} not aligned', (255, 255, 255))]):
        cv2.putText(img, text, (12, 32 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, text, (12, 32 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    return img


def read_markers(depth_path):
    """Markers einstar-rgbd wrote for this frame (metres, camera frame), or an empty array."""
    path = depth_path.replace('/d_', '/m_')[:-4] + '.txt'
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 3))
    return np.loadtxt(path, ndmin=2) / 1000.0


def rigid_fit(src, dst):
    """Least-squares rotation and translation mapping src onto dst (Kabsch), as a 4x4."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    s = np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))])
    r = vt.T @ s @ u.T
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, cd - r @ cs
    return t


class MarkerMap:
    """Retro-reflective markers in model coordinates, built up while scanning.

    A frame is located by matching its markers to the map under a predicted
    pose and fitting the rigid motion; the map is global, so the pose does
    not accumulate drift the way frame-to-frame alignment does."""

    def __init__(self, match_radius=0.004, max_rms=0.001, min_markers=3):
        self.points = np.zeros((0, 3))
        self.counts = np.zeros(0)
        self.match_radius, self.max_rms, self.min_markers = match_radius, max_rms, min_markers

    def _match(self, world, min_count=0):
        if not len(self.points) or not len(world):
            return np.full(len(world), -1)
        d = np.linalg.norm(world[:, None] - self.points[None], axis=2)
        d[:, self.counts < min_count] = np.inf
        j = d.argmin(1)
        j[d[np.arange(len(world)), j] > self.match_radius] = -1
        # one frame marker per map marker: keep the closest
        for k in set(j[j >= 0]):
            rows = np.where(j == k)[0]
            if len(rows) > 1:
                j[rows[d[rows, k] > d[rows, k].min()]] = -1
        return j

    def locate(self, markers, guess):
        """Pose (camera to model) from the markers, or None when they do not fix it."""
        if len(markers) < self.min_markers:
            return None
        pose = guess
        for _ in range(3):
            world = markers @ pose[:3, :3].T + pose[:3, 3]
            # only markers seen in several frames: one-off detections are mostly glints
            j = self._match(world, min_count=3 if self.counts.size and self.counts.max() >= 3 else 0)
            ok = j >= 0
            if ok.sum() < self.min_markers:
                return None
            pose = rigid_fit(markers[ok], self.points[j[ok]])
            res = np.linalg.norm(markers[ok] @ pose[:3, :3].T + pose[:3, 3] - self.points[j[ok]], axis=1)
            if res.max() > 2 * self.max_rms and ok.sum() > self.min_markers:
                bad = np.where(ok)[0][res.argmax()]  # drop the worst match and refit
                ok[bad] = False
                pose = rigid_fit(markers[ok], self.points[j[ok]])
                res = np.linalg.norm(markers[ok] @ pose[:3, :3].T + pose[:3, 3] - self.points[j[ok]], axis=1)
        return pose if np.sqrt((res ** 2).mean()) <= self.max_rms else None

    def add(self, markers, pose):
        """Average matched markers into the map and add new ones."""
        world = markers @ pose[:3, :3].T + pose[:3, 3]
        j = self._match(world)
        for w, k in zip(world, j):
            if k >= 0:
                self.counts[k] += 1
                self.points[k] += (w - self.points[k]) / self.counts[k]
            elif not len(self.points) or np.linalg.norm(self.points - w, axis=1).min() > 1.5 * self.match_radius:
                self.points = np.vstack([self.points, w])
                self.counts = np.append(self.counts, 1)


def close_loops(clouds, poses, max_corr, max_angle_deg=60, max_shift=0.15, min_fitness=0.4):
    """Pose-graph optimisation over keyframes (poses: camera to model).

    Consecutive keyframes are linked by the tracked motion. Every other pair
    whose poses suggest overlapping views is aligned directly; a good
    alignment adds an uncertain edge. Returns the optimised poses and the
    number of accepted edges between non-consecutive keyframes."""
    legacy = o3d.pipelines.registration
    reg = o3d.t.pipelines.registration
    pts = [c.voxel_down_sample(0.003) for c in clouds]
    for c in pts:
        c.estimate_normals(max_nn=20, radius=0.009)
    lpts = [c.to_legacy() for c in pts]

    def edge(i, j, transform, uncertain):
        info = legacy.get_information_matrix_from_point_clouds(lpts[i], lpts[j], max_corr, transform)
        return legacy.PoseGraphEdge(i, j, transform, info, uncertain=uncertain)

    graph = legacy.PoseGraph()
    for p in poses:
        graph.nodes.append(legacy.PoseGraphNode(p))
    for i in range(len(poses) - 1):
        graph.edges.append(edge(i, i + 1, np.linalg.inv(poses[i + 1]) @ poses[i], False))
    loops = 0
    for i in range(len(poses)):
        for j in range(i + 2, len(poses)):
            guess = np.linalg.inv(poses[j]) @ poses[i]  # maps keyframe i into keyframe j
            angle = np.degrees(np.arccos(np.clip((np.trace(guess[:3, :3]) - 1) / 2, -1, 1)))
            if angle > max_angle_deg or np.linalg.norm(guess[:3, 3]) > max_shift:
                continue
            try:
                r = reg.icp(pts[i], pts[j], 2 * max_corr, o3c.Tensor(guess), reg.TransformationEstimationPointToPlane(),
                            reg.ICPConvergenceCriteria(1e-6, 1e-6, 30))
                r = reg.icp(pts[i], pts[j], max_corr, r.transformation, reg.TransformationEstimationPointToPlane(),
                            reg.ICPConvergenceCriteria(1e-6, 1e-6, 30))
            except RuntimeError:
                continue
            if r.fitness >= min_fitness:
                graph.edges.append(edge(i, j, r.transformation.numpy(), True))
                loops += 1
    legacy.global_optimization(graph, legacy.GlobalOptimizationLevenbergMarquardt(),
                               legacy.GlobalOptimizationConvergenceCriteria(),
                               legacy.GlobalOptimizationOption(max_correspondence_distance=max_corr,
                                                               edge_prune_threshold=0.25, reference_node=0))
    return [np.asarray(n.pose) for n in graph.nodes], loops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frame_dir')
    ap.add_argument('--follow', action='store_true', help='process frames while einstar-rgbd writes them, with a preview')
    ap.add_argument('--no-preview', action='store_true')
    ap.add_argument('--out', help='output prefix (default FRAME_DIR/scan)')
    ap.add_argument('--max-depth-mm', type=float, default=300, help='ignore depth beyond this (default 300)')
    ap.add_argument('--voxel-mm', type=float, default=1.5, help='TSDF voxel size (default 1.5)')
    ap.add_argument('--max-corr-mm', type=float, default=5, help='ICP correspondence distance (default 5)')
    ap.add_argument('--track', choices=['keyframe', 'model'], default='keyframe',
                    help='align each frame to the last keyframe (default) or to a raycast of the model')
    ap.add_argument('--keyframe-fitness', type=float, default=0.7,
                    help='start a new keyframe when the overlap with the current one drops below this (default 0.7)')
    ap.add_argument('--no-loop-closure', action='store_true',
                    help='skip the final pose-graph optimisation over keyframes (keyframe tracking only)')
    ap.add_argument('--no-markers', action='store_true',
                    help='ignore the retro-reflective markers einstar-rgbd found (m_NNNNN.txt)')
    args = ap.parse_args()

    frame_dir = args.frame_dir
    out = args.out or os.path.join(frame_dir, 'scan')
    max_depth = args.max_depth_mm / 1000
    preview = args.follow and not args.no_preview
    while not os.path.exists(os.path.join(frame_dir, 'intrinsics.txt')):
        if not args.follow:
            sys.exit(f'{frame_dir}/intrinsics.txt not found (write frames with einstar-rgbd)')
        time.sleep(0.2)
    fx, fy, cx, cy, width, height = np.loadtxt(os.path.join(frame_dir, 'intrinsics.txt'))
    width, height = int(width), int(height)

    dev = o3c.Device('CPU:0')
    K = o3c.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3c.Dtype.Float64)
    I4 = o3c.Tensor(np.eye(4))
    reg = o3d.t.pipelines.registration
    criteria = reg.ICPConvergenceCriteria(1e-6, 1e-6, 15)
    voxel = args.voxel_mm / 1000
    trunc = 4.0  # truncation in voxels

    pose = np.eye(4)
    model = o3d.t.pipelines.slam.Model(voxel, 16, 60000, o3c.Tensor(pose), dev)
    frame = o3d.t.pipelines.slam.Frame(height, width, K, dev)
    ray = o3d.t.pipelines.slam.Frame(height, width, K, dev)

    def cloud(depth_img):
        return o3d.t.geometry.PointCloud.create_from_depth_image(depth_img, K, I4, DEPTH_SCALE, max_depth, 2)

    def align(src, tgt, guess):
        """ICP result, or None when there is too little overlap or ICP fails (singular system)."""
        if len(src.point.positions) < 500 or len(tgt.point.positions) < 500:
            return None
        if 'normals' not in tgt.point:
            tgt.estimate_normals(max_nn=20, radius=0.006)
        try:
            r = reg.icp(src, tgt, args.max_corr_mm / 1000, o3c.Tensor(guess),
                        reg.TransformationEstimationPointToPlane(), criteria)
        except RuntimeError:
            return None
        return r if r.fitness >= MIN_FITNESS else None

    keyframe, keyframe_pose = None, np.eye(4)
    marker_map = MarkerMap()
    prev_pose, n_marker_frames = np.eye(4), 0
    kf_clouds, kf_poses = [], []
    fused_frames = []  # (depth path, keyframe index, pose relative to that keyframe)
    poses, n_lost, fused, index = [], 0, 0, 0
    status = 'starting'
    t0 = time.time()
    quit_requested = False
    while not quit_requested:
        paths = frame_paths(frame_dir, index)
        if not paths:
            if not args.follow or os.path.exists(os.path.join(frame_dir, 'done')):
                break
            if preview and cv2.waitKey(50) & 0xFF == ord('q'):
                break
            continue
        if args.follow and len(paths) > 4:
            index += len(paths) - 1  # fell behind the scanner: jump to the newest frame
            paths = paths[-1:]
        for path in paths:
            index += 1
            depth = o3d.t.io.read_image(path)
            frame.set_data_from_image('depth', depth)
            frame.set_data_from_image('color', o3d.t.io.read_image(path.replace('/d_', '/c_')))
            aligned = fused == 0
            new_keyframe = keyframe is None
            markers = np.zeros((0, 3)) if args.no_markers else read_markers(path)
            by_markers = False
            if fused > 0 and len(markers):
                # constant-velocity guess from the last two poses
                m = marker_map.locate(markers, pose @ np.linalg.inv(prev_pose) @ pose)
                if m is not None:
                    pose, aligned, by_markers = m, True, True
                    n_marker_frames += 1
                    new_keyframe = n_marker_frames % 10 == 0  # keep the ICP fallback's reference fresh
            if fused > 0 and not aligned:
                src = cloud(depth).voxel_down_sample(0.002)
                if args.track == 'keyframe':
                    # guess: the previous pose, relative to the keyframe
                    r = align(src, keyframe, np.linalg.inv(keyframe_pose) @ pose)
                    if r is not None:
                        pose = keyframe_pose @ r.transformation.numpy()
                        aligned = True
                        new_keyframe = r.fitness < args.keyframe_fitness
                if not aligned:
                    r = align(src, cloud(ray.get_data_as_image('depth')), np.eye(4))  # raycast depth is in input units
                    if r is not None:
                        pose = pose @ r.transformation.numpy()
                        aligned = True
                        new_keyframe = args.track == 'keyframe'
            if aligned and new_keyframe:
                keyframe, keyframe_pose = cloud(depth).voxel_down_sample(0.0015), pose.copy()
                kf_clouds.append(keyframe)
                kf_poses.append(keyframe_pose)
            if aligned:
                model.update_frame_pose(fused, o3c.Tensor(pose))
                model.integrate(frame, DEPTH_SCALE, max_depth, trunc)
                model.synthesize_model_frame(ray, DEPTH_SCALE, DEPTH_MIN, max_depth, trunc, False)
                fused += 1
                prev_pose = poses[-1] if poses else pose.copy()
                poses.append(pose.copy())
                fused_frames.append((path, len(kf_poses) - 1, np.linalg.inv(keyframe_pose) @ pose))
                if len(markers):
                    marker_map.add(markers, pose)
                status = 'tracking (markers)' if by_markers else 'tracking'
            else:
                n_lost += 1
                status = 'NOT ALIGNED - move back to where tracking was'
            if preview:
                d = depth.as_tensor().numpy()[:, :, 0] / DEPTH_SCALE
                left = overlay(cv2.imread(path.replace('/d_', '/c_')), d, max_depth, status, fused, n_lost)
                right = shade(ray.get_data_as_image('depth').as_tensor().numpy()[:, :, 0], fx, fy, cx, cy)
                cv2.imshow('einstar fusion (q to finish)', np.hstack([left, right]))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    quit_requested = True
                    break
            elif index % 100 == 0:
                print(f'frame {index}: fused {fused}, not aligned {n_lost}, {time.time() - t0:.0f} s', flush=True)

    if preview:
        cv2.destroyAllWindows()
    if fused == 0:
        sys.exit('no frames fused')
    if n_marker_frames:
        print(f'markers: {n_marker_frames} of {fused} frames located by markers, {len(marker_map.points)} in the map')
    # markers already tie every frame to one global map; the pose graph is for marker-less scans
    if args.track == 'keyframe' and not args.no_loop_closure and len(kf_poses) >= 3 and n_marker_frames < fused / 2:
        t1 = time.time()
        kf_poses, loops = close_loops(kf_clouds, kf_poses, args.max_corr_mm / 1000)
        # re-fuse every frame at its keyframe's corrected pose
        model = o3d.t.pipelines.slam.Model(voxel, 16, 60000, o3c.Tensor(np.eye(4)), dev)
        poses = []
        for n, (path, k, rel) in enumerate(fused_frames):
            pose = kf_poses[k] @ rel
            frame.set_data_from_image('depth', o3d.t.io.read_image(path))
            frame.set_data_from_image('color', o3d.t.io.read_image(path.replace('/d_', '/c_')))
            model.update_frame_pose(n, o3c.Tensor(pose))
            model.integrate(frame, DEPTH_SCALE, max_depth, trunc)
            poses.append(pose)
        print(f'loop closure: {len(kf_poses)} keyframes, {loops} extra links ({time.time() - t1:.0f} s)')
    mesh = model.voxel_grid.extract_triangle_mesh().to_legacy()
    o3d.io.write_triangle_mesh(out + '_mesh.ply', mesh)
    np.savetxt(out + '_poses.txt', np.array(poses).reshape(len(poses), 16))
    print(f'{index} frames: fused {fused}, not aligned {n_lost}; {len(mesh.triangles)} triangles -> {out}_mesh.ply '
          f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
