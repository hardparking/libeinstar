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

Markers that einstar-rgbd found locate a frame against a map of all markers
seen so far. When the predicted pose finds no match, the frame is searched
in the whole map, so a track that slipped snaps back once known markers are
in view. Only frames the markers located add to the map; a frame aligned by
ICP alone could have slipped and would copy markers to the wrong place. An
ICP alignment is also dropped when the frame disagrees with the surface the
model already holds.

With --turntable the object turns on a turntable with markers on it while
the scanner stands still. The room then does not move with the object and
must not be fused: the frames are tracked by the markers alone, the camera's
circle around the object gives the turntable's axis, and only depth near
the axis and above the turntable is fused. That needs the whole scan, so it
does not work with --follow.

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

    def relocate(self, markers, min_inliers=5, tol=0.0015, triples=20):
        """Pose from the markers alone, without a predicted pose, or None.

        After the track has slipped, the prediction `locate` needs is wrong.
        Triples of frame markers are matched to map triples by their side
        lengths and every match is scored by how many frame markers it puts
        on a map marker. A regular grid of markers fits several poses about
        equally well; then the markers do not fix the pose and None is
        returned."""
        known = self.points[self.counts >= 3]
        if len(markers) < min_inliers or len(known) < min_inliers:
            return None
        dm = np.linalg.norm(known[:, None] - known[None], axis=2)
        df = np.linalg.norm(markers[:, None] - markers[None], axis=2)
        rng = np.random.default_rng(len(markers))
        hyps = []
        for _ in range(triples):
            a, b, c = rng.choice(len(markers), 3, replace=False)
            if min(df[a, b], df[b, c], df[c, a]) < 0.015:
                continue  # a thin triangle barely fixes the rotation
            k, l = np.nonzero(np.abs(dm - df[a, b]) < tol)
            ok = (np.abs(dm[l] - df[b, c]) < tol) & (np.abs(dm[k] - df[c, a]) < tol)
            for i, m in zip(*np.nonzero(ok)):
                t = rigid_fit(markers[[a, b, c]], known[[k[i], l[i], m]])
                world = markers @ t[:3, :3].T + t[:3, 3]
                inl = (np.linalg.norm(world[:, None] - known[None], axis=2).min(1) < self.match_radius).sum()
                if inl >= min_inliers:
                    hyps.append((inl, t))
        if not hyps:
            return None
        hyps.sort(key=lambda h: -h[0])
        best_n, best = hyps[0]
        centre = markers.mean(0)
        for n, t in hyps[1:]:
            # another pose that explains nearly as many markers: ambiguous
            if n >= 0.8 * best_n and np.linalg.norm(t[:3, :3] @ centre + t[:3, 3] - best[:3, :3] @ centre - best[:3, 3]) > 0.005:
                return None
        return self.locate(markers, best)

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


def turntable_axis(poses):
    """A point on the turntable's axis and its direction (up), from camera poses in object coordinates.

    The object turns under a still camera, so in object coordinates the
    camera circles the axis: the axis is the normal of the circle through the
    camera positions, through its centre. None when the scan covers too
    little of a turn to tell."""
    c = np.array([p[:3, 3] for p in poses])
    mid = c.mean(0)
    _, s, vt = np.linalg.svd(c - mid)
    if s[1] < 0.3 * s[0]:
        return None
    axis = vt[2] if np.mean([p[:3, 2] for p in poses], axis=0) @ vt[2] < 0 else -vt[2]  # the camera looks down
    xy = np.c_[(c - mid) @ vt[0], (c - mid) @ vt[1]]
    # circle |x - o|^2 = r^2, linear in o and r^2 - |o|^2
    o = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], (xy ** 2).sum(1), rcond=None)[0]
    return mid + o[0] * vt[0] + o[1] * vt[1], axis


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
    ap.add_argument('--turntable', action='store_true',
                    help='the object turns on a turntable with markers on it, the scanner stands still; '
                         'keeps only what turns (not with --follow)')
    ap.add_argument('--turntable-radius-mm', type=float, default=150,
                    help='with --turntable, keep depth within this distance of the axis (default 150)')
    ap.add_argument('--plate-clearance-mm', type=float, default=3,
                    help='with --turntable, drop depth up to this far above the turntable (default 3)')
    args = ap.parse_args()
    if args.turntable and (args.follow or args.no_markers):
        sys.exit('--turntable needs markers and the whole scan: it cannot be used with --follow or --no-markers')

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

    check = o3d.t.pipelines.slam.Frame(height, width, K, dev)

    def agrees_with_model(depth_img, p, tol=0.005, min_overlap=0.2, min_agree=0.6):
        """Whether a frame at pose p sees the surface the model already holds.

        ICP that slipped still fits its own reference, but then the frame
        disagrees with the model fused so far: the surfaces lie apart where
        both see something. A frame that mostly sees new surface cannot be
        checked and passes."""
        if fused < 10:
            return True
        model.update_frame_pose(fused, o3c.Tensor(p))
        model.synthesize_model_frame(check, DEPTH_SCALE, DEPTH_MIN, max_depth, trunc, False)
        d = depth_img.as_tensor().numpy()[:, :, 0].astype(np.float32) / DEPTH_SCALE
        r = check.get_data_as_image('depth').as_tensor().numpy()[:, :, 0].astype(np.float32) / DEPTH_SCALE
        seen = (d > 0) & (d <= max_depth)
        both = seen & (r > 0)
        if both.sum() < min_overlap * seen.sum():
            return True
        return (np.abs(d[both] - r[both]) < tol).mean() >= min_agree

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

    if args.turntable:
        return fuse_turntable(args, frame_dir, out, max_depth, model, frame, cloud, (fx, fy, cx, cy, width, height), trunc)

    keyframe, keyframe_pose = None, np.eye(4)
    marker_map = MarkerMap()
    prev_pose, n_marker_frames, n_relocated, n_rejected = np.eye(4), 0, 0, 0
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
            src = cloud(depth)
            empty = src.is_empty()  # no depth in range: nothing to align or fuse
            aligned = fused == 0 and not empty
            new_keyframe = keyframe is None
            markers = np.zeros((0, 3)) if args.no_markers else read_markers(path)
            by_markers = relocated = False
            if fused > 0 and len(markers):
                # constant-velocity guess from the last two poses
                m = marker_map.locate(markers, pose @ np.linalg.inv(prev_pose) @ pose)
                if m is None:
                    # the prediction may be off because the track slipped: look in the whole map
                    m = marker_map.relocate(markers)
                    relocated = m is not None
                if m is not None:
                    pose, aligned, by_markers = m, True, True
                    n_marker_frames += 1
                    n_relocated += relocated
                    new_keyframe = relocated or n_marker_frames % 10 == 0  # keep the ICP fallback's reference fresh
            if fused > 0 and not aligned and not empty:
                src = src.voxel_down_sample(0.002)
                if args.track == 'keyframe':
                    # guess: the previous pose, relative to the keyframe
                    r = align(src, keyframe, np.linalg.inv(keyframe_pose) @ pose)
                    if r is not None:
                        p = keyframe_pose @ r.transformation.numpy()
                        if agrees_with_model(depth, p):
                            pose, aligned = p, True
                            new_keyframe = r.fitness < args.keyframe_fitness
                        else:
                            n_rejected += 1
                if not aligned:
                    r = align(src, cloud(ray.get_data_as_image('depth')), np.eye(4))  # raycast depth is in input units
                    if r is not None:
                        p = pose @ r.transformation.numpy()
                        if agrees_with_model(depth, p):
                            pose, aligned = p, True
                            new_keyframe = args.track == 'keyframe'
                        else:
                            n_rejected += 1
            if aligned and new_keyframe and not empty:
                keyframe, keyframe_pose = cloud(depth).voxel_down_sample(0.0015), pose.copy()
                kf_clouds.append(keyframe)
                kf_poses.append(keyframe_pose)
            if aligned:
                model.update_frame_pose(fused, o3c.Tensor(pose))
                model.integrate(frame, DEPTH_SCALE, max_depth, trunc)
                model.synthesize_model_frame(ray, DEPTH_SCALE, DEPTH_MIN, max_depth, trunc, False)
                fused += 1
                prev_pose = poses[-1] if poses and not (by_markers and relocated) else pose.copy()
                poses.append(pose.copy())
                fused_frames.append((path, len(kf_poses) - 1, np.linalg.inv(keyframe_pose) @ pose))
                # A frame aligned by ICP alone may have slipped; its markers would
                # enter the map at the wrong place and a later frame could lock
                # onto that copy. So only frames the markers located extend the
                # map, apart from the frames before the first such frame.
                if len(markers) and (by_markers or n_marker_frames == 0):
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
        print(f'markers: {n_marker_frames} of {fused} frames located by markers ({n_relocated} found again '
              f'without a prediction), {len(marker_map.points)} in the map')
    if n_rejected:
        print(f'{n_rejected} ICP alignments rejected: they disagreed with the model')
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


def fuse_turntable(args, frame_dir, out, max_depth, model, frame, cloud, intrinsics, trunc):
    """Turntable scan: markers track the object, then only what turns is fused.

    The room stands still while the object turns, so depth of the room
    cannot be fused at the markers' poses. The first pass tracks every frame
    by the markers alone; the camera positions then give the turntable's
    axis (see turntable_axis) and the top of the turntable is the height,
    along the axis, at which most of the seen surface lies. The second pass
    fuses only depth within --turntable-radius-mm of the axis and above the
    turntable."""
    t0 = time.time()
    marker_map = MarkerMap()
    tracked = []  # (depth path, pose)
    pose = prev_pose = np.eye(4)
    index = 0
    for path in frame_paths(frame_dir, 0):
        index += 1
        markers = read_markers(path)
        if len(markers) < marker_map.min_markers:
            continue
        if tracked:
            m = marker_map.locate(markers, pose @ np.linalg.inv(prev_pose) @ pose)
            if m is None:
                m = marker_map.relocate(markers)
            if m is None:
                continue
            prev_pose, pose = pose, m
        tracked.append((path, pose))
        marker_map.add(markers, pose)
    print(f'turntable: {len(tracked)} of {index} frames located by markers, {len(marker_map.points)} in the map')
    found = turntable_axis([p for _, p in tracked]) if len(tracked) >= 10 else None
    if found is None:
        sys.exit('turntable: the markers do not show the object turning (too few frames located, or too little of a turn)')
    centre, axis = found
    radius = args.turntable_radius_mm / 1000

    def height_radial(points):
        w = points - centre
        h = w @ axis
        return h, np.linalg.norm(w - h[:, None] * axis, axis=1)

    hs = []
    for path, p in tracked[::5]:
        pts = cloud(o3d.t.io.read_image(path)).point.positions.numpy() @ p[:3, :3].T + p[:3, 3]
        h, r = height_radial(pts)
        hs.append(h[r < radius])
    hs = np.concatenate(hs)
    counts, edges = np.histogram(hs, bins=np.arange(hs.min(), hs.max() + 0.002, 0.002))
    top = edges[counts.argmax() + 1]  # the turntable is the flattest, most seen surface
    cut = top + args.plate_clearance_mm / 1000
    print(f'turntable: axis through {np.round(centre * 1000)} mm, top at {top * 1000:.0f} mm along it')

    fx, fy, cx, cy, width, height = intrinsics
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    rays = np.dstack(((u - cx) / fx, (v - cy) / fy, np.ones((height, width))))
    poses, fused = [], 0
    for path, p in tracked:
        raw = o3d.t.io.read_image(path).as_tensor().numpy()
        pts = (raw[:, :, 0, None] / DEPTH_SCALE * rays).reshape(-1, 3) @ p[:3, :3].T + p[:3, 3]
        h, r = height_radial(pts)
        keep = ((raw[:, :, 0] > 0).ravel() & (h > cut) & (r < radius)).reshape(height, width)
        if keep.sum() < 100:
            continue
        frame.set_data_from_image('depth', o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(
            np.where(keep[:, :, None], raw, 0)))))
        frame.set_data_from_image('color', o3d.t.io.read_image(path.replace('/d_', '/c_')))
        model.update_frame_pose(fused, o3c.Tensor(p))
        model.integrate(frame, DEPTH_SCALE, max_depth, trunc)
        poses.append(p)
        fused += 1
    mesh = model.voxel_grid.extract_triangle_mesh().to_legacy()
    o3d.io.write_triangle_mesh(out + '_mesh.ply', mesh)
    np.savetxt(out + '_poses.txt', np.array(poses).reshape(len(poses), 16))
    print(f'{index} frames: fused {fused}; {len(mesh.triangles)} triangles -> {out}_mesh.ply ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
