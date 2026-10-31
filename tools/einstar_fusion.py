#!/usr/bin/env python3
"""Fuses einstar-rgbd frames into a mesh, KinectFusion-style.

    einstar_fusion.py FRAME_DIR [--follow] [--out PREFIX] [--max-depth-mm 300] [--voxel-mm 1.5]

Each frame is aligned by point-to-plane ICP to a raycast of the model fused
so far, then fused into it (a TSDF volume), which is how EXStar tracks
(ICP against its fused model, 160-400 mm working range). Only depth closer
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frame_dir')
    ap.add_argument('--follow', action='store_true', help='process frames while einstar-rgbd writes them, with a preview')
    ap.add_argument('--no-preview', action='store_true')
    ap.add_argument('--out', help='output prefix (default FRAME_DIR/scan)')
    ap.add_argument('--max-depth-mm', type=float, default=300, help='ignore depth beyond this (default 300)')
    ap.add_argument('--voxel-mm', type=float, default=1.5, help='TSDF voxel size (default 1.5)')
    ap.add_argument('--max-corr-mm', type=float, default=5, help='ICP correspondence distance (default 5)')
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
            if fused > 0:
                src = cloud(depth).voxel_down_sample(0.002)
                tgt = cloud(ray.get_data_as_image('depth'))  # raycast depth is in input units
                if len(src.point.positions) >= 500 and len(tgt.point.positions) >= 500:
                    tgt.estimate_normals(max_nn=20, radius=0.006)
                    r = reg.icp(src, tgt, args.max_corr_mm / 1000, I4,
                                reg.TransformationEstimationPointToPlane(), criteria)
                    if r.fitness >= MIN_FITNESS:
                        pose = pose @ r.transformation.numpy()
                        aligned = True
            if aligned:
                model.update_frame_pose(fused, o3c.Tensor(pose))
                model.integrate(frame, DEPTH_SCALE, max_depth, trunc)
                model.synthesize_model_frame(ray, DEPTH_SCALE, DEPTH_MIN, max_depth, trunc, False)
                fused += 1
                poses.append(pose.copy())
                status = 'tracking'
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
    mesh = model.voxel_grid.extract_triangle_mesh().to_legacy()
    o3d.io.write_triangle_mesh(out + '_mesh.ply', mesh)
    np.savetxt(out + '_poses.txt', np.array(poses).reshape(len(poses), 16))
    print(f'{index} frames: fused {fused}, not aligned {n_lost}; {len(mesh.triangles)} triangles -> {out}_mesh.ply '
          f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
