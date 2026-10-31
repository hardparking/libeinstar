"""Compares einstar-rtabmap odometry with einstar-synth ground truth.

usage: python3 eval_trajectory.py poses.txt odom.txt

poses.txt  ground truth from einstar-synth: id, world<-colour camera (mm)
odom.txt   einstar-rtabmap --poses: id, odom<-base (m); base -> colour optical
           frame is RTAB-Map's optical rotation

Both trajectories are expressed relative to their first frame in the colour
camera's optical frame, so they are directly comparable.
"""
import math
import sys


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def inverse(m):
    r = [[m[j][i] for j in range(3)] for i in range(3)]
    t = [-sum(r[i][k] * m[k][3] for k in range(3)) for i in range(3)]
    return [r[0] + [t[0]], r[1] + [t[1]], r[2] + [t[2]], [0, 0, 0, 1]]


def from_row(vals, scale):
    t = [v * scale for v in vals[0:3]]
    r = vals[3:12]
    return [r[0:3] + [t[0]], r[3:6] + [t[1]], r[6:9] + [t[2]], [0, 0, 0, 1]]


def load(path, scale):
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) == 13:
            out[int(p[0])] = from_row([float(x) for x in p[1:]], scale)
    return out


def angle(m):
    c = (m[0][0] + m[1][1] + m[2][2] - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    gt = load(sys.argv[1], 1.0)
    est = load(sys.argv[2], 1000.0)  # metres -> mm
    # rtabmap::CameraModel::opticalRotation(): base <- optical. The driver's
    # upright output (images and camera frame both turned 180 degrees about
    # the optical axis) cancels out, so base <- calibration colour frame is
    # still just the optical rotation.
    base_from_opt = [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]

    ids = sorted(set(gt) & set(est))
    if len(ids) < 2:
        print("not enough common frames")
        return 1
    g0 = inverse(gt[ids[0]])
    e0 = inverse(mat_mul(est[ids[0]], base_from_opt))
    errs, rot_errs, path = [], [], 0.0
    prev = None
    for i in ids:
        g = mat_mul(g0, gt[i])
        e = mat_mul(e0, mat_mul(est[i], base_from_opt))
        d = [g[k][3] - e[k][3] for k in range(3)]
        errs.append(math.sqrt(sum(x * x for x in d)))
        rot_errs.append(angle(mat_mul(inverse(g), e)))
        if prev is not None:
            path += math.sqrt(sum((g[k][3] - prev[k][3]) ** 2 for k in range(3)))
        prev = g
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    print(f"frames {len(ids)}, ground-truth path {path:.1f} mm")
    print(f"translation RMSE {rmse:.2f} mm, max {max(errs):.2f} mm, final {errs[-1]:.2f} mm")
    print(f"rotation max {max(rot_errs):.3f} deg, final {rot_errs[-1]:.3f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
