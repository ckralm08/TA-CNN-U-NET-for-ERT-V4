# =============================================================================
# scripts/generate_dataset.py
# Environment : PyGIMLi
# Fungsi      : Generate dataset sintetik ERT.
#
# Output per sampel (3 file):
#   X_xxxx.npy  -> pseudosection grid  (NZ, NX, 1)  [0,1]  (input CNN)
#   y_xxxx.npy  -> true model grid     (NZ, NX, 1)  [0,1]  (label CNN)
#   d_xxxx.npy  -> rhoa vektor         (n_data,)    [0,1]  (label surrogate)
#
# Urutan training :
#   1. python scripts/generate_dataset.py      (env PyGIMLi)
#   2. python scripts/train_forward.py         (env TF)
#   3. python scripts/train_inversion.py       (env TF)
#   4. python scripts/evaluate.py              (env TF)
# =============================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from pathlib import Path
import shutil

from utils.preprocessing import load_config, normalize


# =============================================================================
# INISIALISASI DARI CONFIG
# =============================================================================
CFG     = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))
DOM     = CFG["domain"]
GRID    = CFG["grid"]
DS      = CFG["dataset"]

XMIN, XMAX = DOM["xmin"], DOM["xmax"]
ZMIN, ZMAX = DOM["zmin"], DOM["zmax"]
NZ         = GRID["nz"]
NX         = GRID["nx"]
N_ELECS    = DOM["n_electrodes"]
RHO_MIN    = DOM["rho_min"]
RHO_MAX    = DOM["rho_max"]

# Grid CNN
XG = np.linspace(XMIN, XMAX, NX)
ZG = np.linspace(ZMAX, ZMIN, NZ)
Xg, Zg = np.meshgrid(XG, ZG)

# Skema Wenner-Alpha
ELECS  = np.linspace(XMIN, XMAX, N_ELECS)
SCHEME = ert.createData(elecs=ELECS, schemeName=DOM["scheme"])

# Counter distribusi mode
mode_counter = {0: 0, 1: 0, 2: 0}


# =============================================================================
# MODE SAMPLER
# =============================================================================
class ModeSampler:
    def __init__(self, n_models: int, ratio: list = None):
        ratio  = ratio or DS["mode_ratio"]
        r      = np.array(ratio, dtype=float)
        frac   = r / r.sum()
        counts = np.floor(frac * n_models).astype(int)

        remainder = n_models - counts.sum()
        for i in np.argsort(-frac):
            if remainder <= 0:
                break
            counts[i] += 1
            remainder -= 1

        self.modes = [0]*counts[0] + [1]*counts[1] + [2]*counts[2]
        np.random.shuffle(self.modes)
        self.idx = 0

    def sample(self) -> int:
        if self.idx >= len(self.modes):
            np.random.shuffle(self.modes)
            self.idx = 0
        m = self.modes[self.idx]
        self.idx += 1
        return m


# =============================================================================
# RANDOM MODEL GENERATOR
# =============================================================================
def create_random_model(sampler=None):
    mode = sampler.sample() if sampler else np.random.choice([0, 1, 2])
    mode_counter[mode] = mode_counter.get(mode, 0) + 1

    world  = mt.createWorld(start=[XMIN, ZMIN], end=[XMAX, ZMAX], worldMarker=True)
    bodies = []

    if mode == 0:
        x = np.random.uniform(-30, 30)
        z = np.random.uniform(-15, -5)
        w = np.random.uniform(10, 17.5)
        h = np.random.uniform(4, 7.5)
        bodies.append(mt.createRectangle(
            start=[x - w/2, z - h/2], end=[x + w/2, z + h/2], marker=2))

    elif mode == 1:
        for k in range(2):
            x = np.random.uniform(-35, 35)
            z = np.random.uniform(-15, -5)
            w = np.random.uniform(7.5, 15)
            h = np.random.uniform(4, 7.5)
            bodies.append(mt.createRectangle(
                start=[x - w/2, z - h/2], end=[x + w/2, z + h/2], marker=2+k))

    elif mode == 2:
        x = np.random.uniform(-10, 10)
        z = np.random.uniform(-12.5, -6)
        w = np.random.uniform(40, 60)
        h = np.random.uniform(4, 7.5)
        bodies.append(mt.createRectangle(
            start=[x - w/2, z - h/2], end=[x + w/2, z + h/2], marker=2))

    geom = world
    for b in bodies:
        geom += b

    mesh    = mt.createMesh(geom, quality=34, area=1.5)
    rho_bg  = np.random.uniform(40, 120)
    rho_an  = np.random.uniform(5, 30)
    markers = np.array([c.marker() for c in mesh.cells()])
    model   = np.full(mesh.cellCount(), rho_bg, dtype=np.float32)
    model[markers == 2] = rho_an
    model[markers == 3] = rho_an
    return mesh, model


# =============================================================================
# BUILD DATA: X (pseudosection grid), y (model grid), d (vektor rhoa)
# =============================================================================
def build_sample(mesh, true_model):
    """
    Returns
    -------
    X     : (NZ, NX, 1) float32  pseudosection grid ternormalisasi  [input CNN]
    y     : (NZ, NX, 1) float32  true model grid ternormalisasi     [label CNN]
    d_obs : (n_data,)   float32  vektor rhoa ternormalisasi          [label surrogate]
    """
    # Forward modeling dengan noise
    data = ert.simulate(
        mesh       = mesh,
        scheme     = SCHEME,
        res        = true_model,
        noiseLevel = np.random.uniform(DS["noise_level_min"], DS["noise_level_max"]),
        noiseAbs   = np.random.uniform(DS["noise_abs_min"],   DS["noise_abs_max"]),
    )

    # Koordinat pseudosection
    s        = np.array(data.sensors())
    A, B     = s[data["a"]], s[data["b"]]
    M, N     = s[data["m"]], s[data["n"]]
    x_mid    = (A[:,0] + B[:,0] + M[:,0] + N[:,0]) / 4
    a_space  = np.abs(A[:,0] - M[:,0])
    z_pseudo = -0.519 * a_space

    # Normalisasi rhoa 
    rhoa      = np.clip(np.array(data["rhoa"]), RHO_MIN, RHO_MAX)
    rhoa_norm = normalize(rhoa, CFG)   # -> [0, 1]

    # Vektor d_obs (untuk surrogate)
    d_obs = rhoa_norm.astype(np.float32)

    # Interpolasi pseudosection ke grid CNN 
    rho_interp = griddata(
        (x_mid, z_pseudo), rhoa_norm,
        (Xg, Zg), method="linear", fill_value=np.nan
    )

    # Mask area valid
    x_unique = np.unique(x_mid)
    z_env    = np.array([z_pseudo[x_mid == x].min() for x in x_unique])
    z_env    = savgol_filter(z_env, 11 if len(z_env) > 11 else 5, 2)
    z_env_g  = np.interp(XG, x_unique, z_env)
    mask     = (Zg <= 0) & (Zg >= z_env_g[np.newaxis, :])

    X       = np.zeros_like(Xg)
    valid   = mask & np.isfinite(rho_interp)
    X[valid] = rho_interp[valid]
    X        = X.reshape(NZ, NX, 1).astype(np.float32)

    # True model grid (label y) 
    cc       = np.array(mesh.cellCenters())
    valid_cc = cc[:, 1] <= 0
    y        = griddata(
        (cc[valid_cc, 0], cc[valid_cc, 1]),
        true_model[valid_cc],
        (Xg, Zg), method="nearest"
    )
    y = np.clip(y, RHO_MIN, RHO_MAX)
    y = normalize(y, CFG)
    y = gaussian_filter(np.nan_to_num(y), 1.0)
    y = y.reshape(NZ, NX, 1).astype(np.float32)

    return X, y, d_obs


# =============================================================================
# DATASET GENERATOR
# =============================================================================
def generate_split(n_samples: int, out_dir: Path, split: str, sampler=None) -> int:
    X_dir = out_dir / "X"
    y_dir = out_dir / "y"
    d_dir = out_dir / "d_obs"
    for d in [X_dir, y_dir, d_dir]:
        d.mkdir(parents=True, exist_ok=True)

    valid = 0
    for i in range(n_samples):
        try:
            mesh, model    = create_random_model(sampler)
            X, y, d_obs    = build_sample(mesh, model)

            np.save(X_dir / f"X_{valid:04d}.npy",  X)
            np.save(y_dir / f"y_{valid:04d}.npy",  y)
            np.save(d_dir / f"d_{valid:04d}.npy",  d_obs)

            print(f"[OK] {split.upper()} {valid:04d}  d_obs shape: {d_obs.shape}",
                  flush=True)
            valid += 1
        except Exception as e:
            print(f"[SKIP] {split.upper()} attempt {i}: {e}", flush=True)

    return valid


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    base = Path(DS["processed_dir"])

    if base.exists():
        shutil.rmtree(base)

    N_TRAIN = DS["n_train"]
    N_VAL   = DS["n_val"]
    N_TEST  = DS["n_test"]

    print("\n========== DATASET CONFIGURATION ==========")
    print(f"TRAIN : {N_TRAIN}")
    print(f"VAL   : {N_VAL}")
    print(f"TEST  : {N_TEST}")
    print(f"NZ={NZ}  NX={NX}  N_ELECS={N_ELECS}")
    print("============================================\n")

    print(">>> Generating TRAIN")
    train_ok = generate_split(N_TRAIN, base/"train", "train",
                               sampler=ModeSampler(N_TRAIN))

    print("\n>>> Generating VAL")
    val_ok = generate_split(N_VAL, base/"val", "val")

    print("\n>>> Generating TEST")
    test_ok = generate_split(N_TEST, base/"test", "test")

    print("\n========== MODE DISTRIBUTION (TRAIN) ==========")
    total = sum(mode_counter.values())
    for m in sorted(mode_counter):
        n   = mode_counter[m]
        lbl = {0:"A (1 kompak)", 1:"B (2 anomali)", 2:"C (lebar)"}[m]
        print(f"  Mode {m} [{lbl}]: {n:4d}  ({100*n/total:.1f}%)")

    print(f"\n========== SELESAI ==========")
    print(f"TRAIN: {train_ok}  VAL: {val_ok}  TEST: {test_ok}")
