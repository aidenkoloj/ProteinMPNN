"""
synCP_animation.py
==================
Creates an animated GIF / MP4 cycling through p_num = 1 … 185.
Each frame shows:
  - Top panel  : reordered ProteinMPNN unconditional probability heatmap
  - Bottom panel: position-wise KL divergence vs the original (p_num=0) probs

USAGE
-----
Run from the directory that contains the ProteinMPNN outputs, e.g.:
    cd /home/ubuntu
    python synCP_animation.py

OUTPUT
------
  synCP_animation.gif   – animated GIF  (always produced)
  synCP_animation.mp4   – MP4 video     (produced if ffmpeg is available)

ADJUST
------
  BASE_DIR   – root folder where the .npz files live
  ORIG_NPZ   – path to the original (p_num=0) probabilities
  PERM_NPZ   – f-string template for permutation files
  FPS        – frames per second in the animation
  SAVE_GIF / SAVE_MP4 – toggle output formats
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import kl_div

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = "/home/ubuntu"
#PDB_NAME = "1wcwA_insert"
ORIG_NPZ = os.path.join(BASE_DIR,
    f"ProteinMPNN/outputs/1wcwA_outputs/1wcw_permutation_0/unconditional_probs_only/1wcw_permutation_0.npz")
# PERM_NPZ = os.path.join(BASE_DIR,
#     "ProteinMPNN/outputs/1wcwA_insert_outputs/",
#     "1wcwA_insert_permutation_{p}/conditional_probs_only/1wcwA_insert_permutation_{p}.npz")

PERM_NPZ = os.path.join(BASE_DIR,
    "ProteinMPNN/outputs/1wcwA_outputs/1wcw_permutation_{p}/unconditional_probs_only/1wcw_permutation_{p}.npz")


P_START  = 0
P_END    = 253 # HAVE TO MANUALLY ASSIGN -- Make code that dynamically determines length from input pdb
FPS      = 10        # frames per second  (lower = slower animation)
SAVE_GIF = True
SAVE_MP4 = False

# ── Alphabet ───────────────────────────────────────────────────────────────
alphabet = list("ACDEFGHIKLMNPQRSTVWYX")

# ── Load original probs ────────────────────────────────────────────────────
print("Loading original probabilities …")
d0      = np.load(ORIG_NPZ)
probs_0 = np.exp(d0["log_p"])   # (1, L, 21)
A       = probs_0[0]            # (L, 21)
L       = A.shape[0]

# ── Pre-compute all frames ─────────────────────────────────────────────────
print(f"Pre-computing frames for p_num = {P_START} … {P_END} …")
all_probs_reord = []
all_kl          = []

for p in range(P_START, P_END + 1):
    path = PERM_NPZ.format(p=p)
    try:
        d     = np.load(path)
        probs = np.exp(d["log_p"])          # (1, L, 21)
        probs_reord = np.roll(probs, p, axis=1)
        B     = probs_reord[0]              # (L, 21)
        kl_pp = np.sum(kl_div(A, B), axis=-1)   # (L,)
    except FileNotFoundError:
        print(f"  WARNING: file not found for p_num={p}, using zeros.")
        probs_reord = np.zeros_like(probs_0)
        kl_pp       = np.zeros(L)

    all_probs_reord.append(probs_reord[0])
    all_kl.append(kl_pp)
    if p % 20 == 0:
        print(f"  … p_num {p} done")

print("All frames computed.")

# ── Figure / axes setup ────────────────────────────────────────────────────
fig, (ax_heat, ax_kl) = plt.subplots(
    2, 1, figsize=(28, 12),
    gridspec_kw={"height_ratios": [2, 1]},
)
fig.subplots_adjust(hspace=0.45)

xtick_pos    = range(0, L, 2)
xtick_labels = range(1, L + 1, 2)

# -- heatmap --
im = ax_heat.imshow(
    all_probs_reord[0].T,
    aspect="auto",
    vmin=0, vmax=1,
    cmap="viridis",
)
ax_heat.set_yticks(range(21))
ax_heat.set_yticklabels(alphabet, fontsize=9)
ax_heat.set_xticks(list(xtick_pos))
ax_heat.set_xticklabels(list(xtick_labels), rotation=0, fontsize=8)
ax_heat.set_xlabel("Position", fontsize=11)
ax_heat.set_ylabel("Amino acid", fontsize=11)
heat_title = ax_heat.set_title(
    f"ProteinMPNN Unconditional Probs — permutation 1 (rolled)",
    fontsize=13,
)
fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02, label="Probability")

# -- KL plot --
(line_kl,) = ax_kl.plot([], [], color="steelblue", lw=1.5)
mean_line   = ax_kl.axhline(0, color="tomato", lw=1.2, linestyle="--", label="mean KL")
kl_title    = ax_kl.set_title("Position-wise KL divergence  (mean = —)", fontsize=12)
ax_kl.set_xlim(0, L - 1)
kl_max = max(kl.max() for kl in all_kl) * 1.05
ax_kl.set_ylim(0, kl_max)
ax_kl.set_xlabel("Position", fontsize=11)
ax_kl.set_ylabel("KL(original ‖ permuted)", fontsize=11)
ax_kl.set_xticks(list(xtick_pos))
ax_kl.set_xticklabels(list(xtick_labels), rotation=0, fontsize=8)
ax_kl.grid(True, alpha=0.3)
ax_kl.legend(fontsize=9, loc="upper right")

# ── Animation update function ──────────────────────────────────────────────
def update(frame_idx):
    p   = P_START + frame_idx
    prb = all_probs_reord[frame_idx]
    kl  = all_kl[frame_idx]
    mk  = float(np.mean(kl))

    im.set_data(prb.T)
    heat_title.set_text(
        f"ProteinMPNN unconditional Probs — permutation {p} (rolled)"
    )

    line_kl.set_data(range(L), kl)
    mean_line.set_ydata([mk, mk])
    kl_title.set_text(f"Position-wise KL divergence  (mean = {mk:.4f})")

    return im, line_kl, mean_line, heat_title, kl_title

# ── Build animation ────────────────────────────────────────────────────────
n_frames = P_END - P_START + 1
print(f"Building animation ({n_frames} frames) …")
ani = animation.FuncAnimation(
    fig, update,
    frames=n_frames,
    interval=1000 // FPS,
    blit=False,
)

# ── Save ───────────────────────────────────────────────────────────────────
out_dir = BASE_DIR

if SAVE_GIF:
    gif_path = os.path.join(out_dir, "1wcw_unconditional_synCP_animation10fps.gif")
    print(f"Saving GIF → {gif_path}  (this may take a minute) …")
    ani.save(gif_path, writer="pillow", fps=FPS)
    print("  GIF saved.")

if SAVE_MP4:
    mp4_path = os.path.join(out_dir, "1wcwA_insert_conditional_synCP_animation10fps.mp4")
    try:
        ani.save(mp4_path, writer="ffmpeg", fps=FPS,
                 extra_args=["-vcodec", "libx264", "-crf", "22"])
        print(f"  MP4 saved → {mp4_path}")
    except Exception as e:
        print(f"  MP4 skipped (ffmpeg not available): {e}")

plt.close(fig)
print("Done.")
