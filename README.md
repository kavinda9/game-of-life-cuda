# Conway's Game of Life — CPU vs CUDA

Three implementations, all seeded with a **Gosper Glider Gun** (a pattern that
fires an endless stream of gliders — good for both correctness-checking and
a visually interesting video):

- `cpu/gol_cpu.cpp`      — plain 2D-array, single-threaded CPU version
- `cuda/gol_naive.cu`    — CUDA, one thread per cell, global memory only
- `cuda/gol_shared.cu`   — CUDA, same logic but uses shared memory tiles (halo cells) to cut redundant global memory reads ~8x

All three support two modes:
- `bench` — large grid, timed, no image output (fair timing)
- `visualize` — small grid, writes one `.pgm` frame per generation (for your video/GIF)

---

## 1. Colab setup

```
Runtime -> Change runtime type -> Hardware accelerator -> GPU (T4)
```

Then in a cell:
```python
!nvidia-smi
```
Confirms the GPU is attached.

Upload this whole folder to Colab (drag into the file browser, or `git clone` /
zip-upload), then `%cd` into it:
```python
%cd gol_project
```

---

## 2. Build everything

```python
!g++ -O2 -o cpu/gol_cpu cpu/gol_cpu.cpp
!nvcc -O2 -o cuda/gol_naive cuda/gol_naive.cu
!nvcc -O2 -o cuda/gol_shared cuda/gol_shared.cu
```

---

## 3. Correctness check (small grid, visual)

```python
!./cpu/gol_cpu visualize 64 64 120 frames_cpu
!./cuda/gol_naive visualize 64 64 120 frames_naive 16 16
!./cuda/gol_shared visualize 64 64 120 frames_shared 16 16
```

All three should produce visually identical gliders escaping the gun —
that's your proof the CUDA kernels are correct (matching the CPU baseline).

Convert frames to a GIF for your report/video:
```python
!pip install imageio -q
import imageio, glob
frames = sorted(glob.glob('frames_cpu/*.pgm'))
imgs = [imageio.v2.imread(f) for f in frames]
imageio.mimsave('gun_cpu.gif', imgs, fps=15)
```
(repeat for `frames_naive` / `frames_shared` if you want side-by-side GIFs)

---

## 4. Performance runs (100 generations, no image I/O)

```python
!./cpu/gol_cpu bench 1024 1024 100
!./cuda/gol_naive bench 1024 1024 100 16 16
!./cuda/gol_shared bench 1024 1024 100 16 16
```

---

## 5. Full benchmark sweep (grid sizes x block sizes -> CSV)

```python
!bash scripts/benchmark.sh
```

This produces `results.csv` with columns:
`version, rows, cols, iters, block_x, block_y, total_ms, avg_ms_per_gen`

Load and plot it for your report:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')

# CPU vs GPU (best block size) across grid sizes
best = df[df.version != 'cpu'].loc[df.groupby(['version','rows'])['total_ms'].idxmin()]
cpu = df[df.version == 'cpu']

fig, ax = plt.subplots()
for v, g in pd.concat([cpu, best]).groupby('version'):
    ax.plot(g['rows'], g['total_ms'], marker='o', label=v)
ax.set_xlabel('Grid size (NxN)')
ax.set_ylabel('Total time for 100 generations (ms)')
ax.set_yscale('log')
ax.legend()
plt.savefig('cpu_vs_gpu.png', dpi=150)
plt.show()

# Block size effect (naive vs shared), fixed grid size
sub = df[(df.rows == 1024) & (df.version != 'cpu')]
fig, ax = plt.subplots()
for v, g in sub.groupby('version'):
    labels = g['block_x'].astype(str) + 'x' + g['block_y'].astype(str)
    ax.bar(labels + ('_' + v), g['total_ms'])
ax.set_ylabel('Total time for 100 generations (ms) @ 1024x1024')
plt.xticks(rotation=45)
plt.savefig('block_size_effect.png', dpi=150)
plt.show()
```

---

## 6. What to put in the report

- Table/plot from `results.csv`: CPU vs naive-CUDA vs shared-CUDA, across grid sizes
- Table/plot: effect of block size (8x8 / 16x16 / 32x32) on naive vs shared
- Speedup numbers: `cpu_time / gpu_time` for each grid size
- Explanation:
  - CPU processes ~cells sequentially, one core
  - Naive CUDA parallelizes across thousands of threads but every thread
    re-reads its 8 neighbors from slow global memory independently
  - Shared-memory version loads each block's tile into fast on-chip memory
    once, so neighboring threads reuse cached values instead of hitting
    global memory repeatedly -> fewer memory transactions, higher throughput
  - Note any point where GPU overhead (kernel launch, small-grid underutilization)
    makes the CPU competitive or faster -- small grids are a good place to
    show this honestly

## 7. What to put in the video (<=3 min)

1. Show the glider gun GIF running (CPU or GPU output) — ~30s
2. Briefly show the code structure: CPU loop -> naive kernel -> shared-memory kernel — ~60s
3. Show `results.csv` / the plots and state the speedup numbers — ~60s
4. One sentence on what shared memory bought you and why — ~30s
