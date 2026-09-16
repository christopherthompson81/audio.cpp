# audio.cpp CUDA build wall time — investigation log

A CUDA build takes long enough to be a tax on iteration. This log measures where
the time goes and what actually moves it, on one machine, with everything else
held still.

**Host:** 16 logical cores, 125 GB RAM, RTX 3090 (sm_86), CUDA 12.8, GCC via
`/usr/bin/c++`, NVMe. Build trees on `/mnt/data` (the root filesystem is at 92%).

**Method note:** every run is a cold configure into an empty directory unless the
entry says otherwise, `-j 16`, and wall time is measured around
`cmake --build`. Page cache is warm for the source tree in all runs, which
flatters every number equally and so does not affect the comparisons.

## Run 0 — 2026-09-15 21:30 — what the current build is doing

Read off the existing configured tree, before changing anything.

```
CMAKE_GENERATOR         Unix Makefiles
CMAKE_BUILD_TYPE        Release
CMAKE_CUDA_COMPILER     /usr/local/cuda-12.8/bin/nvcc   (12.8.93)
CMAKE_*_COMPILER_LAUNCHER   (unset)
objects: 976 total — 779 C++, 141 CUDA
```

Three things stand out, none of them subtle.

**1. Every CUDA translation unit is compiled eight times.** The default arch list
expands to eight `--generate-code` entries:

```
1  arch=compute_50,code=[compute_50]     5  arch=compute_80,code=[compute_80]
2  arch=compute_61,code=[compute_61]     6  arch=compute_86,code=[sm_86]
3  arch=compute_70,code=[compute_70]     7  arch=compute_89,code=[sm_89]
4  arch=compute_75,code=[sm_75]          8  arch=compute_90,code=[compute_90]
```

Only #6 can ever run on this host. The other seven are portability for machines
this build will never reach, paid for on every local compile.

**2. ccache is off, and forced.** `CMakeLists.txt:316`:

```cmake
set(GGML_CCACHE OFF CACHE BOOL "Do not probe for ccache in this local setup" FORCE)
```

`FORCE` means `-DGGML_CCACHE=ON` does not override it. ccache IS installed
(`/usr/bin/ccache`) and its stats show 3685 calls from other projects on this
host, so the tool works here — this build simply never calls it.
`CMAKE_CXX_COMPILER_LAUNCHER` bypasses ggml's probe and is not set either.

**3. Unix Makefiles, and no fast linker.** `mold` and `lld` are absent; `gold` is
present but unused. The shared object is 209 MB, so link time is worth measuring
rather than assuming.

**Implication:** measure the baseline, then take the arch multiplier first — it
is the only one whose size can be predicted from first principles (7/8 of the
CUDA work is for hardware that is not here).

## Run 1 — 2026-09-15 21:05 — baseline, and where the cores actually go

```
cmake -S . -B e1-make-8arch -DCMAKE_BUILD_TYPE=Release \
      -DENGINE_ENABLE_CUDA=ON -DAUDIOCPP_BUILD_C_API=ON
cmake --build e1-make-8arch -j 16
```

A fresh configure asks for **nine** architectures, one more than the tree that
was already configured:

```
CMAKE_CUDA_ARCHITECTURES=50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;
                         86-real;89-real;90-virtual;120a-real
CMAKE_CUDA_ARCHITECTURES_NATIVE=86-real
```

`configure_s=10`.

**Mid-build process census, at 16%:**

```
cicc     (nvcc front end)  16 processes
cc1plus  (C++ front end)    0 processes
```

**Finding: the CUDA phase is the build.** Every core is on `cicc` and none on
C++, a third of the way through a tree with 779 C++ objects and 141 CUDA ones.
The 141 are a sixth of the object count and, at this point, all of the wall
time — which is what a 9x per-TU multiplier buys.

nvcc also has an opinion about three of the nine:

```
nvcc warning : Support for offline compilation for architectures prior to
'<compute/sm/lto>_75' will be removed in a future release
```

That is `compute_50`, `compute_61` and `compute_70` — not merely unused on this
host but deprecated by the toolkit that is compiling them.

**Implication for the experiment plan.** At this rate the baseline is roughly
70-80 minutes, so four full builds is most of a working day. Ordering the
remaining runs by information per minute rather than by tidiness: the
architecture list first, since its effect is both the largest and the one that
can be predicted from first principles, and the generator after.
