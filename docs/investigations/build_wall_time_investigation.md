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

## Run 2 — 2026-09-15 21:37 — the architecture list, measured

Same tree, same options, one variable changed:

```
cmake -S . -B e2-make-native ... -DCMAKE_CUDA_ARCHITECTURES=86-real
  -- Using CMAKE_CUDA_ARCHITECTURES=86-real
  --generate-code entries per CUDA TU: 9 -> 1
```

|                    | baseline (9 arch) | one arch | change |
|--------------------|------------------:|---------:|-------:|
| full build         |           1361 s  |   401 s  | **-70.5%, 3.4x** |
| no-op rebuild      |              1 s  |     0 s  | – |
| build tree         |            1.4 G  |   638 M  | -56% |
| `libaudiocpp.so`   |            201 M  |    69 M  | **-66%** |
| objects / CUDA objs|        953 / 141  | 953 / 141| identical |

**Finding: 22.7 minutes becomes 6.7, and the artifact gets three times smaller.**

### Why it beats the naive estimate

CUDA is ~70% of the baseline's wall time, so removing 8/9 of that work predicts
about -62%. The measured -70.5% is better, and the process census says why.

```
baseline, first 17 min:   cicc=16  cc1plus=0     <- C++ starved
one arch, 1 min in:       cicc=6   cc1plus=5     <- overlapped
```

In the baseline the CUDA work saturates all 16 cores and the 779 C++ objects
cannot start; the C++ tail is *serialised behind* CUDA rather than overlapped
with it. Cutting the CUDA work does not just remove its own time, it lets the
rest of the build fill the cores that CUDA was monopolising. That second-order
effect is the difference between -62% and -70.5%.

### ⚠ A fast build of a broken artifact would be worth nothing

So the single-arch `.so` was run, not just weighed — the same C model test the
suite uses, forced onto the CUDA backend:

```
$ audiocpp_c_api_model_test /mnt/data/models/audiocpp sample_16k.wav cuda ...
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24090 MiB)
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
parity:kokoro_tts:audio_rate=24000  audio_seconds=3.275  audio_peak=0.3531
parity:parakeet_tdt:text=Some call me Nature. Others call me Mother Nature...
ran=5 skipped=0 failures=0   EXIT=0
```

Five families, CUDA graphs live, zero failures.

**What it costs.** The resulting binary runs on sm_86 and nothing else — no PTX
for forward compatibility either, since `86-real` omits it. That is correct for
a local development loop and wrong as a repository default. Two different
proposals, and only the second needs to survive a reviewer with a different GPU.

### Negative result: the generator is not the problem here

`noop_s` is **1 second** for Unix Makefiles on 953 objects. The usual argument
for Ninja on a tree this size — faster no-op and incremental rebuilds — does not
apply, because there is nothing to win. Dropped as a lever; it would have been
invisible had the runner only timed full builds.

## Run 3 — 2026-09-15 21:48 — the linker is not the problem, and ccache is

### Linker: bounded, then dropped

Deleted the shared object and timed the relink alone, in both trees:

```
201 MB .so (9 arch)   relink_s=2      "Linking CXX shared library bin/libaudiocpp.so"
 69 MB .so (1 arch)   relink_s=1
```

**Finding: linking is 2 seconds of a 1361-second build — 0.15%.** An
*instantaneous* linker would save 0.15%. `mold` and `lld` are both absent from
this host and installing either needs root; on this evidence there is no reason
to ask. Lever dropped, cheaply, before anyone was inconvenienced.

### ccache: 2% to pay, 93% to gain

ccache 4.9.1 is installed and this build never calls it (Run 0). Wired via
`CMAKE_{C,CXX,CUDA}_COMPILER_LAUNCHER`, which bypasses the forced
`GGML_CCACHE OFF`, with an isolated `CCACHE_DIR` so the host's own 3685-entry
cache was neither read nor cleared.

| phase | build | hit rate | vs no ccache (401 s) |
|-------|------:|---------:|---------------------:|
| cold cache, empty     | 409 s |   3/951  (0.32%) | **+8 s (2% tax)** |
| warm cache, build dir destroyed |  **29 s** | 932/951 (98.00%) | **-93%, 14x** |

Cache footprint for the whole tree: **0.1 GB**.

**Finding: a from-scratch rebuild goes from 6.7 minutes to 29 seconds.** The
2% cold-build tax is the entire downside, and it is paid once per genuinely new
compilation rather than per build.

This is the lever for the case the arch list does not help: blow away `build/`,
switch branches and come back, bisect, or reconfigure with different options.
Those are all full rebuilds today and all near-free with a warm cache.

### ⚠ The two levers answer different questions

- arch list -> a **cold** build, 3.4x
- ccache    -> a **rebuild**, 14x

Neither substitutes for the other, and quoting one number as "the build got
faster" would misrepresent both.

### Process note

The monitor polled every 180 s, so the completion was noticed roughly three
minutes after it happened and the session sat idle. Poll interval should track
the expected phase length, not be set once and forgotten.

## Run 4 — 2026-09-15 21:57 — the model set, and where it stops helping

`AUDIOCPP_MODEL_SET=custom` (README:258) compiles only the families asked for.
Measured on top of the single-arch build, with ccache off so the variable is
isolated:

```
-DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=kokoro_tts,parakeet_tdt
  -- audio.cpp model composite: custom selected [kokoro_tts;parakeet_tdt]
```

| tree                       | objects | CUDA | C++ | .so  | build |
|----------------------------|--------:|-----:|----:|-----:|------:|
| baseline, 9 arch, full     |     953 |  141 | 812 | 201M | 1361 s|
| 1 arch, full               |     953 |  141 | 812 |  69M |  401 s|
| 1 arch, 2 families         |     415 |  141 | 274 |  42M |  285 s|

**Finding, as predicted: the 141 CUDA objects are a floor.** ggml's CUDA backend
is not model-specific, so no model-set choice touches it. The lever moves C++
only, 812 -> 274, worth a further 29%.

## Run 5 — 2026-09-15 22:09 — all three, and what the release path already does

Combined (1 arch + 2 families + ccache): **cold 292 s, warm 21 s** (95.4% hits).

| configuration                        |  cold | warm rebuild |
|--------------------------------------|------:|-------------:|
| baseline                             | 1361 s|       1361 s |
| + single arch                        |  401 s|        401 s |
| + single arch + ccache, **full set** |  409 s|     **29 s** |
| + single arch + custom set + ccache  |  292 s|     **21 s** |

Keeping the **full** model set costs ~117 s cold and ~8 s warm against narrowing
it. That is a cheap price for not having to think about which families are
compiled in, and it is the configuration worth recommending.

### ⚠ The default arch list is a recent, deliberate fix — do not revert it

`62735ea` "Fix CUDA arch selection: provide a portable default arch list (#280)",
2026-08-23:

> Without an explicit CMAKE_CUDA_ARCHITECTURES, enable_language(CUDA)
> (CMP0104 NEW) seeds it from nvcc's default arch (sm_75 on CUDA 13, sm_52 on
> CUDA 12) - **it does not query the local GPU** - so every build without an
> explicit list was single-arch, e.g. sm_75 even on an RTX 5090.

So "make configure detect the GPU" is not a free win: the previous default only
*looked* like host detection and was in fact nvcc's fixed fallback, producing
wrong-arch binaries silently. #280's own conclusion is the shape to keep:
**"portable default, with native or an explicit arch list for fast local
builds."** Making native *easy* is aligned with that; making it *automatic* is
not.

### A stale comment found while checking this

`.github/workflows/release.yml:501` still says:

> CMAKE_CUDA_ARCHITECTURES must be pinned: audio.cpp defaults to "native" (the
> build host GPU), which is meaningless on GPU-less CI runners...

Both halves are now false — the default is the portable list, and the old
default never queried the host GPU either. The *conclusion* (pin it) is still
right, for a different reason. Worth a one-line correction.

Every release path does pin it, which is the safety evidence that mattered here:
`release.yml:271`, `release.yml:514`, `scripts/build_linux.sh:420`,
`scripts/build_windows.ps1:632`.

### ccache is switched off by a line that reads like a local hack

`CMakeLists.txt:316`:

```cmake
set(GGML_CCACHE OFF CACHE BOOL "Do not probe for ccache in this local setup" FORCE)
```

Upstream ggml probes for ccache and uses it when found. This turns that off for
everybody, and `FORCE` means no `-D` can override it. On the measurements above
that line costs a 14x rebuild for no stated benefit.
