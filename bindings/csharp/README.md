# C# bindings

.NET bindings for the C ABI in `include/audiocpp.h`.

These live in the fork rather than being proposed upstream: the C ABI proposal
([#525](https://github.com/0xShug0/audio.cpp/issues/525)) deliberately keeps
language bindings out of scope.

| project | |
|---|---|
| `AudioCpp` | the binding — `net10.0`, AOT-compatible, no package references |
| `AudioCpp.PathTest` | the C# counterpart of `tests/capi/path_test.c` |
| `AudioCpp.ModelTest` | the C# counterpart of `tests/capi/model_test.c` |

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAUDIOCPP_BUILD_C_API=ON
cmake --build build --target audiocpp audiocpp_c_api_model_test

./bindings/csharp/run-tests.sh build /path/to/models [threads]
```

`run-tests.sh` runs both C# tests and then checks that the bindings report
exactly what the C tests report for the same models. It prints the native
directory it resolved, since multi-config generators (Visual Studio, Xcode,
Ninja Multi-Config) place outputs in a per-configuration subdirectory. Exit codes follow CTest:
0 pass, 1 fail, 77 skip.

Both sides get the same thread count, defaulting to the host's cores, so the
cross-language diff compares like with like. The expensive model run happens once
per language rather than once per purpose — two runs is the floor for a
cross-language comparison, since each language has to do the work to be compared.

The native library is built by CMake rather than packaged, so it will not sit
beside the managed assembly. `AUDIOCPP_NATIVE_DIR` points the resolver at it;
without that, normal platform probing applies.

## Usage

```csharp
using var registry = AudioCppRegistry.Create();
using var model = registry.Load("models/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf", "kokoro_tts");
using var session = model.CreateSession("tts", "offline", new BackendConfig("cuda", 0, 8));

using var request = new AudioCppRequest()
    .SetText("Hello from audio.cpp.", "en-us")
    .SetVoiceId("af_heart");

using var result = session.Run(request);
if (result.Audio is { } audio)
{
    Console.WriteLine($"{audio.Duration:F2}s at {audio.SampleRate} Hz");
}
```

Model families are discovered rather than hardcoded:

```csharp
foreach (var option in model.GetOptions(AudioCppOptionScope.Request))
{
    Console.WriteLine($"{option.Name} default={option.DefaultValue} " +
                      $"range=[{option.MinValue},{option.MaxValue}]");
}
```

## Notes

**Disposal order does not matter.** A session holds its model, which holds its
registry, so the ABI keeps parents alive. This is the reason the C ABI was built
that way: .NET finalizer order is not deterministic, so an API that required an
order would be unusable from here regardless of how carefully a caller wrote
their `using` statements. `AudioCpp.PathTest` disposes the model and registry
and then keeps using the session.

**Everything is copied on the way out.** Every `const char *` and `const float *`
the ABI returns is borrowed from the handle that produced it, so accessors copy
rather than wrap. Returned strings are declared as `IntPtr` in the interop layer
for the same reason — letting the marshaller build a `string` from a return
value would also let it free a pointer it does not own.

**Errors carry the native detail.** A non-OK status becomes an
`AudioCppException` with `Status` and `Detail`. The detail is thread-local on the
native side and a later call may clear it, so it is read synchronously on the
calling thread.

**Threads.** The native library never calls `omp_set_num_threads` — that is
process-global state — so `BackendConfig.Threads` is the only place it is set.
