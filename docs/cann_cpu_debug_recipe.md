# CANN 9.2.0 CPU-debug compile recipe (no NPU)

**Final verdict: WORKS-WITH-WORKAROUNDS.**

A known-good Ascend C kernel (`AddCustom`) can be compiled and run in CANN CPU-debug mode on an x86_64 host with no NPU, using the ASC language support shipped with CANN 9.2.0. The only blocker encountered on this machine is a glibc/gcc-15 header incompatibility (`_Float128` / `__TC__` machine mode) that is bypassed with a one-line header shim.

## What was verified

- Host: Ubuntu, CANN 9.2.0 at `/usr/local/Ascend`, system `g++ 15.2`, no NPU.
- Sample: `AddCustom` kernel adapted from `https://gitee.com/ascend/samples`.
- Flow: single `.asc` source → `find_package(ASC)` CMake project → `cmake -DCMAKE_ASC_RUN_MODE=cpu ...` → executable that runs on the host CPU.
- Result: build succeeds and the executable prints `CPU debug run: PASS`.

## Files in the working example

A self-contained copy lives at `data/cann_cpu_debug_example/`:

```
data/cann_cpu_debug_example/
├── CMakeLists.txt
├── add.asc                 # kernel + host launch/verify
├── add_custom_tiling.h     # tiny tiling struct + GET_TILING_DATA macro
└── shim/
    └── bits/
        └── floatn.h        # glibc float128 workaround
```

## One-time workaround: float128 shim

On glibc >= 2.41 hosts, the bisheng compiler pulls in `<bits/floatn.h>`, which defines `_Float128` / `_Complex _Float128` with the `__TC__` machine mode. The aicore target does not support that mode, so compilation fails with:

```text
/usr/include/x86_64-linux-gnu/bits/floatn.h:83:52: error: unsupported machine mode '__TC__'
typedef _Complex float __cfloat128 __attribute__ ((__mode__ (__TC__)));
                                                   ^
/usr/include/x86_64-linux-gnu/bits/floatn.h:97:9: error: __float128 is not supported on this target
typedef __float128 _Float128;
        ^
```

Workaround: place a stub `bits/floatn.h` on an `-isystem` path so it is found before the real system header.

```bash
mkdir -p /tmp/cann-shim/bits
cat > /tmp/cann-shim/bits/floatn.h <<'EOF'
#ifndef _BITS_FLOATN_H
#define _BITS_FLOATN_H
#define __HAVE_FLOAT128 0
#define __HAVE_DISTINCT_FLOAT128 0
#endif
EOF
```

The example in `data/cann_cpu_debug_example/` already contains the same shim under `shim/bits/floatn.h` and passes `-isystem${CMAKE_CURRENT_SOURCE_DIR}/shim`.

## Reproducible recipe

### 1. Source the CANN environment

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

`ASCEND_HOME_PATH` must be set; the CMake `ASC` package uses it.

### 2. Create the shim (if not using the bundled one)

```bash
mkdir -p /tmp/cann-shim/bits
cat > /tmp/cann-shim/bits/floatn.h <<'EOF'
#ifndef _BITS_FLOATN_H
#define _BITS_FLOATN_H
#define __HAVE_FLOAT128 0
#define __HAVE_DISTINCT_FLOAT128 0
#endif
EOF
```

### 3. Configure and build

Using the bundled example:

```bash
cd data/cann_cpu_debug_example
rm -rf build
cmake -B build -S . \
    -DCMAKE_ASC_RUN_MODE=cpu \
    -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build build -j4
```

Expected result:

```text
[ 50%] Building ASC object CMakeFiles/add.dir/add.asc.o
[100%] Linking ASC executable add
[100%] Built target add
```

### 4. Run

```bash
./build/add
```

Expected result:

```text
[ascendc_acl_stub.cpp][check_interface:330]
[TmSim]: Run in serial mode.
[SUCCESS][CORE_0][pid ...] exit success!
CPU debug run: PASS
```

## How to turn this into a pass/fail gate

For `domino/pipeline/ascend_c/compile_check.py` (or any wrapper), the contract is:

1. **Compile verdict**
   - Run `cmake --build build -j4`.
   - Exit code `0` and `build/add` exists → `response_syntex = "right"`.
   - Any non-zero exit → `response_syntex = "wrong"`.

2. **Run verdict (optional)**
   - Run `./build/add` (or a generated test harness) with a timeout.
   - Exit code `0` and expected output → run `PASS`.
   - Non-zero exit, crash, or wrong output → run `FAIL`.

The compile step alone is the reliable gate: if CANN accepts the kernel, it is syntactically/semantically valid Ascend C. The run step additionally catches logic errors when a golden reference is available.

## CMake boilerplate for a generic kernel

```cmake
cmake_minimum_required(VERSION 3.16.0)
find_package(ASC REQUIRED)
project(my_kernel LANGUAGES ASC CXX)

add_executable(my_kernel my_kernel.asc)

target_compile_options(my_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:-isystem/path/to/shim>
    # define DTYPE_ macros if the kernel uses them
    $<$<COMPILE_LANGUAGE:ASC>:-DDTYPE_X=float>
    $<$<COMPILE_LANGUAGE:ASC>:-DDTYPE_Y=float>
    $<$<COMPILE_LANGUAGE:ASC>:-DDTYPE_Z=float>
)
```

Notes:

- `find_package(ASC REQUIRED)` must appear **before** `project(... LANGUAGES ASC CXX)` so that the `ASC` language module is registered before CMake enables the language.
- `CMAKE_ASC_RUN_MODE=cpu` selects host CPU execution; `CMAKE_ASC_ARCHITECTURES=dav-2201` targets the Ascend 910B1 architecture (other supported values include `dav-3510`).
- Kernels that use `GET_TILING_DATA` need a small tiling header (see `add_custom_tiling.h`) because the CPU-debug single-file path does not run the msopgen tiling generator.

## Verified environment details

```text
CANN version: 9.2.0
ASC compiler: /usr/local/Ascend/cann-9.2.0/bin/bisheng
Host CXX:     /usr/bin/g++ (Ubuntu 15.2.0-16ubuntu1)
CMake:        4.2.3
Architecture: dav-2201 (Ascend910B1)
```

## What is NOT covered here

- Full `msopgen` operator packaging (host + tiling + `.run` package). That path is not needed for a compile-only gate and was not tested.
- NPU execution. CPU debug validates kernel code shape and basic arithmetic; final correctness must still be checked on real Ascend hardware.
- Non-`float` dtypes. The recipe uses `float` for `DTYPE_X/Y/Z`; other dtypes need matching `AscendC::Add` support and buffer sizes.

## References

- Official CPU-debug sample/docs: `https://gitcode.com/cann/asc-tools` (`docs/01_cpu_debug.md`, `examples/02_cpudebug/`).
- Gitee `AddCustom` sample: `https://gitee.com/ascend/samples/blob/master/operator/ascendc/tutorials/AddCustomSample/FrameworkLaunch/AddCustom/op_kernel/add_custom.cpp`.
- CANN cmake module: `/usr/local/Ascend/cann-9.2.0/x86_64-linux/tikcpp/ascendc_kernel_cmake/asc_modules/FindASC.cmake`.
