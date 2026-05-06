# PortLBM — GPU-accelerated Lattice Boltzmann Simulations
Real-time visualized, GPU-accelerated fluid simulation using the Lattice Boltzmann Method (D2Q9/BGK) with SYCL via AdaptiveCpp.

## Getting started

### Dependencies

Only two dependencies must be installed manually. Everything else is either auto-fetched by CMake or installed via the provided script.

| Software | Version | How to obtain |
| -------- | ------- | ------------- |
| [llvm](https://github.com/llvm/llvm-project) | 18.1.8 | `spack install llvm@18.1.8 +llvm_dylib -gold +clang` |
| [glfw](https://github.com/glfw/glfw) | any | Only required with `-DWITH_VISUALIZATION=ON`. Auto-fetched by CMake if not found, but requires X11 dev headers (`sudo apt install libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev`). Alternatively: `spack install glfw` |

> **Note:** Building LLVM with Spack can take more than one hour depending on your machine.

[nlohmann/json](https://github.com/nlohmann/json) and [fmt](https://github.com/fmtlib/fmt) are fetched automatically by CMake if not found on the system.

### Installing AdaptiveCpp

AdaptiveCpp is installed via the provided script, which clones the source and builds it into the build directory:

```bash
# NVIDIA GPU (default)
./install_adaptivecpp.sh

# AMD GPU
./install_adaptivecpp.sh --rocm

# CPU / OpenMP only (no GPU required)
./install_adaptivecpp.sh --cpu

# Custom build directory (default: ./build)
./install_adaptivecpp.sh --build-dir /path/to/build
```

CMake automatically picks up the installation from the build directory — no additional flags needed.

### Building

```bash
mkdir build && cd build

# Without GUI
cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DWITH_NAN_PROTECTION=OFF ..
make
./parallel_lbm

# With GUI
cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DWITH_VISUALIZATION=ON -DWITH_NAN_PROTECTION=ON ..
make
./parallel_lbm
```

## Compile options in detail
In the following, the different CMake compilation options are explained in more detail.

| Option | Default value | Explanation |
| ------ | ------------- | ----------- |
| `-DWITH_VISUALIZATION` | `OFF` | If enabled, the program is built in GUI-mode. |
| `-DWITH_NAN_PROTECTION`| `ON` | If enabled, density and velocity values are set to zero if they are ever `NaN` or ridiculously large |
| `-DUSE_FLOAT`          | `OFF` | If enabled, the program uses single-precision floating-point numbers instead of double-precision |
| `-DFORCE_USE_CPU` | `OFF` | If enabled, the program uses the CPU even if a more performant device is available |
| `-DBENCHMARK_MODE` | `OFF` | Depending on the value of `WITH_VISUALIZATION`, enables the visualized or non-visualized benchmark |
| `-DPORTLBM_BUILD_EXECUTABLE` | `ON` | If disabled, only the `portlbm_core` library is built without the `parallel_lbm` driver executable |

## Simulation settings
The settings of a simulation are specified in `settings/settings.json`. 
A correct file may look like this:

```
{
    "algorithmic": {
        "algorithm": "gpu-two-lattice",
        "dataLayout": "stream",
        "debugMode": false,
        "frameUpdateInterval": 1,
        "timeSteps": 100000,
        "workGroupSize": 1024
    },
    "domain": {
        "horizontalNodes": 2000,
        "scenario": "Hagen-Poiseuille",
        "verticalNodes": 500
    },
    "physical": {
        "inletDensity": 1.2,
        "inletVelocity": {
            "x": 0.0,
            "y": 0.0
        },
        "outletDensity": 1.0,
        "outletVelocity": {
            "x": 0.0,
            "y": 0.0
        },
        "relaxationTime": 0.6
    }
}
```

The parameter `debugMode` can only be set within this file. Setting it to `true` enables a debug mode in which verbose information about the simulation is printed to the console. Furthermore, velocity absolutes and density values are printed directly into the corresponding plots if the GUI is enabled. This debug mode is only meant for finding and fixing bugs for small (!) simulation domains. For reference, the magnitude for which it is meant to be used is a domain with some 10 x 10 nodes, and a work-group size of 16. Please make sure that the parameter `debugMode` is set to `false` when starting a benchmark.

All other parameters can be set from the GUI. In the following, the parameters and possible values are explained more thoroughly.

### Parameter `"algorithm"`
This parameter sets the algorithm that is used in the simulation. The following values can be set:

| Value | Explanation |
| ----- | ----------- |
| `gpu-two-lattice` | Two-lattice implementation with non-linear indexation, that is, two-dimensional iteration; custom work decomposition |
| `gpu-two-lattice-linear` | Two-lattice implementation with linear node indexation, that is, one-dimensional iteration; automatic work decomposition by SYCL runtime |
| `gpu-two-lattice-buffered` | Space-efficient two-lattice implementation with buffers; no second lattice is permanently stored |
| `gpu-swap` | Swap implementation with non-linear indexation and buffering; uses `local` memory |

### Parameter `"dataLayout"`
This parameter sets the data layout of the distribution values. All data layouts were proposed by [Mattila et al.](https://doi.org/10.1016/j.camwa.2007.08.001). The following values can be set:

| Value | Comment |
| ----- | ----------- |
| `stream` | N/A |
| `collision` | N/A |
| `bundle` | N/A |

### Parameter `"frameUpdateInterval"`
Every `"frameUpdateInterval"` iterations, the macroscopic observables are updated on the CPU. Only the values present on the CPU are used for visualization. Any value ranging from `1` to `std::numeric_limits<unsigned int>::max()` is possible.

### Parameter `"timeSteps"`
The algorithm will execute this many iterations. Any value ranging from `1` to `std::numeric_limits<unsigned int>::max()` is possible.

### Parameter `"workGroupSize"`
This parameter sets the work-group size for the SYCL algorithms. Except for the swap algorithm, any value between `1` and `queue.get_device().get_info<sycl::info::device::max_work_group_size>()` is possible. The latter value depends on your device. The swap algorithm requires a work-group size of at least `6`.

### Parameter `"scenario"`
This parameter sets the solid structures of the domain, creating a certain scenario. All scenarios involve an upper and lower pipe boundary. The following values are possible.

| Value | Comment |
| ----- | ----------- |
| `"Hagen-Poiseuille"` | Pipe flow with no inner obstacles |
| `"walls"` | Pipe flow with an obstacle arrangement resembling cascades / a labyrinth |
| `"circle"` | Pipe flow with a circular obstacle in the front |
| `"square"` | Pipe flow with a square obstacle in the front |
| `"wing"` | Pipe flow around a wing |
| `"skyscraper"` | Pipe flow around a three-story skyscraper; higher storys are taller and narrower |
| `"porous"` | Flow through a porous medium that is bounded at the top and bottom; randomized CPU-generated stencil |
| `"plate"` | Pipe flow with a plate obstacle in the front |

### Parameter `"relaxationTime"`
This parameter is related to the viscosity of the fluid. In theory, all values ranging from `0` to `std::numeric_limits<real_type>::max()` are possible. In practice, the simulation is only rarely stable for relaxation times below `0.6` and absurdly high viscosities.

### Parameters  `"horizontalNodes"` and `"verticalNodes"`
Sets the horizontal and vertical extents of the simulated area. Notice that this is not the size of the expanded domain that the algorithms operate on. The simulation domain is always expanded by an outer layer of ghost nodes, and potentially buffer nodes or dummy nodes to match the work-group size. Any size resulting in a total of at most `std::numeric_limits<unsigned int>::max()` is allowed but the minimum extent in each direction is `1`.

### Parameters `"inletDensity"` and `"outletDensity"`
Sets the inlet and outlet density, which is connected to the pressure for the D2Q9I model. For reference: at a standstill equilibrium state, a density of `1.0` is assumed for the fluid. Only positive values are allowed.

### Parameters `"inletVelocity"` and `"outletVelocity"`
The inlet and outlet velocity each consist of two components, `x` and `y`. In theory, any value in the range `std::numeric_limits<real_type>::min()` and `std::numeric_limits<real_type>::max()` is allowed. However, any absolute close to `1 / sqrt(3)` is unlikely to yield a stable simulation. The outlet velocity is without effect, as the [boundary condition by Zou and He](https://doi.org/10.1063/1.869307) with a specified density is assumed. If it is manually disabled within the code, the outlet boundary condition can be set in the same way as the inlet condition.

## Benchmark settings
The settings of the non-visualized benchmark series are set in `settings/benchmark.json`. 
