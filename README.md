# Task-based-Lattice-Boltzmann
This repository contains the software I created in my bachelor thesis.

## Getting started
This instruction is still in a work-in-progress state and will be updated soon!
To ensure that you can build and run the code, please ensure that you have the following software installed and ready to be found by CMake.
All software has been tested with the default Spack setup or the minimum necessary additions unless mentioned otherwise:
- AdaptiveCpp and any further software that is required for the configuration you want to use this software with (tested with AdaptedCpp 24.10)
- HPX 1.10.0 and any further software that is required 
- NLohmann-JSON
- fmt 
Once you have the software set up, the following commands should suffice to build a runnable program.

If you want to build without the GUI, that is, in benchmark mode, use the following lines while in the build directory:
```
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_NAN_PROTECTION=OFF ../ # NaN protection only necessary for the GUI
make
./parallel_lbm # run the code
```

If you want to build with the GUI use the following lines while in the build directory:
```
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_VISUALIZATION=ON -DWITH_NAN_PROTECTION=ON ../
make
./parallel_lbm # run the code
```

