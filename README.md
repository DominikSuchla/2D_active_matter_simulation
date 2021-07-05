# 2D_active_matter_simulation
A numeric simulation code for 2D active matter based on the continuum model outlined in https://doi.org/10.1103/PhysRevFluids.3.061101. This code is CUDA-accelerated and was developed for a future publication.

Compilation:
- You will need a working CUDA environment.
- Main file is "ActFlow_V3-2.cu", you need to include "cuComplexBinOp.h" and "cudaErr.h", and link "cufft".
- See the example makefile for reference.

Running:
- The program needs a path to a directory as its only command line argument.
- This directory will be used to save snapshots of the simulated vorticity field as well as a file with the total energy in the system after every step.
- This directory needs to contain a text file named "specifications.txt" that contains the specifications for the simulation. An example is provided.

Evaluation:
- The snapshots contain the vorticity field and as last two values the mean velocity in x- and y-direction.
- A minimal python script is provided as an example on how to read them.

---------------------------
Dominik Suchla, 05.07.2021
