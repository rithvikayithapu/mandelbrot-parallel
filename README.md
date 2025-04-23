# Parallelizing Mandelbrot Generation

## Project Description

The Mandelbrot set is a famous fractal defined by iterating the function
$$\[
    z_{n+1} = z_n^2 + c
\]$$

where $\(c\)$ is a point in the complex plane. This fractal has an infinitely detailed boundary and provides a rich environment for exploring parallel computing due to its parallel structure. Each pixel in the image corresponds to an independent calculation of whether the sequence remains bounded below a certain threshold.

Traditionally, generating high-resolution Mandelbrot images in a serial manner can be computationally expensive. The independence of pixel calculations, however, makes it an ideal problem for parallelization. By dividing the image (or computational domain) among multiple processes, the workload can be distributed efficiently with minimal communication overhead.

We implement and compare four versions of Mandelbrot set generation in this project. The 4 versions are:

- Sequential
- MPI
- OpenMP
- CUDA

Along with comparing runtimes of the four implementations we shall also explore why one implementation does better than the other and explore trends in the four versions.

## Steps to Run the Code

1. `ssh` into the ARC cluster.
2. `git clone` the repository to ARC.
3. To run a specific implementation, `cd` (change directory) into that implementation's directory.
4. Run `sbatch run_mbrot.batch`.

The batch script will allocate required hardware from the ARC HPC cluster, compile the code and run it. The mandelbrot fractal shall be written in .ppm format. The benchmarks shall be written into .txt files.

## Team Members

- sayitha (Sai Rithvik Ayithapu)
- bdahir (Bhavya Ahir)
