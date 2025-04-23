#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// Image dimensions
#define WIDTH 2560
#define HEIGHT 1440

// Complex plane boundaries
#define RE_MIN -2.5
#define RE_MAX 1.0
#define IM_MIN -1.2
#define IM_MAX 1.2

// Maximum iterations and escape radius
#define MAX_ITER 1000
#define ESCAPE_RADIUS 2.0
#define ESCAPE_RADIUS_SQ (ESCAPE_RADIUS * ESCAPE_RADIUS)

// Color lookup table size
#define COLOR_TABLE_SIZE 4096

// CUDA block size
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Color structure - byte-aligned for better memory access
typedef struct {
    unsigned char r, g, b;
} __attribute__((packed)) Color;

// Get time in milliseconds
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

// Check CUDA errors
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Host-side color table
Color h_color_table[COLOR_TABLE_SIZE];

// Device-side color table
__constant__ Color d_color_table[COLOR_TABLE_SIZE];

// Initialize color table (host side)
void init_color_table() {
    for (int i = 0; i < COLOR_TABLE_SIZE; i++) {
        double normalized_i = (double)i / COLOR_TABLE_SIZE;
        double hue = fmod(normalized_i * MAX_ITER * 15, 360.0);
        double saturation = 0.8;
        double value = 1.0 - 0.6 * normalized_i;
        
        // HSV to RGB conversion
        double c = value * saturation;
        double x = c * (1 - fabs(fmod(hue / 60.0, 2) - 1));
        double m = value - c;
        
        double r = 0, g = 0, b = 0;
        
        if (hue < 60) {
            r = c; g = x; b = 0;
        } else if (hue < 120) {
            r = x; g = c; b = 0;
        } else if (hue < 180) {
            r = 0; g = c; b = x;
        } else if (hue < 240) {
            r = 0; g = x; b = c;
        } else if (hue < 300) {
            r = x; g = 0; b = c;
        } else {
            r = c; g = 0; b = x;
        }
        
        h_color_table[i].r = (unsigned char)((r + m) * 255);
        h_color_table[i].g = (unsigned char)((g + m) * 255);
        h_color_table[i].b = (unsigned char)((b + m) * 255);
    }
    
    // Copy color table to device constant memory
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_color_table, h_color_table, 
                                         sizeof(Color) * COLOR_TABLE_SIZE));
}

// CUDA kernel for Mandelbrot set computation
__global__ void mandelbrot_kernel(Color* output, int width, int height,
                                  double re_min, double re_max, 
                                  double im_min, double im_max) {
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if within image bounds
    if (x >= width || y >= height) return;
    
    // Map pixel to complex plane
    double re_scale = (re_max - re_min) / (width - 1);
    double im_scale = (im_max - im_min) / (height - 1);
    
    double cr = re_min + x * re_scale;  // real component
    double ci = im_max - y * im_scale;  // imaginary component
    
    // Initialize z = 0
    double zr = 0.0;
    double zi = 0.0;
    double zr2 = 0.0;
    double zi2 = 0.0;
    
    // Iterate until escape or max iterations
    int iter;
    for (iter = 0; iter < MAX_ITER; iter++) {
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        zr2 = zr * zr;
        zi2 = zi * zi;
        
        // Check escape condition
        if (zr2 + zi2 > ESCAPE_RADIUS_SQ)
            break;
    }
    
    // Calculate smooth color
    double smooth_iter;
    if (iter == MAX_ITER) {
        smooth_iter = (double)MAX_ITER;
    } else {
        // Smooth coloring formula
        double mag = sqrt(zr2 + zi2);
        smooth_iter = iter + 1.0 - log(log(mag)) / log(2.0);
    }
    
    // Get color from lookup table
    Color color;
    if (smooth_iter >= MAX_ITER) {
        color = {0, 0, 0};  // Black for points in the set
    } else {
        int idx = (int)(smooth_iter * COLOR_TABLE_SIZE / MAX_ITER) % COLOR_TABLE_SIZE;
        color = d_color_table[idx];
    }
    
    // Write color to output buffer
    output[y * width + x] = color;
}

// Write the image data to a PPM file
void write_ppm(const char* filename, Color* image_data, int width, int height) {
    double start_time = get_time_ms();
    
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Unable to open file %s for writing\n", filename);
        return;
    }
    
    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    
    // Write image data in one large block
    fwrite(image_data, sizeof(Color), width * height, fp);
    
    fclose(fp);
    double end_time = get_time_ms();
    printf("Image written to %s (%.2f ms)\n", filename, end_time - start_time);
}

// Print device information
void print_device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Memory clock rate: %d kHz\n", prop.memoryClockRate);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments for dimensions and output file
    int width = (argc > 1) ? atoi(argv[1]) : WIDTH;
    int height = (argc > 2) ? atoi(argv[2]) : HEIGHT;
    const char *outfile = (argc > 3) ? argv[3] : "mandelbrot_cuda.ppm";
    
    // Start timing
    double total_start_time = get_time_ms();
    
    // Print device information
    print_device_info();
    
    // Initialize color table
    double init_start_time = get_time_ms();
    init_color_table();
    double init_end_time = get_time_ms();
    printf("Color table initialization: %.2f ms\n", init_end_time - init_start_time);
    
    // Allocate host memory for image
    Color* h_image_data = (Color*)malloc(width * height * sizeof(Color));
    if (h_image_data == NULL) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    
    // Allocate device memory for image
    Color* d_image_data;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_image_data, width * height * sizeof(Color)));
    
    // Calculate grid and block dimensions
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);
    
    printf("Grid dimensions: %d x %d blocks\n", gridDim.x, gridDim.y);
    printf("Block dimensions: %d x %d threads\n", blockDim.x, blockDim.y);
    printf("Total threads: %d\n\n", gridDim.x * gridDim.y * blockDim.x * blockDim.y);
    
    // Start kernel timing
    double kernel_start_time = get_time_ms();
    
    // Launch CUDA kernel
    mandelbrot_kernel<<<gridDim, blockDim>>>(d_image_data, width, height, 
                                            RE_MIN, RE_MAX, IM_MIN, IM_MAX);
    
    // Synchronize and check for errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    double kernel_end_time = get_time_ms();
    double kernel_time = kernel_end_time - kernel_start_time;
    printf("CUDA kernel execution: %.2f ms\n", kernel_time);
    
    // Copy result back to host
    double copy_start_time = get_time_ms();
    CHECK_CUDA_ERROR(cudaMemcpy(h_image_data, d_image_data, 
                              width * height * sizeof(Color), 
                              cudaMemcpyDeviceToHost));
    double copy_end_time = get_time_ms();
    printf("Device to host transfer: %.2f ms\n", copy_end_time - copy_start_time);
    
    // Write the image to file
    write_ppm(outfile, h_image_data, width, height);
    
    // Clean up
    free(h_image_data);
    cudaFree(d_image_data);
    
    // End timing
    double total_end_time = get_time_ms();
    double total_time = total_end_time - total_start_time;
    
    // Print performance summary
    printf("\n--- Performance Summary ---\n");
    printf("Image resolution: %d x %d (%d pixels)\n", width, height, width * height);
    printf("Maximum iterations: %d\n", MAX_ITER);
    printf("Kernel execution: %.2f ms (%.2f MPixels/sec)\n", 
           kernel_time, (width * height) / (kernel_time * 1000));
    printf("Total runtime: %.2f ms (%.2f MPixels/sec)\n", 
           total_time, (width * height) / (total_time * 1000));
    
    return 0;
}
