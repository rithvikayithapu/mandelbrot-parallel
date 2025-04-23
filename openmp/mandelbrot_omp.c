#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

// Image dimensions - using powers of 2 can help with cache alignment
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

// Tile size for cache-friendly processing
#define TILE_WIDTH 64
#define TILE_HEIGHT 64

// Color structure - byte-aligned for better memory access
typedef struct {
    unsigned char r, g, b;
} __attribute__((packed)) Color;

// Buffer for the entire image
Color* image_buffer;

// Get time in milliseconds
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

// Pre-computed color lookup table for better performance
#define COLOR_TABLE_SIZE 4096
Color color_table[COLOR_TABLE_SIZE];

// Initialize the color lookup table
void init_color_table() {
    #pragma omp parallel for
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
        
        color_table[i].r = (unsigned char)((r + m) * 255);
        color_table[i].g = (unsigned char)((g + m) * 255);
        color_table[i].b = (unsigned char)((b + m) * 255);
    }
}

// Get color from pre-computed table
inline Color get_color(double smooth_iter) {
    if (smooth_iter >= MAX_ITER)
        return (Color){0, 0, 0}; // Points in the set are black

    // Map the iteration value to the color table
    int idx = (int)(smooth_iter * COLOR_TABLE_SIZE / MAX_ITER) % COLOR_TABLE_SIZE;
    return color_table[idx];
}

// Optimized smooth iteration calculation
inline double smooth_iteration(double cr, double ci) {
    // Use direct real/imaginary components instead of complex type for better performance
    double zr = 0;
    double zi = 0;
    double zr2 = 0;
    double zi2 = 0;
    int iter;
    
    // Main iteration loop - avoiding complex number library for better performance
    for (iter = 0; iter < MAX_ITER; iter++) {
        zi = 2 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        zr2 = zr * zr;
        zi2 = zi * zi;
        
        // Early bailout check
        if (zr2 + zi2 > ESCAPE_RADIUS_SQ)
            break;
    }
    
    // Return early if we hit max iterations (in the set)
    if (iter == MAX_ITER)
        return (double)MAX_ITER;
    
    // Smooth coloring formula using logarithmic escape time
    double mag = sqrt(zr2 + zi2);
    return iter + 1 - log(log(mag)) / log(2.0);
}

// Process a single tile of the image
void process_tile(int tile_x, int tile_y, double re_scale, double im_scale) {
    int x_start = tile_x * TILE_WIDTH;
    int y_start = tile_y * TILE_HEIGHT;
    int x_end = x_start + TILE_WIDTH;
    int y_end = y_start + TILE_HEIGHT;
    
    // Clamp to image boundaries
    if (x_end > WIDTH) x_end = WIDTH;
    if (y_end > HEIGHT) y_end = HEIGHT;
    
    // Process pixels in the tile
    for (int y = y_start; y < y_end; y++) {
        double ci = IM_MAX - y * im_scale;
        
        for (int x = x_start; x < x_end; x++) {
            double cr = RE_MIN + x * re_scale;
            
            // Calculate smooth iteration count
            double smooth_iter = smooth_iteration(cr, ci);
            
            // Get color based on iteration value
            image_buffer[y * WIDTH + x] = get_color(smooth_iter);
        }
    }
}

// Write the image data to a PPM file
void write_ppm(const char* filename) {
    double start_time = get_time_ms();
    
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Unable to open file %s for writing\n", filename);
        return;
    }
    
    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    
    // Write image data in one large block for better I/O performance
    fwrite(image_buffer, sizeof(Color), WIDTH * HEIGHT, fp);
    
    fclose(fp);
    double end_time = get_time_ms();
    printf("Image written to %s (%.2f ms)\n", filename, end_time - start_time);
}

int main(int argc, char* argv[]) {
    // Start overall timing
    double total_start_time = get_time_ms();
    
    // Use all available threads by default
    int num_threads = omp_get_max_threads();
    
    // Allow overriding thread count from command line
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        if (num_threads <= 0) num_threads = omp_get_max_threads();
    }
    
    omp_set_num_threads(num_threads);
    printf("Using %d OpenMP threads\n", num_threads);
    
    // Timing for memory allocation and initialization
    double init_start_time = get_time_ms();
    
    // Allocate memory for image as a single contiguous block for better cache performance
    image_buffer = (Color*)aligned_alloc(64, WIDTH * HEIGHT * sizeof(Color));
    if (!image_buffer) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
    }
    
    // Initialize color lookup table
    init_color_table();
    
    double init_end_time = get_time_ms();
    printf("Initialization: %.2f ms\n", init_end_time - init_start_time);
    
    // Calculate pixel scale
    double re_scale = (RE_MAX - RE_MIN) / (WIDTH - 1);
    double im_scale = (IM_MAX - IM_MIN) / (HEIGHT - 1);
    
    printf("Generating Mandelbrot set image %dx%d...\n", WIDTH, HEIGHT);
    
    // Start timing for Mandelbrot calculation
    double calc_start_time = get_time_ms();
    
    // Calculate number of tiles
    int num_tiles_x = (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH;
    int num_tiles_y = (HEIGHT + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int total_tiles = num_tiles_x * num_tiles_y;
    
    // Process tiles in parallel using dynamic scheduling for better load balancing
    int tiles_processed = 0;
    
    #pragma omp parallel
    {
        #pragma omp single
        printf("Starting calculation with %d threads\n", omp_get_num_threads());
        
        #pragma omp for schedule(dynamic, 1)
        for (int tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
            int tile_x = tile_idx % num_tiles_x;
            int tile_y = tile_idx / num_tiles_x;
            
            process_tile(tile_x, tile_y, re_scale, im_scale);
            
            // Update progress
            #pragma omp atomic
            tiles_processed++;
            
            if (tiles_processed % 10 == 0) {
                #pragma omp critical
                {
                    printf("Progress: %.1f%% (%d/%d tiles, %.2f ms elapsed)\r", 
                          100.0 * tiles_processed / total_tiles, 
                          tiles_processed, total_tiles,
                          get_time_ms() - calc_start_time);
                    fflush(stdout);
                }
            }
        }
    }
    
    double calc_end_time = get_time_ms();
    double calc_time = calc_end_time - calc_start_time;
    printf("\nMandelbrot calculation: %.2f ms (%.2f MPixels/sec)\n", 
           calc_time,
           (WIDTH * HEIGHT) / (calc_time));
    
    // Write the image to file
    write_ppm("mandelbrot_parallel.ppm");
    
    // Timing for memory cleanup
    double cleanup_start_time = get_time_ms();
    free(image_buffer);
    double cleanup_end_time = get_time_ms();
    
    // End overall timing
    double total_end_time = get_time_ms();
    printf("\n--- Performance Summary ---\n");
    printf("Total execution time: %.2f ms\n", total_end_time - total_start_time);
    printf("Rendering resolution: %d x %d (%d pixels)\n", WIDTH, HEIGHT, WIDTH * HEIGHT);
    printf("Maximum iterations: %d\n", MAX_ITER);
    printf("Thread count: %d\n", num_threads);
    printf("Rendering speed: %.2f MPixels/sec\n", 
           (WIDTH * HEIGHT) / (calc_time * 1000.0));
    printf("Average time per pixel: %.6f μs\n", 
           (calc_time * 1000.0) / (WIDTH * HEIGHT));
    printf("Average time per tile: %.2f ms\n", 
           calc_time / total_tiles);
    
    return 0;
}
