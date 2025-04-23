#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <sys/time.h>

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

// Color structure
typedef struct {
    unsigned char r, g, b;
} Color;

// Get time in milliseconds
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

// HSV to RGB conversion (for smooth coloring)
Color hsv_to_rgb(double h, double s, double v) {
    double c = v * s;
    double x = c * (1 - fabs(fmod(h / 60.0, 2) - 1));
    double m = v - c;
    
    double r = 0, g = 0, b = 0;
    
    if (h >= 0 && h < 60) {
        r = c; g = x; b = 0;
    } else if (h >= 60 && h < 120) {
        r = x; g = c; b = 0;
    } else if (h >= 120 && h < 180) {
        r = 0; g = c; b = x;
    } else if (h >= 180 && h < 240) {
        r = 0; g = x; b = c;
    } else if (h >= 240 && h < 300) {
        r = x; g = 0; b = c;
    } else {
        r = c; g = 0; b = x;
    }
    
    Color color;
    color.r = (unsigned char)((r + m) * 255);
    color.g = (unsigned char)((g + m) * 255);
    color.b = (unsigned char)((b + m) * 255);
    
    return color;
}

// Calculate smooth iteration count for a point
double smooth_iteration(double complex c) {
    double complex z = 0;
    int iter;
    
    for (iter = 0; iter < MAX_ITER; iter++) {
        z = z * z + c;
        
        // Check if point escapes
        if (cabs(z) > ESCAPE_RADIUS)
            break;
    }
    
    // Return early if we hit max iterations (in the set)
    if (iter == MAX_ITER)
        return (double)iter;
    
    // Smooth coloring formula using logarithmic escape time
    double log_zn = log(cabs(z));
    double nu = log(log_zn / log(ESCAPE_RADIUS)) / log(2.0);
    return iter + 1 - nu;
}

// Get color for a specific smooth iteration value
Color get_color(double smooth_iter) {
    if (smooth_iter >= MAX_ITER)
        return (Color){0, 0, 0}; // Points in the set are black
    
    // Map the iteration value to a color using HSV
    // This creates a cyclic color pattern with smooth transitions
    double hue = fmod(smooth_iter * 15, 360.0);
    double saturation = 0.8;
    double value = smooth_iter < MAX_ITER ? 1.0 : 0.0;
    
    // More iterations = darker color
    if (smooth_iter > 0)
        value = 1.0 - 0.6 * (smooth_iter / MAX_ITER);
        
    return hsv_to_rgb(hue, saturation, value);
}

// Write the image data to a PPM file
void write_ppm(const char* filename, Color** image) {
    double start_time = get_time_ms();
    
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Unable to open file %s for writing\n", filename);
        return;
    }
    
    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    
    // Write image data
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            fwrite(&image[y][x], 1, 3, fp);
        }
    }
    
    fclose(fp);
    double end_time = get_time_ms();
    printf("Image written to %s (%.2f ms)\n", filename, end_time - start_time);
}

int main() {
    // Start overall timing
    double total_start_time = get_time_ms();
    
    // Timing for memory allocation
    double alloc_start_time = get_time_ms();
    
    // Allocate memory for image
    Color** image = (Color**)malloc(HEIGHT * sizeof(Color*));
    for (int i = 0; i < HEIGHT; i++) {
        image[i] = (Color*)malloc(WIDTH * sizeof(Color));
    }
    
    double alloc_end_time = get_time_ms();
    printf("Memory allocation: %.2f ms\n", alloc_end_time - alloc_start_time);
    
    // Calculate pixel scale
    double re_scale = (RE_MAX - RE_MIN) / (WIDTH - 1);
    double im_scale = (IM_MAX - IM_MIN) / (HEIGHT - 1);
    
    printf("Generating Mandelbrot set image %dx%d...\n", WIDTH, HEIGHT);
    
    // Start timing for Mandelbrot calculation
    double calc_start_time = get_time_ms();
    
    // Generate the Mandelbrot set
    // #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < HEIGHT; y++) {
        double im = IM_MAX - y * im_scale;
        
        for (int x = 0; x < WIDTH; x++) {
            double re = RE_MIN + x * re_scale;
            double complex c = re + im * I;
            
            // Calculate smooth iteration count
            double smooth_iter = smooth_iteration(c);
            
            // Get color based on iteration value
            image[y][x] = get_color(smooth_iter);
        }
        
        // Show progress
        if (y % 100 == 0) {
            printf("Progress: %.1f%% (%.2f ms elapsed)\r", 
                  100.0 * y / HEIGHT, get_time_ms() - calc_start_time);
            fflush(stdout);
        }
    }
    
    double calc_end_time = get_time_ms();
    printf("\nMandelbrot calculation: %.2f ms (%.2f pixels/ms)\n", 
           calc_end_time - calc_start_time,
           (WIDTH * HEIGHT) / (calc_end_time - calc_start_time));
    
    // Write the image to file
    write_ppm("mandelbrot.ppm", image);
    
    // Timing for memory cleanup
    double cleanup_start_time = get_time_ms();
    
    // Free allocated memory
    for (int i = 0; i < HEIGHT; i++) {
        free(image[i]);
    }
    free(image);
    
    double cleanup_end_time = get_time_ms();
    printf("Memory cleanup: %.2f ms\n", cleanup_end_time - cleanup_start_time);
    
    // End overall timing
    double total_end_time = get_time_ms();
    printf("\n--- Performance Summary ---\n");
    printf("Total execution time: %.2f ms\n", total_end_time - total_start_time);
    printf("Rendering resolution: %d x %d (%d pixels)\n", WIDTH, HEIGHT, WIDTH * HEIGHT);
    printf("Maximum iterations: %d\n", MAX_ITER);
    printf("Average rendering speed: %.2f pixels/ms\n", 
           (WIDTH * HEIGHT) / (calc_end_time - calc_start_time));
    printf("Average time per pixel: %.6f ms\n", 
           (calc_end_time - calc_start_time) / (WIDTH * HEIGHT));
    
    return 0;
}
