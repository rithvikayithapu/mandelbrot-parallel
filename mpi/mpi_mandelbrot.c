// mandelbrot_mpi.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Image dimensions (tweak as needed)
#define WIDTH 2560
#define HEIGHT 1440

// Complex plane boundaries
#define RE_MIN   -2.5
#define RE_MAX    1.0
#define IM_MIN   -1.2
#define IM_MAX    1.2

// Max iterations & escape radius
#define MAX_ITER 1000
#define ESCAPE_R 2.0
#define ESCAPE_R2 (ESCAPE_R*ESCAPE_R)

// Gradient table size
#define COLOR_TABLE_SIZE 4096

// Tile size for locality (optional)
#define TILE_WIDTH  64
#define TILE_HEIGHT 64

// RGB color struct (packed)
typedef struct {
    unsigned char r, g, b;
} __attribute__((packed)) Color;

// Global lookup table
static Color color_table[COLOR_TABLE_SIZE];

// Convert HSV→RGB (helper for init_color_table)
static Color hsv_to_rgb(double hue, double sat, double val) {
    double c = val * sat;
    double x = c * (1 - fabs(fmod(hue/60.0,2) - 1));
    double m = val - c;
    double rp=0, gp=0, bp=0;
    if (hue <  60) { rp=c; gp=x; bp=0; }
    else if (hue < 120) { rp=x; gp=c; bp=0; }
    else if (hue < 180) { rp=0; gp=c; bp=x; }
    else if (hue < 240) { rp=0; gp=x; bp=c; }
    else if (hue < 300) { rp=x; gp=0; bp=c; }
    else               { rp=c; gp=0; bp=x; }
    return (Color){
        (unsigned char)((rp+m)*255),
        (unsigned char)((gp+m)*255),
        (unsigned char)((bp+m)*255)
    };
}

// Build the gradient table in parallel
void init_color_table() {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < COLOR_TABLE_SIZE; i++) {
        double t = (double)i / COLOR_TABLE_SIZE;
        double hue = fmod(t * MAX_ITER * 15, 360.0);
        double sat = 0.8;
        double val = 1.0 - 0.6*t;
        color_table[i] = hsv_to_rgb(hue, sat, val);
    }
}

// Smooth escape‑time iteration
static double smooth_iteration(double cr, double ci) {
    double zr=0, zi=0, zr2=0, zi2=0;
    int iter;
    for (iter=0; iter<MAX_ITER; iter++) {
        zi  = 2*zr*zi + ci;
        zr  = zr2 - zi2 + cr;
        zr2 = zr*zr;
        zi2 = zi*zi;
        if (zr2 + zi2 > ESCAPE_R2) break;
    }
    if (iter == MAX_ITER) return (double)MAX_ITER;
    double mag = sqrt(zr2 + zi2);
    return iter + 1 - log(log(mag))/log(2.0);
}

// Map a smooth iteration to Color
static inline Color get_color(double it) {
    if (it >= MAX_ITER) return (Color){0,0,0};
    int idx = (int)(it * COLOR_TABLE_SIZE / MAX_ITER) % COLOR_TABLE_SIZE;
    return color_table[idx];
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // Command‑line overrides?
    int width  = (argc>1 ? atoi(argv[1]) : WIDTH);
    int height = (argc>2 ? atoi(argv[2]) : HEIGHT);
    const char *outfile = (argc>3 ? argv[3] : "mandelbrot_mpi.ppm");

    // Determine row partition
    int base = height/size, rem = height%size;
    int start = rank*base + (rank<rem ? rank : rem);
    int rows  = base + (rank<rem ? 1 : 0);

    // Timing markers
    double t0 = MPI_Wtime();

    // Init gradient table (shared, but cheap)
    init_color_table();

    double t1 = MPI_Wtime();

    // Allocate local buffer
    Color *local_buf = malloc(rows * width * sizeof(Color));
    if (!local_buf) {
        fprintf(stderr,"Rank %d: malloc failed\n",rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Precompute scaling
    double re_scale = (RE_MAX - RE_MIN)/(width-1);
    double im_scale = (IM_MAX - IM_MIN)/(height-1);

    // Compute your tile chunk
    double t_comp0 = MPI_Wtime();
    for (int y=0; y<rows; y++) {
        int py = start + y;
        double ci = IM_MAX - py*im_scale;
        for (int x=0; x<width; x++) {
            double cr = RE_MIN + x*re_scale;
            double it = smooth_iteration(cr,ci);
            local_buf[y*width + x] = get_color(it);
        }
    }
    double t_comp1 = MPI_Wtime();

    // Prepare gather
    int local_count = rows * width * sizeof(Color);
    int *counts = NULL, *displs = NULL;
    if (rank==0) {
        counts = malloc(size*sizeof(int));
        displs = malloc(size*sizeof(int));
    }
    MPI_Gather(&local_count,1,MPI_INT, counts,1,MPI_INT, 0, MPI_COMM_WORLD);

    if (rank==0) {
        displs[0]=0;
        for (int i=1; i<size; i++)
            displs[i] = displs[i-1] + counts[i-1];
    }

    // Gather all into full_buf on root
    Color *full_buf = NULL;
    if (rank==0) {
        full_buf = malloc(height*width*sizeof(Color));
        if (!full_buf) MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Gatherv(local_buf, local_count, MPI_BYTE,
                full_buf, counts, displs, MPI_BYTE,
                0, MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    // Rank 0 writes PPM
    if (rank==0) {
        FILE *fp = fopen(outfile,"wb");
        fprintf(fp,"P6\n%d %d\n255\n", width, height);
        fwrite(full_buf, sizeof(Color), width*height, fp);
        fclose(fp);
    }
    double t3 = MPI_Wtime();

    // Print performance on rank 0
    if (rank==0) {
        double init_time   = (t1 - t0)*1000;
        double comp_time   = (t_comp1 - t_comp0)*1000;
        double gather_time = (t2 - t_comp1)*1000;
        double io_time     = (t3 - t2)*1000;
        double total_time  = (t3 - t0)*1000;
        double mpixels     = (width*(double)height)/1e6;

        printf("\n--- MPI Mandelbrot Performance ---\n");
        printf("Ranks           : %d\n", size);
        printf("Image resolution: %d x %d (%d pixels)\n", width, height, width*height);
        printf("Max iterations  : %d\n\n", MAX_ITER);

        printf("Initialization: %.2f ms\n",             init_time);
        printf("Computation   : %.2f ms (%.2f MPixels/sec)\n",
               comp_time, mpixels/(comp_time/1000.0));
        printf("Gather + I/O   : %.2f ms\n",            gather_time + io_time);
        printf("  ‣ Gather     : %.2f ms\n",            gather_time);
        printf("  ‣ Write PPM  : %.2f ms\n\n",          io_time);
        printf("Total runtime  : %.2f ms\n",            total_time);
        printf("Overall speed  : %.2f MPixels/sec\n",
               mpixels/(total_time/1000.0));
    }

    // Clean up
    free(local_buf);
    if (rank==0) {
        free(full_buf);
        free(counts);
        free(displs);
    }
    MPI_Finalize();
    return 0;
}


// mpicc -fopenmp mpi_mandelbrot.c -O3 -o mpi_mandelbrot
// mpirun -np 4 ./mpi_mandelbrot 2560 1440 mpi_mandelbrot.ppm
