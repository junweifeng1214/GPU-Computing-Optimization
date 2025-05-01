/* ==================================================================
    Programmer: Yicheng Tu (ytu@cse.usf.edu)
    The basic SDH algorithm implementation for 3D data with CUDA extension
    To compile: nvcc SDH.cu -o SDH in the GAIVI machines
   ==================================================================
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#define BOX_SIZE 23000 /* size of the data box on one dimension */
/* descriptors for single atom in the tree */
typedef struct atomdesc {
    double x_pos;
    double y_pos;
    double z_pos;
} atom;
typedef struct hist_entry {
    long long d_cnt; /* need a long long type as the count might be huge */
} bucket;
/* Global variables */
bucket *histogram, *cpu_histogram; /* list of all buckets in the histogram */
long long PDH_acnt;                /* total number of data points */
int num_buckets;                    /* total number of buckets in the histogram */
double PDH_res;                     /* value of w */
atom *atom_list;                    /* list of all data points */
/* Timing variables */
struct timezone Idunno;
struct timeval startTime, endTime;
/*
    distance of two points in the atom_list
*/
double p2p_distance(int ind1, int ind2) {
    double x1 = atom_list[ind1].x_pos;
    double x2 = atom_list[ind2].x_pos;
    double y1 = atom_list[ind1].y_pos;
    double y2 = atom_list[ind2].y_pos;
    double z1 = atom_list[ind1].z_pos;
    double z2 = atom_list[ind2].z_pos;
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}
/*
    brute-force SDH solution in a single CPU thread
*/
int PDH_baseline() {
    int i, j, h_pos;
    double dist;
    for (i = 0; i < PDH_acnt; i++) {
        for (j = i + 1; j < PDH_acnt; j++) {
            dist = p2p_distance(i, j);
            h_pos = (int)(dist / PDH_res);
            if (h_pos < num_buckets) {
                histogram[h_pos].d_cnt++;
            }
        }
    }
    return 0;
}
/*
    set a checkpoint and show the (natural) running time in seconds
*/
double report_running_time() {
    long sec_diff, usec_diff;
    gettimeofday(&endTime, &Idunno);
    sec_diff = endTime.tv_sec - startTime.tv_sec;
    usec_diff = endTime.tv_usec - startTime.tv_usec;
    if (usec_diff < 0) {
        sec_diff--;
        usec_diff += 1000000;
    }
    printf("Running time: %ld.%06ld\n", sec_diff, usec_diff);
    return (double)(sec_diff * 1.0 + usec_diff / 1000000.0);
}
/*
    print the counts in all buckets of the histogram
*/
void output_histogram(bucket *histogram) {
    int i;
    long long total_cnt = 0;
    for (i = 0; i < num_buckets; i++) {
        if (i % 5 == 0) /* we print 5 buckets in a row */
            printf("\n%02d: ", i);
        printf("%15lld ", histogram[i].d_cnt);
        total_cnt += histogram[i].d_cnt;
        /* we also want to make sure the total distance count is correct */
        if (i == num_buckets - 1)
            printf("\n T:%lld \n", total_cnt);
        else
            printf("| ");
    }
}
/*
    CUDA Kernel: Computes the spatial distance histogram (SDH) on GPU.
    Each thread processes a particle and calculates the distance to all other particles.
    It then updates the corresponding bucket in the histogram using atomic operations.
*/
__global__ void computeSDH_CUDA(atom *d_atoms, bucket *d_histogram, int N, double bucket_width, int num_buckets) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return; // Out-of-bounds check
    for (int j = i + 1; j < N; j++) {
        double dx = d_atoms[i].x_pos - d_atoms[j].x_pos;
        double dy = d_atoms[i].y_pos - d_atoms[j].y_pos;
        double dz = d_atoms[i].z_pos - d_atoms[j].z_pos;
        double distance = sqrt(dx * dx + dy * dy + dz * dz);
        int h_pos = (int)(distance / bucket_width);
        if (h_pos < num_buckets) {
            // Critical part: Atomic addition to avoid race conditions
            atomicAdd((unsigned long long *)&d_histogram[h_pos].d_cnt, 1ULL);
        }
    }
}
/*
    GPU Version: Allocate device memory, copy data to GPU, launch kernel, and copy results back to CPU.
*/
void run_CUDA_version() {
    atom *d_atom_list;
    bucket *d_histogram;
    // Allocate memory on GPU for atoms and histogram
    cudaMalloc(&d_atom_list, PDH_acnt * sizeof(atom));
    cudaMalloc(&d_histogram, num_buckets * sizeof(bucket));
    // Copy atom data to GPU
    cudaMemcpy(d_atom_list, atom_list, PDH_acnt * sizeof(atom), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, num_buckets * sizeof(bucket));
    // Launch kernel with threads and blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (PDH_acnt + threadsPerBlock - 1) / threadsPerBlock;
    computeSDH_CUDA<<<blocksPerGrid, threadsPerBlock>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res, num_buckets);
    cudaDeviceSynchronize();
    // Copy results back to CPU
    cudaMemcpy(histogram, d_histogram, num_buckets * sizeof(bucket), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_atom_list);
    cudaFree(d_histogram);
}
/*
    Compares the CPU and GPU histograms by computing and displaying the difference for each bucket.
*/
void compare_histograms(bucket *cpu_hist, bucket *gpu_hist) {
    printf("\nDifference between CPU and GPU histograms:\n");
    for (int i = 0; i < num_buckets; i++) {
        long long diff = cpu_hist[i].d_cnt - gpu_hist[i].d_cnt;
        if (i % 5 == 0)
            printf("\n%02d: ", i);
        printf("%15lld ", diff);
        if (i != num_buckets - 1)
            printf("| ");
    }
    printf("\n");
}
int main(int argc, char **argv) {
    int i;
    PDH_acnt = atoi(argv[1]);
    PDH_res = atof(argv[2]);
    num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
    histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
    cpu_histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
    atom_list = (atom *)malloc(sizeof(atom) * PDH_acnt);
    srand(1);
    /* generate data following a uniform distribution */
    for (i = 0; i < PDH_acnt; i++) {
        atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
        atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
        atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
    }
    // CPU version
    gettimeofday(&startTime, &Idunno);
    PDH_baseline();
    gettimeofday(&endTime, &Idunno);
    report_running_time();
    memcpy(cpu_histogram, histogram, sizeof(bucket) * num_buckets);
    output_histogram(cpu_histogram);
    // GPU version
    gettimeofday(&startTime, &Idunno);
    run_CUDA_version();
    gettimeofday(&endTime, &Idunno);
    report_running_time();
    output_histogram(histogram);
    // Compare CPU and GPU results
    compare_histograms(cpu_histogram, histogram);
    free(histogram);
    free(cpu_histogram);
    free(atom_list);
    return 0;
}
