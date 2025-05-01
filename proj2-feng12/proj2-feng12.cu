#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BOX_SIZE 23000   /* 3D space size */

/* Structure for an atom (3D point) */
typedef struct {
    double x, y, z;
} atom;

/* Histogram bucket structure */
typedef struct {
    unsigned long long int d_cnt;
} bucket;

/* Global variables */
long long PDH_acnt;  /* Number of atoms */
double PDH_res;      /* Bucket width */
int num_buckets;     /* Number of histogram buckets */
int BLOCK_SIZE;      /* Dynamic block size */
atom *atom_list;     /* List of atoms */
bucket *cpu_histogram, *gpu_histogram;  /* Histograms */

/* Timing variables */
struct timeval startTime, endTime;

/* Measure execution time */
double report_running_time() {
    long sec_diff, usec_diff;
    gettimeofday(&endTime, NULL);
    sec_diff = endTime.tv_sec - startTime.tv_sec;
    usec_diff = endTime.tv_usec - startTime.tv_usec;
    if (usec_diff < 0) {
        sec_diff--;
        usec_diff += 1000000;
    }
    return (double)(sec_diff + usec_diff / 1000000.0);
}

/* Compute Euclidean distance (CPU version) */
double p2p_distance_cpu(atom a1, atom a2) {
    double dx = a1.x - a2.x;
    double dy = a1.y - a2.y;
    double dz = a1.z - a2.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

/* CPU version of SDH computation */
void PDH_baseline() {
    for (int i = 0; i < PDH_acnt; i++) {
        for (int j = i + 1; j < PDH_acnt; j++) {  // Avoid double counting
            double dist = p2p_distance_cpu(atom_list[i], atom_list[j]);
            int h_pos = (int)(dist / PDH_res);
            if (h_pos < num_buckets) {
                cpu_histogram[h_pos].d_cnt++;
            }
        }
    }
}

/* CUDA Kernel: Compute SDH using Shared Memory Optimization */
__global__ void computeSDH_CUDA(atom *d_atoms, bucket *d_histogram, int N, double bucket_width, int num_buckets) {
    extern __shared__ unsigned long long local_hist[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory histogram
    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Compute distances for each atom
    if (tid < N) {
        atom a1 = d_atoms[tid];

        for (int j = tid + 1; j < N; j++) {  
            atom a2 = d_atoms[j];

            // Compute distance
            double dist = sqrt((a1.x - a2.x) * (a1.x - a2.x) +
                               (a1.y - a2.y) * (a1.y - a2.y) +
                               (a1.z - a2.z) * (a1.z - a2.z));

            int h_pos = (int)(dist / bucket_width);
            if (h_pos < num_buckets) {
                atomicAdd(&local_hist[h_pos], 1ULL);
            }
        }
    }
    __syncthreads();

    // Write back to global memory
    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
        atomicAdd(&d_histogram[i].d_cnt, local_hist[i]);
    }
}

/* Run the CUDA version */
void run_CUDA_version() {
    atom *d_atom_list;
    bucket *d_histogram;
    cudaMalloc(&d_atom_list, PDH_acnt * sizeof(atom));
    cudaMalloc(&d_histogram, num_buckets * sizeof(bucket));

    cudaMemcpy(d_atom_list, atom_list, PDH_acnt * sizeof(atom), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, num_buckets * sizeof(bucket));

    int blocksPerGrid = (PDH_acnt + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    size_t sharedMemSize = num_buckets * sizeof(unsigned long long);
    computeSDH_CUDA<<<blocksPerGrid, BLOCK_SIZE, sharedMemSize>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res, num_buckets);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\n******** Total Running Time of GPU Kernel = %.5f ms *******\n", milliseconds);

    cudaMemcpy(gpu_histogram, d_histogram, num_buckets * sizeof(bucket), cudaMemcpyDeviceToHost);

    cudaFree(d_atom_list);
    cudaFree(d_histogram);
}

/* Print histogram */
void print_histogram(bucket *hist) {
    unsigned long long total = 0;
    for (int i = 0; i < num_buckets; i += 5) {
        for (int j = 0; j < 5 && (i + j) < num_buckets; j++) {
            printf("%02d: %12llu | ", i + j, hist[i + j].d_cnt);
            total += hist[i + j].d_cnt;
        }
        printf("\n");
    }
    printf(" T:%llu\n", total);
}

/* Compare CPU and GPU histograms */
void compare_histograms() {
    printf("\nDifference between CPU and GPU histograms:\n\n");
    for (int i = 0; i < num_buckets; i += 5) {
        for (int j = 0; j < 5 && (i + j) < num_buckets; j++) {
            printf("%02d: %12lld | ", i + j, (long long)(cpu_histogram[i + j].d_cnt - gpu_histogram[i + j].d_cnt));
        }
        printf("\n");
    }
}

/* Main function */
int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: ./proj2 {#of_samples} {bucket_width} {block_size}\n");
        return 1;
    }

    PDH_acnt = atoll(argv[1]);
    PDH_res = atof(argv[2]);
    BLOCK_SIZE = atoi(argv[3]);  // Read block size from command-line

    num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;

    cpu_histogram = (bucket *)calloc(num_buckets, sizeof(bucket));
    gpu_histogram = (bucket *)calloc(num_buckets, sizeof(bucket));
    atom_list = (atom *)malloc(PDH_acnt * sizeof(atom));

    srand(1);
    for (int i = 0; i < PDH_acnt; i++) {
        atom_list[i].x = ((double)rand() / RAND_MAX) * BOX_SIZE;
        atom_list[i].y = ((double)rand() / RAND_MAX) * BOX_SIZE;
        atom_list[i].z = ((double)rand() / RAND_MAX) * BOX_SIZE;
    }

    gettimeofday(&startTime, NULL);
    PDH_baseline();
    double cpu_time = report_running_time();
    printf("\n******** Total Running Time of CPU = %.5f sec *******\n", cpu_time);

    print_histogram(cpu_histogram);

    run_CUDA_version();

    print_histogram(gpu_histogram);
    compare_histograms();

    free(cpu_histogram);
    free(gpu_histogram);
    free(atom_list);

    return 0;
}
