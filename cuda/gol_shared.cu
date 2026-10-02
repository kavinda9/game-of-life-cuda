#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

static const int GUN[][2] = {
    {0,24},
    {1,22},{1,24},
    {2,12},{2,13},{2,20},{2,21},{2,34},{2,35},
    {3,11},{3,15},{3,20},{3,21},{3,34},{3,35},
    {4,0},{4,1},{4,10},{4,16},{4,20},{4,21},
    {5,0},{5,1},{5,10},{5,14},{5,16},{5,17},{5,22},{5,24},
    {6,10},{6,16},{6,24},
    {7,11},{7,15},
    {8,12},{8,13}
};
static const int GUN_N = sizeof(GUN) / sizeof(GUN[0]);

void seed_glider_gun(std::vector<unsigned char>& g, int rows, int cols, int offr = 2, int offc = 2) {
    std::fill(g.begin(), g.end(), 0);
    for (int i = 0; i < GUN_N; ++i) {
        int r = (GUN[i][0] + offr) % rows;
        int c = (GUN[i][1] + offc) % cols;
        g[r * cols + c] = 1;
    }
}

void write_pgm(const std::string& path, const unsigned char* g, int rows, int cols) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fprintf(f, "P5\n%d %d\n255\n", cols, rows);
    std::vector<unsigned char> row(cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) row[c] = g[r * cols + c] ? 255 : 0;
        fwrite(row.data(), 1, cols, f);
    }
    fclose(f);
}

// Shared memory tile = blockDim + 2 in each dimension (1-cell halo on every side).
// Dynamically sized shared memory, allocated by the launch call (extern __shared__).
extern __shared__ unsigned char tile[];

__global__ void gol_step_shared(const unsigned char* cur, unsigned char* next, int rows, int cols) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockDim.x, by = blockDim.y;
    int tileW = bx + 2;

    int c = blockIdx.x * bx + tx;   // this thread's global column
    int r = blockIdx.y * by + ty;   // this thread's global row

    // Load the halo tile into shared memory. Each thread loads its own cell,
    // plus threads on the tile border also load the extra halo cells around it.
    for (int dy = ty; dy < by + 2; dy += by) {
        int rr = (blockIdx.y * by + dy - 1 + rows) % rows;
        for (int dx = tx; dx < bx + 2; dx += bx) {
            int cc = (blockIdx.x * bx + dx - 1 + cols) % cols;
            tile[dy * tileW + dx] = cur[rr * cols + cc];
        }
    }
    __syncthreads();

    if (r >= rows || c >= cols) return;

    // In shared memory, this thread's own cell sits at (ty+1, tx+1) because
    // of the 1-cell halo offset.
    int sy = ty + 1, sx = tx + 1;
    int cnt = 0;
    #pragma unroll
    for (int dr = -1; dr <= 1; ++dr) {
        #pragma unroll
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;
            cnt += tile[(sy + dr) * tileW + (sx + dc)];
        }
    }
    unsigned char alive = tile[sy * tileW + sx];
    unsigned char result = 0;
    if (alive && (cnt == 2 || cnt == 3)) result = 1;
    else if (!alive && cnt == 3) result = 1;
    next[r * cols + c] = result;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage:\n  %s bench <rows> <cols> <iters> [blockX blockY]\n"
               "  %s visualize <rows> <cols> <iters> <outdir> [blockX blockY]\n", argv[0], argv[0]);
        return 1;
    }
    std::string mode = argv[1];

    if (mode == "bench") {
        int rows = argc > 2 ? atoi(argv[2]) : 1024;
        int cols = argc > 3 ? atoi(argv[3]) : 1024;
        int iters = argc > 4 ? atoi(argv[4]) : 100;
        int bx = argc > 5 ? atoi(argv[5]) : 16;
        int by = argc > 6 ? atoi(argv[6]) : 16;

        size_t bytes = (size_t)rows * cols * sizeof(unsigned char);
        std::vector<unsigned char> h_grid(rows * cols);
        seed_glider_gun(h_grid, rows, cols);

        unsigned char *d_cur, *d_next;
        CUDA_CHECK(cudaMalloc(&d_cur, bytes));
        CUDA_CHECK(cudaMalloc(&d_next, bytes));
        CUDA_CHECK(cudaMemcpy(d_cur, h_grid.data(), bytes, cudaMemcpyHostToDevice));

        dim3 block(bx, by);
        dim3 grid((cols + bx - 1) / bx, (rows + by - 1) / by);
        size_t shmem = (size_t)(bx + 2) * (by + 2) * sizeof(unsigned char);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            gol_step_shared<<<grid, block, shmem>>>(d_cur, d_next, rows, cols);
            std::swap(d_cur, d_next);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        printf("CUDA shared | grid=%dx%d | block=%dx%d | iters=%d | kernel-only total=%.3f ms | avg/gen=%.4f ms\n",
               rows, cols, bx, by, iters, ms, ms / iters);

        cudaFree(d_cur);
        cudaFree(d_next);

    } else if (mode == "visualize") {
        int rows = argc > 2 ? atoi(argv[2]) : 64;
        int cols = argc > 3 ? atoi(argv[3]) : 64;
        int iters = argc > 4 ? atoi(argv[4]) : 120;
        std::string outdir = argc > 5 ? argv[5] : "frames_shared";
        int bx = argc > 6 ? atoi(argv[6]) : 16;
        int by = argc > 7 ? atoi(argv[7]) : 16;
        std::string mkcmd = "mkdir -p " + outdir;
        system(mkcmd.c_str());

        size_t bytes = (size_t)rows * cols * sizeof(unsigned char);
        std::vector<unsigned char> h_grid(rows * cols);
        seed_glider_gun(h_grid, rows, cols);

        unsigned char *d_cur, *d_next;
        CUDA_CHECK(cudaMalloc(&d_cur, bytes));
        CUDA_CHECK(cudaMalloc(&d_next, bytes));
        CUDA_CHECK(cudaMemcpy(d_cur, h_grid.data(), bytes, cudaMemcpyHostToDevice));

        dim3 block(bx, by);
        dim3 grid((cols + bx - 1) / bx, (rows + by - 1) / by);
        size_t shmem = (size_t)(bx + 2) * (by + 2) * sizeof(unsigned char);

        for (int i = 0; i < iters; ++i) {
            CUDA_CHECK(cudaMemcpy(h_grid.data(), d_cur, bytes, cudaMemcpyDeviceToHost));
            char fname[256];
            snprintf(fname, sizeof(fname), "%s/frame_%04d.pgm", outdir.c_str(), i);
            write_pgm(fname, h_grid.data(), rows, cols);
            gol_step_shared<<<grid, block, shmem>>>(d_cur, d_next, rows, cols);
            std::swap(d_cur, d_next);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("Wrote %d frames to %s/\n", iters, outdir.c_str());

        cudaFree(d_cur);
        cudaFree(d_next);
    } else {
        printf("Unknown mode: %s\n", mode.c_str());
        return 1;
    }
    return 0;
}
