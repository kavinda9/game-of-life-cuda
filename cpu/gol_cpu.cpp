// gol_cpu.cpp
// Conway's Game of Life - plain 2D array (single-threaded CPU) implementation.
//
// Modes:
//   bench      : run N generations on a (possibly large) grid, print timing only
//   visualize  : run on a smaller grid, write one .pgm image per generation
//
// Build:  g++ -O2 -o gol_cpu gol_cpu.cpp
// Run:    ./gol_cpu bench 1024 1024 100
//         ./gol_cpu visualize 64 64 120 frames

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

using Grid = std::vector<unsigned char>;

static inline int idx(int r, int c, int cols) { return r * cols + c; }

// Classic Gosper Glider Gun cell offsets (36x9 bounding box).
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

void seed_glider_gun(Grid& g, int rows, int cols, int offr = 2, int offc = 2) {
    std::fill(g.begin(), g.end(), 0);
    for (int i = 0; i < GUN_N; ++i) {
        int r = (GUN[i][0] + offr) % rows;
        int c = (GUN[i][1] + offc) % cols;
        g[idx(r, c, cols)] = 1;
    }
}

// Toroidal (wrap-around) neighbor count.
static inline int count_neighbors(const Grid& g, int r, int c, int rows, int cols) {
    int cnt = 0;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;
            int rr = (r + dr + rows) % rows;
            int cc = (c + dc + cols) % cols;
            cnt += g[idx(rr, cc, cols)];
        }
    }
    return cnt;
}

void step(const Grid& cur, Grid& next, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int n = count_neighbors(cur, r, c, rows, cols);
            unsigned char alive = cur[idx(r, c, cols)];
            unsigned char result = 0;
            if (alive && (n == 2 || n == 3)) result = 1;
            else if (!alive && n == 3) result = 1;
            next[idx(r, c, cols)] = result;
        }
    }
}

void write_pgm(const std::string& path, const Grid& g, int rows, int cols) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fprintf(f, "P5\n%d %d\n255\n", cols, rows);
    std::vector<unsigned char> row(cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) row[c] = g[idx(r, c, cols)] ? 255 : 0;
        fwrite(row.data(), 1, cols, f);
    }
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage:\n  %s bench <rows> <cols> <iters>\n  %s visualize <rows> <cols> <iters> <outdir>\n", argv[0], argv[0]);
        return 1;
    }
    std::string mode = argv[1];

    if (mode == "bench") {
        int rows = argc > 2 ? atoi(argv[2]) : 1024;
        int cols = argc > 3 ? atoi(argv[3]) : 1024;
        int iters = argc > 4 ? atoi(argv[4]) : 100;

        Grid a(rows * cols), b(rows * cols);
        seed_glider_gun(a, rows, cols);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            step(a, b, rows, cols);
            std::swap(a, b);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        printf("CPU 2D-array | grid=%dx%d | iters=%d | total=%.3f ms | avg/gen=%.4f ms\n",
               rows, cols, iters, ms, ms / iters);

    } else if (mode == "visualize") {
        int rows = argc > 2 ? atoi(argv[2]) : 64;
        int cols = argc > 3 ? atoi(argv[3]) : 64;
        int iters = argc > 4 ? atoi(argv[4]) : 120;
        std::string outdir = argc > 5 ? argv[5] : "frames";
        std::string mkcmd = "mkdir -p " + outdir;
        system(mkcmd.c_str());

        Grid a(rows * cols), b(rows * cols);
        seed_glider_gun(a, rows, cols);

        for (int i = 0; i < iters; ++i) {
            char fname[256];
            snprintf(fname, sizeof(fname), "%s/frame_%04d.pgm", outdir.c_str(), i);
            write_pgm(fname, a, rows, cols);
            step(a, b, rows, cols);
            std::swap(a, b);
        }
        printf("Wrote %d frames to %s/\n", iters, outdir.c_str());
    } else {
        printf("Unknown mode: %s\n", mode.c_str());
        return 1;
    }
    return 0;
}
