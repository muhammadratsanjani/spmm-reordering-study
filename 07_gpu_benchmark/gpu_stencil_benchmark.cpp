#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdio>
#include <cstring> // memcpy

// TOGGLE SHUFFLE: Comment baris ini untuk mendapatkan "Natural Ordering"
#define ENABLE_SHUFFLE 1

// --- 1. DATA STRUCTURES (HOST SIDE) ---
struct HostCSR {
    int num_rows;
    int num_nnz;
    std::vector<int> row_map;
    std::vector<int> col_idx;
    std::vector<double> values;
};

// --- 2. GENERATOR GRID 3D (Natural or Shuffled) ---
HostCSR generate_3d_stencil_shuffled(int nx, int ny, int nz) {
    int N = nx * ny * nz;
    HostCSR mat;
    mat.num_rows = N;
    
    // Adjacency List Construction (Explicit for clarity)
    // Walaupun boros memori host, ini paling mudah dipahami untuk shuffled study
    std::vector<std::vector<int>> adj(N);
    
    // Helper lambda index (x,y,z) -> linear ID
    // Hati-hati: x + y*nx + z*nx*ny
    for(int z=0; z<nz; z++) {
        for(int y=0; y<ny; y++) {
            for(int x=0; x<nx; x++) {
                int u = x + (y*nx) + (z*nx*ny);
                
                // 7-Point Stencil Neighbors
                if(x>0)     adj[u].push_back( (x-1) + (y*nx) + (z*nx*ny) );
                if(x<nx-1)  adj[u].push_back( (x+1) + (y*nx) + (z*nx*ny) );
                
                if(y>0)     adj[u].push_back( x + ((y-1)*nx) + (z*nx*ny) );
                if(y<ny-1)  adj[u].push_back( x + ((y+1)*nx) + (z*nx*ny) );
                
                if(z>0)     adj[u].push_back( x + (y*nx) + ((z-1)*nx*ny) );
                if(z<nz-1)  adj[u].push_back( x + (y*nx) + ((z+1)*nx*ny) );
                
                adj[u].push_back(u); // Include self
            }
        }
    }

    // Prepare Node Ordering
    std::vector<int> p(N);
    for(int i=0; i<N; i++) p[i] = i; // Default: Natural Order

#ifdef ENABLE_SHUFFLE
    // Jika didefinisikan, acak urutannya
    printf("[INFO] Shuffle ENABLED. Randomizing Node IDs...\n");
    std::mt19937 rng(12345);
    std::shuffle(p.begin(), p.end(), rng);
#else
    printf("[INFO] Shuffle DISABLED. Using Natural 3D Ordering.\n");
#endif

    // Inverse Permutation: inv_p[new_id] = old_id
    std::vector<int> inv_p(N);
    for(int i=0; i<N; i++) inv_p[p[i]] = i;

    // Build CSR based on New Ordering
    mat.row_map.push_back(0);
    int current_nnz = 0;
    
    for(int i=0; i<N; i++) {
        int old_u = inv_p[i]; // Siapa pemilik sah baris 'i' ini dulu?
        std::vector<int> neighbors;
        
        // Ambil tetangga (dalam ID baru)
        for(int old_v : adj[old_u]) {
            neighbors.push_back(p[old_v]); 
        }
        std::sort(neighbors.begin(), neighbors.end()); // CSR wajib urut kolom
        
        for(int col : neighbors) {
            mat.col_idx.push_back(col);
            mat.values.push_back(1.0); // Dummy Value
            current_nnz++;
        }
        mat.row_map.push_back(current_nnz);
    }
    mat.num_nnz = current_nnz;
    return mat;
}

// --- 3. GPU BENCHMARK FUNCTION ---
void run_benchmark(int grid_dim) {
    long long n_nodes = (long long)grid_dim * grid_dim * grid_dim;
    printf("Generating %d^3 Grid (%lld Nodes)...\n", grid_dim, n_nodes);
    
    HostCSR h_mat = generate_3d_stencil_shuffled(grid_dim, grid_dim, grid_dim);
    int N = h_mat.num_rows;
    int NNZ = h_mat.num_nnz;
    printf("Matrix Size: %d Rows, %d NNZ. Moving to Device...\n", N, NNZ);

    // Device Views (Memory Space Otomatis Cuda jika di-compile dgn Cuda)
    typedef Kokkos::DefaultExecutionSpace::memory_space MemSpace;
    Kokkos::View<int*, MemSpace> row_map("row_map", N+1);
    Kokkos::View<int*, MemSpace> col_idx("col_idx", NNZ);
    Kokkos::View<double*, MemSpace> values("values", NNZ);
    Kokkos::View<double*, MemSpace> x("x", N);
    Kokkos::View<double*, MemSpace> y("y", N);

    // Host Mirrors
    auto h_row = Kokkos::create_mirror_view(row_map);
    auto h_col = Kokkos::create_mirror_view(col_idx);
    auto h_val = Kokkos::create_mirror_view(values);
    
    // Copy vector -> host view
    // (Note: this is slow serial copy, but OK for initialization)
    for(int i=0; i<=N; i++) h_row(i) = h_mat.row_map[i];
    for(int i=0; i<NNZ; i++) h_col(i) = h_mat.col_idx[i];
    for(int i=0; i<NNZ; i++) h_val(i) = h_mat.values[i];

    // Deep Copy Host -> Device
    Kokkos::deep_copy(row_map, h_row);
    Kokkos::deep_copy(col_idx, h_col);
    Kokkos::deep_copy(values, h_val);
    Kokkos::deep_copy(x, 1.0); 

    // Kernel Policy
    typedef Kokkos::TeamPolicy<> policy_t;
    typedef policy_t::member_type member_t;
    
    // Warmup
    Kokkos::parallel_for("Warmup", policy_t(N, Kokkos::AUTO), KOKKOS_LAMBDA(const member_t& team) {
        if(team.league_rank() == 0) double temp = 0.0; 
    });
    Kokkos::fence();

    // Measurement
    Kokkos::Timer timer;
    int repeat = 20; 
    for(int iter=0; iter<repeat; iter++) {
        Kokkos::parallel_for("SpMV_GPU", policy_t(N, Kokkos::AUTO), KOKKOS_LAMBDA(const member_t& team) {
            int row = team.league_rank();
            double sum = 0.0;
            int start = row_map(row);
            int len = row_map(row+1) - start;
            
            // Team Thread Range: Semua thread di block ini gotong royong hitung 1 baris
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, len), 
                [=] (const int k_off, double& lsum) {
                    lsum += values(start + k_off) * x(col_idx(start + k_off));
                }, sum);
            
            if(team.team_rank()==0) y(row) = sum;
        });
    }
    Kokkos::fence();
    
    double avg_time = timer.seconds() / repeat;
    double gflops = (2.0 * NNZ * 1e-9) / avg_time;
    printf(">>> Result: %d^3 | Time: %.5f s | Perf: %.2f GFLOPs\n\n", grid_dim, avg_time, gflops);
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        printf("=== KOKKOS SPMV GPU BENCHMARK (3D STENCIL) ===\n");
        printf("Backend: %s\n\n", typeid(Kokkos::DefaultExecutionSpace).name());
        
        // Scaling Study
        run_benchmark(50);   // 125k
        run_benchmark(80);   // 512k
        run_benchmark(100);  // 1M
    }
    Kokkos::finalize();
    return 0;
}
