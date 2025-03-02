#include "Mesh.h"
#include "helper_math.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void voxelize(float* faces, size_t n_faces, size_t dim_x, size_t dim_y, size_t dim_z, unsigned int* v_table, float3 spacing, float3 bbox_min) {
    size_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;

    float3 delta_p = make_float3(spacing.x, spacing.y, spacing.z);
    int3 grid_size = make_int3(dim_x - 1, dim_y - 1, dim_z - 1);

    if (thread_idx < n_faces) {
        // We have 1 thread x triangle and each triangle is composed by 3 vertices.
        // Each vertex is composed by 3 float so the first vertex will be at thread_idx * 9
        size_t i = thread_idx * 9;

        // TRIANGLE INFORMATION
        float3 v0 = make_float3(faces[i], faces[i + 1], faces[i + 2]) - make_float3(bbox_min.x, bbox_min.y, bbox_min.z);
        float3 v1 = make_float3(faces[i + 3], faces[i + 4], faces[i + 5]) - make_float3(bbox_min.x, bbox_min.y, bbox_min.z);
        float3 v2 = make_float3(faces[i + 6], faces[i + 7], faces[i + 8]) - make_float3(bbox_min.x, bbox_min.y, bbox_min.z);

        float3 e0 = v1 - v0;
        float3 e1 = v2 - v1;
        float3 e2 = v0 - v2;

        float3 n = normalize(cross(e0, e1));

        // Compute current triangle bbox
        Mesh::AABBox<float3> t_bbox{ fminf(v0, fminf(v1, v2)), fmaxf(v0, fmaxf(v1, v2)) };

        // Compute current triangle bbox in voxel grid coordinates
        Mesh::AABBox<int3> t_bbox_grid{};
        t_bbox_grid.p_min = clamp(make_int3(t_bbox.p_min / delta_p), make_int3(0, 0, 0), grid_size);
        t_bbox_grid.p_max = clamp(make_int3(t_bbox.p_max / delta_p), make_int3(0, 0, 0), grid_size);

        // SETUP STAGE
        // Compute critical point
        float3 c{};
        if (n.x > .0f) { c.x = delta_p.x; }
        if (n.y > .0f) { c.y = delta_p.y; }
        if (n.z > .0f) { c.z = delta_p.z; }

        // Compute d1 and d2 for plane overlap test
        float d1 = dot(n, c - v0);
        float d2 = dot(n, (delta_p - c) - v0);

        // Loop unrolling to prepare projection test properties
        // XY plane
        float2 e0_xy = make_float2(-1.f * e0.y, e0.x);
        float2 e1_xy = make_float2(-1.f * e1.y, e1.x);
        float2 e2_xy = make_float2(-1.f * e2.y, e2.x);
        if (n.z < 0.0f) {
            e0_xy = -e0_xy;
            e1_xy = -e1_xy;
            e2_xy = -e2_xy;
        }

        float d_e0_xy = (-1.0f * dot(e0_xy, make_float2(v0.x, v0.y)) + max(.0f, delta_p.x * e0_xy.x) + max(.0f, delta_p.y * e0_xy.y));
        float d_e1_xy = (-1.0f * dot(e1_xy, make_float2(v1.x, v1.y)) + max(.0f, delta_p.x * e1_xy.x) + max(.0f, delta_p.y * e1_xy.y));
        float d_e2_xy = (-1.0f * dot(e2_xy, make_float2(v2.x, v2.y)) + max(.0f, delta_p.x * e2_xy.x) + max(.0f, delta_p.y * e2_xy.y));

        // YZ plane
        float2 e0_yz = make_float2(-1.f * e0.z, e0.y);
        float2 e1_yz = make_float2(-1.f * e1.z, e1.y);
        float2 e2_yz = make_float2(-1.f * e2.z, e2.y);
        if (n.x < 0.0f) {
            e0_yz = -e0_yz;
            e1_yz = -e1_yz;
            e2_yz = -e2_yz;
        }

        float d_e0_yz = (-1.0f * dot(e0_yz, make_float2(v0.y, v0.z)) + max(.0f, delta_p.y * e0_yz.x) + max(.0f, delta_p.z * e0_xy.y));
        float d_e1_yz = (-1.0f * dot(e1_yz, make_float2(v1.y, v1.z)) + max(.0f, delta_p.y * e1_yz.x) + max(.0f, delta_p.z * e1_xy.y));
        float d_e2_yz = (-1.0f * dot(e2_yz, make_float2(v2.y, v2.z)) + max(.0f, delta_p.y * e2_yz.x) + max(.0f, delta_p.z * e2_xy.y));

        // ZX plane
        float2 e0_zx = make_float2(-1.f * e0.x, e0.z);
        float2 e1_zx = make_float2(-1.f * e1.x, e1.z);
        float2 e2_zx = make_float2(-1.f * e2.x, e2.z);
        if (n.y < 0.0f) {
            e0_zx = -e0_zx;
            e1_zx = -e1_zx;
            e2_zx = -e2_zx;
        }

        float d_e0_zx = (-1.0f * dot(e0_zx, make_float2(v0.z, v0.x)) + max(.0f, delta_p.x * e0_zx.x) + max(.0f, delta_p.z * e0_xy.y));
        float d_e1_zx = (-1.0f * dot(e1_zx, make_float2(v1.z, v1.x)) + max(.0f, delta_p.x * e1_zx.x) + max(.0f, delta_p.z * e1_xy.y));
        float d_e2_zx = (-1.0f * dot(e2_zx, make_float2(v2.z, v2.x)) + max(.0f, delta_p.x * e2_zx.x) + max(.0f, delta_p.z * e2_xy.y));

        // OVERLAP TEST
        // For each voxel in the triangle bbox
        for (int z = t_bbox_grid.p_min.z; z <= t_bbox_grid.p_max.z; ++z) {
            for (int y = t_bbox_grid.p_min.y; y <= t_bbox_grid.p_max.y; ++y) {
                for (int x = t_bbox_grid.p_min.x; x <= t_bbox_grid.p_max.x; ++x) {
                    // Compute minimum corner coordinates
                    float3 p = make_float3(x * spacing.x, y * spacing.y, z * spacing.z);

                    // Triangle plane overlap test
                    if (((dot(n, p) + d1) * (dot(n, p) + d2)) > .0f) { continue; }

                    // Check if the current triangle overlaps the current voxel
                    // XY plane
                    float2 p_xy = make_float2(p.x, p.y);
                    if (dot(e0_xy, p_xy) + d_e0_xy < .0f) { continue; }
                    if (dot(e1_xy, p_xy) + d_e1_xy < .0f) { continue; }
                    if (dot(e2_xy, p_xy) + d_e2_xy < .0f) { continue; }

                    // YZ plane
                    float2 p_yz = make_float2(p.y, p.z);
                    if (dot(e0_yz, p_yz) + d_e0_yz < .0f) { continue; }
                    if (dot(e1_yz, p_yz) + d_e1_yz < .0f) { continue; }
                    if (dot(e2_yz, p_yz) + d_e2_yz < .0f) { continue; }

                    // ZX plane
                    float2 p_zx = make_float2(p.z, p.x);
                    if (dot(e0_zx, p_zx) + d_e0_zx < .0f) { continue; }
                    if (dot(e1_zx, p_zx) + d_e1_zx < .0f) { continue; }
                    if (dot(e2_zx, p_zx) + d_e2_zx < .0f) { continue; }

                    // Set the current voxel as intersected
                    size_t location = (size_t)x + ((size_t)y * (size_t)dim_y) + ((size_t)z * (size_t)dim_y * (size_t)dim_z);
                    atomicAdd(&(v_table[location]), 1);
                }
            }
        }
    }
}

void kernelWrapper(Mesh::Mesh& m, const Mesh::VoxelGrid& v_grid, unsigned int* v_table, float& time) {
    unsigned int* v_table_d = nullptr;
    float* faces_d = nullptr;

    cudaEvent_t start;
    cudaEvent_t end;
    float elapsed_time;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&end));

    // Setup voxel table inside GPU
    gpuErrchk(cudaMalloc((void**)&v_table_d, sizeof(unsigned int) * (size_t)v_grid.dim_x * (size_t)v_grid.dim_y * (size_t)v_grid.dim_z));
    gpuErrchk(cudaMemcpy(v_table_d, v_table, sizeof(unsigned int) * (size_t)v_grid.dim_x * (size_t)v_grid.dim_y * (size_t)v_grid.dim_z, cudaMemcpyHostToDevice));

    // Setup mesh faces inside GPU
    //gpuErrchk(cudaMalloc((void**)&faces_d, sizeof(float) * m.faces_idx.size() * 3));
    //gpuErrchk(cudaMemcpy(faces_d, )));

    // Compute the grid and block dimensions
    dim3 block(32, 1, 1);
    dim3 grid((m.faces_idx.size() + 32 - 1) / 32, 1, 1);

    gpuErrchk(cudaEventRecord(start, 0));

    // Launch the voxelization kernel
    voxelize <<<grid, block>>> (faces_d, m.faces_idx.size(), (size_t)v_grid.dim_x, (size_t)v_grid.dim_y, (size_t)v_grid.dim_z, v_table_d, make_float3(v_grid.spacing[0], v_grid.spacing[1], v_grid.spacing[2]), make_float3(v_grid.aabb.p_min[0], v_grid.aabb.p_min[1], v_grid.aabb.p_min[2]));
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaEventRecord(end, 0));
    gpuErrchk(cudaEventSynchronize(end));
    gpuErrchk(cudaEventElapsedTime(&elapsed_time, start, end));
    time = elapsed_time;

    // Copy the GPU memory to RAM
    
    // Free the GPU memory
    gpuErrchk(cudaFree(v_table_d));
}