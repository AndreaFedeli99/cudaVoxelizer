#include "CpuVoxelizer.h"

int clamp(const int min, const int max, const int v);

void CpuVoxelizer::voxelizeMesh(const Mesh::VoxelGrid& v_grid, Mesh::Mesh& m, unsigned int* v_table) {
	// For each triangle in the mesh
	for (size_t i = 0; i < m.vertices.size(); ++i) {
		// Translate mesh verticies to bbox origin
		m.vertices[i] = m.vertices[i] - v_grid.aabb.p_min;
	}

	Vec3::Vec3i grid_size{ v_grid.dim_x - 1, v_grid.dim_y - 1, v_grid.dim_z - 1 };

	// For each triangle in the mesh
	for (unsigned int i = 0; i < (unsigned int)m.faces_idx.size(); ++i) {
		// TRIANGLE INFORMATION
		Vec3::Vec3 v0{ m.vertices[m.faces_idx[i][0]] };
		Vec3::Vec3 v1{ m.vertices[m.faces_idx[i][1]] };
		Vec3::Vec3 v2{ m.vertices[m.faces_idx[i][2]] };

		Vec3::Vec3 edge0 = v1 - v0;
		Vec3::Vec3 edge1 = v2 - v1;
		Vec3::Vec3 edge2 = v0 - v2;

		Vec3::Vec3 n = Vec3::unit_vector(Vec3::cross(edge0, edge1));

		// Compute current triangle bbox
		Mesh::AABBox<Vec3::Vec3> t_bbox{ Vec3::min(v0, Vec3::min(v1, v2)), Vec3::max(v0, Vec3::max(v1, v2)) };
		
		// Compute current triangle bbox in voxel grid coordinates
		Mesh::AABBox<Vec3::Vec3i> t_bbox_grid{};
		
		int p_min_x = clamp(0, grid_size[0], (int)(t_bbox.p_min[0] / v_grid.spacing[0]));
		int p_min_y = clamp(0, grid_size[1], (int)(t_bbox.p_min[1] / v_grid.spacing[1]));
		int p_min_z = clamp(0, grid_size[2], (int)(t_bbox.p_min[2] / v_grid.spacing[2]));
		t_bbox_grid.p_min = Vec3::Vec3i{ p_min_x, p_min_y, p_min_z };

		int p_max_x = clamp(0, grid_size[0], (int)(t_bbox.p_max[0] / v_grid.spacing[0]));
		int p_max_y = clamp(0, grid_size[1], (int)(t_bbox.p_max[1] / v_grid.spacing[1]));
		int p_max_z = clamp(0, grid_size[2], (int)(t_bbox.p_max[2] / v_grid.spacing[2]));
		t_bbox_grid.p_max = Vec3::Vec3i{ p_max_x, p_max_y, p_max_z };

		// SETUP STAGE
		Vec3::Vec3 delta_p = v_grid.spacing;

		// Compute the critical point
		Vec3::Vec3 c{};
		if (n[0] > .0f) { c[0] = delta_p[0]; }
		if (n[1] > .0f) { c[1] = delta_p[1]; }
		if (n[2] > .0f) { c[2] = delta_p[2]; }

		// Compute d1 and d2 for plane overlap test
		float d1 = Vec3::dot(n, c - v0);
		float d2 = Vec3::dot(n, (delta_p - c) - v0);

		// OVERLAP TEST
		// For each voxel in the triangle bbox
		for (int z = t_bbox_grid.p_min[2]; z <= t_bbox_grid.p_max[2]; ++z) {
			for (int y = t_bbox_grid.p_min[1]; y <= t_bbox_grid.p_max[1]; ++y) {
				for (int x = t_bbox_grid.p_min[0]; x <= t_bbox_grid.p_max[0]; ++x) {

					// Compute minimum corner coordinates
					Vec3::Vec3 p{ x * v_grid.spacing[0], y * v_grid.spacing[1], z * v_grid.spacing[2] };

					// Triangle plane overlap test
					if (((Vec3::dot(n, p) + d1) * (Vec3::dot(n, p) + d2)) > .0f) { continue; }

					// For all 3 planes XY, YZ, ZX
					for (int j = 0; j < 3; ++j) {
						int axis0 = j;
						int axis1 = (j + 1) % 3;
						int axis2 = (j + 2) % 3;

						// Compute the projection of the each triangle's edge on the n-th plane
						Vec2::Vec2 e0_proj{ -1.f * edge0[axis1], edge0[axis0] };
						Vec2::Vec2 e1_proj{ -1.f * edge1[axis1], edge1[axis0] };
						Vec2::Vec2 e2_proj{ -1.f * edge2[axis1], edge2[axis0] };
						if (n[axis2] < .0f) {	
							e0_proj = -e0_proj;
							e1_proj = -e1_proj;
							e2_proj = -e2_proj;
						}

						float d_e0 = ( -1.0f * Vec2::dot(e0_proj, Vec2::Vec2{ v0[axis0], v0[axis1] }) ) + std::max(.0f, delta_p[axis0] * e0_proj[0]) + std::max(.0f, delta_p[axis1] * e0_proj[1]);
						float d_e1 = ( -1.0f * Vec2::dot(e1_proj, Vec2::Vec2{ v1[axis0], v1[axis1] }) ) + std::max(.0f, delta_p[axis0] * e1_proj[0]) + std::max(.0f, delta_p[axis1] * e1_proj[1]);
						float d_e2 = ( -1.0f * Vec2::dot(e2_proj, Vec2::Vec2{ v2[axis0], v2[axis1] }) ) + std::max(.0f, delta_p[axis0] * e2_proj[0]) + std::max(.0f, delta_p[axis1] * e2_proj[1]);

						// Check if the current triangle overlaps the current voxel
						Vec2::Vec2 p_proj{ p[axis0], p[axis1] };
						if (Vec2::dot(e0_proj, p_proj) + d_e0 < .0f) { continue; }
						if (Vec2::dot(e1_proj, p_proj) + d_e1 < .0f) { continue; }
						if (Vec2::dot(e2_proj, p_proj) + d_e2 < .0f) { continue; }

						// Set the current voxel as intersected
						size_t location = (size_t)x + ((size_t)y * (size_t)v_grid.dim_y) + ((size_t)z * (size_t)v_grid.dim_y * (size_t)v_grid.dim_z);
						v_table[location] = 1;

						continue;
					}
				}
			}
		}
	}
}

int clamp(const int min, const int max, const int v) {
	return std::max(min, std::min(max, v));
}