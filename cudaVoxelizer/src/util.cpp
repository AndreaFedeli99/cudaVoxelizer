#include "util.h"

std::vector<Vec3::Vec3> unit_cube_v = {
		Vec3::Vec3{ 0, 0, 0 },
		Vec3::Vec3{ 1, 0, 0 },
		Vec3::Vec3{ 1, 1, 0 },
		Vec3::Vec3{ 0, 1, 0 },
		Vec3::Vec3{ 0, 0, 1 },
		Vec3::Vec3{ 1, 0, 1 },
		Vec3::Vec3{ 1, 1, 1 },
		Vec3::Vec3{ 0, 1, 1 }
};

std::vector<Vec3::Vec3i> unit_cube_f = {
	Vec3::Vec3i{ 0, 3, 1 },
	Vec3::Vec3i{ 1, 3, 2 },
	Vec3::Vec3i{ 5, 4, 0 },
	Vec3::Vec3i{ 5, 0, 1 },
	Vec3::Vec3i{ 6, 5, 1 },
	Vec3::Vec3i{ 1, 2, 6 },
	Vec3::Vec3i{ 3, 6, 2 },
	Vec3::Vec3i{ 3, 7, 6 },
	Vec3::Vec3i{ 4, 3, 0 },
	Vec3::Vec3i{ 4, 7, 3 },
	Vec3::Vec3i{ 7, 4, 5 },
	Vec3::Vec3i{ 7, 5, 6 },
};

void util::writeVertex(std::ofstream& out, const Vec3::Vec3& v) {
	out << "v " << v[0] << " " << v[1] << " " << v[2] << std::endl;
}

void util::writeVertexNormal(std::ofstream& out, const Vec3::Vec3& v) {
	out << "vn " << v[0] << " " << v[1] << " " << v[2] << std::endl;
}

void util::writeFace(std::ofstream& out, const Vec3::Vec3i f) {
	out << "f " << f[0] + 1 << " " << f[1] + 1 << " " << f[2] + 1 << std::endl;
}

void util::makeCube(Mesh::Mesh& m, const Vec3::Vec3 v) {
	m = Mesh::Mesh{ unit_cube_v, unit_cube_f };
	Vec3::Vec3 spacing{ 1, 1, 1 };
	
	for (size_t i = 0; i < m.vertices.size(); ++i) {
		m.vertices[i][0] = v[0] + (spacing[0] * m.vertices[i][0]);
		m.vertices[i][1] = v[1] + (spacing[1] * m.vertices[i][1]);
		m.vertices[i][2] = v[2] + (spacing[2] * m.vertices[i][2]);
	}
}

void util::saveObj(const unsigned int* v_table, const Mesh::VoxelGrid& v_grid, const std::string filename) {
	Mesh::Mesh voxel_mesh{};
	Mesh::Mesh cube{};

	for (int x = 0; x < v_grid.dim_x; ++x) {
		for (int y = 0; y < v_grid.dim_y; ++y) {
			for (int z = 0; z < v_grid.dim_z; ++z) {
				size_t location = (size_t)x + ((size_t)y * (size_t)v_grid.dim_y) + ((size_t)z * (size_t)v_grid.dim_y * (size_t)v_grid.dim_z);
				
				if (v_table[location]) {
					Vec3::Vec3 v{ 1.f * x , 1.f * y, 1.f * z };
					makeCube(cube, v);

					voxel_mesh.append(cube);
				}
			}
		}
	}

	std::ofstream out{ filename, std::ofstream::out };
	for (size_t i = 0; i < voxel_mesh.vertices.size(); ++i) {
		writeVertex(out, voxel_mesh.vertices[i]);
	}

	for (size_t i = 0; i < voxel_mesh.normals.size(); ++i) {
		writeVertexNormal(out, voxel_mesh.normals[i]);
	}

	for (size_t i = 0; i < voxel_mesh.faces_idx.size(); ++i) {
		writeFace(out, voxel_mesh.faces_idx[i]);
	}

	out.close();
}