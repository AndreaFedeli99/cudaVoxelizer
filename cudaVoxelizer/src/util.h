#pragma once

#include "Vec3.h"
#include "Mesh.h"

#include <fstream>
#include <string>

namespace util {
	void writeVertex(std::ofstream& out, const Vec3::Vec3& v);
	void writeVertexNormal(std::ofstream& out, const Vec3::Vec3& v);
	void writeFace(std::ofstream& out, const Vec3::Vec3i f);

	void makeCube(Mesh::Mesh& m, const Vec3::Vec3 v);
	void saveObjGPU(const unsigned int* v_taable, const Mesh::VoxelGrid& v_grid, const std::string filename);
	void saveObj(const unsigned int* v_table, const Mesh::VoxelGrid& v_grid, const std::string filename);
}