#pragma once

#include "Mesh.h"

#include <cmath>

namespace CpuVoxelizer {
	
	void voxelizeMesh(const Mesh::VoxelGrid& v_grid, Mesh::Mesh& m, unsigned int* v_table);
}