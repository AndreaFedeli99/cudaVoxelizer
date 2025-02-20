#pragma once

#include "Mesh.h"

namespace CpuVoxelizer {
	
	void voxelizeMesh(const Mesh::VoxelGrid& v_grid, Mesh::Mesh& m, bool* v_table);
}