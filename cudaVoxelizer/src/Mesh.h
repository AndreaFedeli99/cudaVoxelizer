#pragma once

#include <vector>
#include <array>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Vec2.h"
#include "Vec3.h"

namespace Mesh {

	template <typename T> struct AABBox {
		__host__ __device__ AABBox<T>() : p_min{ T{} }, p_max{ T{} } {};
		__host__ __device__ AABBox<T>(const T p_min, const T p_max) : p_min{ p_min }, p_max{ p_max } {};

		T p_min;
		T p_max;
	};

	template <typename T> inline std::ostream& operator<<(std::ostream& os, const AABBox<T>& aabb) {
		os << "Min: " << aabb.p_min << "\tMax: " << aabb.p_max;
		return os;
	}

	class Mesh {
	public:
		Mesh();
		Mesh(const std::vector<Vec3::Vec3> vert, const std::vector<Vec3::Vec3i> face_idx);
		Mesh(const std::vector<Vec3::Vec3> vert, const std::vector<Vec3::Vec3i> face_idx, const std::vector<Vec2::Vec2> tex, const std::vector<Vec3::Vec3i> tex_idx);
		Mesh(const Mesh& other);
		Mesh(Mesh&& other) noexcept;

		std::vector<std::array<Vec3::Vec3, 3>> get_faces() const;
		void compute_normals();
		void append(const Mesh& m);
		AABBox<Vec3::Vec3> get_AABBox() const;

		Mesh& operator=(const Mesh& m);
		Mesh& operator=(Mesh&& m) noexcept;

		std::vector<Vec3::Vec3> vertices;
		std::vector<Vec3::Vec3i> faces_idx;
		std::vector<Vec3::Vec3> normals;
		std::vector<Vec2::Vec2> tex_coords;
		std::vector<Vec3::Vec3i> tex_idx;
	};

	struct VoxelGrid {
		VoxelGrid();
		VoxelGrid(const AABBox<Vec3::Vec3> bbox, const int dim_x, const int dim_y, const int dim_z, const Vec3::Vec3 spacing);

		AABBox<Vec3::Vec3> aabb;
		int dim_x, dim_y, dim_z;
		Vec3::Vec3 spacing;
	};
}