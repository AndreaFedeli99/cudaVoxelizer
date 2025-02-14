#pragma once

#include <vector>
#include <array>
#include <iostream>

#include "Vec2.h"
#include "Vec3.h"

namespace Mesh {

	struct AABBox;

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
		AABBox get_AABBox() const;

		Mesh& operator=(const Mesh& m);
		Mesh& operator=(Mesh&& m) noexcept;

		std::vector<Vec3::Vec3> vertices;
		std::vector<Vec3::Vec3i> faces_idx;
		std::vector<Vec3::Vec3> normals;
		std::vector<Vec3::Vec3i> normal_idx;
		std::vector<Vec2::Vec2> tex_coords;
		std::vector<Vec3::Vec3i> tex_idx;
	};

	struct AABBox {
		AABBox();
		AABBox(const Vec3::Vec3 p_min, const Vec3::Vec3 p_max);

		float get_length_x() const;
		float get_length_y() const;
		float get_length_z() const;

		Vec3::Vec3 p_min;
		Vec3::Vec3 p_max;
	};

	inline std::ostream& operator<<(std::ostream& os, const AABBox& aabb) {
		os << "Min: " << aabb.p_min << "\tMax: " << aabb.p_max;
		return os;
	}

	struct VoxelGrid {
		VoxelGrid();
		VoxelGrid(const AABBox bbox, const unsigned int dim_x, const unsigned int dim_y, const unsigned int dim_z, const float spacing);

		AABBox aabb;
		unsigned int dim_x, dim_y, dim_z;
		float spacing;
	};
}