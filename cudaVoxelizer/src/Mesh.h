#pragma once

#include <vector>
#include <array>
#include <iostream>

#include "Vec2.h"
#include "Vec3.h"

struct AABBox;

class Mesh {
public:
	Mesh();
	Mesh(std::vector<Vec3::Vec3> vert, std::vector<Vec3::Vec3i> face_idx);
	Mesh(std::vector<Vec3::Vec3> vert, std::vector<Vec3::Vec3i> face_idx, std::vector<Vec2::Vec2> tex, std::vector<Vec3::Vec3i> tex_idx);
	Mesh(const Mesh& other);
	Mesh(Mesh&& other) noexcept;

	std::vector<std::array<Vec3::Vec3, 3>> const get_faces();
	void compute_normals();
	void append(const Mesh& m);
	AABBox get_AABBox();

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
	AABBox(Vec3::Vec3 p_min, Vec3::Vec3 p_max);

	float get_length_x();
	float get_length_y();
	float get_length_z();

	Vec3::Vec3 p_min;
	Vec3::Vec3 p_max;
};

inline std::ostream& operator<<(std::ostream& os, const AABBox& aabb) {
	os << "Min: " << aabb.p_min << "\tMax: " << aabb.p_max;
	return os;
}