#include "Mesh.h"

#include <vector>
#include <array>

Mesh::Mesh::Mesh() : vertices{}, faces_idx{}, normals{}, tex_coords{}, tex_idx{} {}
Mesh::Mesh::Mesh(const std::vector<Vec3::Vec3> vert, const std::vector<Vec3::Vec3i> face_idx) : vertices{ vert }, faces_idx{ face_idx }, normals{}, tex_coords{}, tex_idx{} {
	this->compute_normals();
}
Mesh::Mesh::Mesh(const std::vector<Vec3::Vec3> vert, const std::vector<Vec3::Vec3i> face_idx, const std::vector<Vec2::Vec2> tex, const std::vector<Vec3::Vec3i> tex_idx) : vertices{ vert }, faces_idx{ face_idx }, normals{}, tex_coords{ tex }, tex_idx{ tex_idx } {
	this->compute_normals();
}
Mesh::Mesh::Mesh(Mesh&& other) noexcept : vertices{ std::move(other.vertices) }, faces_idx{ std::move(other.faces_idx) }, normals{ std::move(other.normals) }, tex_coords{ std::move(other.tex_coords) }, tex_idx{ std::move(other.tex_idx) } {}
Mesh::Mesh::Mesh(const Mesh& other) : vertices{ other.vertices }, faces_idx{ other.faces_idx }, normals{ other.normals }, tex_coords{ other.tex_coords }, tex_idx{ other.tex_idx } {}

std::vector <std::array<Vec3::Vec3, 3>> Mesh::Mesh::get_faces() const {
	std::vector<std::array<Vec3::Vec3, 3>> faces{};

	faces.reserve(this->faces_idx.size());
	for (Vec3::Vec3i idx : this->faces_idx) {
		std::array<Vec3::Vec3, 3> triangle = { Vec3::Vec3{}, Vec3::Vec3{}, Vec3::Vec3{}, };
		for (unsigned int i = 0; i < 3; ++i) {
			triangle[i] = this->vertices[idx[i]];
		}
		faces.push_back(triangle);
	}

	return faces;
}

void Mesh::Mesh::compute_normals() {
	Vec3::Vec3 norms[3] = { Vec3::Vec3{}, Vec3::Vec3{}, Vec3::Vec3{} };
	Vec3::Vec3 v1{};
	Vec3::Vec3 v2;

	this->normals.resize(this->vertices.size(), Vec3::Vec3{ .0f, .0f, .0f });

	for (Vec3::Vec3i idx : this->faces_idx) {
		for (unsigned int i = 0; i < 3; ++i) {
			v1 = this->vertices[idx[(i + 1) % 3]] - this->vertices[idx[i]];
			v2 = this->vertices[idx[(i + 2) % 3]] - this->vertices[idx[i]];
			norms[i] = Vec3::cross(v1, v2);
			this->normals[idx[i]] += norms[i];
		}
	}

	for (size_t i = 0; i < this->vertices.size(); ++i) {
		this->normals[i] = Vec3::unit_vector(this->normals[i]);
	}

	return;
}

void Mesh::Mesh::append(const Mesh& m) {
	const unsigned int offset = (unsigned int)this->vertices.size();

	this->vertices.insert(this->vertices.end(), m.vertices.begin(), m.vertices.end());
	this->faces_idx.insert(this->faces_idx.end(), m.faces_idx.begin(), m.faces_idx.end());
	this->normals.insert(this->normals.end(), m.normals.begin(), m.normals.end());
	this->tex_coords.insert(this->tex_coords.end(), m.tex_coords.begin(), m.tex_coords.end());
	this->tex_idx.insert(this->tex_idx.end(), m.tex_idx.begin(), m.tex_idx.end());

	for (size_t i = this->faces_idx.size() - m.faces_idx.size(); i < this->faces_idx.size(); ++i) {
		this->faces_idx[i] += offset;
		
		if (this->tex_idx.size() > 0)
			this->tex_idx[i] += offset;
	}

	return;
}

Mesh::AABBox<Vec3::Vec3> Mesh::Mesh::get_AABBox() const {
	Vec3::Vec3 v_min{ vertices[0] };
	Vec3::Vec3 v_max{ vertices[0] };

	for (auto v : this->vertices) {
		for (unsigned int i = 0; i < 3; ++i) {
			if (v[i] < v_min[i])
				v_min[i] = v[i];
			if (v[i] > v_max[i])
				v_max[i] = v[i];
		}
	}

	return AABBox<Vec3::Vec3>{ v_min, v_max };
}

Mesh::Mesh& Mesh::Mesh::operator=(const Mesh& m) {
	this->vertices = m.vertices;
	this->faces_idx = m.faces_idx;
	this->normals = m.normals;
	this->tex_coords = m.tex_coords;
	this->tex_idx = m.tex_idx;
	return *this;
}

Mesh::Mesh& Mesh::Mesh::operator=(Mesh&& m) noexcept {
	if (this != &m) {
		this->vertices = std::move(m.vertices);
		this->faces_idx = std::move(m.faces_idx);
		this->normals = std::move(m.normals);
		this->tex_coords = std::move(m.tex_coords);
		this->tex_idx = std::move(m.tex_idx);
	}
	return *this;
}

Mesh::VoxelGrid::VoxelGrid() : aabb{ AABBox<Vec3::Vec3>{} }, dim_x{ 1 }, dim_y{ 1 }, dim_z{ 1 }, spacing{ Vec3::Vec3( 1.f, 1.f, 1.f )} {}
Mesh::VoxelGrid::VoxelGrid(const AABBox<Vec3::Vec3> bbox, const int dim_x, const int dim_y, const int dim_z, const Vec3::Vec3 spacing) 
	: aabb{ bbox }, dim_x{ dim_x }, dim_y{ dim_y }, dim_z{ dim_z }, spacing{ spacing } {}