#include "Vec3.h"
#include "Mesh.h"

/*
* Disable error/warnings when using experimental/filesystem library
*/
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <experimental/filesystem>

const std::string PROGRAM_NAME = "#CudaVoxelizer - ";

bool loadMesh(const std::experimental::filesystem::path& file_path, Mesh& m, std::string& log_msg);

int main(int argc, char *argv[]) {
	std::string msg{};

	if (argc != 3) {
		std::cerr << PROGRAM_NAME << "ERROR::VOXELIZER::wrong number of parameters." << std::endl << "\tPlease insert 2 parameters" << std::endl;
		return 1;
	}

	std::experimental::filesystem::path filepath{ argv[1] };
	std::string vox_filename{ argv[2] };

	std::cout << PROGRAM_NAME << "Loading mesh from " << filepath.filename().string() << "..." << std::endl;

	Mesh m{};
	if (!loadMesh(filepath, m, msg)) {
		std::cerr << PROGRAM_NAME << msg;
		return 1;
	}

	std::cout << PROGRAM_NAME << filepath.filename().string() << " (" << m.faces_idx.size() << " triangles) succesfully loaded..." << std::endl;

	std::cout << PROGRAM_NAME << m.get_AABBox() << std::endl;



	std::cin.get();
	return 0;
}

bool loadMesh(const std::experimental::filesystem::path& file_path, Mesh& m, std::string& msg) {
	std::string line{};
	std::ifstream fs{};
	std::string token{};
	bool ok{ true };

	if (ok) {
		if (file_path.has_extension() && file_path.extension().string() != ".obj") {
			msg = PROGRAM_NAME + "ERROR::VOXELIZER::wrong file format.\n\tPlease provide an.obj file\n";
			ok = false;
		}
	}

	if (ok) {
		fs.open(file_path);
		if (!fs.good()) {
			msg = PROGRAM_NAME + "ERROR::VOXELIZER::file doesn't exist.\n\tPlease provide a valid path\n";
			ok = false;
		}
	}

	if (ok) {
		while (!std::getline(fs, line).eof()) {
			std::istringstream ss{ line };
			ss >> token;

			// Vertices
			if (token == "v") {
				Vec3::Vec3 vert{};
				ss >> vert[0] >> vert[1] >> vert[2];
				m.vertices.push_back(vert);
			}
			// Texture coordinates
			else if (token == "vt") {
				Vec2::Vec2 text_coords{};
				ss >> text_coords[0] >> text_coords[1];
				m.tex_coords.push_back(text_coords);
			}
			// Faces
			else if (token == "f") {
				Vec3::Vec3i face_idx{};
				Vec3::Vec3i text_idx{};
				std::string indeces{};
				int idx{ 0 };

				// Read the indeces of the face
				std::getline(ss, indeces);

				std::istringstream idx_token_ss{ indeces };

				// Cycle the face indeces
				for (unsigned int i = 0; i < 3; ++i) {
					idx_token_ss >> indeces;
					std::replace(indeces.begin(), indeces.end(), '/', ' ');
					std::istringstream idx_ss{ indeces };

					// vertex idx
					idx_ss >> idx;
					if (idx > 0)
						face_idx[i] = idx - 1;
					else
						face_idx[i] = (unsigned int)m.vertices.size() + idx;

					// If texture coords are specified
					if (m.tex_coords.size() != 0) {
						// texture idx
						idx_ss >> idx;
						if (idx > 0)
							text_idx[i] = idx - 1;
						else
							text_idx[i] = (unsigned int)m.vertices.size() + idx;
					}

					// Reset the stringstream
					idx_ss.str("");
					idx_ss.clear();
				}

				// If there are still more then 3 vertices
				if (idx_token_ss.rdbuf()->in_avail() != 0) {
					msg = PROGRAM_NAME + "ERROR::VOXELIZER::not triangular mesh.\n\tPlease provide a triangular mesh\n";
					ok = false;
					idx_token_ss.clear();
					break;
				}
				else {
					m.faces_idx.push_back(face_idx);

					if (m.tex_coords.size() != 0)
						m.tex_idx.push_back(text_idx);
				}

				// Reset the stringstream
				idx_token_ss.str("");
				idx_token_ss.clear();
			}
		
			ss.str("");
			ss.clear();
		}
	}

	if (ok)
		m.compute_normals();

	fs.close();

	return ok;
}

