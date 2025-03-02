#pragma once

#include <cmath>
#include <iostream>

namespace Vec3
{
	class Vec3 {
	public:
		Vec3() : coords{ .0f, .0f, .0f } {}
		Vec3(const float x, const float y, const float z) : coords{ x, y, z } {}

		float x() const { return coords[0]; }
		float y() const { return coords[1]; }
		float z() const { return coords[2]; }

		Vec3 operator-() const { return Vec3(-coords[0], -coords[1], -coords[2]); }
		float operator[](unsigned int i) const { return coords[i]; }
		float& operator[](unsigned int i) { return coords[i]; }

		Vec3& operator=(const Vec3& v) {
			if (this == &v)
				return *this;
			coords[0] = v[0];
			coords[1] = v[1];
			coords[2] = v[2];
			return *this;
		}

		Vec3& operator+=(const Vec3& v) {
			coords[0] += v.x();
			coords[1] += v.y();
			coords[2] += v.z();
			return *this;
		}

		Vec3& operator*=(const float t) {
			coords[0] *= t;
			coords[1] *= t;
			coords[2] *= t;
			return *this;
		}

		Vec3& operator/=(const float t) {
			return *this *= 1 / t;
		}

		float length() const { return sqrtf(length_squared()); }

		float length_squared() const { return coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2]; }

	private:
		float coords[3];
	};

	typedef struct Vec3i {
		Vec3i() : elem{ 0, 0, 0 } {}
		Vec3i(const int x, const int y, const int z) : elem{ x, y, z } {}

		int operator[](const int i) const { return elem[i]; }
		int& operator[](const int i) { return elem[i]; }

		Vec3i& operator+=(const int x) {
			elem[0] += x;
			elem[1] += x;
			elem[2] += x;
			return *this;
		}

		Vec3i& operator-=(const int x) {
			elem[0] -= x;
			elem[1] -= x;
			elem[2] -= x;
			return *this;
		}

		int elem[3];
	} Vec3i;

	inline std::ostream& operator<<(std::ostream& out, const Vec3& v) {
		return out << v[0] << ' ' << v[1] << ' ' << v[2];
	}

	inline Vec3 operator+(const Vec3& v1, const Vec3& v2) {
		return Vec3(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
	}

	inline Vec3 operator-(const Vec3& v1, const Vec3& v2) {
		return Vec3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
	}

	inline Vec3 operator*(const float t, const Vec3& v) {
		return Vec3(t * v[0], t * v[1], t * v[2]);
	}

	inline Vec3 operator*(const Vec3& v, const float t) {
		return t * v;
	}

	inline Vec3 operator/(const Vec3& v, const float t) {
		return (1 / t) * v;
	}

	inline float dot(const Vec3& v1, const Vec3& v2) {
		return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
	}

	inline Vec3 cross(const Vec3& v1, const Vec3& v2) {
		return Vec3((v1[1] * v2[2]) - (v1[2] * v2[1]),
					(v1[2] * v2[0]) - (v1[0] * v2[2]),
					(v1[0] * v2[1]) - (v1[1] * v2[0]));
	}

	inline Vec3 unit_vector(const Vec3& v) {
		return v / v.length();
	}

	inline Vec3 min(const Vec3& v1, const Vec3& v2) {
		return Vec3{ std::min(v1[0], v2[0]), std::min(v1[1], v2[1]), std::min(v1[2], v2[2])};
	}

	inline Vec3 max(const Vec3& v1, const Vec3& v2) {
		return Vec3{ std::max(v1[0], v2[0]), std::max(v1[1], v2[1]), std::max(v1[2], v2[2]) };
	}
}