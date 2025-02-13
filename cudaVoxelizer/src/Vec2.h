#pragma once

#include <iostream>
#include <cmath>

namespace Vec2 
{
	class Vec2 {
	public:
		Vec2() : coords{ 0, 0 } {}
		Vec2(float x, float y) : coords{ x, y } {}

		float x() const { return coords[0]; }
		float y() const { return coords[1]; }

		Vec2 operator-() const { return Vec2(-coords[0], -coords[1]); }
		float operator[](unsigned int i) const { return coords[i]; }
		float& operator[](unsigned int i) { return coords[i]; }

		Vec2& operator+=(const Vec2& v) {
			coords[0] += v.x();
			coords[1] += v.y();
			return *this;
		}

		Vec2& operator*=(const float t) {
			coords[0] *= t;
			coords[1] *= t;
			return *this;
		}

		Vec2& operator/=(const float t) {
			return *this *= 1 / t;
		}

		float length() const { return sqrtf(length_squared()); }

		float length_squared() const { return coords[0] * coords[0] + coords[1] * coords[1]; }

	private:
		float coords[2];
	};

	typedef struct Vec2i {
		Vec2i() : elem{ 0, 0 } {}
		Vec2i(const unsigned int x, const unsigned int y, const unsigned int z) : elem{ x, y } {}

		unsigned int operator[](const unsigned int i) const { return elem[i]; }
		unsigned int& operator[](const unsigned int i) { return elem[i]; }

		Vec2i& operator+=(const unsigned int x) {
			elem[0] += x;
			elem[1] += x;
			return *this;
		}

		Vec2i& operator-=(const unsigned int x) {
			elem[0] -= x;
			elem[1] -= x;
			return *this;
		}

		unsigned int elem[2];
	} Vec2i;

	inline std::ostream& operator<<(std::ostream& out, const Vec2& v) {
		return out << v[0] << ' ' << v[1];
	}

	inline Vec2 operator+(const Vec2& v1, const Vec2& v2) {
		return Vec2(v1[0] + v2[0], v1[1] + v2[1]);
	}

	inline Vec2 operator-(const Vec2& v1, const Vec2& v2) {
		return Vec2(v1[0] - v2[0], v1[1] - v2[1]);
	}

	inline Vec2 operator*(const float t, const Vec2& v) {
		return Vec2(t * v[0], t * v[1]);
	}

	inline Vec2 operator*(const Vec2& v, const float t) {
		return t * v;
	}

	inline Vec2 operator/(const Vec2& v, const float t) {
		return (1 / t) * v;
	}

	inline float dot(const Vec2& v1, const Vec2& v2) {
		return v1[0] * v2[0] + v1[1] * v2[1];
	}

}