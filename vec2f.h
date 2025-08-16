#pragma once

struct Vec2f
{
    float x = 0.0f;
    float y = 0.0f;

    __host__ __device__ Vec2f operator + (const Vec2f& vec) const
    {
        return
        {
            x + vec.x,
            y + vec.y
        };
    }

    __host__ __device__ Vec2f operator * (const float& value) const
    {
        return
        {
            x * value,
            y * value
        };
    }
};

inline float length(const Vec2f& vec)
{
    float lenSq = (vec.x * vec.x) + (vec.y * vec.y);
    return sqrt(lenSq);
}

inline void normalize(Vec2f& vec)
{
    float len = length(vec);
    vec.x /= len;
    vec.y /= len;
}