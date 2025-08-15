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
};

