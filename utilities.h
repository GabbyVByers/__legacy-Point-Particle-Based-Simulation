#pragma once



inline float randf(float min, float max)
{
    return ((rand() / (float)RAND_MAX) * (max - min)) + min;
}

inline Vec2f randVec2f(float min, float max)
{
    float x = randf(min, max);
    float y = randf(min, max);
    return { x, y };
}