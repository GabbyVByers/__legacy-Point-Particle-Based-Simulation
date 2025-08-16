#pragma once

#include "vec2f.h"
#include "sharedarray.h"

struct Particle
{
	Vec2f position;
	Vec2f velocity;
};

struct GlobalState
{
	int width = -1;
	int height = -1;
	uchar4* pixels = nullptr;
	SharedArray<Particle> particles;
};

