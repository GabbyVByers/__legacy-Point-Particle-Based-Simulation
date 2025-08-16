#pragma once

#include "vec2f.h"
#include "sharedarray.h"
#include "utilities.h"

struct Particle
{
	Vec2f pos;
	Vec2f vel;
};

struct GlobalState
{
	int width = -1;
	int height = -1;
	uchar4* pixels = nullptr;
	SharedArray<Particle> particles;
};

inline void initParticles(GlobalState& globalState, int numParticles)
{
	for (int i = 0; i < numParticles; i++)
	{
		Particle particle;
		particle.pos = randVec2f(-1.0f, 1.0f);
		particle.vel = randVec2f(-0.01f, 0.01f);
		globalState.particles.add(particle);
	}
	globalState.particles.updateHostToDevice();
}

