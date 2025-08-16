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

	float interactionRadius = 0.52f;
	float forceScale = 0.000005f;
	float velocityDampening = 0.95f;
};

inline void initParticles(GlobalState& globalState, int numParticles)
{
	for (int i = 0; i < numParticles; i++)
	{
		Particle part;
		part.pos = randVec2f(-1.0f, 1.0f);
		part.vel = randVec2f(-0.0001f, 0.0001f);
		globalState.particles.add(part);
	}
	globalState.particles.updateHostToDevice();
}

