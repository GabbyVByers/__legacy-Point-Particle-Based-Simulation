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

	Vec2f mousePos;
	bool isMouseLeft = false;

	float pauliExclusionPower = 12.0f;
	float attractiveDispersionPower = 6.0f;
	float interactionRadius = 0.01f;
	float forceScaling = 0.1f;
	float velocityDampening = 0.99f;
	float mouseAttraction = 0.001f;
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

