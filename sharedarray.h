#pragma once

#include "cuda_runtime.h"

template<typename type>

class SharedArray
{
private:

	size_t capacity = 1;

public:

	size_t size = 0;
	type* hostPointer = nullptr;
	type* devicePointer = nullptr;

	SharedArray()
	{
		hostPointer = new type[capacity];
		cudaMalloc((void**)&devicePointer, sizeof(type) * capacity);
	}

	void doubleCapacity()
	{
		type* newHostPointer = nullptr;
		type* newDevicePointer = nullptr;

		newHostPointer = new type[capacity * 2];
		cudaMalloc((void**)&newDevicePointer, sizeof(type) * capacity * 2);

		memcpy(newHostPointer, hostPointer, sizeof(type) * capacity);
		cudaMemcpy(newDevicePointer, devicePointer, sizeof(type) * capacity, cudaMemcpyDeviceToDevice);

		delete[] hostPointer;
		cudaFree(devicePointer);

		hostPointer = newHostPointer;
		devicePointer = newDevicePointer;

		capacity = capacity * 2;
	}

	void remove(size_t index)
	{
		if (index >= size)
			return;

		if (index == size - 1)
		{
			size--;
			return;
		}

		for (size_t i = index + 1; i < size; i++)
		{
			hostPointer[i - 1] = hostPointer[i];
		}
		size--;
	}

	void add(type element)
	{
		if (size == capacity)
			doubleCapacity();

		hostPointer[size] = element;
		size++;
	}

	void clear()
	{
		delete[] hostPointer;
		cudaFree(devicePointer);

		size = 0;
		capacity = 1;

		hostPointer = new type[capacity];
		cudaMalloc((void**)&devicePointer, sizeof(type) * capacity);
	}

	void updateHostToDevice()
	{
		cudaMemcpy(devicePointer, hostPointer, size * sizeof(type), cudaMemcpyHostToDevice);
	}

	void updateDeviceToHost()
	{
		cudaMemcpy(hostPointer, devicePointer, size * sizeof(type), cudaMemcpyDeviceToHost);
	}

	void free()
	{
		delete[] hostPointer;
		cudaFree(devicePointer);
	}
};

