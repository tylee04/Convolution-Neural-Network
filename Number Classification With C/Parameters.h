#pragma once
#include "Tensor.h"
#include <stdlib.h>
#include <time.h>

typedef struct Parameter_b
{
	Tensor2d W;
	Tensor2d b;
} Parameter_b;

typedef struct Parameter
{
	Tensor2d W;
} Parameter;

double rands()
{
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if (phase == 0)
	{
		do
		{
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;
			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1 || S == 0);
		X = V1 * sqrt(-2 * log(S) / S);

	}
	else X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;
	return X;
}

Parameter CreateParameter(int h, int w)
{
	Parameter params;
	params.W = Create2dTensor(h, w, 0);

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			params.W.tensor[i][j] = rands();
		}
	}

	return params;
}

Parameter_b CreateParameter_b(int units, int input_units)
{
	Parameter_b params;
	params.W = Create2dTensor(units, input_units, 0);
	params.b = Create2dTensor(units, 1, 0);

	for (int i = 0; i < units; i++)
	{
		for (int j = 0; j < input_units; j++)
		{
			params.W.tensor[i][j] = rands();
		}
		params.b.tensor[i][0] = rands();
	}

	return params;
}

Parameter_b CreateParameter_b_init(int units, int input_units, int init)
{
	Parameter_b params;
	params.W = Create2dTensor(units, input_units, 0);
	params.b = Create2dTensor(units, 1, 0);

	for (int i = 0; i < units; i++)
	{
		for (int j = 0; j < input_units; j++)
		{
			params.W.tensor[i][j] = init;
		}
		params.b.tensor[i][0] = init;
	}

	return params;
}

int getparams_b(Parameter_b params)
{
	return params.W.Height * params.W.Width + params.b.Height * params.b.Width;
}

void PrintParams_b(Parameter_b params)
{
	printf("[Weight]\n");
	Print2dTensor(params.W);
	printf("[bias]\n");
	Print2dTensor(params.b);
}