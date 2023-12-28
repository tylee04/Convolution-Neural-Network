#pragma once
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#define EXTRA_MEMORY 64 //메모리해제시 여분의 메모리를 할당하여 런타임 오류방지

typedef struct Tensor2d
{
	double** tensor;
	int Height;
	int Width;
} Tensor2d;

Tensor2d Create2dTensor(int h, int w, double init)
{
	Tensor2d tensor;
	tensor.Height = h;
	tensor.Width = w;

	tensor.tensor = (double**)malloc(sizeof(double*) * h + EXTRA_MEMORY);
	for (int i = 0; i < h; i++)
	{
		tensor.tensor[i] = (double*)malloc(sizeof(double) * w + EXTRA_MEMORY);
	}

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			tensor.tensor[i][j] = init;
		}
	}

	return tensor;
}

void Print2dTensor(Tensor2d tensor)
{
	printf("{ ");
	for (int i = 0; i < tensor.Height; i++)
	{
		if (i != 0)
		{
			printf("  ");
		}
		printf("{ ");
		for (int j = 0; j < tensor.Width; j++)
		{
			printf("%10f, ", tensor.tensor[i][j]);
		}
		printf("}");
		if (i != tensor.Height - 1)
		{
			printf(",\n");
		}
	}
	printf(" }\n");
}

void Print2dTensor_Shape(Tensor2d tensor)
{
	printf("(%d, %d)\n", tensor.Height, tensor.Width);
}

void Free2dTensor(Tensor2d tensor)
{
	for (int i = 0; i < tensor.Height; i++)
	{
		free(tensor.tensor[i]);
	}
	free(tensor.tensor);
}

bool checkshape_mul(Tensor2d  a, Tensor2d b)
{
	if (a.Width != b.Height)
	{
		printf("a: (%d, %d) || b: (%d, %d)\n", a.Height, a.Width, b.Height, b.Width);
		return false;
	}
	return true;
}

bool checkshape(Tensor2d a, Tensor2d b)
{
	if (a.Height != b.Height || a.Width != b.Width)
	{
		printf("a: (%d, %d) || b: (%d, %d)\n", a.Height, a.Width, b.Height, b.Width);
		return false;
	}
	return true;
}

Tensor2d multensor(Tensor2d a, Tensor2d b)
{
	if (!checkshape_mul(a, b))
	{
		printf("[Tensor Multiplication Error]: 두 행렬의 행과 열이 맞지 않습니다\n");
		exit(0);
	}

	Tensor2d temp = Create2dTensor(a.Height, b.Width, 0);

	for (int i = 0; i < a.Height; i++)
	{
		for (int j = 0; j < b.Width; j++)
		{
			double sum = 0;
			for (int k = 0; k < a.Width; k++)
			{
				sum += a.tensor[i][k] * b.tensor[k][j];
			}
			temp.tensor[i][j] = sum;
		}
	}

	return temp;
}

Tensor2d product(Tensor2d a, Tensor2d b)
{
	if (!checkshape(a, b))
	{
		printf("[Hadamard Product Error]: 두 행렬의 형상이 맞지 않습니다\n");
		exit(0);
	}

	Tensor2d temp = Create2dTensor(a.Height, a.Width, 0);

	for (int i = 0; i < a.Height; i++)
	{
		for (int j = 0; j < a.Width; j++)
		{
			temp.tensor[i][j] = a.tensor[i][j] * b.tensor[i][j];
		}
	}

	return temp;
}

Tensor2d sub(Tensor2d a, Tensor2d b)
{
	if (!checkshape(a, b))
	{
		printf("[Tensor Subtraction Error]: 두 행렬의 형상이 맞지 않습니다\n");
		exit(0);
	}

	Tensor2d temp = Create2dTensor(a.Height, b.Width, 0);

	for (int i = 0; i < a.Height; i++)
	{
		for (int j = 0; j < b.Width; j++)
		{
			temp.tensor[i][j] = a.tensor[i][j] - b.tensor[i][j];
		}
	}

	return temp;
}

Tensor2d add(Tensor2d a, Tensor2d b)
{
	if (!checkshape(a, b))
	{
		printf("[Tensor Addition Error]: 두 행렬의 형상이 맞지 않습니다\n");
		exit(0);
	}

	Tensor2d temp = Create2dTensor(a.Height, b.Width, 0);

	for (int i = 0; i < a.Height; i++)
	{
		for (int j = 0; j < b.Width; j++)
		{
			temp.tensor[i][j] = a.tensor[i][j] + b.tensor[i][j];
		}
	}

	return temp;
}

Tensor2d transposed(Tensor2d tensor)
{
	Tensor2d temp = Create2dTensor(tensor.Width, tensor.Height, 0);

	for (int i = 0; i < tensor.Width; i++)
	{
		for (int j = 0; j < tensor.Height; j++)
		{
			temp.tensor[i][j] = tensor.tensor[j][i];
		}
	}

	return temp;
}

Tensor2d broadcasting(Tensor2d x, int num, int axis)
{
	Tensor2d temp;
	if (axis == 0)// 행으로 브로드캐스팅
	{
		temp = Create2dTensor(x.Height, x.Width * num, 0);
		for (int i = 0; i < temp.Height; i++)
		{
			for (int j = 0; j < temp.Width; j++)
			{
				temp.tensor[i][j] = x.tensor[i % x.Height][j % x.Width];
			}
		}
	}
	else if (axis == 1)// 열으로 브로드캐스팅
	{
		temp = Create2dTensor(x.Height * num, x.Width, 0);
		for (int i = 0; i < temp.Height; i++)
		{
			for (int j = 0; j < temp.Width; j++)
			{
				temp.tensor[i][j] = x.tensor[i % x.Height][j % x.Width];
			}
		}
	}
	else
	{
		printf("[Broadcasting Error]: Wrong axis\n");
		exit(0);
	}

	return temp;
}

Tensor2d* tensor_division(Tensor2d a, int division_num, int axis)
{
	Tensor2d *temp = (Tensor2d*)malloc(sizeof(Tensor2d*) * division_num + EXTRA_MEMORY);
	if (axis == 0)
	{
		if (a.Width % division_num != 0)
		{
			Print2dTensor_Shape(a);
			printf("[Division Error]: Width of Matrix doesn't divide by %d\n", division_num);
			exit(0);
		}
		for (int i = 0; i < division_num; i++)
		{
			temp[i] = Create2dTensor(a.Height, a.Width / division_num, 0);
			for (int j = 0; j < a.Height; j++)
			{
				for (int k = 0; k < a.Width / division_num; k++)
				{
					//printf("[%d][%d][%d] [%d][%d]\n", i, j, k, j, i * (division_num + 1) + k);
					temp[i].tensor[j][k] = a.tensor[j][i * (temp[i].Width) + k];
				}
			}
		}
	}
	else if (axis == 1)
	{
		if (a.Height % division_num != 0)
		{
			Print2dTensor(a);
			printf("[Division Error]: Height of Matrix doesn't divide by %d\n", division_num);
			exit(0);
		}
		for (int i = 0; i < division_num; i++)
		{
			temp[i] = Create2dTensor(a.Height / division_num, a.Width, 0);
			for (int j = 0; j < a.Height / division_num; j++)
			{
				for (int k = 0; k < a.Width; k++)
				{
					//printf("[%d][%d]\n", i * (temp[i].Height) + j, k);
					temp[i].tensor[j][k] = a.tensor[i * (a.Height / division_num) + j][k];
				}
			}
		}
	
	}
	else
	{
		printf("[Tensor Division Error]: Wrong axis\n");
		exit(0);
	}

	return temp;
}