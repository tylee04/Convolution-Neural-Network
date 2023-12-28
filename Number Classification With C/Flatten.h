#pragma once
#include "Tensor.h"

Tensor2d flatten_matrix(Tensor2d tensor)
{
	Tensor2d temp = Create2dTensor(1, tensor.Height * tensor.Width, 0);
	int idx = 0;
	for (int i = 0; i < tensor.Height; i++)
	{
		for (int j = 0; j < tensor.Width; j++)
		{
			temp.tensor[0][idx] = tensor.tensor[i][j];
			idx++;
		}
	}

	return temp;
}

Tensor2d union_tensor(Tensor2d a, Tensor2d b, int axis)
{
	Tensor2d temp;
	if (axis == 0)// 행으로 브로드캐스팅
	{
		temp = Create2dTensor(a.Height, a.Width + b.Width, 0);
		for (int i = 0; i < temp.Height; i++)
		{
			for (int j = 0; j < temp.Width; j++)
			{
				if (j + 1 > a.Width)
				{
					temp.tensor[i][j] = b.tensor[i][j % b.Width];
				}
				else
				{
					temp.tensor[i][j] = a.tensor[i][j];
				}
			}
		}
	}
	else if (axis == 1)// 열으로 브로드캐스팅
	{
		temp = Create2dTensor(a.Height + b.Height, a.Width, 0);
		for (int i = 0; i < temp.Height; i++)
		{
			for (int j = 0; j < temp.Width; j++)
			{
				if (i + 1 > a.Height)
				{
					temp.tensor[i][j] = b.tensor[i % b.Height][j];
				}
				else
				{
					temp.tensor[i][j] = a.tensor[i][j];
				}
			}
		}
	}
	else
	{
		printf("[Tensor Union Error]: Wrong axis\n");
		exit(0);
	}

	return temp;
}

Tensor2d flatten(Tensor2d *mat, int mat_size)
{
	Tensor2d temp = flatten_matrix(mat[0]);
	for (int i = 1; i < mat_size; i++)
	{
		temp = union_tensor(temp, flatten_matrix(mat[i]), 0);
	}
	return temp;
}

Tensor2d un_flatten(Tensor2d flatten_tensor, int width, int height)
{
	Tensor2d temp = Create2dTensor(width, height, 0);
	if (flatten_tensor.Height == 1 && flatten_tensor.Width % width == 0)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				temp.tensor[i][j] = flatten_tensor.tensor[0][i * width + j];
			}
		}
	}
	else if (flatten_tensor.Width == 1)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				temp.tensor[i][j] = flatten_tensor.tensor[i * width + j][0];
			}
		}
	}
	else
	{
		printf("[Un-flatten Error]: Input Tensor shape has wrong shape\n");
		exit(0);
	}
	return temp;
}
