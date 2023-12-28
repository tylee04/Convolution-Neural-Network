#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include <math.h>
#include <stdlib.h>
#include "Load DataSets.h"

void one_hot_encoding(Tensor2d y[BATCH_SIZE], Tensor2d y_out[BATCH_SIZE])
{
	for (int i = 0; i < BATCH_SIZE; i++)
	{
		y_out[i] = Create2dTensor(LABELS, 1, 0);

        y_out[i].tensor[(int)y[i].tensor[0][0]][0] = 1.0f;
	}
}

Tensor2d sigmoid(Tensor2d x) {
    Tensor2d temp = Create2dTensor(x.Height, x.Width, 0);

    for (int i = 0; i < x.Height; i++)
    {
        for (int j = 0; j < x.Width; j++)
        {
            temp.tensor[i][j] = 1 / (1 + exp(-x.tensor[i][j]));
        }
    }

    return temp;
}

Tensor2d grads_sigmoid(Tensor2d x) {
    Tensor2d temp = Create2dTensor(x.Height, x.Width, 0);
    for (int i = 0; i < x.Height; i++)
    {
        for (int j = 0; j < x.Width; j++)
        {
            temp.tensor[i][j] = x.tensor[i][j] * (1 - x.tensor[i][j]);
        }
    }
    return temp;
}

double sum_squares_error(Tensor2d y, Tensor2d t) {
    if (y.Width != 1 || t.Width != 1 || !checkshape(y, t))
    {
        printf("y: (%d, %d) || t: (%d, %d)\n", y.Height, y.Width, t.Height, t.Width);
        printf("[Sum Squares Error Error]: 두 행렬이 형상이 맞지 않거나 1차원 행렬이 아닙니다\n");
        exit(0);
    }

    Tensor2d temp = sub(y, t);
    double sum = 0;
    for (int i = 0; i < temp.Width; i++)
    {
        sum += pow(temp.tensor[i][0], 2);
    }

    return 0.5f * sum;
}

Tensor2d sum_squares_error_backward(Tensor2d predict_y, Tensor2d t)
{
    Tensor2d grads_sigmoid_mat = grads_sigmoid(predict_y);
    Tensor2d grads = sub(predict_y, t);

    grads = product(grads, grads_sigmoid_mat);

    Free2dTensor(grads_sigmoid_mat);

    return grads;
}