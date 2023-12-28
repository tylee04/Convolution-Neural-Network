#pragma once
#include "Parameters.h"
#include "functions.h"
#include "Tensor.h"
#include "Conv2d.h"
typedef struct MaxPooling2d
{
	int features;
	int pooling_h;
	int pooling_w;
	Tensor2d *pooling;
	Tensor2d *max_map;
} MaxPooling2d;

MaxPooling2d maxpooling2d(int features, int pooling_h, int pooling_w)
{
	MaxPooling2d maxpooling2d;
	maxpooling2d.features = features;
	maxpooling2d.pooling_h = pooling_h;
	maxpooling2d.pooling_w = pooling_w;
	maxpooling2d.pooling = (Tensor2d*)malloc(sizeof(Tensor2d) * features + EXTRA_MEMORY);
	maxpooling2d.max_map = (Tensor2d*)malloc(sizeof(Tensor2d) * features + EXTRA_MEMORY);
	for (int i = 0; i < features; i++)
	{
		maxpooling2d.pooling[i] = Create2dTensor(pooling_h, pooling_w, 0);
	}
	return maxpooling2d;
}

void maxpooling2d_forward(Conv2d *conv2d, MaxPooling2d *maxpooling2d)
{
	//conv2d의 높이와 폭이 maxpooling의 높이와 폭과의 나머지가 0이여야함
	for (int f = 0; f < maxpooling2d->features; f++)
	{
		maxpooling2d->max_map[f] = Create2dTensor(conv2d->feature_map->Height, conv2d->feature_map->Width, 0);
		//printf("%d\n", f);
		for (int h_offset = 0; h_offset < conv2d->kernel_size_h; h_offset += maxpooling2d->pooling_h)
		{
			for (int w_offset = 0; w_offset < conv2d->kernel_size_h; w_offset += maxpooling2d->pooling_w)
			{
				int* max_h = (int*)malloc(sizeof(int) * maxpooling2d->pooling_w * maxpooling2d->pooling_h + EXTRA_MEMORY);
				int* max_w = (int*)malloc(sizeof(int) * maxpooling2d->pooling_w * maxpooling2d->pooling_h + EXTRA_MEMORY);
				memset(max_h, -1, maxpooling2d->pooling_w * maxpooling2d->pooling_h * sizeof(int));
				memset(max_w, -1, maxpooling2d->pooling_w * maxpooling2d->pooling_h * sizeof(int));
				double max = -100;
				int cnt = 0;
				for (int i = 0; i < maxpooling2d->pooling_h; i++)
				{
					for (int j = 0; j < maxpooling2d->pooling_w; j++)
					{
						//printf("[%d][%d][%d]\n", f, h_offset + i, w_offset + j);
						//Print2dTensor(conv2d->feature_map[f]);
						//printf("\n");
						if (conv2d->feature_map[f].tensor[h_offset + i][w_offset + j] > max)
						{
							cnt = 0;
							max = conv2d->feature_map[f].tensor[h_offset + i][w_offset + j];
							max_h[cnt] = h_offset + i;
							max_w[cnt] = w_offset + j;
						}
						else if (conv2d->feature_map[f].tensor[h_offset + i][w_offset + j] == max)
						{
							cnt++;
							max_h[cnt] = h_offset + i;
							max_w[cnt] = w_offset + j;
						}
					}
				}
				maxpooling2d->pooling[f].tensor[h_offset / maxpooling2d->pooling_h][w_offset / maxpooling2d->pooling_w] = max;
				for (int i = 0; i <= cnt; i++)
				{
					//printf("[%d][%d]\n", max_h[i], max_w[i]);
					maxpooling2d->max_map[f].tensor[max_h[i]][max_w[i]] = 1;
				}
			}
		}
	}
}