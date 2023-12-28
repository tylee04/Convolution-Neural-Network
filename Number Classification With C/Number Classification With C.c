#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include "Load DataSets.h"
#include "functions.h"
#include "Tensor.h"
#include "Parameters.h"
#include "Dense.h"
#include "Conv2d.h"
#include "Flatten.h"
#include "Model.h"
#include "MaxPooling2d.h"

//Hyper Parameters
#define EPOCHS 100
#define LEARNING_RATE 0.2

int main()
{
	srand((unsigned int)time(NULL));

	BITMAPFILEHEADER hf;
	BITMAPINFOHEADER hinfo;

	Tensor2d x_train[BATCH_SIZE], y[BATCH_SIZE], y_train[BATCH_SIZE];
	Load_DataSets_BITMAP_GRAY_SCALE("../Datasets", x_train, y);
	
	// loss function계산시 transposed를 생략하기 위해 세로로 one hot encoding함
	one_hot_encoding(y, y_train);
	for (int i = 0; i < BATCH_SIZE; i++)
	{
		Free2dTensor(y[i]);
	}

	// 예측률을 높이기 위해 x_train의 값을 Normalization해야함
	for (int i = 0; i < BATCH_SIZE; i++)
	{
		for (int h = 0; h < x_train[i].Height; h++)
		{
			for (int w = 0; w < x_train[i].Width; w++)
			{
				x_train[i].tensor[h][w] /= 255.0f;
			}
		}
	}

	Conv2d conv = conv2d(3, 3, 3);
	MaxPooling2d maxpooling = maxpooling2d(conv.filters, 2, 2);
	Dense dense_layer = dense(3, maxpooling.features * maxpooling.pooling_h * maxpooling.pooling_w);

	for (int epoch = 0; epoch < EPOCHS; epoch++)
	{
		double loss = 0;
		int correct_cnt = 0;
		double accuracy = 0.0f;
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			//Forward Propagation
			conv2d_forward(x_train[batch], &conv);
			maxpooling2d_forward(&conv, &maxpooling);
			Tensor2d flatten_ = flatten(maxpooling.pooling, maxpooling.features);
			flatten_ = transposed(flatten_);
			dense_forward(&dense_layer, flatten_);
			loss += sum_squares_error(dense_layer.a, y_train[batch]);

			//Back Propagation
			Tensor2d delta = sum_squares_error_backward(dense_layer.a, y_train[batch]);
			dense_layer.delta_params.b = add(dense_layer.delta_params.b, delta);
			Tensor2d reshape_delta_w, reshape_delta_a, grads_W;
			reshape_delta_w = broadcasting(delta, dense_layer.delta_params.W.Width, 0);
			
			Tensor2d flatten_transposed = transposed(flatten_);
			reshape_delta_a = broadcasting(flatten_transposed, dense_layer.delta_params.W.Height, 1);
			grads_W = product(reshape_delta_w, reshape_delta_a);
			dense_layer.delta_params.W = add(dense_layer.delta_params.W, grads_W);
			//PrintParams_b(dense_layer.delta_params);

			Tensor2d transposed_dense_w, grads_a, kernel_delta;
			transposed_dense_w = transposed(dense_layer.params.W);
			//Print2dTensor(transposed_dense_w);
			reshape_delta_w = multensor(transposed_dense_w, delta);
			//Print2dTensor(reshape_delta_w);
			grads_a = grads_sigmoid(flatten_);
			kernel_delta = product(reshape_delta_w, grads_a);
			//Print2dTensor(kernel_delta);
			Tensor2d* delta_kernel_division = tensor_division(kernel_delta, maxpooling.features, 1);
			Tensor2d* delta_kernel = (Tensor2d*)malloc(sizeof(Tensor2d) * maxpooling.features + EXTRA_MEMORY);
			for (int f = 0; f < maxpooling.features; f++)
			{
				delta_kernel_division[f] = un_flatten(delta_kernel_division[f], maxpooling.pooling_h, maxpooling.pooling_w);
				delta_kernel[f] = Create2dTensor(maxpooling.max_map[f].Height, maxpooling.max_map[f].Width, 0);

				for (int h_offset = 0; h_offset < maxpooling.max_map[f].Height / maxpooling.pooling_h; h_offset++)
				{
					for (int w_offset = 0; w_offset < maxpooling.max_map[f].Width / maxpooling.pooling_w; w_offset++)
					{
						for (int h = 0; h < maxpooling.pooling_h; h++)
						{
							for (int w = 0; w < maxpooling.pooling_w; w++)
							{
								delta_kernel[f].tensor[h_offset * maxpooling.pooling_h + h][w_offset * maxpooling.pooling_w + w] = delta_kernel_division[f].tensor[h_offset][w_offset] * maxpooling.max_map[f].tensor[h_offset * maxpooling.pooling_h + h][w_offset * maxpooling.pooling_w + w];
								//delta kernel bias
								conv.delta_b[f] += delta_kernel[f].tensor[h_offset * maxpooling.pooling_h + h][w_offset * maxpooling.pooling_w + w];
							}
						}
					}
				}

				for (int h_offset = 0; h_offset < conv.kernel_size_h; h_offset++)
				{
					for (int w_offset = 0; w_offset < conv.kernel_size_w; w_offset++)
					{
						double z = 0;
						for (int i = 0; i < delta_kernel[f].Height; i++)
						{
							for (int j = 0; j < delta_kernel[f].Width; j++)
							{
								//printf("[%d][%d][%d][%d][%d]\n", f, i + h_offset, j + w_offset, i, j);
								z += x_train[batch].tensor[i + h_offset][j + w_offset] * delta_kernel[f].tensor[i][j];
							}
						}
						conv.delta_params[f].W.tensor[h_offset][w_offset] += z;
					}
				}
				//Print2dTensor(conv.delta_params[f].W);
				//printf("%lf\n", conv.delta_b[f]);
			}
		}

		Update_Dense_params(&dense_layer, -LEARNING_RATE);
		Update_Conv2d_params(&conv, -LEARNING_RATE);		

		//Write_Kernel_Map_BMP(&conv, epoch);
		//Write_Feature_Map_BMP(&conv, epoch);

		//accuracy
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			//Forward Propagation
			conv2d_forward(x_train[batch], &conv);
			maxpooling2d_forward(&conv, &maxpooling);
			Tensor2d flatten_ = flatten(maxpooling.pooling, maxpooling.features);
			flatten_ = transposed(flatten_);
			dense_forward(&dense_layer, flatten_);

			double max = 0;
			int max_idx = 0;
			int labels = 0;
			for (int label = 0; label < LABELS; label++)
			{
				if (max < dense_layer.a.tensor[label][0])
				{
					max = dense_layer.a.tensor[label][0];
					max_idx = label;
				}
				if (y_train[batch].tensor[label][0] == 1)
				{
					labels = label;
				}
			}
			if (y_train[batch].tensor[max_idx][0] == 1)
			{
				correct_cnt++;
			}
		}
		accuracy = (double)correct_cnt / (double)BATCH_SIZE;

		printf("# Epoch  %2d/%d\n", epoch + 1, EPOCHS);
		printf("# Loss:  %f  -  Accuracy:  %f\n", loss, accuracy);
	}

	printf("========================Convolution===================\n");
	for (int i = 0; i < conv.filters; i++)
	{
		printf("Kernel %d:\n", i);
		printf("[Weights]\n");
		Print2dTensor(conv.params[i].W);
		printf("[Bias]\n");
		printf("[%10f]\n", conv.b[i]);
	}
	printf("=======================Maxpooling====================\n");
	for (int i = 0; i < maxpooling.features; i++)
	{
		printf("Maxpooling %d:\n", i);
		Print2dTensor(maxpooling.pooling[i]);
	}
	printf("=========================Dense========================\n");
	PrintParams_b(dense_layer.params);
	printf("fin.\n");
	return 0;
}