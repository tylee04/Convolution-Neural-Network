#pragma once
#include "Parameters.h"
#include "functions.h"
#include "Tensor.h"
#include "Flatten.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef struct Conv2d
{
	int filters;
	int kernel_size_h;
	int kernel_size_w;
	Parameter *params;
	double* b;
	Tensor2d *feature_map;
	Parameter *delta_params;
	double* delta_b;
} Conv2d;

Conv2d conv2d(int filters, int kernel_size_h, int kernel_size_w)
{
	Conv2d conv2d;
	conv2d.filters = filters;
	conv2d.kernel_size_h = kernel_size_h;
	conv2d.kernel_size_w = kernel_size_w;
	conv2d.b = (double*)malloc(sizeof(double) * filters + EXTRA_MEMORY);

	conv2d.params = (Parameter*)malloc(sizeof(Parameter) * filters + EXTRA_MEMORY);
	conv2d.feature_map = (Tensor2d*)malloc(sizeof(Tensor2d) * conv2d.filters + EXTRA_MEMORY);
	conv2d.delta_params = (Parameter*)malloc(sizeof(Parameter) * filters + EXTRA_MEMORY);
	conv2d.delta_b = (double*)malloc(sizeof(double) * filters + EXTRA_MEMORY);
	for (int i = 0; i < filters; i++)
	{
		//conv2d.params[i] = CreateParameter(kernel_size_h, kernel_size_w);
		conv2d.params[i].W = Create2dTensor(kernel_size_h, kernel_size_w, 0);
		conv2d.b[i] = rands();

		conv2d.delta_params[i].W = Create2dTensor(kernel_size_h, kernel_size_w, 0);
		conv2d.delta_b[i] = 0.0f;
	}
	return conv2d;
}

void conv2d_forward(Tensor2d input, Conv2d *conv2d)
{
	for (int f = 0; f < conv2d->filters; f++)
	{
		conv2d->feature_map[f] = Create2dTensor(input.Height - conv2d->kernel_size_h + 1, input.Width - conv2d->kernel_size_w + 1, 0);
		for (int h_offset = 0; h_offset < input.Height - conv2d->kernel_size_h + 1; h_offset++)
		{
			for (int w_offset = 0; w_offset < input.Width - conv2d->kernel_size_w + 1; w_offset++)
			{
				double z = 0;
				for (int i = 0; i < conv2d->kernel_size_h; i++)
				{
					for (int j = 0; j < conv2d->kernel_size_w; j++)
					{
						z += input.tensor[i + h_offset][j + w_offset] * conv2d->params[f].W.tensor[i][j];
					}
				}
				conv2d->feature_map[f].tensor[h_offset][w_offset] = z + conv2d->b[f];
			}
		}
		conv2d->feature_map[f] = sigmoid(conv2d->feature_map[f]);
	}
}

Tensor2d conv2d_backward(Conv2d conv2d, Tensor2d input)
{
	for (int f = 0; f < conv2d.filters; f++)
	{
		for (int h_offset = 0; h_offset < input.Height - conv2d.kernel_size_h + 1; h_offset++)
		{
			for (int w_offset = 0; w_offset < input.Width - conv2d.kernel_size_w + 1; w_offset++)
			{
				double z = 0;
				for (int i = 0; i < conv2d.kernel_size_h; i++)
				{
					for (int j = 0; j < conv2d.kernel_size_w; j++)
					{
						z += input.tensor[i + h_offset][j + w_offset] * conv2d.params[f].W.tensor[i][j];
					}
				}
				conv2d.delta_params[f].W.tensor[h_offset][w_offset] = z;
			}
		}
	}
}

void Update_Conv2d_params(Conv2d *conv2d, double learning_late)
{
	Tensor2d lr_W = Create2dTensor(conv2d->kernel_size_h, conv2d->kernel_size_w, learning_late);
	for (int i = 0; i < conv2d->filters; i++)
	{
		conv2d->delta_params[i].W = product(conv2d->delta_params[i].W, lr_W);
		conv2d->delta_b[i] *= learning_late;

		conv2d->params[i].W = add(conv2d->params[i].W, conv2d->delta_params[i].W);
		conv2d->delta_b[i] += conv2d->delta_b[i];
	}
}

void itoaSub(int num, char* str, int radix) {
	int tmp = num;
	int cnt = 0;

	while (tmp != 0) {
		tmp /= 10;
		cnt++;
	}

	str[cnt] = '\0';
	do {
		cnt--;
		str[cnt] = (char)(num % 10 + 48);
		num = num / 10;
	} while (num != 0);
}

Tensor2d paramstopixel(Tensor2d tensor)
{
	Tensor2d temp = Create2dTensor(tensor.Width, tensor.Width, 0);
	float min, max;
	max = min = tensor.tensor[0][0];
	for (int i = 0; i < tensor.Height; i++)
	{
		for (int j = 0; j < tensor.Width; j++)
		{
			if (min > tensor.tensor[i][j])
			{
				min = tensor.tensor[i][j];
			}
			if (max < tensor.tensor[i][j])
			{
				max = tensor.tensor[i][j];
			}
		}
	}
	//Print2dTensor(tensor);
	for (int i = 0; i < tensor.Height; i++)
	{
		for (int j = 0; j < tensor.Width; j++)
		{
			temp.tensor[i][j] = fabs(255.0f * ((tensor.tensor[i][j] + fabs(min)) / (fabs(max) + fabs(min))));
		}
	}
	//Print2dTensor(temp);
	return temp;
}

void Write_Kernel_Map_BMP(Conv2d *conv2d, int epoch)
{
	BITMAPFILEHEADER hf;
	BITMAPINFOHEADER hinfo;
	FILE* readf = NULL;
	readf = fopen("C:/Users/reete/Desktop/Number Classification With C/kernel_map.bmp", "rb");
	if (readf == NULL)
	{
		printf("kernel map fail to read file");
		return NULL;
	}
	fread(&hf, sizeof(BITMAPFILEHEADER), 1, readf);
	fread(&hinfo, sizeof(BITMAPINFOHEADER), 1, readf);

	fclose(readf);
	//printf("%d %d\n", hinfo->biHeight, hinfo->biWidth);
	BYTE* OutputImg = NULL;
	FILE* writef = NULL;
	for (int f = 0; f < conv2d->filters; f++)
	{
		char filename[1000] = "C:/Users/reete/Desktop/Number Classification With C/kernel_map/";
		char filter[100], num[100];
		sprintf(filter, "%d", f);
		sprintf(num, "%05d", epoch);
		strcat(filename, filter);
		strcat(filename, "/");
		strcat(filename, num);
		strcat(filename, ".bmp");
		printf("%s\n", filename);

		writef = fopen(filename, "wb");

		OutputImg = (BYTE*)malloc(sizeof(BYTE) * hinfo.biSizeImage);
		//printf("Size: %d\n", hinfo->biSizeImage);
		fwrite(&hf, sizeof(BITMAPFILEHEADER), 1, writef);
		fwrite(&hinfo, sizeof(BITMAPINFOHEADER), 1, writef);

		int cnt = 0;
		//Print2dTensor(conv2d->params[f].W);
		Tensor2d flatten_ = paramstopixel(conv2d->params[f].W);
		flatten_ = flatten_matrix(flatten_);
		//Print2dTensor(flatten_);
		//Tensor2d flatten_ = flatten(conv2d->params[f].W);
		for (int i = 0; i < hinfo.biSizeImage; i += 3)
		{
			//printf("[%03f]\n", flatten_.tensor[0][cnt]);
			OutputImg[i] = (BYTE)(flatten_.tensor[0][cnt]);
			OutputImg[i + 1] = (BYTE)(flatten_.tensor[0][cnt]);
			OutputImg[i + 2] = (BYTE)(flatten_.tensor[0][cnt]);
			cnt++;
		}
		fwrite(OutputImg, sizeof(BYTE), hinfo.biSizeImage, writef);
		//printf("\n");
	}
	fclose(writef);
}

void Write_Feature_Map_BMP(Conv2d* conv2d, int epoch)
{
	BITMAPFILEHEADER hf;
	BITMAPINFOHEADER hinfo;
	FILE* readf = NULL;
	readf = fopen("C:/Users/User/Desktop/Number Classification With C/feature_map.bmp", "rb");
	if (readf == NULL)
	{
		printf("feature_map fail to read file\n");
		return NULL;
	}
	fread(&hf, sizeof(BITMAPFILEHEADER), 1, readf);
	fread(&hinfo, sizeof(BITMAPINFOHEADER), 1, readf);

	fclose(readf);
	//printf("%d %d\n", hinfo->biHeight, hinfo->biWidth);
	BYTE* OutputImg = NULL;
	FILE* writef = NULL;
	for (int f = 0; f < conv2d->filters; f++)
	{
		char filename[1000] = "C:/Users/User/Desktop/Number Classification With C/feature_map/";
		char filter[100], num[100];
		sprintf(filter, "%d", f);
		sprintf(num, "%05d", epoch);
		strcat(filename, filter);
		strcat(filename, "/");
		strcat(filename, num);
		strcat(filename, ".bmp");
		printf("%s\n", filename);

		writef = fopen(filename, "wb");
		OutputImg = (BYTE*)malloc(sizeof(BYTE) * hinfo.biSizeImage);
		//printf("Size: %d\n", hinfo->biSizeImage);
		fwrite(&hf, sizeof(BITMAPFILEHEADER), 1, writef);
		fwrite(&hinfo, sizeof(BITMAPINFOHEADER), 1, writef);

		int cnt = 0;
		Tensor2d flatten_ = paramstopixel(conv2d->params[f].W);
		flatten_ = flatten_matrix(flatten_);
		for (int i = 0; i < hinfo.biSizeImage; i += 3)
		{
			//printf("[%2d] %d\n", i, (BYTE)(255 * flatten_.tensor[0][cnt]));
			OutputImg[i] = (BYTE)(flatten_.tensor[0][cnt]);
			OutputImg[i + 1] = (BYTE)(flatten_.tensor[0][cnt]);
			OutputImg[i + 2] = (BYTE)(flatten_.tensor[0][cnt]);
			cnt++;
		}
		fwrite(OutputImg, sizeof(BYTE), hinfo.biSizeImage, writef);

	}
	fclose(writef);
}