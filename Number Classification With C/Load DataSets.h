#pragma once
#define _CRT_SECURE_NO_WARNINGS 
#include "Tensor.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <io.h> 
#include <stdlib.h>
#include "dirent.h"
#include <sys/stat.h>
#include <Windows.h>

#define MAX_LEN 1000
#define BATCH_SIZE 96
#define LABELS 3

char temp[MAX_LEN][MAX_LEN];

char* FileSearch(const char* file_path)
{
	temp[0][0] = '\0';
	DIR* dir_info = NULL;
	struct dirent* dir_entry = NULL;

	dir_info = opendir(file_path);
	if (NULL != dir_info)
	{
		int cnt = 0;
		while (dir_entry = readdir(dir_info)) {
			if (dir_entry->d_name[0] != '.')
			{
				strcpy(temp[cnt], dir_entry->d_name);
				cnt++;
			}
		}
	}
	closedir(dir_info);

	return (char*)temp;
}

char(*labels)[MAX_LEN];//배열길이의 최댓값을 넘어서기 때문에 전역변수로 선언
char(*imgs)[MAX_LEN];
char label_cpy[MAX_LEN][MAX_LEN];

int char2int(char* char_)
{
	int length = 0;
	int temp = 0;

	for (int i = 0; i < strlen(char_); i++)
	{
		if (char_[i] == '\0' || char_[i] == '\n') break;
		length++;
	}

	for (int i = 0; i < length; i++)
	{
		temp += ((int)char_[i] - 48) * pow(10, length - i - 1);
	}

	return temp;
}

double char2double(char* char_)
{
	int length = 0;
	double temp = 0.0f;

	for (int i = 0; i < strlen(char_); i++)
	{
		if (char_[i] == '\0' || char_[i] == '\n') break;
		length++;
	}

	for (int i = 0; i < length; i++)
	{
		temp += ((int)char_[i] - 48) * pow(10, length - i - 1);
	}

	return temp;
}

double RGB2GRAY_SCALE(BYTE r, BYTE g, BYTE b)
{
	return 0.212f * (double)r + 0.701f * (double)g + 0.087f * (double)b;
}

BYTE* readBMP(BITMAPFILEHEADER* hf, BITMAPINFOHEADER* hinfo, char filename[])
{
	int width = 0, height = 0;
	FILE* f = NULL;
	f = fopen(filename, "rb");
	if (f == NULL)
	{
		printf("fail to read file");
		return NULL;
	}
	fread(hf, sizeof(BITMAPFILEHEADER), 1, f);
	fread(hinfo, sizeof(BITMAPINFOHEADER), 1, f);
	//fread(hRGB, sizeof(RGBQUAD), 256, f);

	width = hinfo->biWidth;
	height = hinfo->biHeight;

	//printf("Width :%d \t Height :%d\t Image Size :%d\n", width, height, hinfo->biSizeImage);

	BYTE* InputImg = NULL;

	InputImg = (BYTE*)malloc(sizeof(BYTE) * hinfo->biSizeImage);
	fread(InputImg, sizeof(BYTE), hinfo->biSizeImage, f);

	fclose(f);
	return InputImg;
}

void Load_DataSets_BITMAP_GRAY_SCALE(const char* file_path, Tensor2d x[BATCH_SIZE], Tensor2d y[BATCH_SIZE])
{
	int img_size[2] = { 0, };
	labels = FileSearch(file_path);

#pragma region Copy Label path
	for (int label = 0; label < LABELS; label++)
	{
		if (labels[label][0] == NULL) break;

		strcpy(label_cpy[label], file_path);
		strcat(label_cpy[label], "\\");
		strcat(label_cpy[label], labels[label]);
	}
#pragma endregion

	int cnt = 0;
	for (int label = 0; label < LABELS; label++)
	{
		if (label_cpy[label][0] == NULL) break;
		imgs = FileSearch(label_cpy[label]);
		for (int img = 0; img < MAX_LEN; img++)
		{
			if (imgs[img][0] == NULL) break;
			char img_path[1000];
			strcpy(img_path, label_cpy[label]);
			strcat(img_path, "\\");
			strcat(img_path, imgs[img]);
#pragma region Image2Arr
			BITMAPFILEHEADER hf;
			BITMAPINFOHEADER hinfo;

			//printf("%s\n", img_path);
			BYTE* img = readBMP(&hf, &hinfo, img_path);
			if (img == NULL) return 0;

			x[cnt] = Create2dTensor(hinfo.biHeight, hinfo.biWidth, 0);
			y[cnt] = Create2dTensor(1, 1, 0);

			int padding = hinfo.biWidth % 4;
			for (int i = 0; i < hinfo.biHeight; i++)
			{
				for (int j = 0; j < hinfo.biWidth; j++)
				{
					BYTE b = img[i * (hinfo.biWidth * 3 + padding) + j * 3];
					BYTE g = img[i * (hinfo.biWidth * 3 + padding) + j * 3 + 1];
					BYTE r = img[i * (hinfo.biWidth * 3 + padding) + j * 3 + 2];
					//printf("[%f] " , RGB2GRAY_SCALE(r, g, b));
					x[cnt].tensor[i][j] = RGB2GRAY_SCALE(r, g, b);
					y[cnt].tensor[0][0] = label;
				}
				//printf("\n");
			}
			//printf("\n");
#pragma endregion
			cnt++;
		}
	}
}