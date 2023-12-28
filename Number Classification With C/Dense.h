#pragma once
#include "Tensor.h"
#include "Parameters.h"
#include "functions.h"

typedef struct Dense
{
	Parameter_b params;
	Parameter_b delta_params;
	Tensor2d a;
} Dense;

Dense dense(int units, int input_units)
{
	Dense dense;
	dense.delta_params = CreateParameter_b_init(units, input_units, 0);
	dense.params = CreateParameter_b(units, input_units);

	//PrintParams_b(dense.params);

	return dense;
}

void dense_forward(Dense *unit, Tensor2d prev_params)
{
	unit->a = multensor(unit->params.W, prev_params);
	unit->a = add(unit->a, unit->params.b);
	unit->a = sigmoid(unit->a);

	return unit;
}

Tensor2d dense_backward(Dense unit, Tensor2d next_delta_tensor, Tensor2d next_layer)
{
	Tensor2d grads_sigmoid_tensor = grads_sigmoid(next_layer);
	Tensor2d tranposed_params = transposed(unit.params.W);
	Tensor2d grads = multensor(tranposed_params, next_delta_tensor);

	return grads;
}

void Update_Dense_params(Dense *dense, double learning_rate)
{
	Tensor2d lr_W = Create2dTensor(dense->delta_params.W.Height, dense->delta_params.W.Width, learning_rate);
	Tensor2d lr_b = Create2dTensor(dense->delta_params.b.Height, dense->delta_params.b.Width, learning_rate);
	dense->delta_params.W = product(dense->delta_params.W, lr_W);
	dense->delta_params.b = product(dense->delta_params.b, lr_b);
	dense->params.W = add(dense->params.W, dense->delta_params.W);
	dense->params.b = add(dense->params.b, dense->delta_params.b);

	Free2dTensor(lr_W);
	Free2dTensor(lr_b);
}