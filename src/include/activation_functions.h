#pragma once

#include <eigen3/Eigen/Eigen>

double sigmoid(double x)
{
	return 1./(1.+exp(-x));
}
