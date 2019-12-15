#pragma once

#include <eigen3/Eigen/Eigen>

namespace Activations
{

double sigmoid(double x)
{
	return 1./(1.+exp(-x));
}

}
