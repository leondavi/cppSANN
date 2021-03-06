
#pragma once

#include <Eigen/Core>

using namespace Eigen;

namespace Normalization
{
/**
 * Softmax normalization function
 * - Make sure that one value is very close to 1 and the rest are close to 0
 * - Move all the output nodes values to the range [0,1]
 * - The sum of all output node values equals to 1.
 * This function can be applied to layer's neurons for example.
 * Don't pass this function as activation function of layer.
 * It can be applied to layer's neurons only by set_new_val_to_neurons function
 */
inline VectorXd softmax(VectorXd &y)
{
	VectorXd y_exp = y.array().exp();
	double sum_y_exp = y_exp.sum();
	return y_exp.array()/sum_y_exp;
}

}
