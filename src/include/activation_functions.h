#pragma once

#include <eigen3/Eigen/Eigen>

namespace Activations
{


/**
 * Sigmoid
 * Pros:
	- Squashes numbers to range [0,1]
	- Interpretation as a saturating “firing rate” of a neuron
	Cons:
	- Saturated neurons “kill” the gradients
	- Sigmoid outputs are not zero-centered
	- exp() is a bit compute expensive
 */
double Sigmoid(double x)
{
	return 1./(1.+exp(-x));
}

/**
 * Rectified Linear Unit
 * Pros:
	- Does not saturate
	- Very computationally efficient
	- Converges much faster than
	- sigmoid/tanh in practice (e.g. 6x)
	- Actually more biologically
	- plausible than sigmoid
	Cons:
	- Not zero-centered output
 */
double ReLU(double x)
{
	return std::max(0.,x);
}

/**
 * Note: argument a should be set here or use lambda function to envelope it
 */
double LeakyReLU(double x,double a = 0.5)
{
	return std::max(a*x,x);
}

/**
 * Exponential Linear Units
 * Pros:
	- All benefits of ReLU
	- Closer to zero mean outputs
	- Negative saturation regime compared with leaky ReLU adds some robustness to noise

	Note: argument a should be set here or use lambda function to envelope it
 */
double ELU(double x, double a = 0.5) // a should be set here
{
	return x>0 ? x : a*(exp(x)-1);
}

/**
 * Tanh
 *
 * Pros:
 * 		- Squashes numbers to range [-1,1]
 * 		- Zero centered
 * Cons:
 * 		- Kills gradients when saturated.
 *
 */
double Tanh(double x)
{
	return tanh(x);
}


}
