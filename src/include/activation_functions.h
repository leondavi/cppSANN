#pragma once

#include <eigen3/Eigen/Eigen>

#define DEFAULT_ACTIVATION_FUNC Activations::ReLU

typedef std::function<double(double)> t_activation_func;

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
inline double Sigmoid(double x)
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
inline double ReLU(double x)
{
	return std::max(0.,x);
}

/**
 * Note: Must be used with lambda function that envelope this function and set alpha param
 * Example: [](double x){return Activations::LeakyReLU(x,0.01);}
 */
inline double LeakyReLU(double x,double a = 0.5)
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
inline double ELU(double x, double a = 0.5) // a should be set here
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
inline double Tanh(double x)
{
	return tanh(x);
}

inline double None(double x)
{
	return x;
}


}
