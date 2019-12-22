#pragma once

#include <eigen3/Eigen/Eigen>

typedef std::function<double(double)> t_activation_func;


namespace Activations
{

class ActivationFunction
{
public:
	virtual double function(double x) = 0;
	virtual double function_derivative(double x) = 0;

	virtual ~ActivationFunction() {}

	virtual t_activation_func get_func() { return [this](double x){return this->function(x);}; }
	virtual t_activation_func get_Dfunc() { return [this](double x){return this->function_derivative(x);}; }

};



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
class Sigmoid : public ActivationFunction
{

public:
	inline double function(double x)
	{
		return 1./(1.+exp(-x));
	}

	inline double function_derivative(double x)
	{
		double sigmoid_x = function(x);
		return sigmoid_x*(1-sigmoid_x);
	}
};

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
class ReLU : public ActivationFunction
{
public:
	inline double function(double x)
	{
		return std::max(0.,x);
	}

	inline double function_derivative(double x)
	{
		return x>0 ? 1 : 0;
	}
};



/**
 * Note: Must be used with lambda function that envelope this function and set alpha param
 * Example: [](double x){return Activations::LeakyReLU(x,0.01);}
 */
class LeakyReLU : public ActivationFunction
{

	double a_; //alpha of leaky relue

public:
	inline double function(double x)
	{
		return std::max(a_*x,x);
	}

	inline double function_derivative(double x)
	{
		return x > 0 ? 1 : a_;
	}

	LeakyReLU(double a = 0.1) : a_(a) {};

};

class None : public ActivationFunction
{
public:
	inline double function(double x)
	{
		return x;
	}

	inline double function_derivative(double x)
	{
		return x;
	}
};

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


} //Activations end

//=================================================================================================================//

namespace Derivatives
{

/**
 * Derivative of leaky ReLU
 * Note: Must be used with lambda function that envelope this function and set alpha param
 * Example: [](double x){return Activations::LeakyReLU(x,0.01);}
 */
inline double DLeakyReLU(double x,double a = 0.5)
{
	return x > 0 ? 1 : a;
}

/**
 * Derivative of ELU
 */
inline double DELU(double x, double a = 0.5) // a should be set here
{
	return x > 0 ? 1 : a*exp(x);
}

/**
 * Derivative of hiperbolic tangent
 *
 */
inline double DTanh(double x)
{
	double r = tanh(x);
	return 1-r*r;
}

}

typedef std::shared_ptr<Activations::ActivationFunction> ActivationFunctionPtr ;
#define DEFAULT_ACTIVATION_FUNC std::make_shared<Activations::ReLU>()

