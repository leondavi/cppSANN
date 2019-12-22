#pragma once

#include <eigen3/Eigen/Eigen>



namespace Activations
{

typedef std::function<double(double)> t_activation_func;

class ActivationFunction
{
public:
	virtual double function(double x) = 0;
	virtual double function_derivative(double x) = 0; //first derivative of function

	virtual ~ActivationFunction() {}

	virtual t_activation_func get_func() { return [this](double x){return this->function(x);}; }
	virtual t_activation_func get_Dfunc() { return [this](double x){return this->function_derivative(x);}; } //first derivative of function

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

/**
 * Exponential Linear Units
 * Pros:
	- All benefits of ReLU
	- Closer to zero mean outputs
	- Negative saturation regime compared with leaky ReLU adds some robustness to noise

	Note: argument a should be set here or use lambda function to envelope it
 */

class ELU : public ActivationFunction
{

	double a_; //alpha of leaky relue

public:

	inline double function(double x)
	{
		return x>0 ? x : a_*(exp(x)-1);
	}

	inline double function_derivative(double x)
	{
		return x > 0 ? 1 : a_*exp(x);
	}

	ELU(double a = 0.1) : a_(a) {};
};

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
class Tanh : public ActivationFunction
{
public:

	inline double function(double x)
	{
		return tanh(x);
	}

	inline double function_derivative(double x)
	{
		double r = tanh(x);
		return 1-r*r;
	}
};

/**
 * Doesn't impact anything - Mainly for debug or layers without activation function such as input layer
 */
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


typedef std::shared_ptr<Activations::ActivationFunction> ActivationFunctionPtr ;
#define DEFAULT_ACTIVATION_FUNC std::make_shared<Activations::ReLU>()

} //Activations end



