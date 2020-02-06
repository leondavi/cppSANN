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
 * Swish
 * y = x*Sigmoid(x)
 */
class Swish : public ActivationFunction
{
private:
	double B_;
	Sigmoid sigmoid_;

public:
	inline double function(double x)
	{
		return B_*x*sigmoid_.function(x);
	}

	inline double function_derivative(double x)
	{
		double sigmoid_x = sigmoid_.function(x);
		return B_*(sigmoid_x+x*sigmoid_x*(1-sigmoid_x));
	}

	Swish(double b = 1) : B_(b) {};
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
 * Doesn't impact anything - Mainly for debug or layers without activation function such as output layer
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
		return 1;
	}
};

	typedef enum {ACT_NONE,ACT_SIGMOID,ACT_RELU,ACT_LEAKY_RELU,ACT_SWISH,ACT_ELU} act_t;

} //end of Activation namespace

typedef std::shared_ptr<Activations::ActivationFunction> ActivationFunctionPtr;

/**
 * Select actiation by enum in Optimizers::opt_t
 */
inline ActivationFunctionPtr select_activation(Activations::act_t ActVal)
{
	ActivationFunctionPtr chosen_act;
	switch (ActVal)
	{
		case Activations::ACT_NONE         : { chosen_act = std::make_shared<Activations::None>(); break; }
		case Activations::ACT_SIGMOID      : { chosen_act = std::make_shared<Activations::Sigmoid>(); break; }
		case Activations::ACT_RELU         : { chosen_act = std::make_shared<Activations::ReLU>(); break; }
		case Activations::ACT_LEAKY_RELU   : { chosen_act = std::make_shared<Activations::LeakyReLU>(); break; }
		case Activations::ACT_SWISH        : { chosen_act = std::make_shared<Activations::Swish>(); break; }
		case Activations::ACT_ELU          : { chosen_act = std::make_shared<Activations::ELU>(); break; }
	}

	return chosen_act;
}

#define DEFAULT_ACTIVATION_FUNC std::make_shared<Activations::Sigmoid>()





