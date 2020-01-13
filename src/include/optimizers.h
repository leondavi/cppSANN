#pragma once

#include <eigen3/Eigen/Eigen>

using namespace Eigen;

namespace Optimizers
{

class Optimizer
{
public:
	virtual ~Optimizer() {};

	virtual void optimize(MatrixXd &Weights,const MatrixXd &W_grad, double &bias, const double bias_diff, double lr) = 0; //overwrite weights
};

class StochasticGradientDescent : public Optimizer
{
public:
	void optimize(MatrixXd &Weights,const MatrixXd &W_grad, double &bias, const double bias_diff, double lr) override
	{
		Weights -= lr*W_grad;
		bias -= bias_diff;
	}
};

/**
 * Calculates the mean gradient of the mini-batch
 * Use the mean gradient we calculated to update the weights
 *
 */
class MiniBatchGradientDescent : public Optimizer
{
private:
	uint32_t batch_size_;
	uint32_t curr_batch_;
	double bias_acc_;
	MatrixXd grad_acc_;//accumulator of gradients

public:

	MiniBatchGradientDescent(uint32_t batch_size = 50) : Optimizer(),
							batch_size_(batch_size),curr_batch_(0),bias_acc_(0) {};

	void optimize(MatrixXd &Weights,const MatrixXd &W_grad, double &bias, const double bias_diff, double lr) override
	{
		if(curr_batch_ == 0)
		{
			bias_acc_ = bias_diff;
			grad_acc_ = W_grad;
		}
		else if (curr_batch_ >= batch_size_)
		{
			//updating weights with gradient
			Weights -= lr*(grad_acc_/batch_size_); //take step by mean of all last gradients
			bias -= lr*(bias_acc_/batch_size_);
			curr_batch_ = 0;
		}
		else
		{
			bias_acc_ += bias_diff;
			grad_acc_ += W_grad;
		}
		curr_batch_++;

	}
};

/**
 * Momentum Optimizer
 * Accelerates SGD with momentum factor based on past step.
 * v_ is the current velocity v(t)
 * v_p_ is past velocity v(t-1)
 * v(t) = gamma_*v(t-1)+lr*W_grad
 * W = W-v(t)
 */
class Momentum : public Optimizer
{
private:
	MatrixXd v_;
	MatrixXd v_p_;

	double gamma_;
	double v_bias_;
	double v_p_bias_;
	bool init;

public:

	Momentum() : Optimizer() , gamma_(0.9),v_bias_(0),v_p_bias_(0),init(true)
	{

	}

	void optimize(MatrixXd &Weights,const MatrixXd &W_grad, double &bias, const double bias_diff, double lr) override
	{
		if (init)
		{
			v_p_ = Eigen::MatrixXd::Zero(W_grad.rows(),W_grad.cols());
			v_bias_ = bias_diff;
			v_p_bias_ = 0;
			init = false;
		}

		//calculating current v(t) and v_bias(t)
		v_ = lr*W_grad+gamma_*v_p_;
		v_bias_ = lr*bias_diff+gamma_*v_p_bias_;

		//update weights and bias
		Weights -= v_;//W-V(t)
		bias -= v_bias_;//b-v(t)

		//update past temrs for next calculation
		v_p_ = v_;
		v_p_bias_ = v_bias_;
	}

};



}
typedef std::shared_ptr<Optimizers::Optimizer> OptimizerPtr ;

#define DEFAULT_OPTIMIZER std::make_shared<Optimizers::StochasticGradientDescent>()
