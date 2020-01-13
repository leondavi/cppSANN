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




}
typedef std::shared_ptr<Optimizers::Optimizer> OptimizerPtr ;

#define DEFAULT_OPTIMIZER std::make_shared<Optimizers::StochasticGradientDescent>()
