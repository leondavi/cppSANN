#pragma once

#include <eigen3/Eigen/Eigen>
#include "loss_functions.h"
#include <iostream>

#define MINI_BATCH_GRADIENT_DEFAULT_BATCH_SIZE 50
#define MOMENTUM_DEFAULT_GAMMA_VAL 0.9

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
	/**
	 * Overwrites Weights with result of gradients change
	 */
	void optimize(MatrixXd &Weights,const MatrixXd &W_grad, double &bias, const double bias_diff, double lr) override
	{
	//	std::cout<<"in Weights:\n"<<Weights<<std::endl;
		Weights -= lr*W_grad;
	//	std::cout<<"out Weights:\n"<<Weights<<std::endl;

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

	MiniBatchGradientDescent(uint32_t batch_size = MINI_BATCH_GRADIENT_DEFAULT_BATCH_SIZE) : Optimizer(),
							batch_size_(batch_size),curr_batch_(0),bias_acc_(0) {};
	/**
	 * Overwrites Weights
	 */
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

	Momentum(double gamma_val = MOMENTUM_DEFAULT_GAMMA_VAL) : Optimizer() , gamma_(gamma_val),v_bias_(0),v_p_bias_(0),init(true)
	{

	}
	/**
	 * Overwrites Weights
	 */
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

/**
 * Nestrov Accelerated Gradient
 *
 *	TODO - test this function
 *	TODO - Improve implementation
 */
class NAG : public Optimizer
{
private:
	MatrixXd v_;
	MatrixXd v_p_;

	double gamma_;
	double v_bias_;
	double v_p_bias_;
	bool init;

	LossFunctionPtr loss_func_;

	/**
	 * Loss is given only for vectors
	 * This is a workaround to compute the loss of two matrices
	 */
	MatrixXd get_loss_derivative_between_mats(MatrixXd &y_mat,MatrixXd &y_pred_mat)
	{
		MatrixXd res(y_mat.rows(),y_mat.cols());
		VectorXd y_vec,y_pred_vec;
		for (int r = 0 ; r < res.rows(); r++)
		{
			y_vec = y_mat.row(r);
			y_pred_vec = y_pred_mat.row(r);
			res.row(r) = loss_func_->derivative(y_vec,y_pred_vec);
		}
		return res;
	}

public:

	NAG(LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>(),double gamma_val = MOMENTUM_DEFAULT_GAMMA_VAL) : Optimizer() ,
		gamma_(gamma_val),v_bias_(0),v_p_bias_(0),init(true),loss_func_(loss_func)
	{

	}
	/**
	 * Overwrites Weights
	 */
	void optimize(MatrixXd &Weights,const MatrixXd &W_grad, double &bias, const double bias_diff, double lr) override
	{
		if (init)
		{
			v_p_ = Eigen::MatrixXd::Zero(W_grad.rows(),W_grad.cols());
			v_bias_ = bias_diff;
			v_p_bias_ = 0;
			init = false;
		}

		MatrixXd gamma_times_v_p_ = gamma_*v_p_;
		MatrixXd  next_pos_grad = get_loss_derivative_between_mats(Weights,gamma_times_v_p_);
		//calculating current v(t) and v_bias(t)
		v_ = lr*next_pos_grad+gamma_times_v_p_;
		v_bias_ = lr*bias_diff+gamma_*v_p_bias_;

		//update weights and bias
		Weights -= v_;//W-V(t)
		bias -= v_bias_;//b-v(t)

		//update past temrs for next calculation
		v_p_ = v_;
		v_p_bias_ = v_bias_;
	}
};

class Adagrad : public Optimizer
{
private:

	double epsilon_;
	bool init;

	MatrixXd Gt_;


public:

	Adagrad(double epsilon = 1e-8) : Optimizer(),epsilon_(epsilon),init(true)
		{}

	void optimize(MatrixXd &Weights,const MatrixXd &W_grad, double &bias, const double bias_diff, double lr) override
	{
		if (init)
		{
			Gt_ = W_grad.array().pow(2);
			init = false;
		}
		else
		{
	//		std::cout<<"Gt before pow: \n"<<Gt_<<std::endl;
			Gt_.array() += W_grad.array().pow(2);
	//		std::cout<<"Gt after pow: \n"<<Gt_<<std::endl;
		}
		MatrixXd denominator(Gt_);
		denominator += epsilon_*MatrixXd::Ones(denominator.rows(),denominator.cols());
	//	std::cout<<"Gt before sqrt: \n"<<denominator<<std::endl;
		denominator = denominator.cwiseSqrt();
	//	std::cout<<"Gt after sqrt: \n"<<denominator<<std::endl;
		denominator = denominator.cwiseInverse();//TODO find better implementation
	//	std::cout<<"Gt after cwiseinverse: \n"<<denominator<<std::endl;

		Weights -= lr*denominator.cwiseProduct(W_grad);

		bias -= bias_diff;

		//std::cout<<"weights: \n"<<Weights<<std::endl;


	}

};




}
typedef std::shared_ptr<Optimizers::Optimizer> OptimizerPtr ;

#define DEFAULT_OPTIMIZER std::make_shared<Optimizers::StochasticGradientDescent>()
