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
/**
 * Stochastic Gradient Descent
 */
class SGD : public Optimizer
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

		bias -= lr*bias_diff;
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
			bias_acc_ += lr*bias_diff;
			grad_acc_ += lr*W_grad;
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

/**
 * mt = b1*m_former_t+(1-b1)*gt   Calculating moving average, gt is the gradient, mt is first moment
 * vt = b2*v_former_t+(1-b2)*gt^2  Calculating moving average, gt^2 is the squared gradient, mt is second moment
 */
class Adam : public Optimizer
{
private:

	double lr_; //learning rate

	double b1_; //hyper param beta 1
	double b2_; //hyper param beta 2
	double b1_pow_t_;
	double b2_pow_t_;

	double epsilon_;
	bool init_;

	uint64_t t_; //current time

	MatrixXd mt_; //first moment estimate of gradient (biased)
	MatrixXd vt_; //second moment estimate of gradient (biased)
	MatrixXd mt_former_; //moving average of mean - former iteration result
	MatrixXd vt_former_; //moving average of the second moment - former iteration result


public:

	Adam(double lr = 0.01, double b1 = 0.9,double b2 = 0.999,double eps = 1e-8) : Optimizer() , lr_(lr), b1_(b1) ,b2_(b2),
																			  b1_pow_t_(b1), b2_pow_t_(b2),epsilon_(eps),
																			  init_(false),t_(0)
	{

	}

	void optimize(MatrixXd &Weights,const MatrixXd &W_grad, double &bias, const double bias_diff, double lr) override
	{
		t_++;
		b1_pow_t_*=b1_pow_t_;
		b2_pow_t_*=b2_pow_t_;

		if (!init_)
		{
			mt_ = MatrixXd::Zero(W_grad.rows(),W_grad.cols());
			vt_ = MatrixXd::Zero(W_grad.rows(),W_grad.cols());
			mt_former_ = MatrixXd::Zero(W_grad.rows(),W_grad.cols());
			vt_former_ = MatrixXd::Zero(W_grad.rows(),W_grad.cols());
			init_ = true;
		}

		MatrixXd grad_squared;
		std::cout<<"before\n"<<W_grad<<std::endl;
		grad_squared = W_grad.array()*W_grad.array();
		std::cout<<"after\n"<<grad_squared<<std::endl;

		mt_ = b1_*mt_former_ +  (1-b1_)*W_grad;
		vt_ = b2_*vt_former_ +  (1-b2_)*grad_squared;

		MatrixXd mt_bc; // Bias-corrected first moment estimate
		MatrixXd vt_bc_sqr; // Bias-corrected second moment estimate

		mt_bc = mt_/(1-b1_pow_t_);
		vt_bc_sqr = (vt_/(1-b2_pow_t_)).cwiseSqrt();

		Weights = Weights - (MatrixXd)(lr_*mt_bc.array()/(vt_bc_sqr.array()+epsilon_));

		mt_former_ = mt_;
		vt_former_ = vt_;
	}

};

typedef enum {OPT_SGD,OPT_MINI_BATCH_SGD,OPT_MOMENTUM,OPT_NAG,OPT_ADAGRAD,OPT_ADAM} opt_t;

}
typedef std::shared_ptr<Optimizers::Optimizer> OptimizerPtr ;

/**
 * Select actiation by enum in Optimizers::opt_t
 */
inline OptimizerPtr select_optimizer(Optimizers::opt_t OptVal)
{
	OptimizerPtr chosen_opt;
	switch (OptVal)
	{
		case Optimizers::OPT_SGD            : { chosen_opt = std::make_shared<Optimizers::SGD>(); break; }
		case Optimizers::OPT_MINI_BATCH_SGD : { chosen_opt = std::make_shared<Optimizers::MiniBatchGradientDescent>(); break; }
		case Optimizers::OPT_MOMENTUM       : { chosen_opt = std::make_shared<Optimizers::Momentum>(); break; }
		case Optimizers::OPT_NAG            : { chosen_opt = std::make_shared<Optimizers::NAG>(); break; }
		case Optimizers::OPT_ADAGRAD        : { chosen_opt = std::make_shared<Optimizers::Adagrad>(); break; }
		case Optimizers::OPT_ADAM           : { chosen_opt = std::make_shared<Optimizers::Adam>(); break; }
	}

	return chosen_opt;
}

#define DEFAULT_OPTIMIZER std::make_shared<Optimizers::SGD>()
