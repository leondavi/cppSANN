#pragma once

#include <eigen3/Eigen/Eigen>
#include "math_methods.h"
#include "normalization_functions.h"
#include <iostream>

using namespace Eigen;

namespace LossFunctions
{

typedef enum {LOSS_CROSS_ENTROPY,LOSS_MSE} loss_t;


	class Loss
	{
	public:

		virtual ~Loss() {};

		virtual VectorXd func(VectorXd &y, VectorXd &y_pred) = 0;
		virtual VectorXd derivative(VectorXd &y, VectorXd &y_pred) = 0;

		virtual loss_t loss_type() = 0;

	};

	class CrossEntropy : public Loss
	{
	public:
		// Softargmax  f(x) = exp(xi)/sum(exp(xj))
		inline VectorXd softmax(VectorXd &x)
		{
			double sum_e = x.array().exp().sum();
			return x.array().exp()/sum_e;
		}

		inline VectorXd func(VectorXd &y_pred, VectorXd &y) override
		{
			VectorXd res = -y.array()*y_pred.array().log();
			return res;

		}
		inline VectorXd derivative(VectorXd &y_pred, VectorXd &y) override
		{
			VectorXd res;
			return res;
		}

		loss_t loss_type()
		{
			return LOSS_CROSS_ENTROPY;
		}
	};

	class MSELoss : public Loss
	{
	public:
		inline VectorXd func(VectorXd &y_pred, VectorXd &y) override
		{
			VectorXd tmp = y_pred-y;
			tmp = tmp.array().pow(2);
			return tmp;
		}
		inline VectorXd derivative(VectorXd &y_pred,VectorXd &y) override
		{
			return y_pred -y; //Output - Target
		}

		loss_t loss_type()
		{
			return LOSS_MSE;
		}
	};

}

typedef std::shared_ptr<LossFunctions::Loss> LossFunctionPtr ;


/**
 * Select loss function by enum in LossFunctions::loss_t
 */
inline LossFunctionPtr select_loss_function(LossFunctions::loss_t lossVal)
{
	LossFunctionPtr chosen_loss_function;
	switch (lossVal)
	{
		case LossFunctions::LOSS_CROSS_ENTROPY : { chosen_loss_function = std::make_shared<LossFunctions::CrossEntropy>(); break; }
		case LossFunctions::LOSS_MSE      	   : { chosen_loss_function = std::make_shared<LossFunctions::MSELoss>(); break; }
	}

	return chosen_loss_function;
}


