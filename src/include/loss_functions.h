#pragma once

#include <eigen3/Eigen/Eigen>
#include "math_methods.h"
#include "normalization_functions.h"

using namespace Eigen;

namespace LossFunctions
{

	class Loss
	{
	public:

		virtual ~Loss() {};

		virtual double func(VectorXd &y, VectorXd &y_pred) = 0;
	};

	/**
	 * This is the LogLoss function.
	 * It gives a penately for wrong binary classifications.
	 * It consists out of two loss fucntions for positive class Y=1 and negative class Y=0:
	 * Positive class: Loss = -log(Y_pred)
	 * Negative class: Loss = -log(1-Y_pred)
	 */
	class LogLoss
	{
	public:
		inline VectorXd func(VectorXd &y, VectorXd &y_pred)
		{
			return y.array()*y_pred.array().log()*(-1)+(1-y.array())*(1-y_pred.array()).log()*(-1);
		}
	};

	/**
	 * CategoricalCrossEntropyLoss
	 *
	 * LogLoss(y,softmax(Y_pred))
	 */
	class CategoricalCrossEntropyLoss : Loss
	{
	public:
			inline double func(VectorXd &y, VectorXd &y_pred) override
			{
				LogLoss logloss;
				VectorXd y_pred_softmax = Normalization::softmax(y_pred);
				return logloss.func(y,y_pred_softmax).sum();
			}
	};

	class MSELoss : Loss
	{
	public:
		inline double func(VectorXd &y, VectorXd &y_pred) override
		{
			return ExtMath::mse(y,y_pred);
		}

	};

}
