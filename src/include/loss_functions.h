#pragma once

#include <eigen3/Eigen/Eigen>

using namespace Eigen;

namespace LossFunctions
{

	class Loss
	{
	public:

		virtual ~Loss() {};

		virtual VectorXd func(VectorXd y, VectorXd y_pred) = 0;
	};

	/**
	 * This is the LogLoss function.
	 * It gives a penately for wrong binary classifications.
	 * It consists out of two loss fucntions for positive class Y=1 and negative class Y=0:
	 * Positive class: Loss = -log(Y_pred)
	 * Negative class: Loss = -log(1-Y_pred)
	 */
	class LogLoss : Loss
	{
	public:
		inline VectorXd func(VectorXd y, VectorXd y_pred) override
		{
			return y.array()*y_pred.array().log()*(-1)+(1-y.array())*(1-y_pred.array()).log()*(-1);
		}
	};

}
