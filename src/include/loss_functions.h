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

		virtual VectorXd func(VectorXd &y, VectorXd &y_pred) = 0;
		virtual VectorXd derivative(VectorXd &y, VectorXd &y_pred) = 0;

	};


	class MSELoss : public Loss
	{
	public:
		inline VectorXd func(VectorXd &y, VectorXd &y_pred) override
		{
			VectorXd tmp = y-y_pred;
			tmp = tmp.array().pow(2);
			return tmp;
		}
		inline VectorXd derivative(VectorXd &y_pred,VectorXd &y) override
		{
			return y_pred - y;
		}



	};

}

typedef std::shared_ptr<LossFunctions::Loss> LossFunctionPtr ;

