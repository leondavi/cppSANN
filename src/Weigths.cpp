/*
 * Weigths.cpp
 *
 *  Created on: Dec 6, 2019
 *      Author: david
 */

#include "include/Weights.h"

namespace ANN
{

namespace WeightsNormalization
{
	/**
	 * Normalizing the weights by the standard deviation of weigths_in matrix
	 */
	void NormalizedByInputSize(MatrixXd &weights_in)
	{
		double normalization_factor = sqrt((double)weights_in.cols());
		ExtMath::change_vals_by_func(weights_in,[normalization_factor](double cell_val)
				{
					return cell_val/normalization_factor;
				});
	}
}

Weights::Weights(uint32_t rows, uint32_t cols,int val) : bias_(0)
{
	this->weights_mat_= val*MatrixXd::Ones(rows,cols);
}

Weights::Weights(uint32_t rows, uint32_t cols,double bias, double mu, double sig)
{
	this->weights_mat_= ExtMath::randn(rows,cols,mu,sig);
	WeightsNormalization::NormalizedByInputSize(this->weights_mat_);
	bias_ = bias;
}

}
