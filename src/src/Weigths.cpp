/*
 * Weigths.cpp
 *
 *  Created on: Dec 6, 2019
 *      Author: david
 */

#include "../include/Weights.h"

namespace WeightsNormalization
{
	/**
	 * Normalizing the weights by the standard deviation of weigths_in matrix
	 */
	void NormalizeByInputSize(MatrixXd &weights_in)
	{
		double normalization_factor = sqrt(weights_in.cols());
		change_vals_by_func(weights_in,[normalization_factor](double cell_val)
				{
					return cell_val/normalization_factor;
				});
	}
}
