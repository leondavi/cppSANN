/*
 * Weigths.h
 *
 *  Created on: Dec 6, 2019
 *      Author: david
 */

#ifndef SRC_WEIGTHS_H_
#define SRC_WEIGTHS_H_

#include <eigen3/Eigen/Dense>
#include "math_methdos.h"

using namespace Eigen;

namespace ANN
{

namespace WeightsNormalization
{
	void NormalizedByInputSize(MatrixXd &weights_in);
}



class Weights
{

	MatrixXd weights_mat_;
	double bias_;

public:

	//TODO add custom weights by function
	Weights(uint32_t rows, uint32_t cols, double bias = 1, double mu = 0, double sig = 1);
	Weights(MatrixXd weights_mat,double bias) : weights_mat_(weights_mat),bias_(bias) { }

	~Weights() {}

	//---- getters -----//

	//---- setters ----//
	void set_bias(double bias_val) { this->bias_ = bias_val; }

};

}

#endif /* SRC_WEIGTHS_H_ */
