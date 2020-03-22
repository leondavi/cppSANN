/*
 * Weigths.h
 *
 *  Created on: Dec 6, 2019
 *      Author: david
 */

#ifndef SRC_WEIGTHS_H_
#define SRC_WEIGTHS_H_

#include <eigen3/Eigen/Dense>

#include "math_methods.h"

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
	VectorXd bias_;

public:

	//TODO add custom weights by function
	Weights(uint32_t rows, uint32_t cols,bool random_init = true, double mu = 0, double sig = 1);
	Weights(uint32_t rows, uint32_t cols,double weights_val,double bias_val);
	Weights(MatrixXd weights_mat,VectorXd bias) : weights_mat_(weights_mat),bias_(bias) { }

	~Weights() {}

	//---- getters -----//

	MatrixXd* get_weights_mat_ptr() { return &(this->weights_mat_); }
	VectorXd* get_bias_ptr() { return &(this->bias_); }


	//---- setters ----//
	void set_bias(VectorXd bias_val) { this->bias_ = bias_val; }
	void set_weights(MatrixXd new_weights) { this->weights_mat_ = new_weights; }
	void subtract_weights(MatrixXd diff_weights) { this->weights_mat_ -= diff_weights; }
	void subtract_bias(VectorXd diff) {this->bias_ -= diff;}

	VectorXd dot(VectorXd given_vec)
	{
		return ExtMath::dot(this->weights_mat_,given_vec);
	}



};

}

#endif /* SRC_WEIGTHS_H_ */
