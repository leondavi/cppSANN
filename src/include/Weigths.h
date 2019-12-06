/*
 * Weigths.h
 *
 *  Created on: Dec 6, 2019
 *      Author: david
 */

#ifndef SRC_WEIGTHS_H_
#define SRC_WEIGTHS_H_

#include <eigen3/Eigen/Dense>

using namespace Eigen;

class Weigths {
	MatrixXd weights_mat;
public:
	Weigths();
	virtual ~Weigths();
};

#endif /* SRC_WEIGTHS_H_ */
