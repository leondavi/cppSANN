/*
 * Layer.h
 *
 *  Created on: Dec 6, 2019
 *      Author: David Leon
 */

#ifndef SRC_LAYER_H_
#define SRC_LAYER_H_

#include <eigen3/Eigen/Dense>
#include <stdlib.h>
#include <memory>
#include "Weigths.h"

using namespace Eigen;

class Layer {


private:
	std::weak_ptr<Layer> former_layer;
	std::weak_ptr<Layer> later_layer;
public:
	Layer();
	virtual ~Layer();
};

#endif /* SRC_LAYER_H_ */
