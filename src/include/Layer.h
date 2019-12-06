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

/**
 * This class represents a single layer
 */
class Layer
{
private:
	std::weak_ptr<Layer> previous_layer_ptr_;
	std::weak_ptr<Layer> next_layer_ptr_;
public:
	Layer(std::weak_ptr<Layer> previous_layer_ptr,std::weak_ptr<Layer> next_layer_ptr) :
		previous_layer_ptr_(previous_layer_ptr),next_layer_ptr_(next_layer_ptr)

	{

	}
	virtual ~Layer();

	inline std::weak_ptr<Layer> get_previous_layer_ptr() { return this->previous_layer_ptr_; }
	inline std::weak_ptr<Layer> get_next_layer_ptr() { return this->next_layer_ptr_; }

};//end of Layer


class InputLayer :  public Layer
{
private:


public:

};//end of InputLayer

#endif /* SRC_LAYER_H_ */
