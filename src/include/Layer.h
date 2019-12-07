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

#include "Weights.h"

using namespace Eigen;

/**
 * This class represents a single layer
 */
class Layer
{
private:
	uint32_t layer_size_;
	VectorXd neurons_;

	std::weak_ptr<Layer> previous_layer_ptr_;
	std::weak_ptr<Layer> next_layer_ptr_;
	std::weak_ptr<Weights> input_weights_ptr_;
	std::weak_ptr<Weights> output_weights_ptr_;




public:

	Layer(uint32_t layer_size,std::weak_ptr<Layer> previous_layer_ptr,std::weak_ptr<Layer> next_layer_ptr) :
		layer_size_(layer_size),neurons_(layer_size),
		previous_layer_ptr_(previous_layer_ptr),next_layer_ptr_(next_layer_ptr)
	{

	}
	virtual ~Layer() {}

	//---- getters ----//
	virtual inline std::weak_ptr<Layer> get_previous_layer_ptr() { return this->previous_layer_ptr_; }
	virtual inline std::weak_ptr<Layer> get_next_layer_ptr() { return this->next_layer_ptr_; }
	virtual inline std::weak_ptr<Weights> get_input_weights_ptr() { return this->input_weights_ptr_; }
	virtual inline std::weak_ptr<Weights> get_output_weights_ptr() { return this->output_weights_ptr_; }


	//---- setters ----//



};//end of Layer


class InputLayer :  public Layer
{
private:


public:

	InputLayer(uint32_t layer_size,std::weak_ptr<Layer> next_layer);

	//---- getters ----//

	inline std::weak_ptr<Layer> get_previous_layer_ptr() override { return std::weak_ptr<Layer>(); }//No previous in this layer
	inline std::weak_ptr<Weights> get_input_weights_ptr() override { return std::weak_ptr<Weights>(); }//No input weights to this layer


};//end of InputLayer


class OutputLayer :  public Layer
{
private:


public:
	OutputLayer(uint32_t layer_size,std::weak_ptr<Layer> previous_layer);

	//---- getters ----//

	inline std::weak_ptr<Layer> get_next_layer_ptr() override { return std::weak_ptr<Layer>(); }//No next layer after the output layer
	inline std::weak_ptr<Weights> get_output_weights_ptr() override { return std::weak_ptr<Weights>(); }//No output weights to this layer


};//end of InputLayer

#endif /* SRC_LAYER_H_ */
