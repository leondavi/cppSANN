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

namespace ANN
{
/**
 * This class represents a single layer
 */
class Layer : std::enable_shared_from_this<Layer>
{
private:
	uint32_t layer_size_;
	VectorXd neurons_; //column vector
	bool connected;

	std::weak_ptr<Layer> previous_layer_ptr_;
	std::weak_ptr<Layer> next_layer_ptr_;
	std::shared_ptr<Weights> input_weights_ptr_;
	std::shared_ptr<Weights> output_weights_ptr_;

	inline void set_connected() { this->connected = true; }
	inline void set_disconnected() { this->connected = false; }

	std::shared_ptr<Layer> getptr() { return std::shared_ptr<Layer>(this); }


public:

	Layer(uint32_t layer_size,std::weak_ptr<Layer> previous_layer_ptr,std::weak_ptr<Layer> next_layer_ptr) :
		layer_size_(layer_size),neurons_(layer_size),connected(false),
		previous_layer_ptr_(previous_layer_ptr),next_layer_ptr_(next_layer_ptr)
	{

	}
	virtual ~Layer() {}

	//---- getters ----//
	virtual inline std::weak_ptr<Layer> get_previous_layer_ptr() { return this->previous_layer_ptr_; }
	virtual inline std::weak_ptr<Layer> get_next_layer_ptr() { return this->next_layer_ptr_; }
	virtual inline std::shared_ptr<Weights> get_input_weights_ptr() { return this->input_weights_ptr_; }
	virtual inline std::shared_ptr<Weights> get_output_weights_ptr() { return this->output_weights_ptr_; }

	inline uint32_t get_layer_size() { return this->layer_size_; }

	bool is_connected() { return this->connected; }


	virtual bool connect_next(std::weak_ptr<Layer> next_layer,
							  std::weak_ptr<Weights> output_weights_ptr_ =  std::weak_ptr<Weights>());

	//---- setters ----//

	virtual inline void set_previous_layer_ptr(std::weak_ptr<Layer> previous_layer_ptr) { this->previous_layer_ptr_ = previous_layer_ptr; }




};//end of Layer


class InputLayer :  public Layer
{
private:


public:

	InputLayer(uint32_t layer_size,std::weak_ptr<Layer> next_layer);

	//---- getters ----//

	inline std::weak_ptr<Layer> get_previous_layer_ptr() override { return std::weak_ptr<Layer>(); }//No previous in this layer
	inline std::shared_ptr<Weights> get_input_weights_ptr() override { return std::shared_ptr<Weights>(); }//No input weights to this layer

	inline void set_previous_layer_ptr(std::weak_ptr<Layer> previous_layer_ptr) override { Layer::set_previous_layer_ptr(std::weak_ptr<Layer>()); }


};//end of InputLayer


class OutputLayer :  public Layer
{
private:


public:
	OutputLayer(uint32_t layer_size,std::weak_ptr<Layer> previous_layer);

	//---- getters ----//

	inline std::weak_ptr<Layer> get_next_layer_ptr() override { return std::weak_ptr<Layer>(); }//No next layer after the output layer
	inline std::shared_ptr<Weights> get_output_weights_ptr() override { return std::shared_ptr<Weights>(); }//No output weights to this layer

	inline bool connect_next(std::weak_ptr<Layer> next_layer,
								  std::shared_ptr<Weights> output_weights_ptr_ =  std::shared_ptr<Weights>())
	{
		throw std::runtime_error("No connect next for OutpoutLayer");
	}

};//end of InputLayer

}//end of namespace ANN

#endif /* SRC_LAYER_H_ */
