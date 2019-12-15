

#include "include/definitions.hh"
#include "include/Layer.h"
#include "include/activation_functions.h"

int main(int ac, char** av);
/**
 * Examples how to use
 */

int main(int ac, char** av)
{
	MatrixXd m(3,3);
	m << 1, 2, 3,
	     4, 5, 6,
	     7, 8, 9;
	ANN::Weights new_weight(m);
	VectorXd sample_vec(3); sample_vec << 1,2,3;
	std::cout<<"Before: "<<*new_weight.get_weights_mat()<<std::endl;
	MatrixXd dprod_vec = new_weight.dot(sample_vec);

	//sigmoid(dprod_vec);

	std::cout<<"After: "<<*new_weight.get_weights_mat()<<std::endl;
	std::cout<<"sample_vec: \n"<<sample_vec<<std::endl;
	std::cout<<"dprod vec: \n"<<dprod_vec<<std::endl;


	return 0;

}
