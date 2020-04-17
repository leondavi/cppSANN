


#include "tests/default_test.h"
#include <cerrno>

std::vector<std::string> test_type = {"Default","Autoencoder"};
enum {DEFAULT_TEST,AUTOENCODER_TEST};

int main(int argc, char * argv[])
{
	uint32_t test_case = 0;
	if(argc>1)
	{
		std::string input(argv[1]);
		for(uint32_t i=0; i<test_type.size(); i++)
		{
			if (input == test_type[i])
			{
				test_case = i;
			}
		}
	}

	std::cout<<"Running "<<test_type[test_case]<<" test"<<std::endl;



	switch(test_case)
	{
		case DEFAULT_TEST: { return deafault_test(); }
		case AUTOENCODER_TEST: { return 0; }
	}

	return EINVAL;

}
