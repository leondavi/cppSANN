# cppSANN

This library is a simple implementation of artificial neural network.
It is designed as a light library in order to make the integration of it easier.
It is based on [Eigen library](https://eigen.tuxfamily.org/dox/). 
The components of the neural network accompained with comments and explanation
to mitigate the learning curve. Expanding this library should be also simple.  

### Requiremnts:

1. Eigen library located in default location (or use symbolic link to real location). 
   In ubuntu: ```sudo apt install libeigen3-dev ```
2. Boost library: ```sudo apt-get install libboost-all-dev```
3. scons build tool: ```sudo apt-get install -y scons```

### Build Clean and Run:

1. Build by running: ```scons``` in main directory. 
2. Build as shared library by running: ```scons shared=1```.
3. Run tests by calling ```build/cppSANN_exec```.
4. Clean the project: ```scons -c``` in main directory. 

## License 
cppSANN is given for free. (You should inspect Eigen library license also), 
Please cite cppSANN if you use it in your research or application. 
We don't take any responsibility for using this library. 


You are welcome to submit issues. 
