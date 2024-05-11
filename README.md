# Delta Connect 
The goal is to build a connect four simulator game in C++, expose the classes to python and then use python ML libraries to create deep learning models and optimize a winning connect 4 strategy through self-play.

To generate the wheel files using cmake:
* git clone https://github.com/pybind/pybind11.git
* mkdir build
* cd build
* cmake ..
* make game
* make ConnectFourBitboard
* if all the above worked then: "python ../setup.py bdist_wheel" should work
* pip install ../dist/ConnectFourBitboard[version].whl