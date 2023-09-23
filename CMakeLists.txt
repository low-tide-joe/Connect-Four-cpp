cmake_minimum_required(VERSION 3.4...3.18)
project(ConnectFourBitboard)

add_library(game STATIC src/game.cpp)
set_target_properties(game PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(pybind11)
pybind11_add_module(ConnectFourBitboard src/binder.cpp)
target_link_libraries(ConnectFourBitboard PRIVATE game)
