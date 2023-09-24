#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "game.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(ConnectFourBitboard, m) {

  py::class_<ConnectFourBitboard>(m, "ConnectFourBitboard")
        .def(py::init<>())
        .def("printBoard", &ConnectFourBitboard::printBoard)
        .def("makeMove", &ConnectFourBitboard::makeMove)
        .def("reset" , &ConnectFourBitboard::reset)
        .def("getPlayerBoardState", &ConnectFourBitboard::getPlayerBoardState)
        .def("getAvailableActions", &ConnectFourBitboard::getAvailableActions)
        .def_readwrite("gameState", &ConnectFourBitboard::gameState)
        .def_readwrite("currentPlayer", &ConnectFourBitboard::currentPlayer);
}