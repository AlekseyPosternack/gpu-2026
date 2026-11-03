#ifndef __GELU_OMP_H
#define __GELU_OMP_H
#pragma once
#include <cmath>
#include <omp.h>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float> &input);

#endif // __GELU_OMP_H