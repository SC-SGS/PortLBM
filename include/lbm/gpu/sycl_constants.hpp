/**
 * @brief       This header file contains constants for SYCL functionality used throughout the project.
 *
 * @copyright   Copyright (c) 2026 Alexander Strack
 */

#ifndef SYCL_CONSTANTS_HPP
#define SYCL_CONSTANTS_HPP

#include <sycl/sycl.hpp>

namespace lbm
{

namespace gpu
{

namespace constants
{

constexpr auto read = sycl::access::mode::read;
constexpr auto write = sycl::access::mode::write;
constexpr auto read_write = sycl::access::mode::read_write;

}  // namespace constants

}  // namespace gpu

}  // namespace lbm

#endif  // ! SYCL_CONSTANTS_HPP
