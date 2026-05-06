/**
 * @file        sycl_constants.hpp
 *
 * @author      Marcel Graf
 *
 * @brief       This header file contains constants for SYCL functionality used throughout the project.
 *
 * @version     1.0
 *
 * @date        November 2024
 *
 * @copyright Copyright (c) 2024
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
