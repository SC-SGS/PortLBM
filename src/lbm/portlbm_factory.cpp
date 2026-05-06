/**
 * @file        portlbm_factory.cpp
 *
 * @author      Marcel Graf
 *
 * @brief       Implementation of the PortLBM factory functions.
 *
 *              This is the **only** translation unit that includes
 *              `sycl_algorithm_handler.hpp`, and therefore the only one that
 *              pulls in SYCL headers on behalf of library consumers.  Everything
 *              else in the public API remains SYCL-free.
 *
 * @version     1.0
 *
 * @date        May 2025
 *
 * @copyright   Copyright (c) 2025 Marcel Graf
 */

#include "lbm/portlbm_factory.hpp"

// This include is the only place in the public surface that drags in SYCL.
#include "lbm/execution/sycl_algorithm_handler.hpp"

std::unique_ptr<lbm::execution::AlgorithmHandler> lbm::create_handler(const lbm::core::Properties &props)
{
    return std::make_unique<lbm::execution::SYCLAlgorithmHandler>(props);
}

std::unique_ptr<lbm::execution::AlgorithmHandler> lbm::create_handler(const std::string &settings_path)
{
    return std::make_unique<lbm::execution::SYCLAlgorithmHandler>(settings_path);
}
