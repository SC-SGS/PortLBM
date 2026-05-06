/**
 * @file        portlbm.hpp
 *
 * @brief       Umbrella include for the PortLBM library.
 *              Including this single header pulls in the complete public API.
 *
 *              Core simulation infrastructure:
 *                - lbm::core::Properties  — simulation parameters
 *                - lbm::core::Domain      — grid/decomposition metadata
 *                - lbm::core::Simulation  — top-level data container
 *                - lbm::core::Control     — iteration control / progress
 *                - lbm::core::Results     — CPU/GPU result arrays
 *
 *              Execution:
 *                - lbm::execution::AlgorithmHandler      — abstract interface
 *                - lbm::execution::SYCLAlgorithmHandler  — SYCL implementation
 *
 *              Utilities:
 *                - lbm::file_interaction::*  — JSON ↔ Properties helpers
 *                - lbm::exceptions::*        — exception hierarchy
 *
 * @copyright   Copyright (c) Marcel Graf
 */

#ifndef PORTLBM_HPP
#define PORTLBM_HPP

// Version
#include "version.hpp"

// Core (SYCL-free headers first — safe to include without a device)
#include "core/constants.hpp"
#include "core/domain.hpp"
#include "core/properties.hpp"
// Core (SYCL-dependent)
#include "core/access.hpp"
#include "core/domain_initialization.hpp"
#include "core/simulation.hpp"
#include "core/timer.hpp"

// Execution
#include "execution/algorithm_handler.hpp"
#include "execution/sycl_algorithm.hpp"
#include "execution/sycl_algorithm_handler.hpp"

// File I/O
#include "file_interaction/file_interaction.hpp"

// Exceptions
#include "exceptions/exceptions.hpp"

#endif  // !PORTLBM_HPP
