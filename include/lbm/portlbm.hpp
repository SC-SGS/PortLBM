/**
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
 *                - lbm::create_handler()                 — factory (hides SYCL)
 *
 *              Utilities:
 *                - lbm::file_interaction::*  — JSON ↔ Properties helpers
 *                - lbm::exceptions::*        — exception hierarchy
 *
 * @copyright   Copyright (c) 2025 Marcel Graf
 *              Copyright (c) 2026 Alexander Strack
 */

#ifndef PORTLBM_HPP
#define PORTLBM_HPP

// Version
#include "version.hpp"

// Core — SYCL-free (safe to include without a device)
#include "core/access.hpp"
#include "core/constants.hpp"
#include "core/domain.hpp"
#include "core/domain_initialization.hpp"
#include "core/properties.hpp"
// Core — SYCL-dependent
#include "core/simulation.hpp"
#include "core/timer.hpp"

// Execution
#include "execution/algorithm_handler.hpp"
#include "execution/sycl_algorithm.hpp"
#include "execution/sycl_algorithm_handler.hpp"

// Factory — preferred entry point; hides SYCL from call sites
#include "portlbm_factory.hpp"

// File I/O
#include "file_interaction/file_interaction.hpp"

// Exceptions
#include "exceptions/exceptions.hpp"

#endif  // !PORTLBM_HPP
