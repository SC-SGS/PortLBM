/**
 * @brief       Factory functions for creating `AlgorithmHandler` instances.
 *
 *              This header intentionally has **no SYCL dependency**. It exposes
 *              only the abstract `AlgorithmHandler` interface and the SYCL-free
 *              `Properties` type.  The SYCL headers are confined to the
 *              accompanying `portlbm_factory.cpp` translation unit, so consumers
 *              do not need to link against or compile any SYCL code just to call
 *              the factory.
 *
 *              Typical usage (programmatic):
 *              @code
 *              lbm::core::Properties props("lptl", "stream",
 *                  false, 64, 1000, 10, "Hagen-Poiseuille", 62, 254,
 *                  0.0, 0.0, 1.005, 0.0, 0.0, 1.0, 1.0);
 *              props.validate();
 *              auto handler = lbm::create_handler(props);
 *              handler->start();
 *              handler->block_until_finished();
 *              @endcode
 *
 *              Typical usage (JSON file):
 *              @code
 *              auto handler = lbm::create_handler("settings/settings.json");
 *              @endcode
 *
 * @copyright   Copyright (c) 2026 Alexander Strack
 */

#ifndef LBM_PORTLBM_FACTORY_HPP
#define LBM_PORTLBM_FACTORY_HPP

#include "execution/algorithm_handler.hpp"  // includes properties.hpp + domain.hpp
#include <memory>
#include <string>

namespace lbm
{

/**
 * @brief   Creates a ready-to-use `AlgorithmHandler` from a pre-built
 *          `Properties` object.
 *
 * The returned handler is fully initialised and ready for `start()`.
 * Call `props.validate()` before passing to get clear error messages
 * for any out-of-range or unknown values.
 *
 * The concrete type of the returned object is `SYCLAlgorithmHandler`, but
 * callers should treat it as the opaque `AlgorithmHandler` interface so that
 * future backends can be added without changing call sites.
 *
 * @param[in]   props   fully-populated simulation parameters
 *
 * @return  owning pointer to the initialised handler
 *
 * @throws  lbm::exceptions::json::PropertyArgumentException   if `work_group_size`
 *          exceeds the device maximum
 */
std::unique_ptr<execution::AlgorithmHandler> create_handler(const core::Properties &props);

/**
 * @brief   Creates a ready-to-use `AlgorithmHandler` from a JSON settings file.
 *
 * Equivalent to reading the file with `file_interaction::json_to_properties()`
 * and forwarding to the Properties overload, but avoids the extra include.
 *
 * @param[in]   settings_path   path to the JSON settings file
 *
 * @return  owning pointer to the initialised handler
 *
 * @throws  lbm::exceptions::json::PropertyArgumentException   on invalid settings
 */
std::unique_ptr<execution::AlgorithmHandler> create_handler(const std::string &settings_path);

}  // namespace lbm

#endif  // LBM_PORTLBM_FACTORY_HPP
