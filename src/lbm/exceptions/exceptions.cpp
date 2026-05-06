/**
 * @file        exceptions.cpp
 *
 * @author      Marcel Graf
 *
 * @brief       This source file contains the definitions of the LBM exceptions.
 *
 * @version     1.2
 *
 * @date        March 2025
 *
 * @copyright   Copyright (c) Marcel Graf
 *
 */

#include "../../../include/lbm/exceptions/exceptions.hpp"

lbm::exceptions::Exception::Exception(
    const std::string &message, const std::string_view &exception_name, const std::source_location &source_location) :
    std::runtime_error(message),
    exception_name(exception_name),
    source_location(source_location){};

std::string lbm::exceptions::Exception::to_string() const
{
    return fmt::format(
        "{} thrown:\n"
        "-------------------------------------------------------------------------------\n"
        "What:\t\t{}\n"
        "File:\t\t{}\n"
        "Function:\t{}\n"
        "Line:\t\t{}\n",
        exception_name,
        this->what(),
        source_location.file_name(),
        source_location.function_name(),
        source_location.line());
}

lbm::exceptions::json::PropertyArgumentException::PropertyArgumentException(
    const std::string &message, const std::source_location &source_location) :
    lbm::exceptions::Exception(message, "JSON Property Argument Exception", source_location){};

lbm::exceptions::json::MissingPropertyException::MissingPropertyException(const std::string &message,
                                                                          const std::source_location &source_location) :
    lbm::exceptions::Exception(message, "JSON Missing Property Exception", source_location){};

lbm::exceptions::json::UnknownPropertyException::UnknownPropertyException(const std::string &message,
                                                                          const std::source_location &source_location) :
    lbm::exceptions::Exception(message, "JSON Unknown Property Exception", source_location){};

lbm::exceptions::observables::OutOfBoundsException::OutOfBoundsException(const std::string &message,
                                                                         const std::source_location &source_location) :
    lbm::exceptions::Exception(message, "Observables Out Of Bounds Exception", source_location){};

lbm::exceptions::domain::OutOfDomainException::OutOfDomainException(const std::string &message,
                                                                    const std::source_location &source_location) :
    lbm::exceptions::Exception(message, "Out Of Domain Exception", source_location){};

lbm::exceptions::algorithm::WaitException::WaitException(const std::string &message,
                                                         const std::source_location &source_location) :
    lbm::exceptions::Exception(message, "Wait Exception", source_location){};
