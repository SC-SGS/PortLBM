/**
 * @file        exceptions.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declaration of several exceptions for the lattice Boltzmann implementation.
 * 
 * @version     1.1
 * 
 * @date        December 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP

// Format
#include <fmt/core.h>

// Standary library
#include <stdexcept>                                                   
#include <string_view>                            

/*
 * Careful: At the time this file was created, the used Clang 18.1.3 compiler includes the `source_location.hpp` header
 *          as an experimental feature. This may change in future versions of Clang, and the include direction may change
 *          to `<source_location>`.
 */
//#include <experimental/source_location>
#include <source_location> // tested under clang 18.1.8 with AMDGPU target

namespace lbm
{
    /**
     * @brief This namespace contains various exceptions for more precise error handling related to the lattice Boltzmann
     *        implementation.
     */
    namespace exceptions
    {
        /**
         * @brief This general exception class is essentially a `std::runtime_error` with a `std::source_location`. 
         */
        class Exception : public std::runtime_error
        {
            std::string_view exception_name;
            std::source_location source_location;
            
            public:

                explicit Exception
                (
                    const std::string &message, 
                    const std::string_view &exception_name = "Exception", 
                    const std::source_location &source_location = std::source_location::current()
                );

                /**
                 * @brief Returns a string containing relevant information on the thrown exception.
                 *        It is intended to be printed to the console.
                 *        The following information is included (in this order):
                 *  
                 *        - The name of the exception
                 * 
                 *        - The `what()` information of the underlying `std::runtime_error`
                 * 
                 *        - The file in which the concerned line is located
                 * 
                 *        - The concerned function causing the exception to be thrown.
                 * 
                 *        - The concerned line 
                 * 
                 * @return a string containing the aforementioned information
                 */
                std::string to_string() const;
        };

        /**
         * @brief This namespace contains exceptions related to the interaction with JSON files.
         */
        namespace json
        {
            /**
             * @brief This exception is thrown when a read JSON file has one or more properties with illegal values.
             *        For example, this includes negative domain extents or density values, or `NULL`.
             */
            class PropertyArgumentException : public Exception
            {
                public:
                    
                    explicit PropertyArgumentException
                    (
                        const std::string &message,
                        const std::source_location &source_location = std::source_location::current()
                    );
            };

            /**
             * @brief This exception is thrown if one or more properties are not specified in a read JSON file.
             */
            class MissingPropertyException : public Exception
            {
                public:

                    explicit MissingPropertyException
                    (
                        const std::string &message,
                        const std::source_location &source_location = std::source_location::current()
                    );
            };

            /**
             * @brief This exception is thrown if one or more unknown properties are specified in a read JSON file.
             */
            class UnknownPropertyException : public Exception
            {
                public:

                    explicit UnknownPropertyException
                    (
                        const std::string &message, 
                        const std::source_location &source_location = std::source_location::current()
                    );
            };
        } // ! namespace json

        /**
         * @brief This namespace contains exceptions related to illegal values of macroscopic observables.
         */
        namespace observables
        {
            /**
             * @brief This exception is thrown when the value of a macroscopic observable is outside its allowed range.
             *        The rules of physics and the defined range of `double` are considered, and the stronger condition is guiding.
             *        The following restrictions apply:
             * 
             *        - Densities:  Must be larger than `0` but not `nan` or `inf` 
             * 
             *        - Velocities: Must not include a component that is `nan` or `inf` 
             *
             */
            class OutOfBoundsException : public Exception
            {
                public:

                    explicit OutOfBoundsException
                    (
                        const std::string &message, 
                        const std::source_location &source_location= std::source_location::current()
                    );
            };
        } // ! namespace observables

        /**
         * @brief This namespace contains exceptions related to illegal interactions with the simulation domain.
         */
        namespace domain
        {
            /**
             * @brief This exception is thrown when an access to a node outside of the simulation domain is attempted.
             *        Possible cases include negative or beyond-limitation node coordinates, or otherwise illegal requests.
             */
            class OutOfDomainException : public Exception
            {
                public:

                    explicit OutOfDomainException
                    (
                        const std::string &message, 
                        const std::source_location &source_location = std::source_location::current()
                    );
            };
        } // ! namespace domain

        /**
         * @brief This namespace contains exceptions related to the execution of LBM algorithms.
         */
        namespace algorithm
        {
            /**
             * @brief   This exception is thrown when a thread waits for an algorithm to finish its execution although
             *          the target algorithm has not commenced execution or it will never be able to finish.
             */
            class WaitException : public Exception
            {
                public:

                    explicit WaitException
                    (
                        const std::string &message, 
                        const std::source_location &source_location = std::source_location::current()
                    );
            };
        } // ! namespace algorithm

    } // ! namespace exceptions

} // ! namespace lbm

#endif // ! EXCEPTIONS_HPP