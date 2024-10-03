#ifndef UTILS_HPP
#define UTILS_HPP

#include "access.hpp"
#include "defines.hpp"

#include <array>
#include <vector>
#include <set>
#include <iostream>
#include <iomanip>
#include <string>

/**
 * @brief This function is used to access matrices that are stored in a row-major-order container.
 * 
 * @param row row of the element that is to be accessed
 * @param column column of the element that is to be accessed
 * @param column_count number of columns stored in the data structure
 * @return unsigned int the index of the requested element
 */
inline unsigned int matrix_access
(
    unsigned int row, 
    unsigned int column, 
    unsigned int column_count)
{
    return row * column_count + column;
}

/**
 * @brief This namespace contains multiple utility functions that allow the user
 *        to print a data structure to the console.
 */
namespace to_console
{

    /**
     * @brief Allows to print out a vector representing a data layout whose column count is HORIZONTAL_NODES.
     *        Notice that the vector is assumed to represent a matrix.
     * 
     * @tparam T the type of the objects the specified vector holds (must be numeric)
     * @param vector the vector that is to be printed in the console
     */
    template <typename T>
    void print_vector(const std::vector<T> &vector)
    {
        for(auto y = VERTICAL_NODES; y-- > 0; )
        {
            for(auto x = 0; x < HORIZONTAL_NODES; ++x)
            {
                if(x == 0 && y == 0) std::cout << "\033[31m";
                else if(x == (HORIZONTAL_NODES - 1) && y == (VERTICAL_NODES -1)) std::cout << "\033[34m";
                std::cout << vector[matrix_access(y,x, HORIZONTAL_NODES)];
                std::cout << "\t\033[0m";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    } 

    /**
     * @brief Adapted version of print_vector that supports a custom column count.
     *        In order to enforce that the vector is printed as a row vector, specify a column count
     *        that exceeds the length of the vector, e.g. std::numeric_limits<int>::max_value()
     * 
     * @tparam T the type of the objects the specified vector holds (must be numeric)
     * @param vector the vector that is to be printed in the console
     * @param row_length a line break will be set after this many entries
     */
    template <typename T>
    void print_vector(const std::vector<T> &vector, unsigned int row_length)
    {
        int current_value_count = 0;
        std::cout << "[";
        for(auto i = 0; i < vector.size(); ++i)
        {
            if(current_value_count == row_length)
            {
                current_value_count = 0;
                std::cout << std::endl;
            }
            std::cout << vector[i];
            std::cout << "\t";
            current_value_count++;
        }
        std::cout << "]";
        std::cout << std::endl;
    }

    /**
     * @brief Adapted version of print_vector that prints out the phase vector of a lattice.
     *        If a node is solid (i.e. the entry is true), it is represented by #.
     *        If a node is fluid (i.e. the entry is false), it is represented by ~.
     * 
     * @param vector the phase vector
     */
    inline void print_phase_vector(const std::vector<bool> &vector)
    {
        for(auto y = VERTICAL_NODES; y-- > 0; )
        {
            for(auto x = 0; x < HORIZONTAL_NODES; ++x)
            {
                if(vector[matrix_access(y,x, HORIZONTAL_NODES)]) std::cout << "\033[32m#\033[0m";
                else std::cout << "\033[34m~\033[0m"; 
                std::cout << " ";
            }
            std::cout << std::endl;
        }
    } 

    /**
     * @brief Prints all velocity values in the lattice to the console.
     *        All values are printed in order, i.e. the origin is located at the lower left corner of the output.
     * 
     * @param vector a vector containing all velocity values
     */
    inline void print_velocity_vector(const std::vector<velocity> &vector)
    {
        for(auto y = VERTICAL_NODES; y-- > 0; )
        {
            for(auto x = 0; x < HORIZONTAL_NODES; ++x)
            {
                if(x == 0 && y == 0) std::cout << "\033[31m";
                else if(x == (HORIZONTAL_NODES - 1) && y == (VERTICAL_NODES -1)) std::cout << "\033[34m";
                std::cout << "("<< vector[matrix_access(y,x, HORIZONTAL_NODES)][0] << ", " << vector[matrix_access(y,x, HORIZONTAL_NODES)][1] << ")";
                std::cout << "\t  \033[0m";
                std::cout << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    } 

    /**
     * @brief Prints the contents of a set to the console.
     * 
     * @tparam T the type of the objects the set holds (must be compatible with the console)
     * @param set the set whose contents are to be printed
     */
    template <typename T>
    void print_set(const std::set<T> &set)
    {
        std::cout << "(";
        for(const auto element : set)
        {
            std::cout << element;
            std::cout << ", ";
        }
        std::cout << ")";
        std::cout << std::endl;
    }

    /**
     * @brief Prints all distribution values in to the console.
     *        They are displayed in the original order, i.e. the origin is located at the lower left corner of the printed distribution chart.
     * 
     * @param distribution_values a vector containing the distribution values of all nodes 
     * @param access_function the function used to access the distribution values
     */
    template <class T> inline 
    void print_distribution_values
    (
        const std::vector<double> &distribution_values, 
        const T &lbm_accessor
    )
    {
        std::cout << "Trying to print results " << std::endl;
        std::vector<std::vector<unsigned int>> print_dirs = {{6,7,8}, {3,4,5}, {0,1,2}};
        unsigned int current_node_index = 0;
        unsigned int previous_direction = 0;
        std::vector<double> current_values(9,0);
        std::cout << std::setprecision(3) << std::fixed;

        for(auto y = VERTICAL_NODES; y-- > 0; )
        {
            for(auto i = 0; i < 3; ++i)
            {
                auto current_row = print_dirs[i];
                for(auto x = 0; x < HORIZONTAL_NODES; ++x)
                {
                    //std::cout << "Currently at node with coords (" << x << ", " << y << ")" << std::endl;
                    if(x == 0 && y == 0) std::cout << "\033[31m";
                    else if(x == (HORIZONTAL_NODES - 1) && y == (VERTICAL_NODES -1)) std::cout << "\033[34m";
                    current_node_index = lbm_access::get_node_index(x, y);
                    current_values = lbm_access::get_distribution_values_of(distribution_values, current_node_index, lbm_accessor);

                    for(auto j = 0; j < 3; ++j)
                    {
                        auto direction = current_row[j];
                        std::cout << current_values[direction] << "  ";
                    }
                    std::cout << "\t\033[0m";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }

    /**
     * @brief Prints all distribution values in to the console.
     *        They are displayed in the original order, i.e. the origin is located at the lower left corner of the printed distribution chart.
     * 
     * @param distribution_values a vector containing the distribution values of all nodes 
     * @param access_function the function used to access the distribution values
     */
    inline void print_distribution_values
    (
        const std::vector<double> &distribution_values, 
        const access_function access_function
    )
    {
        std::cout << "Trying to print results " << std::endl;
        std::vector<std::vector<unsigned int>> print_dirs = {{6,7,8}, {3,4,5}, {0,1,2}};
        unsigned int current_node_index = 0;
        unsigned int previous_direction = 0;
        std::vector<double> current_values(9,0);
        std::cout << std::setprecision(3) << std::fixed;

        for(auto y = VERTICAL_NODES; y-- > 0; )
        {
            for(auto i = 0; i < 3; ++i)
            {
                auto current_row = print_dirs[i];
                for(auto x = 0; x < HORIZONTAL_NODES; ++x)
                {
                    //std::cout << "Currently at node with coords (" << x << ", " << y << ")" << std::endl;
                    if(x == 0 && y == 0) std::cout << "\033[31m";
                    else if(x == (HORIZONTAL_NODES - 1) && y == (VERTICAL_NODES -1)) std::cout << "\033[34m";
                    current_node_index = lbm_access::get_node_index(x, y);
                    current_values = lbm_access::get_distribution_values_of(distribution_values, current_node_index, access_function);

                    for(auto j = 0; j < 3; ++j)
                    {
                        auto direction = current_row[j];
                        std::cout << current_values[direction] << "  ";
                    }
                    std::cout << "\t\033[0m";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }

    /**
     * @brief Prints the greeting message displayed when running an algorithm.
     * 
     * @param algorithm_name A string containing the display name of the algorithm
     * @param iterations This many iterations are to be executed
     */
    inline void print_run_greeting(const std::string &algorithm_name, const unsigned int iterations)
    {
        std::cout << "------------------------------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "Now running " << algorithm_name << " for " << iterations << " iterations." << std::endl;
        std::cout << std::endl;
    } 

    /**
     * @brief Prints the simulation results, i.e. the velocity vectors and density values, for all time steps.
     * 
     * @param results a vector containing the simulation data tuples.
     */
    inline void print_simulation_results(std::vector<sim_data_tuple> &results)
    {
        unsigned int iterations = results.size();
        std::cout << std::endl;
        std::cout << "Velocity values: " << std::endl;
        std::cout << std::endl;
        for(auto i = 0; i < iterations; ++i)
        {
            std::cout << "t = " << i << std::endl;
            std::cout << "-------------------------------------------------------------------------------- " << std::endl;
            to_console::print_velocity_vector(std::get<0>(results[i]));
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;

        std::cout << "Density values: " << std::endl;
        std::cout << std::endl;
        
        for(auto i = 0; i < iterations; ++i)
        {
            std::cout << "t = " << i << std::endl;
            std::cout << "-------------------------------------------------------------------------------- " << std::endl;
            to_console::print_vector(std::get<1>(results[i]));
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    /**
     * @brief Prints the message to the console that explains what the colors in the console output stand for.
     * 
     */
    inline void print_ansi_color_message()
    {
        std::cout << "This program utilizes ANSI color codes to output colored text. " 
                  << "If your command line does not support those codes, your output may be corrupted." 
                  << std::endl;
        std::cout << "In all following prints showing the entire simulation domain, "; 
        std::cout << "the origin will be marked in \033[31mred\033[0m and the outmost coordinate will be marked in \033[34mblue\033[0m." << std::endl;
        std::cout << "Milestones will be marked in \033[33myellow\033[0m." << std::endl;
        std::cout << "In the case of parallel implementations, buffer nodes will be marked in \033[32mgreen\033[0m." << std::endl;
        std::cout << std::endl;
    }

    /**
     * @brief This namespace contains functions that are meant for the visualization of the parallel algorithms.
     * 
     */
    namespace buffered
    {

        /**
         * @brief Prints all velocity values in the lattice to the console.
         *        All values are printed in order, i.e. the origin is located at the lower left corner of the output.
         * 
         * @param vector a vector containing all velocity values
         */
        inline void print_velocity_vector(const std::vector<velocity> &vector)
        {
            std::cout << std::setprecision(5) << std::fixed;
            unsigned int line_counter = 0;
        
            for(auto y = VERTICAL_NODES; y-- > 0; )
            {
                if(line_counter == SUBDOMAIN_HEIGHT) std::cout << "\033[32m";
                for(auto x = 0; x < HORIZONTAL_NODES; ++x)
                {
                    if(x == 0 && y == 0) std::cout << "\033[31m";
                    else if(x == (HORIZONTAL_NODES - 1) && y == (VERTICAL_NODES -1)) std::cout << "\033[34m";
                    std::cout << "("<< vector[matrix_access(y,x, HORIZONTAL_NODES)][0] << ", " << vector[matrix_access(y,x, HORIZONTAL_NODES)][1] << ")";
                    if (line_counter == SUBDOMAIN_HEIGHT) std::cout << "  \t";
                    else std::cout << "  \t\033[0m";
                    std::cout << " ";
                }
                if (line_counter == SUBDOMAIN_HEIGHT) line_counter = 0;
                else line_counter++;
                std::cout << std::endl;
                std::cout << "\033[0m";
            }
            std::cout << std::endl;
            std::cout << std::setprecision(3) << std::fixed;
        } 

        /**
         * @brief Allows to print out a vector representing a data layout whose column count is HORIZONTAL_NODES.
         *        Notice that the vector is assumed to represent a matrix.
         * 
         * @tparam T the type of the objects the specified vector holds (must be numeric)
         * @param vector the vector that is to be printed in the console
         */
        template <typename T>
        void print_vector(const std::vector<T> &vector)
        {
            unsigned int line_counter = 0;

            for(auto y = VERTICAL_NODES ; y-- > 0; )
            {
                if(line_counter == SUBDOMAIN_HEIGHT) std::cout << "\033[32m";
                for(auto x = 0; x < HORIZONTAL_NODES; ++x)
                {
                    if(x == 0 && y == 0) std::cout << "\033[31m";
                    else if(x == (HORIZONTAL_NODES - 1) && y == (VERTICAL_NODES -1)) std::cout << "\033[34m";
                    std::cout << vector[matrix_access(y,x, HORIZONTAL_NODES)];
                    if (line_counter == SUBDOMAIN_HEIGHT) std::cout << "\t";
                    else std::cout << "\t\033[0m";
                }
                if (line_counter == SUBDOMAIN_HEIGHT) line_counter = 0;
                else line_counter++;
                std::cout << std::endl;
                std::cout << "\033[0m";
            }
            std::cout << std::endl;
        } 

        /**
         * @brief Prints the simulation results, i.e. the velocity vectors and density values, for all time steps.
         * 
         * @param results a vector containing the simulation data tuples.
         */
        inline void print_simulation_results(std::vector<sim_data_tuple> &results)
        {
            unsigned int iterations = results.size();
            std::cout << std::endl;
            std::cout << "Velocity values: " << std::endl;
            std::cout << std::endl;
            for(auto i = 0; i < iterations; ++i)
            {
                std::cout << "t = " << i << std::endl;
                std::cout << "-------------------------------------------------------------------------------- " << std::endl;
                to_console::buffered::print_velocity_vector(std::get<0>(results[i]));
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;

            std::cout << "Density values: " << std::endl;
            std::cout << std::endl;
            
            for(auto i = 0; i < iterations; ++i)
            {
                std::cout << "t = " << i << std::endl;
                std::cout << "-------------------------------------------------------------------------------- " << std::endl;
                to_console::buffered::print_vector(std::get<1>(results[i]));
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

    /**
     * @brief Prints all distribution values in to the console.
     *        They are displayed in the original order, i.e. the origin is located at the lower left corner of the printed distribution chart.
     * 
     * @param distribution_values a vector containing the distribution values of all nodes 
     * @param access_function the function used to access the distribution values
     */
    inline void print_distribution_values
    (
        const std::vector<double> &distribution_values, 
        const access_function access_function
    )
    {
        std::vector<std::vector<unsigned int>> print_dirs = {{6,7,8}, {3,4,5}, {0,1,2}};
        unsigned int current_node_index = 0;
        unsigned int previous_direction = 0;
        std::vector<double> current_values(9,0);
        std::cout << std::setprecision(3) << std::fixed;
        unsigned int line_counter = 0;

        for(auto y = VERTICAL_NODES ; y-- > 0; )
        {
            if(line_counter == SUBDOMAIN_HEIGHT)
            {
                std::cout << "\033[32m";
            }
            for(auto i = 0; i < 3; ++i)
            {
                auto current_row = print_dirs[i];
                for(auto x = 0; x < HORIZONTAL_NODES; ++x)
                {
                    //std::cout << "Currently at node with coords (" << x << ", " << y << ")" << std::endl;
                    if(x == 0 || x == 1 || x == HORIZONTAL_NODES - 1 || x == HORIZONTAL_NODES - 2) std::cout << std::setprecision(5) << std::fixed;
                    if(x == 0 && y == 0) std::cout << "\033[31m";
                    else if(x == (HORIZONTAL_NODES - 1) && y == (VERTICAL_NODES - 1)) std::cout << "\033[34m";
                    current_node_index = lbm_access::get_node_index(x, y);
                    current_values = lbm_access::get_distribution_values_of(distribution_values, current_node_index, access_function);

                    for(auto j = 0; j < 3; ++j)
                    {
                        auto direction = current_row[j];
                        std::cout << current_values[direction] << "  ";
                    }
                    std::cout << "\t";
                    if((x == 0 && y == 0) || (x == (HORIZONTAL_NODES - 1) && y == (VERTICAL_NODES - 1))) std::cout << "\033[0m";
                    if(x == 0 || x == 1 || x == HORIZONTAL_NODES - 1 || x == HORIZONTAL_NODES - 2) std::cout << std::setprecision(3) << std::fixed;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
            if(line_counter == SUBDOMAIN_HEIGHT) line_counter = 0;
            else line_counter++;
            std::cout << "\033[0m";
        }
        std::cout << std::setprecision(3) << std::fixed;
    }
    }
}

/**
 * @brief This namespace contains some utility functions for vectors.
 */
namespace vec_utils
{
    /**
     * @brief Swaps the entries of the specified vector.
     */
    inline void swap(std::vector<double> &vector, unsigned int a, unsigned int b)
    {
        double temp = vector[a];
        vector[a] = vector[b];
        vector[b] = temp;
    }
}

/**
 * @brief This namespace contains some functions that are convenient for mathematical computations.
 */
namespace math_utils
{
    /**
     * @brief Calculates the dot product of the specified vectors
     */
    template <unsigned long int d>
    double dot(const std::array<double, d> &x, const std::array<double, d> &y)
    {
        double result = 0;
        for(auto i = 0; i < d; ++i) result += x[i] * y[i];
        return result;
    }

    /**
     * @brief Calculates the contraction of two matrices stored as arrays in row-major order.
     */
    template<unsigned int d>
    double contraction(const std::array<double, d> &x, const std::array<double, d> &y)
    {
        double result = 0;
        for(auto i = 0; i < d; ++i) result += x[i] * y[i];
        return result;
    }

    /**
     * @brief Calculates the outer product of the specified two arrays.
     * @return the resulting matrix stored as an array in row-major order.
     */
    template<unsigned int d>
    std::array<double, d*d> outer(const std::array<double, d> &x, const std::array<double, d> &y)
    {
        std::array<double, d*d> result;
        for(auto i = 0; i < d; ++i)
        {
            for(auto j = 0; j < d; ++j)
            {
                result[matrix_access(i,j,d)] = x[i] * y[j];
            }
        }
        return result;
    }
}

/**
 * @brief This debug handler determines whether or not the user wishes to enter a debug mode.
 * 
 * @param input user input
 */
inline bool debug_handler(std::string &input)
{
    std::cout << std::endl;
    if(input == "debug")
    {
        return true;
    }
    else
    {
        std::string selection = input;
        while(selection != "debug" && selection != "quit")
        {
            std::cout << std::endl;
            std::cout << "You have executed this program with the argument '" << selection << "' which is not supported. "; 
            std::cout << "The following keywords are supported: " << std::endl;
            std::cout << "\tType 'debug' to enable debug mode for this program." << std::endl;
            std::cout << "\tType 'no_debug' to disable debug mode for this program." << std::endl;
            std::cout << "\tType anything else to exit this program." << std::endl;
            std::cout << "Please choose one of these options. Running the program without any arguments yields a non-debug run." << std::endl;
            std::cout << std::endl;
            std::cout << "Your selection: ";
            std::cin >> selection;
        }
        if(selection == "debug")
        {
            return true;
        }
        else if(selection == "no_debug")
        {
            return false;
        }
        else
        {
            exit(0);
        }
    }
}

inline std::tuple<bool, bool, unsigned int, unsigned int, double, unsigned int> setup_assistant()
{
    // Additions for the future: Access function, boundary conditions
    bool done = false;
    std::string done_input = "";

    bool enable_debug_mode = false;
    std::string debug_mode_input = "";

    bool enable_debug_distributions = false;
    std::string debug_distributions_input = "";

    unsigned int vertical_nodes = 0;
    unsigned int horizontal_nodes = 0;
    unsigned int relaxation_time = 0;
    unsigned int time_steps = 0;
    bool answered = false;

    std::tuple<bool, bool, unsigned int, unsigned int, double, unsigned int> result(
        enable_debug_mode, 
        enable_debug_distributions, 
        vertical_nodes, 
        horizontal_nodes, 
        relaxation_time, 
        time_steps);

    std::cout << "You have opted for an assisted setup of the simulation. " << std::endl; 
    std::cout << "This assistent will help you specify the crucial parameters of the simulation. " << std::endl; 
    std::cout << "You will be able to confirm that all specifications are okay." << std::endl; 
    std::cout << "If they are not, you will be given the option to re-enter them from the start." << std::endl; 
    std::cout << "In case you have stated an incompatible value, you will be asked for an answer again." << std::endl; 
    std::cout << "-------------------------------------------------------------------------------------------" << std::endl; 

    while(!done)
    {
        answered = false;
    
        while(!answered)
        {
            std::cout << "Enable debug mode? ('y' for 'yes' or 'n' for 'no'): ";
            std::cin >> debug_mode_input; 
            if(debug_mode_input == "y")
            {
                std::get<0>(result) = true;
                answered = true;
            }
            else if(debug_mode_input == "n")
            {
                std::get<0>(result) = false;
                answered = true;
            }
        }

        answered = false;

        while(!answered)
        {
            std::cout << "Enable debug distributions? ('y' for 'yes' or 'n' for 'no'): ";
            std::cin >> debug_distributions_input; 
            if(debug_distributions_input == "y")
            {
                std::get<1>(result) = true;
                answered = true;
            }
            else if(debug_distributions_input == "n")
            {
                std::get<1>(result) = false;
                answered = true;
            }
        }

        answered = false;
        
        while(!answered)
        {
            std::cout << "Vertical nodes (enter an integer value greater than 2): ";
            std::cin >> vertical_nodes; 
            if(vertical_nodes > 2)
            {
                std::get<2>(result) = vertical_nodes;
                answered = true;
            }
        }

        answered = false;

        while(!answered)
        {
            std::cout << "Horizontal nodes (enter an integer value greater than 2): ";
            std::cin >> horizontal_nodes; 
            if(horizontal_nodes > 2)
            {
                std::get<3>(result) = horizontal_nodes;
                answered = true;
            }
        }

        answered = false;

        while(!answered)
        {
            std::cout << "Relaxation time (enter a float value greater than 0, will be cut after 3rd decimal place)" << std::endl;
            std::cout << "[Recommendation: 1.4] > ";
            std::cin >> relaxation_time; 
            if(relaxation_time > 0)
            {
                std::get<4>(result) = relaxation_time;
                answered = true;
            }
        }
            
        answered = false;

        while(!answered)
        {
            std::cout << "Number of iterations (enter an integer value greater than 0): " << std::endl;
            std::cin >> time_steps; 
            if(time_steps > 0)
            {
                std::get<4>(result) = horizontal_nodes;
                answered = true;
            }
            else if(time_steps <= 0)
            {
                std::cout << "Congratulations! You have completed the achievement 'What am I doing here?'" << std::endl;
            }
        }

        answered = false;

        while(!answered)
        {
            std::cout << "Do you want to start the simulation with these parameters?" << std::endl;
            std::cout << "Type 'y' to start the simulation." << std::endl;
            std::cout << "Type 'n' to re-enter all parameters." << std::endl;
            std::cout << "Type 'q' to quit the program." << std::endl;
            std::cin >> done_input; 
            if(done_input == "y")
            {
                done = true;
                answered = true;
            }
            else if(done_input == "n")
            {
                done = false;
                answered = true;
                std::cout << "----------------------------------------------------------------------------------------------------" << std::endl;
            }
            else if(done_input == "q")
            {
                done = true;
                answered = true;
                std::cout << "Exiting program..." << std::endl;
                exit(0);
            }
        }
    }
    std::cout << std::endl;
    std::cout << "Preparing simulation..." << std::endl;
    std::cout << std::endl;
}

#endif