/**
 * @file        access.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       Updated version of the lattice Boltzmann access functions first introduced in my SimTech project work:
 *              https://github.com/MarcelGraf0710/Task-based-Lattice-Boltzmann.
 *              The major change is that accessor objects are used to achieve different access patterns since SYCL
 *              is not compatible with function pointers.
 *              The data layouts were proposed by Mattila et al.
 * 
 * @version     2.0
 * 
 * @date        2024-09-27
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef ACCESS_HPP
#define ACCESS_HPP

#include "defines.hpp"

/**
 * @brief This namespace contains functions that map input values to array index accesses.
 */
namespace lbm_access
{
    /**
     * @brief This abstract non-instanciable accessor class defines the root of an accessor hierarchy.
     *        It is used as the base type of valid accessor objects.
     *        These accessor objects are used to implement substituable access functions.
     */
    class LBMAccessorObject
    {
        unsigned int horizontal_nodes;

        protected:

            /**
             * @brief This constructor is only available to child classes to set up the amount of
             *        horizontal nodes. It is not available otherwise, and no instances of this type
             *        can be constructed.
             * 
             * @param horizontal_nodes the amount of horizontal nodes in the lattice
             */
            explicit LBMAccessorObject(const unsigned int horizontal_nodes);

        public:

            /**
             * @brief Retrieves the index of the distribution value within the according vector
             *        following the data layout of this accessor object.
             * 
             * @param node      index of the node 
             * @param direction index of the direction
             * @return          the index of the corresponding distribution value
             */
            virtual unsigned int get_index(const unsigned int node, const unsigned int direction) const = 0;

            /**
             * @brief Returns the string representation of this class.
             */
            inline static std::string get_layout_string() {return "generic LBM accessor";};   

            /**
            * @brief Retrieves the coordinates of the node with the specified node index.
            * 
            * @param node_index the index of the node in question
            * @return           A tuple containing the x and y coordinate of the specified node.
            */
            inline std::tuple<unsigned int, unsigned int> get_node_coordinates(const unsigned int node_index) const
            {
                return std::make_tuple(node_index % horizontal_nodes, node_index / horizontal_nodes);
            }

            /**
             * @brief Returns the index of the neighbor that is reached when moving in the specified direction.
             * 
             * @param node_index    the index of the current node
             * @param direction     the direction of movement
             * @return              the node index of the neighbor
             */
            inline unsigned int get_neighbor(const unsigned int node_index, const unsigned int direction) const
            {
                int y_offset = direction / 3 - 1;                            // -1 for {0,1,2}, 0 for {3,4,5}, 1 for {6,7,8}
                int x_offset = direction - (3 * y_offset + 4);               // -1 for {0,3,6}, 0 for {1,4,7}, 1 for {2,5,8}
                return node_index + y_offset * horizontal_nodes + x_offset;
            }   
    };

    /**
     * @brief Instances of this class model the access of distribution values according to the collision data layout.
     */
    class LBMCollisionAccessor : public LBMAccessorObject
    {
        public:

            /**
             * @brief Constructs a new collision accessor object.
             * 
             * @param horizontal_nodes the amount of horizontal nodes in the lattice
             */
            explicit LBMCollisionAccessor(unsigned int horizontal_nodes);

            /**
             * @brief Returns the string representation of this class.
             */
            inline static std::string get_layout_string() {return "collision";};

            /**
             * @brief Retrieves the index of the distribution value within the according vector
             *        following the collision layout.
             * 
             * @param node      index of the node 
             * @param direction index of the direction
             * @return          the index of the corresponding distribution value
             */
            inline unsigned int get_index(const unsigned int node, const unsigned int direction) const override
            {
                return DIRECTION_COUNT * node + direction;  
            }
    };

    /**
     * @brief Instances of this class model the access of distribution values according to the stream data layout.
     */
    class LBMStreamAccessor : public LBMAccessorObject
    {
        unsigned int buffered_node_count;

        public:

            /**
             * @brief Constructs a new stream accessor object.
             * 
             * @param horizontal_nodes      the amount of horizontal nodes in the lattice
             * @param buffered_node_count   the total amount of nodes in the lattice including buffers
             */
            explicit LBMStreamAccessor(unsigned int horizontal_nodes, unsigned int buffered_node_count); 

            /**
             * @brief Returns the string representation of this class.
             */
            inline static std::string get_layout_string() {return "stream";};

            /**
             * @brief Retrieves the index of the distribution value within the according vector
             *        following the stream layout.
             * 
             * @param node      index of the node 
             * @param direction index of the direction
             * @return          the index of the corresponding distribution value
             */
            inline unsigned int get_index(const unsigned int node, const unsigned int direction) const override
            {
                return buffered_node_count * direction + node;
            }
  
    };

    /**
     * @brief Instances of this class model the access of distribution values according to the bundle data layout.
     */
    class LBMBundleAccessor : public LBMAccessorObject
    {
        unsigned int buffered_node_count;

        public:

            /**
             * @brief Constructs a new bundle accessor object.
             * 
             * @param horizontal_nodes      the amount of horizontal nodes in the lattice
             * @param buffered_node_count   the total amount of nodes in the lattice including buffers
             */
            explicit LBMBundleAccessor(unsigned int horizontal_nodes, unsigned int buffered_node_count); 

            /**
             * @brief Returns the string representation of this class.
             */
            inline static std::string get_layout_string() {return "bundle";};

            /**
             * @brief Retrieves the index of the distribution value within the according vector
             *        following the bundle layout.
             * 
             * @param node      index of the node 
             * @param direction index of the direction
             * @return          the index of the corresponding distribution value
             */
            inline unsigned int get_index(const unsigned int node, const unsigned int direction) const override
            {
                return 3 * (direction / 3) * buffered_node_count + (direction % 3) + 3 * node; 
            }
    };
    
    ///////////////////////////////////////////////////////////////////////////////////////////////
    //                                                                                           //
    //   Careful! All of the following content is deprecated and will be removed in the future.  //
    //                                                                                           //
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief This function returns the distribution values of the node with the specified index using the specified access pattern.
     * 
     * @param source the distribution values will be read from this vector
     * @param node_index this is the index of the node in the domain
     * @param access this access function will be used
     * @return A vector containing the distribution values
     */
    template<class T>
    std::vector<double> get_distribution_values_of
    (
        const std::vector<double> &source, 
        int node_index, 
        T &lbm_accessor
    )
    {
        std::vector<double> dist_vals(9,0);
        for(auto direction = 0; direction < DIRECTION_COUNT; ++direction)
        {
            dist_vals[direction] = source[lbm_accessor.get_index(node_index, direction)];
        }
        return dist_vals;
    }

    /**
    * @brief Retrieves the coordinates of the node with the specified node index.
    * @return A tuple containing the x and y coordinate of the specified node.
    */
    inline 
    std::tuple<unsigned int, unsigned int> get_node_coordinates
    (
        unsigned int node_index
    )
    {
        return std::make_tuple(node_index % HORIZONTAL_NODES, node_index / HORIZONTAL_NODES);
    }

    /**
     * @brief Returns the index of the neighbor that is reached when moving in the specified direction.
     * 
     * @param node_index the index of the current node
     * @param direction the direction of movement
     * @return the node index of the neighbor
     */
    inline unsigned int get_neighbor(unsigned int node_index, unsigned int direction)
    {
        int y_offset = direction / 3 - 1; // -1 for {0,1,2}, 0 for {3,4,5}, 1 for {6,7,8}
        int x_offset = direction - (3 * y_offset + 4); //-1 for {0,3,6}, 0 for {1,4,7}, 1 for {2,5,8}
        return node_index + y_offset * HORIZONTAL_NODES + x_offset;
    }

    /**
     * @brief Returns the array index of the collision layout
     * @param node the node in the simulation domain
     * @param direction the direction of the velocity vector
     * @return the index of the array storing the distribution values 
     */
    inline unsigned int collision(unsigned int node, unsigned int direction)
    {
        return DIRECTION_COUNT * node + direction;        
    }

    /**
     * @brief Returns the array index of the stream layout
     * 
     * @param node the node in the simulation domain
     * @param direction the direction of the velocity vector
     * @return the index of the array storing the distribution values  
     */
    inline unsigned int stream(unsigned int node, unsigned int direction)
    {
        return TOTAL_NODE_COUNT * direction + node;
    }

    /**
     * @brief Returns the array index of the bundle layout
     * 
     * @param node the node in the simulation domain
     * @param direction the direction of the velocity vector
     * @return the index of the array storing the distribution values  
     */
    inline unsigned int bundle(unsigned int node, unsigned int direction)
    {
        return 3 * (direction / 3) * TOTAL_NODE_COUNT + (direction % 3) + 3 * node; 
    }

    /**
     * @brief Returns the index the desired node has within the array that stores it. 
     *        The origin lies at the lower left corner and enumeration is row-major.
     * 
     * @param x x coordinate
     * @param y y coordinate
     * @return the index of the desired note
     */
    inline unsigned int get_node_index(unsigned int x, unsigned int y)
    {
        return x + y * HORIZONTAL_NODES;
    }

    /**
     * @brief This function returns the distribution values of the node with the specified index using the specified access pattern.
     * 
     * @param source the distribution values will be read from this vector
     * @param node_index this is the index of the node in the domain
     * @param access this access function will be used
     * @return A vector containing the distribution values
     */
    std::vector<double> get_distribution_values_of
    (
        const std::vector<double> &source, 
        int node_index, 
        access_function access
    );

    /**
     * @brief This function sets the distribution values of the node with the specified index to the specified values 
     *        using the specified access pattern.
     * 
     * @param dist_vals a vector containing the values to which the distribution values shall be set
     * @param destination the distribution values will be written to this vector
     * @param node_index this is the index of the node in the domain
     * @param access this access function will be used
     */
    void set_distribution_values_of
    (
        const std::vector<double> &dist_vals, 
        std::vector<double> &destination, 
        int node_index, 
        access_function access
    );

    /**
     * @brief This function sets all distribution values of the node with the specified index to the specified values 
     *        using the specified access pattern.
     * 
     * @param dist_vals a vector containing the values to which the distribution values shall be set
     * @param destination the distribution values will be written to this vector
     * @param node_index this is the index of the node in the domain
     * @param access this access function will be used
     */
    template<class T>
    void set_distribution_values_of
    (
        const std::vector<double> &dist_vals, 
        std::vector<double> &destination, 
        int node_index, 
        T lbm_accessor
    )
    {
        for(auto direction = 0; direction < DIRECTION_COUNT; ++direction)
        {
            destination[lbm_accessor.get_index(node_index, direction)] = dist_vals[direction];
        }
    }
}

#endif