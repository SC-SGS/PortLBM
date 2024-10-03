#include "../include/defines.hpp"
#include "../include/utils.hpp"

bool DEBUG_MODE = false;
bool RESULTS_TO_CSV = false;

unsigned int VERTICAL_NODES = 24;
unsigned int HORIZONTAL_NODES = 7;
unsigned long TOTAL_NODE_COUNT = VERTICAL_NODES * HORIZONTAL_NODES;

double RELAXATION_TIME = 1.4;
unsigned int TIME_STEPS = 50;

unsigned int SUBDOMAIN_HEIGHT = 8;
unsigned int SUBDOMAIN_COUNT = ((VERTICAL_NODES + 1) / (1 + SUBDOMAIN_HEIGHT));
unsigned int BUFFER_COUNT = SUBDOMAIN_COUNT - 1;
unsigned long TOTAL_NODES_EXCLUDING_BUFFERS = (TOTAL_NODE_COUNT - BUFFER_COUNT * HORIZONTAL_NODES);

velocity INLET_VELOCITY = {0.1,0.0};
velocity OUTLET_VELOCITY = {0.00,0.0};
double INLET_DENSITY = 1;
double OUTLET_DENSITY = 1;

unsigned int SHIFT_OFFSET = HORIZONTAL_NODES + 1;
unsigned int SHIFT_DISTRIBUTION_VALUE_COUNT = (TOTAL_NODE_COUNT + (BUFFER_COUNT) * (HORIZONTAL_NODES) + (SUBDOMAIN_COUNT) * (SHIFT_OFFSET));

access_function ACCESS_FUNCTION = lbm_access::collision;

/** Mapping of directions as proposed by Mattila to the corresponding velocity vectors */
const std::map<unsigned int, velocity> VELOCITY_VECTORS =
{
    {6, {-1, 1}},  {7, {0, 1}}, {8, {1, 1}},   
    {3, {-1, 0}},  {4, {0, 0}},  {5, {1, 0}},   
    {0, {-1, -1}}, {1, {0, -1}}, {2, {1, -1}}   
};

/** Mapping of directions as proposed by Mattila to the WEIGHTS of the corresponding distribution function */
const std::map<unsigned int,double> WEIGHTS = 
{
    {6, 1.0/36}, {7, 1.0/9}, {8, 1.0/36},
    {3, 1.0/9},  {4, 4.0/9}, {5, 1.0/9},
    {0, 1.0/36}, {1, 1.0/9}, {2, 1.0/36}
};

/** Vector containing the distribution values that actually change within a streaming step */
const std::vector<unsigned int> STREAMING_DIRECTIONS = {0, 1, 2, 3, 5, 6, 7, 8};

/** Vector containing all directions */
const std::vector<unsigned int> ALL_DIRECTIONS = {0, 1, 2, 3, 4, 5, 6, 7, 8};

/**
 * @brief Returns the Maxwell-Boltzmann-Distribution for all directions in the order proposed by Mattila et al.
 * 
 * @param u two-dimensional velocity vector
 * @param rho density
 * @return the probability of there being a particle with velocity v_direction 
 */
std::vector<double> maxwell_boltzmann_distribution
(
    const velocity &u, 
    const double rho
)
{
    std::vector<double> result(DIRECTION_COUNT,0);
    for(auto direction = 0; direction < DIRECTION_COUNT; ++direction)
    {
        result[direction] = WEIGHTS.at(direction) *
            (
                rho + 3 * math_utils::dot(VELOCITY_VECTORS.at(direction), u) 
                + 9.0/2 * pow(math_utils::dot(VELOCITY_VECTORS.at(direction), u), 2)
                - 3.0/2 * math_utils::dot(u, u)
            );
    }
    return result;
}
