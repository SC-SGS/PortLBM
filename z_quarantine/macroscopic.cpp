#include "../include/macroscopic.hpp"

velocity macroscopic::flow_velocity(const std::vector<double> &distribution_functions)
{
    velocity flow_velocity{0,0};
    velocity velocity_vector{0,0};

    int velocity_x_component = 0; 
    int velocity_y_component = 0; 
    
    for(int i = 0; i < 9; ++i)
    {
        velocity_x_component = i % 3 - 1; 
        velocity_y_component = i / 3 - 1; 
        flow_velocity[0] += distribution_functions[i] * velocity_x_component;
        flow_velocity[1] += distribution_functions[i] * velocity_y_component;
    }
    return flow_velocity;
}
