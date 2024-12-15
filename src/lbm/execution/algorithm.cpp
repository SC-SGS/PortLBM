

#include "../../../include/lbm/execution/algorithm.hpp"

lbm::execution::Algorithm::Algorithm(const sycl::queue &queue) 
: 
future(hpx::async([&]{})), 
queue(std::make_shared<sycl::queue>(queue)), 
simulation(std::make_unique<core::Simulation>())
{
    std::cout << "Constructor of algorithm called!" << std::endl;
};