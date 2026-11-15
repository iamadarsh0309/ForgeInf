#include "forge/tensor/tensor.hpp"
#include <limits>
#include <stdexcept>

namespace forge
{

    Shape::Shape(std::vector<size_t> dimensions)
        : dimensions_(std::move(dimensions))
    {
        for (size_t dimension : dimensions_)
        {
            if (dimension == 0)
            {
                throw std::invalid_argument(
                    "Tensor dimensions must be positive");
            }
        }
    }

    size_t Shape::rank() const
    {
        return dimensions_.size();
    }

    size_t Shape::numel() const
    {
        size_t total = 1;
        for (size_t dimension : dimensions_)
        {
            if (total > std::numeric_limits<size_t>::max() / dimension)
            {
                throw std::overflow_error(
                    "Tensor element count overflow");
            }

            total *= dimension;
        }

        return total;
    }

    const std::vector<size_t> &Shape::dims() const
    {
        return dimensions_;
    }
}