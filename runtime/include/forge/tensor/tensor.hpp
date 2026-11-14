#pragma once
#include <cstddef>
#include <vector>

namespace forge
{

    class Shape
    {
    private:
        std::vector<size_t> dimensions_;

    public:
        explicit Shape(std::vector<size_t> dimensions);
        size_t rank() const;
        size_t numel() const;

        const std::vector<size_t> &dims() const;
    };

}