#!/bin/bash

# Find and format all relevant source files
find . -iname "*.cpp" -o -iname "*.hpp" -o -iname "*.cu" -o -iname "*.h" | xargs clang-format -i

echo "Files formatted successfully."