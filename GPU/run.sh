#!/bin/bash

# Set the build directory
BUILD_DIR="build"

# Check if the build directory exists, if not create it
if [ ! -d "$BUILD_DIR" ]; then
  mkdir "$BUILD_DIR"
fi

# Run meson setup (configure build environment)
meson setup "$BUILD_DIR" || exit 1

# Compile the project
meson compile -C "$BUILD_DIR" || exit 1

# Find the executable (assuming only one main executable)
EXECUTABLE=$(find "$BUILD_DIR" -type f -executable -print -quit)

# Run the executable if it exists
if [ -x "$EXECUTABLE" ]; then
  echo "Running the project:"
  "$EXECUTABLE"
else
  echo "Executable not found in $BUILD_DIR"
fi