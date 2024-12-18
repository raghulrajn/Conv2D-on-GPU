# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -I/usr/include/eigen3

# Target executable name
TARGET := tensor4d

# Source files
SRCS := tensor4d.cpp

# Object files
OBJS := $(SRCS:.cpp=.o)

# Compilation rule
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule to remove compiled files
clean:
	rm -f $(TARGET) $(OBJS)

# Phony targets
.PHONY: all clean
