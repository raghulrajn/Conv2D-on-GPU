# Conv2D-on-GPU

## Conv2D implementation in CPU

### Features

Implemented 4D initialisation of tensor
 - Tensor-Tensor addition, Subtraction, Multiplication, Division
 - Tensor-Scalar addition, Subtraction, Multiplication, Division

### Installation of eigen library for matrix operations

```console
sudo apt-get install libeigen3-dev
```

### To read [pre-trained kernels](https://github.com/milesial/Pytorch-UNet)

```
git clone https://github.com/rogersce/cnpy.git
cd cnpy
mkdir -p build
cd build
cmake ..
make
```
Export cnpy to build path
```
export LD_LIBRARY_PATH=cnpy/build:$LD_LIBRARY_PATH
chmod +r cnpy/build/libcnpy.a
```

### To compile
```
g++ -std=c++17 -I/cnpy -I/usr/include/eigen3 -L/cnpy/build -o conv conv2D.cpp kernel_loader.cpp cnpy/build/libcnpy.a -lz
```

### Link to documentation
[Link](https://docs.google.com/document/d/1qH3mKdrBO7R1P-sYfWqf-k0hHxdn1PYL75xTPcQsFmI/)
