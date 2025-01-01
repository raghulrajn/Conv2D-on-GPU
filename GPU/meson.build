project(
    'conv2d',
    ['cpp', 'c'],
    default_options : [
        'buildtype=debugoptimized',
        'optimization=3',
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

subdir('lib')
opencv_dep = dependency('opencv4', required : true)

executable('conv2d',
    [
        'src/conv2d.cpp',
        xxd.process('src/conv2d.cl'),
    ],
    extra_files: [
        'src/conv2d.cl',
    ],
    dependencies : [
        core_dep,
        opencl_dep,
        opencv_dep
    ],
)