cmake_minimum_required(VERSION 3.16)
project(ParEvalKokkos CXX)

if (NOT DRIVERS_CPP)
    message(FATAL_ERROR "DRIVERS_CPP not set")
endif()
if (NOT PROBLEM_SRC)
    message(FATAL_ERROR "PROBLEM_SRC not set")
endif()

find_package(Kokkos REQUIRED)
add_compile_definitions(USE_KOKKOS)
add_compile_definitions(DRIVER_PROBLEM_SIZE=${DRIVER_PROBLEM_SIZE})

add_executable(a.out ${PROBLEM_SRC} ${DRIVERS_CPP}/models/kokkos-driver.cc)
target_link_libraries(a.out Kokkos::kokkos)
target_include_directories(a.out PRIVATE ${DRIVERS_CPP})
target_include_directories(a.out PRIVATE ${DRIVERS_CPP}/models)
target_include_directories(a.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
