project(OpenANNLibrary)

configure_file(OpenANN.cpp.in ${PROJECT_SOURCE_DIR}/OpenANN.cpp)
add_definitions(${OPENANN_COMPILER_FLAGS})
file(GLOB_RECURSE openann_src "*.cpp")
add_library(openann SHARED ${openann_src})
target_link_libraries(openann alglib)

if(OPENMP_FOUND)
  target_link_libraries(openann gomp)
endif()

if(LAPACK_FOUND)
  add_compile_definitions(OPENANN_HAVE_LAPACK)
  target_link_libraries(openann ${LAPACK_LIBRARIES})
endif()

if(BLAS_FOUND)
  add_compile_definitions(OPENANN_HAVE_BLAS)
  target_link_libraries(openann ${BLAS_LIBRARIES})
endif()

target_link_libraries(openann ${FOUNDATION_LDFLAGS})
target_include_directories(openann PUBLIC ${FOUNDATION_INCLUDE_DIRS})
target_compile_options(openann PUBLIC ${FOUNDATION_CFLAGS_OTHER})
target_link_libraries(openann alglib)
install(TARGETS openann DESTINATION lib)
