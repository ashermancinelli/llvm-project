add_flang_unittest(FlangRuntimeTests
  CharacterTest.cpp
  RuntimeTesting.cpp
)

target_link_libraries(FlangRuntimeTests
  PRIVATE
  FortranRuntime
)
