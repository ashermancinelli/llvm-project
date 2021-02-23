add_flang_unittest(FlangRuntimeTests
  CharacterTest.cpp
  CMakeLists.txt
  ExternalIOTest.cpp
  FormatTest.cpp
  HelloWorldOutputTest.cpp
  ListInputTest.cpp
  RuntimeTesting.cpp
)

target_link_libraries(FlangRuntimeTests
  PRIVATE
  FortranRuntime
)
