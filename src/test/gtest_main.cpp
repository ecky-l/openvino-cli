// Dedicated gtest entry point so the test executable is unambiguously ours
// (avoids clashing with any other main()) and lets ctest run the suite.
#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
