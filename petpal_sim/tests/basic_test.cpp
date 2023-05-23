#include <gtest/gtest.h>

// To run tests: colcon test --ctest-args tests [package_selection_args]
// To view test test results: colcon test-result --all --verbose

TEST(package_name, a_first_test) {
    ASSERT_EQ(1, 1 + 1);
}

int main(int argc, char ** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}