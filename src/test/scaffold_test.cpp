// P0 scaffold test: proves the build wiring (RapidJSON found + linked, GTest
// driving ovcli_core) and the util base64/env helpers we will build on.
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "../main/util.hpp"

using namespace ovcli;

TEST(ovcli_core, base64_roundtrip) {
  auto strs = std::vector<std::string>{"a","b","ab","abc","a]b[", "foobar","hello openvino genesis"};
  for (const std::string& s : strs) {
    EXPECT_EQ(base64_decode(base64_encode(s)), s);
  }
}

TEST(ovcli_core, base64_known_vectors) {
  EXPECT_EQ(base64_encode("foobar"), "Zm9vYmFy");
  EXPECT_EQ(base64_encode("Ma"), "TWE=");
  EXPECT_EQ(base64_encode("M"), "TQ==");
  EXPECT_EQ(base64_encode("f"), "Zg==");
  // Decode is lenient about the URL-safe alphabet.
  EXPECT_EQ(base64_decode("Zm9vYmFy"), "foobar");
}

TEST(ovcli_core, env_or_defaults) {
  setenv("OVCLI_TEST_SETCFG", "configured", 1);
  EXPECT_EQ(env_or("OVCLI_TEST_SETCFG", "def"), "configured");
  EXPECT_EQ(env_or("OVCLI_TEST_NOPE_VAR", "def"), "def");
  // Empty string treated as missing by default.
  setenv("OVCLI_TEST_EMPTY", "", 1);
  EXPECT_EQ(env_or("OVCLI_TEST_EMPTY", "def", true), "def");
  EXPECT_EQ(env_or("OVCLI_TEST_EMPTY", "def", false), "");
  // 'on' style flag truthiness (mirrors _STUB_ON).
  setenv("OVCLI_TEST_FLAG", "on", 1);
  setenv("OVCLI_TEST_FLAG_OFF", "no", 1);
  EXPECT_TRUE(env_flag_on("OVCLI_TEST_FLAG"));
  // Unset => false (matches worker_process._STUB_ON: an unset var is "" which is
  // not one of the truthy spellings). The previous "EXPECT_TRUE" here was the bug.
  EXPECT_FALSE(env_flag_on("OVCLI_TEST_NOPE_VAR"));
  EXPECT_FALSE(env_flag_on("OVCLI_TEST_FLAG_OFF"));
}
