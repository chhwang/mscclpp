// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "../framework.hpp"
#include "debug.h"
#include "logger.hpp"

namespace {

std::string capturedLog;

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    if (const char* previous = std::getenv(name_.c_str())) {
      wasSet_ = true;
      previous_ = previous;
    }
    if (!set(value)) throw std::runtime_error("Failed to set test environment variable");
  }

  ~ScopedEnvVar() {
    if (wasSet_)
      ::setenv(name_.c_str(), previous_.c_str(), 1);
    else
      ::unsetenv(name_.c_str());
  }

  const char* get() const { return std::getenv(name_.c_str()); }
  bool set(const char* value) const {
    return value ? ::setenv(name_.c_str(), value, 1) == 0 : ::unsetenv(name_.c_str()) == 0;
  }

 private:
  std::string name_;
  std::string previous_;
  bool wasSet_ = false;
};

class TemporaryLogFile {
 public:
  TemporaryLogFile() {
    char path[] = "/tmp/mscclpp_logger_test_XXXXXX";
    int fd = mkstemp(path);
    if (fd >= 0) {
      close(fd);
      path_ = path;
    }
  }

  ~TemporaryLogFile() {
    if (!path_.empty()) std::remove(path_.c_str());
  }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

void captureLog(const char* msg) { capturedLog += msg; }

}  // namespace

TEST(LoggerTest, FormatsStringsPointersAndFilters) {
  TemporaryLogFile logFile;
  ASSERT_FALSE(logFile.path().empty());
  const std::string& path = logFile.path();

  mscclpp::LogSubsysSet subsystems;
  subsystems.set(static_cast<size_t>(mscclpp::LogSubsys::NET));
  mscclpp::Logger logger("", mscclpp::LogLevel::INFO, 0, subsystems, path);

  const char* text = "const pointer";
  char mutableText[] = "mutable array";
  const char* nullText = nullptr;
  int value = 0;
  const std::string formatted =
      logger.message<false>("literal ", "literal", " const ", text, " mutable ", mutableText, " null ", nullText,
                            " pointer ", static_cast<const void*>(&value));
  EXPECT_TRUE(formatted.find("literal literal const const pointer mutable mutable array null (null) pointer 0x") == 0);
  EXPECT_FALSE(logger.shouldLog(mscclpp::LogLevel::DEBUG, mscclpp::LogSubsys::NET));
  EXPECT_TRUE(logger.shouldLog(mscclpp::LogLevel::INFO, mscclpp::LogSubsys::NET));
  EXPECT_TRUE(logger.shouldLog(mscclpp::LogLevel::WARN, mscclpp::LogSubsys::NET));
  EXPECT_FALSE(logger.shouldLog(mscclpp::LogLevel::INFO, mscclpp::LogSubsys::ENV));

  if (logger.shouldLog(mscclpp::LogLevel::INFO, mscclpp::LogSubsys::NET)) {
    logger.log("file output");
  }
  std::ifstream inputLogFile(path);
  const std::string contents((std::istreambuf_iterator<char>(inputLogFile)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(contents.find("file output\n") != std::string::npos);
}

TEST(LoggerTest, LegacyEnvAliasesRespectNewSettingPrecedence) {
  // Exercise the same resolver functions used by Env without touching its process-wide singleton.
  ScopedEnvVar legacyLevel("MSCCLPP_DEBUG", "TRACE");
  ScopedEnvVar newLevel("MSCCLPP_LOG_LEVEL", nullptr);
  EXPECT_EQ(mscclpp::detail::resolveLogLevel(newLevel.get(), legacyLevel.get()), "DEBUG");
  EXPECT_TRUE(legacyLevel.set("VERSION"));
  EXPECT_EQ(mscclpp::detail::resolveLogLevel(newLevel.get(), legacyLevel.get()), "ERROR");
  EXPECT_TRUE(legacyLevel.set("WARN"));
  EXPECT_EQ(mscclpp::detail::resolveLogLevel(newLevel.get(), legacyLevel.get()), "WARN");
  EXPECT_TRUE(legacyLevel.set("ABORT"));
  EXPECT_EQ(mscclpp::detail::resolveLogLevel(newLevel.get(), legacyLevel.get()), "INFO");
  EXPECT_TRUE(newLevel.set("ERROR"));
  EXPECT_EQ(mscclpp::detail::resolveLogLevel(newLevel.get(), legacyLevel.get()), "ERROR");
  EXPECT_TRUE(newLevel.set(""));
  EXPECT_EQ(mscclpp::detail::resolveLogLevel(newLevel.get(), legacyLevel.get()), "");

  ScopedEnvVar legacySubsys("MSCCLPP_DEBUG_SUBSYS", "INIT,COLL,P2P,SHM,GRAPH,TUNING,ENV,ALLOC,EXECUTOR,NCCL");
  ScopedEnvVar newSubsys("MSCCLPP_LOG_SUBSYS", nullptr);
  EXPECT_EQ(mscclpp::detail::resolveLogSubsystems(newSubsys.get(), legacySubsys.get()),
            "ENV,NET,CONN,ALGO,CONN,GPU,CONN,EXEC,ALGO,ENV,GPU,EXEC,NCCL");
  EXPECT_TRUE(legacySubsys.set("^P2P,NET"));
  EXPECT_EQ(mscclpp::detail::resolveLogSubsystems(newSubsys.get(), legacySubsys.get()), "^CONN,NET");
  EXPECT_TRUE(newSubsys.set("NCCL"));
  EXPECT_EQ(mscclpp::detail::resolveLogSubsystems(newSubsys.get(), legacySubsys.get()), "NCCL");
  EXPECT_TRUE(newSubsys.set(""));
  EXPECT_EQ(mscclpp::detail::resolveLogSubsystems(newSubsys.get(), legacySubsys.get()), "");

  ScopedEnvVar legacyFile("MSCCLPP_DEBUG_FILE", "legacy_%p_%%.log");
  ScopedEnvVar newFile("MSCCLPP_LOG_FILE", nullptr);
  EXPECT_EQ(mscclpp::detail::resolveLogFile(newFile.get(), legacyFile.get()),
            "legacy_" + std::to_string(getpid()) + "_%.log");
  EXPECT_TRUE(newFile.set("new_%p.log"));
  EXPECT_EQ(mscclpp::detail::resolveLogFile(newFile.get(), legacyFile.get()), "new_%p.log");
  EXPECT_TRUE(newFile.set(""));
  EXPECT_EQ(mscclpp::detail::resolveLogFile(newFile.get(), legacyFile.get()), "");
}

TEST(LoggerTest, PreservesCLogHandlerAndErrorStringExports) {
  capturedLog.clear();
  EXPECT_EQ(mscclppSetLogHandler(captureLog), mscclppSuccess);
  mscclpp::LogSubsysSet allSubsystems;
  allSubsystems.set();
  mscclpp::Logger callbackLogger("", mscclpp::LogLevel::ERROR, 0, allSubsystems, "");
  callbackLogger.log("callback message");
  EXPECT_TRUE(capturedLog.find("callback message\n") != std::string::npos);

  EXPECT_EQ(mscclppSetLogHandler(nullptr), mscclppInvalidArgument);
  EXPECT_EQ(std::string(mscclppGetErrorString(mscclppInvalidUsage)), "invalid usage");
  EXPECT_EQ(mscclppSetLogHandler(mscclppDefaultLogHandler), mscclppSuccess);
  mscclppDefaultLogHandler("default handler output\n");
  mscclpp::detail::setLogHandler(nullptr);
}
