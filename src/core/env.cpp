// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <type_traits>

// clang-format off
#include <mscclpp/env.hpp>
#include <mscclpp/errors.hpp>
// clang-format on

#include "logger.hpp"

template <typename T>
T readEnv(const std::string& envName, const T& defaultValue) {
  const char* envCstr = getenv(envName.c_str());
  if (envCstr == nullptr) return defaultValue;
  if constexpr (std::is_same_v<T, int>) {
    return atoi(envCstr);
  } else if constexpr (std::is_same_v<T, bool>) {
    return (std::string(envCstr) != "0");
  } else {
    return T(envCstr);
  }
}

namespace {

bool equalsIgnoreCase(const std::string& value, const char* expected) {
  const std::string expectedString(expected);
  return value.size() == expectedString.size() &&
         std::equal(value.begin(), value.end(), expectedString.begin(),
                    [](unsigned char lhs, unsigned char rhs) { return std::tolower(lhs) == std::tolower(rhs); });
}

std::string legacyLogLevel(const std::string& debug) {
  if (equalsIgnoreCase(debug, "WARN")) return "WARN";
  if (equalsIgnoreCase(debug, "INFO") || equalsIgnoreCase(debug, "ABORT")) return "INFO";
  if (equalsIgnoreCase(debug, "TRACE")) return "DEBUG";
  if (equalsIgnoreCase(debug, "VERSION")) return "ERROR";
  return "ERROR";
}

std::string legacyLogSubsystems(const std::string& debugSubsys) {
  if (debugSubsys.empty()) return "ALL";

  const bool invert = debugSubsys.front() == '^';
  std::istringstream input(invert ? debugSubsys.substr(1) : debugSubsys);
  std::ostringstream output;
  std::string token;
  bool first = true;
  while (std::getline(input, token, ',')) {
    std::string upper = token;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char value) { return static_cast<char>(std::toupper(value)); });
    const char* mapped = nullptr;
    if (upper == "INIT")
      mapped = "ENV,NET,CONN";
    else if (upper == "COLL")
      mapped = "ALGO";
    else if (upper == "P2P")
      mapped = "CONN";
    else if (upper == "SHM")
      mapped = "GPU,CONN";
    else if (upper == "GRAPH" || upper == "EXECUTOR" || upper == "MSCCLPP_EXECUTOR")
      mapped = "EXEC";
    else if (upper == "TUNING")
      mapped = "ALGO";
    else if (upper == "MSCCLPP_NCCL")
      mapped = "NCCL";
    else if (upper == "ALLOC")
      mapped = "GPU";
    else
      mapped = token.c_str();

    if (!first) output << ',';
    if (invert && first) output << '^';
    output << mapped;
    first = false;
  }
  if (first && invert) return "^";
  return output.str();
}

std::string legacyLogFile(const std::string& path) {
  if (path.empty()) return "";
  std::string hostname(1024, '\0');
  if (::gethostname(hostname.data(), hostname.size()) != 0) hostname.clear();
  if (!hostname.empty()) {
    hostname.resize(std::char_traits<char>::length(hostname.c_str()));
    const size_t dot = hostname.find('.');
    if (dot != std::string::npos) hostname.resize(dot);
  }

  const std::string pid = std::to_string(::getpid());
  std::string expanded;
  expanded.reserve(path.size());
  for (size_t i = 0; i < path.size(); ++i) {
    if (path[i] != '%' || i + 1 == path.size()) {
      expanded += path[i];
      continue;
    }
    switch (path[++i]) {
      case '%':
        expanded += '%';
        break;
      case 'h':
        expanded += hostname;
        break;
      case 'p':
        expanded += pid;
        break;
      default:
        expanded += '%';
        expanded += path[i];
        break;
    }
  }
  return expanded;
}

}  // namespace

namespace mscclpp {
namespace detail {

std::string resolveLogLevel(const char* configured, const std::string& legacy) {
  return configured ? std::string(configured) : legacyLogLevel(legacy);
}

std::string resolveLogSubsystems(const char* configured, const std::string& legacy) {
  return configured ? std::string(configured) : legacyLogSubsystems(legacy);
}

std::string resolveLogFile(const char* configured, const std::string& legacy) {
  return configured ? std::string(configured) : legacyLogFile(legacy);
}

}  // namespace detail
}  // namespace mscclpp

template <typename T>
void readAndSetEnv(const std::string& envName, T& env) {
  const char* envCstr = getenv(envName.c_str());
  if (envCstr == nullptr) return;
  if constexpr (std::is_same_v<T, int>) {
    env = atoi(envCstr);
  } else if constexpr (std::is_same_v<T, bool>) {
    env = (std::string(envCstr) != "0");
  } else {
    env = std::string(envCstr);
  }
}

template <typename T>
void logEnv(const std::string& envName, const T& env) {
  if (!getenv(envName.c_str())) return;
  INFO(mscclpp::ENV, envName, "=", env);
}

namespace mscclpp {

Env::Env()
    : debug(readEnv<std::string>("MSCCLPP_DEBUG", "")),
      debugSubsys(readEnv<std::string>("MSCCLPP_DEBUG_SUBSYS", "")),
      debugFile(readEnv<std::string>("MSCCLPP_DEBUG_FILE", "")),
      logLevel(detail::resolveLogLevel(std::getenv("MSCCLPP_LOG_LEVEL"), debug)),
      logSubsys(detail::resolveLogSubsystems(std::getenv("MSCCLPP_LOG_SUBSYS"), debugSubsys)),
      logFile(detail::resolveLogFile(std::getenv("MSCCLPP_LOG_FILE"), debugFile)),
      hcaDevices(readEnv<std::string>("MSCCLPP_HCA_DEVICES", "")),
      ibvSo(readEnv<std::string>("MSCCLPP_IBV_SO", "")),
      ibvMode(readEnv<std::string>("MSCCLPP_IBV_MODE", "host")),
      hostid(readEnv<std::string>("MSCCLPP_HOSTID", "")),
      socketFamily(readEnv<std::string>("MSCCLPP_SOCKET_FAMILY", "")),
      socketIfname(readEnv<std::string>("MSCCLPP_SOCKET_IFNAME", "")),
      commId(readEnv<std::string>("MSCCLPP_COMM_ID", "")),
      cacheDir(readEnv<std::string>("MSCCLPP_CACHE_DIR", readEnv<std::string>("HOME", "~") + "/.cache/mscclpp")),
      npkitDumpDir(readEnv<std::string>("MSCCLPP_NPKIT_DUMP_DIR", "")),
      cudaIpcUseDefaultStream(readEnv<bool>("MSCCLPP_CUDAIPC_USE_DEFAULT_STREAM", false)),
      ncclSharedLibPath(readEnv<std::string>("MSCCLPP_NCCL_LIB_PATH", "")),
      forceNcclFallbackOperation(readEnv<std::string>("MSCCLPP_FORCE_NCCL_FALLBACK_OPERATION", "")),
      ncclSymmetricMemory(readEnv<bool>("MSCCLPP_NCCL_SYMMETRIC_MEMORY", false)),
      forceDisableNvls(readEnv<bool>("MSCCLPP_FORCE_DISABLE_NVLS", false)),
      forceDisableGdr(readEnv<bool>("MSCCLPP_FORCE_DISABLE_GDR", false)),
      ibGidIndex(readEnv<int>("MSCCLPP_IB_GID_INDEX", 0)) {}

std::shared_ptr<Env> env() {
  static std::shared_ptr<Env> globalEnv = std::shared_ptr<Env>(new Env());
  static bool logged = false;
  if (!logged) {
    logged = true;
    // cannot log inside the constructor because of circular dependency
    logEnv("MSCCLPP_DEBUG", globalEnv->debug);
    logEnv("MSCCLPP_DEBUG_SUBSYS", globalEnv->debugSubsys);
    logEnv("MSCCLPP_DEBUG_FILE", globalEnv->debugFile);
    logEnv("MSCCLPP_LOG_LEVEL", globalEnv->logLevel);
    logEnv("MSCCLPP_LOG_SUBSYS", globalEnv->logSubsys);
    logEnv("MSCCLPP_LOG_FILE", globalEnv->logFile);
    logEnv("MSCCLPP_HCA_DEVICES", globalEnv->hcaDevices);
    logEnv("MSCCLPP_IBV_SO", globalEnv->ibvSo);
    logEnv("MSCCLPP_IBV_MODE", globalEnv->ibvMode);
    logEnv("MSCCLPP_HOSTID", globalEnv->hostid);
    logEnv("MSCCLPP_SOCKET_FAMILY", globalEnv->socketFamily);
    logEnv("MSCCLPP_SOCKET_IFNAME", globalEnv->socketIfname);
    logEnv("MSCCLPP_COMM_ID", globalEnv->commId);
    logEnv("MSCCLPP_CACHE_DIR", globalEnv->cacheDir);
    logEnv("MSCCLPP_NPKIT_DUMP_DIR", globalEnv->npkitDumpDir);
    logEnv("MSCCLPP_CUDAIPC_USE_DEFAULT_STREAM", globalEnv->cudaIpcUseDefaultStream);
    logEnv("MSCCLPP_NCCL_LIB_PATH", globalEnv->ncclSharedLibPath);
    logEnv("MSCCLPP_FORCE_NCCL_FALLBACK_OPERATION", globalEnv->forceNcclFallbackOperation);
    logEnv("MSCCLPP_NCCL_SYMMETRIC_MEMORY", globalEnv->ncclSymmetricMemory);
    logEnv("MSCCLPP_FORCE_DISABLE_NVLS", globalEnv->forceDisableNvls);
    logEnv("MSCCLPP_FORCE_DISABLE_GDR", globalEnv->forceDisableGdr);
    logEnv("MSCCLPP_IB_GID_INDEX", globalEnv->ibGidIndex);
  }
  return globalEnv;
}

}  // namespace mscclpp
