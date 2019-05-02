cc_library(
    name = "idm",
    srcs = [
        "idm.cpp",
    ],
    hdrs = [
        "idm.hpp",
    ],
    deps = [
        "//modules/commons:commons",
        "//modules/world:world",
        "//modules/models/behavior:behavior",
        "//modules/models/dynamic:dynamic"
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "idm_tests",
    srcs = [
        "idm_tests.cpp",
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//modules/geometry",
        "//modules/models/dynamic:dynamic",
        "//modules/models/behavior/idm:idm",
        "@gtest//:main",
    ],
)