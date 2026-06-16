if(TARGET geogram)
    return()
endif()

include(FetchContent)

FetchContent_Declare(
    geogram
    GIT_REPOSITORY https://github.com/BrunoLevy/geogram.git
    GIT_TAG v1.9.2
)

# Opciones importantes
set(GEOGRAM_WITH_GRAPHICS OFF CACHE BOOL "" FORCE)
set(GEOGRAM_WITH_LUA OFF CACHE BOOL "" FORCE)
set(GEOGRAM_WITH_TETGEN OFF CACHE BOOL "" FORCE)
set(GEOGRAM_WITH_HLBFGS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(geogram)
