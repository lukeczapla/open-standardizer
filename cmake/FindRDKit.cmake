# cmake/FindRDKit.cmake
#
# Robust RDKit finder for Debian/Ubuntu:
#  1. Try pkg-config rdkit if available.
#  2. Fallback to manual header/lib search.
#
# Provides:
#   RDKit_FOUND
#   RDKit_INCLUDE_DIRS
#   RDKit_LIBRARIES
#   RDKit::RDKit  (INTERFACE target)

include_guard()

set(RDKit_FOUND FALSE)

# ---------------------------------------------------------
# 1) Try pkg-config (if available)
# ---------------------------------------------------------
find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
    pkg_check_modules(RDKIT_PC QUIET rdkit)

    if(RDKIT_PC_FOUND)
        set(RDKit_INCLUDE_DIRS "${RDKIT_PC_INCLUDE_DIRS}")
        set(RDKit_LIBRARIES    "${RDKIT_PC_LIBRARIES}")
        set(RDKit_FOUND TRUE)
    endif()
endif()

# ---------------------------------------------------------
# 2) Manual fallback if pkg-config failed
# ---------------------------------------------------------
if(NOT RDKit_FOUND)
    # Typical Ubuntu /usr/include/rdkit layout:
    find_path(RDKit_INCLUDE_DIR
        NAMES GraphMol/ROMol.h
        PATH_SUFFIXES rdkit
        PATHS
            /usr/include
            /usr/local/include
    )

    # Try the common monolithic RDKitChem library first, then Chemistry, GraphMol, etc.
    find_library(RDKit_LIBRARY
        NAMES RDKitChem RDKitChemistry RDKitGraphMol
        PATHS
            /usr/lib
            /usr/local/lib
            /usr/lib/x86_64-linux-gnu
    )

    if(RDKit_INCLUDE_DIR AND RDKit_LIBRARY)
        set(RDKit_INCLUDE_DIRS "${RDKit_INCLUDE_DIR}")
        set(RDKit_LIBRARIES    "${RDKit_LIBRARY}")
        set(RDKit_FOUND TRUE)
    endif()
endif()

# ---------------------------------------------------------
# 3) Report or fail
# ---------------------------------------------------------
if(NOT RDKit_FOUND)
    message(FATAL_ERROR
        "Could not find RDKit via pkg-config or manual search. "
        "Make sure librdkit-dev is installed and headers are in /usr/include/rdkit "
        "and libs in /usr/lib/x86_64-linux-gnu or similar."
    )
endif()

# ---------------------------------------------------------
# 4) Create imported target
# ---------------------------------------------------------
if(NOT TARGET RDKit::RDKit)
    add_library(RDKit::RDKit INTERFACE IMPORTED)
    set_target_properties(RDKit::RDKit PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${RDKit_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES     "${RDKit_LIBRARIES}"
    )
endif()
