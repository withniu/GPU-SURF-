# - Find cudpp
# Find the native CUDPP includes and library.
# Once done this will define
#
#  CUDPP_INCLUDE_DIR    - where to find cudpp.h
#  CUDPP_LIBRARIES      - List of libraries when using cudpp.
#  CUDPP_FOUND          - True if cudpp found.
#

#=============================================================================
# Copyright 2001-2009 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

FIND_PACKAGE(CUDA)

FIND_PATH(CUDPP_INCLUDE_DIR cudpp.h)

FIND_LIBRARY(CUDPP_LIBRARY NAMES cudpp)

# handle the QUIETLY and REQUIRED arguments and set CUDPP_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CUDPP DEFAULT_MSG CUDPP_LIBRARY CUDPP_INCLUDE_DIR)

