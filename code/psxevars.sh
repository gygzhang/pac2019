#!/bin/bash
#
# Copyright (C) 2018 Intel Corporation. All Rights Reserved.
# 
# The source code contained or described herein and all documents
# related to the source code ("Material") are owned by Intel Corporation
# or its suppliers or licensors.  Title to the Material remains with
# Intel Corporation or its suppliers and licensors.  The Material is
# protected by worldwide copyright and trade secret laws and treaty
# provisions.  No part of the Material may be used, copied, reproduced,
# modified, published, uploaded, posted, transmitted, distributed, or
# disclosed in any way without Intel's prior express written permission.
# 
# No license under any patent, copyright, trade secret or other
# intellectual property right is granted to or conferred upon you by
# disclosure or delivery of the Materials, either expressly, by
# implication, inducement, estoppel or otherwise.  Any license under
# such intellectual property rights must be express and approved by
# Intel in writing.

# Bourne Shell script for the Intel(R) Parallel Studio XE 2019 Update 1 for Linux* OS

SCRIPTPATH=/opt/intel/parallel_studio_xe_2019.1.053
ROOTPATH=/opt/intel
INTEL_TARGET_ARCH="intel64"
ITAC_PARAM=""
INTEL_PYTHON="3"
if [[ ! -f "$ROOTPATH/intelpython3/bin/activate" ]]; then
    INTEL_PYTHON="2"
fi

while [ $# -gt 0 ]; do
    arg="$1"
    if [ -n "$arg" ]; then
        case "$arg" in
            ia32 )       INTEL_TARGET_ARCH="ia32"    ;;
            intel64 )    INTEL_TARGET_ARCH="intel64" ;;
            impi64 )     ITAC_PARAM="$arg"           ;;
            -python )    INTEL_PYTHON="$2"; shift    ;;
            * )          break                       ;;
        esac
    fi
    shift
done

echo "Intel(R) Parallel Studio XE 2019 Update 1 for Linux*"
echo "Copyright (C) 2009-2018 Intel Corporation. All rights reserved."

if [[ -f "$SCRIPTPATH/compilers_and_libraries_2019/linux/bin/compilervars.sh" ]]; then
    . "$SCRIPTPATH/compilers_and_libraries_2019/linux/bin/compilervars.sh" "$INTEL_TARGET_ARCH"
fi

if [ x"$INTEL_TARGET_ARCH" = x"intel64" ]; then
    if [[ -f "$SCRIPTPATH/clck_2019/bin/clckvars.sh" ]]; then
        . "$SCRIPTPATH/clck_2019/bin/clckvars.sh" 
    fi
fi

if [[ -f "$SCRIPTPATH/itac_2019/bin/itacvars.sh" ]]; then
    . "$SCRIPTPATH/itac_2019/bin/itacvars.sh" "$ITAC_PARAM"
fi

if [[ -f "$SCRIPTPATH/inspector_2019/inspxe-vars.sh" ]]; then
    . "$SCRIPTPATH/inspector_2019/inspxe-vars.sh" quiet
fi

if [[ -f "$SCRIPTPATH/vtune_amplifier_2019/amplxe-vars.sh" ]]; then
    . "$SCRIPTPATH/vtune_amplifier_2019/amplxe-vars.sh" quiet
fi

if [[ -f "$SCRIPTPATH/advisor_2019/advixe-vars.sh" ]]; then
    . "$SCRIPTPATH/advisor_2019/advixe-vars.sh" quiet
fi

if [[ -f "$ROOTPATH/intelpython$INTEL_PYTHON/bin/activate" ]]; then
    . "$ROOTPATH/intelpython$INTEL_PYTHON/bin/activate"
    export PS1=$CONDA_PS1_BACKUP
fi

export PATH="${PATH}:$SCRIPTPATH/bin"

if [ -z "${INTEL_LICENSE_FILE}" ]; then
    export INTEL_LICENSE_FILE="$ROOTPATH/licenses"
else
    export INTEL_LICENSE_FILE="$ROOTPATH/licenses:$INTEL_LICENSE_FILE"
fi
