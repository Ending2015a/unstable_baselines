#!/bin/bash

PROJ_DIR=""
UPLOAD="false"
TOKEN=""
REPORT_CMD="-m"

function load_token() {
    # $1: token path
    if [ -f $1 ]; then
        TOKEN=$(cat $1)
    else
        echo "Token file not found"
    fi
}

function coverage_test() {
    echo "Start running coverage test"
    echo "Project dir: ${PROJ_DIR}"
    RELPATH=$(realpath --relative-to="$PWD" ${PROJ_DIR})
    coverage run --source "${RELPATH}" \
                 --omit "${RELPATH}/unstable_baselines/logger/*","${RELPATH}/test/*"\
                 -m unittest discover -s "${PROJ_DIR}/test"
    echo "Report coverage: ${REPORT_CMD}"
    coverage report ${REPORT_CMD}
    # generate coverage reports
    coverage xml -o "${PROJ_DIR}/coverage.xml"
}

function upload_report() {
    echo "Uploading reports"
    CODECOV="${PROJ_DIR}/tools/codecov"
    if [ ! -f "${CODECOV}" ]; then
    echo "Download codecov"
        # download scripts
        curl -o "${CODECOV}" \
            -s https://uploader.codecov.io/v0.1.0_4653/linux/codecov
        chmod +x ${CODECOV}
    fi
    eval "${CODECOV} -t ${TOKEN}" -R "${PROJ_DIR}"
}

ARGS=`getopt -o p:ut:c: -l path:,upload,token:,cmd: -n "$0" -- "$@"`

if [ $? -ne 0 ]; then
    echo "Terminating ..." >&2
    exit 1
fi

eval set -- "$ARGS"

while true; do
    case "$1" in
    -p|--path) PROJ_DIR="$2"; shift;;
    -u|--upload) UPLOAD="true";;
    -t|--token) TOKEN="$2"; shift;;
    -c|--cmd) REPORT_CMD="$2"; shift;;
    --) shift; break;;
    *)
        echo "Unknown args: $@"
        exit 1
    esac
    shift
done

if [ -z "$PROJ_DIR" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    PROJ_DIR=$(dirname "${SCRIPT_DIR}")
fi

coverage_test

if [ $UPLOAD == "true" ]; then
    # if token not set, load token
    if [ -z "$TOKEN" ]; then
        load_token "${PROJ_DIR}/tools/.token"
    fi
    # check token is not empty
    if [ -z "$TOKEN" ]; then
        echo "Token not found, exit"
        exit 1
    fi
    upload_report
fi