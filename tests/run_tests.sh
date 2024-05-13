#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )" # path to script (root/tests/)
echo "$SCRIPTPATH"
tests_path="$SCRIPTPATH"/../build/tests
pregen_path="$SCRIPTPATH"/pregen_inputs

run_single_test () {
    test_path=$1
    if [ "${test_path:(-4)}" = ".cpp" ]; then
        echo "======================================================"
        base_name=$(basename "$1")
        test_name="${base_name%.*}"
        executable_path="$tests_path"/"$test_name"
        echo "Test: $test_name"
        if [ "${test_name:(-7)}" = "_pregen" ]; then
            # test name ends with _pregen: look for matrix input files in pregen_inputs
            # matrix input files should have names starting with the test name (w/o "_pregen")
            test_name="${test_name%_*}"
            echo "Pregen Test: $test_name"
            for test_input in "$pregen_path"/"$test_name"*; do
                echo "Test Input: $test_input"
                test_output=$("$executable_path" < "$test_input")
                if [[ "$test_output" != *PASSED* ]]; then
                    echo -e "\033[0;31mFAILED, output follows:\033[0m"
                    echo "$test_output"
                else
                    echo -e "\033[0;32mPASSED\033[0m"
                fi
            done
        else
            # not a pregen test
            test_output=$("$executable_path" < "/dev/null")
            if [[ "$test_output" != *PASSED* ]]; then
                echo -e "\033[0;31mFAILED, output follows:\033[0m"
                echo "$test_output"
            else
                echo -e "\033[0;32mPASSED\033[0m"
            fi
        fi
    fi
}

if [ $# = 0 ]
then
    # find source files in tests src dir
    for test_file in "$SCRIPTPATH"/*.cpp # $(find $tests_path -type f -perm -u+x)
    do
        run_single_test "$test_file"
    done
else
    # run executables given by args
    for test_file in "$@"
    do
        run_single_test "$test_file"
    done
fi