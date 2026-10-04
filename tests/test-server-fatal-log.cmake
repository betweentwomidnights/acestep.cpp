# ABOUTME: Verifies that exit() flushes all captured stderr without running destructors.
execute_process(COMMAND "${TEST_SERVER}" --fatal-log
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr TIMEOUT 20)
if(NOT result STREQUAL "17")
    message(FATAL_ERROR "Fatal-log fixture returned ${result}, expected 17")
endif()
string(REPEAT "x" 262144 expected)
string(APPEND expected "FATAL-TAIL")
if(NOT stderr STREQUAL expected)
    string(LENGTH "${stderr}" length)
    message(FATAL_ERROR "Captured stderr was lost or corrupted (${length} bytes)")
endif()
message(STATUS "Fatal exit delivered all 262154 bytes of stderr")
