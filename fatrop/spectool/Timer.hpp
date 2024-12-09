#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <iostream>
#include <time.h>

class Timer {
public:
    explicit Timer() { start(); }
    double getMs() { return static_cast<double>(getNs()) / 1.e6; }
    double getSeconds() { return static_cast<double>(getNs()) / 1.e9; }
private:
    void start() { clock_gettime(CLOCK_MONOTONIC, &_startTime); }
    int64_t getNs() {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        return (int64_t)(now.tv_nsec - _startTime.tv_nsec) +
            1000000000 * (now.tv_sec - _startTime.tv_sec);
    }
    struct timespec _startTime;
};


#endif