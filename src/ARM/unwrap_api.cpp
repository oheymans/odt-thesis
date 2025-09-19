#include "solver.h"
#include <cstring>  // for memcpy

// Renamed to unwrap_phase_ARM
extern "C" void unwrap_phase_ARM(double* wrapped, double* unwrapped,
                                 double* mask, double* iW, double* jW,
                                 int rows, int cols,
                                 double mu, double lambda, int numIter, int ban_OmegaInit) {
    double* result = UnwrapARM(wrapped, mask, iW, jW, unwrapped,
                               mu, lambda, numIter, ban_OmegaInit, cols, rows);

    // copy result into output buffer
    size_t size_image = (size_t)rows * (size_t)cols;
    std::memcpy(unwrapped, result, size_image * sizeof(double));

    free(result); // UnwrapARM mallocs memory
}
