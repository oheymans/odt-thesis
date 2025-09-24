#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "util.h"
#include "extract.h"
#include "grad.h"
#include "pi.h"
#include <time.h>
#include <omp.h>

#define POS_RES     0x01
#define NEG_RES     0x02
#define VISITED     0x04
#define ACTIVE      0x08
#define BRANCH_CUT  0x10
#define BORDER      0x20
#define UNWRAPPED   0x40
#define POSTPONED   0x80
#define RESIDUE     (POS_RES | NEG_RES)
#define AVOID       (BRANCH_CUT | BORDER)

__declspec(dllexport) int goldstein_unwrap_array(float *phase, float *soln, int xsize, int ysize);

int NUM_CORES;

/* ----------------- Helpers ----------------- */

double timediff(clock_t t1, clock_t t2) {
    return ((double)t2 - t1) / CLOCKS_PER_SEC * 1000.0;
}

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void Gradxy(float *phase, float *gradx, float *grady, int xsize, int ysize) {
    int i, j, w;
    #pragma omp parallel for private(j,i,w) shared(xsize,ysize,phase,gradx,grady)
    for (j=0; j<ysize-1; j++) {
        for (i=0; i<xsize-1; i++) {
            w = j*xsize + i;
            gradx[w] = Gradient(phase[w], phase[w+1]);
            grady[w] = Gradient(phase[w], phase[w+xsize]);
        }
    }
}

int GetNextOneToUnwrap(int *a, int *b, int *index_list, int *num_index, int xsize, int ysize) {
    if (*num_index < 1) return 0;
    int index = index_list[*num_index - 1];
    *a = index % xsize;
    *b = index / xsize;
    --(*num_index);
    return 1;
}

void InsertList(float *soln, float val, unsigned char *bitflags,
                int a, int b, int *index_list, int *num_index, int xsize) {
    int index = b*xsize + a;
    soln[index] = val;
    index_list[*num_index] = index;
    ++(*num_index);
    bitflags[index] |= UNWRAPPED;
}

void UpdateList(int x, int y, float val, float *phase, float *soln,
                unsigned char *bitflags, int xsize, int ysize,
                int *index_list, int *num_index) {
    int a, b, k, w;
    float grad;

    a = x - 1; b = y; k = b*xsize + a;
    if (a >= 0 && !(bitflags[k] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
        w = y*xsize + x-1;
        grad = Gradient(phase[w], phase[w+1]);
        InsertList(soln, val + grad, bitflags, a, b, index_list, num_index, xsize);
    }

    a = x + 1; b = y; k = b*xsize + a;
    if (a < xsize && !(bitflags[k] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
        w = y*xsize + x;
        grad = -Gradient(phase[w], phase[w+1]);
        InsertList(soln, val + grad, bitflags, a, b, index_list, num_index, xsize);
    }

    a = x; b = y - 1; k = b*xsize + a;
    if (b >= 0 && !(bitflags[k] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
        w = (y-1)*xsize + x;
        grad = Gradient(phase[w], phase[w+xsize]);
        InsertList(soln, val + grad, bitflags, a, b, index_list, num_index, xsize);
    }

    a = x; b = y + 1; k = b*xsize + a;
    if (b < ysize && !(bitflags[k] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
        w = y*xsize + x;
        grad = -Gradient(phase[w], phase[w+xsize]);
        InsertList(soln, val + grad, bitflags, a, b, index_list, num_index, xsize);
    }
}

/* ----------------- Unwrapping ----------------- */

int UnwrapAroundCutsGoldstein(float *phase, unsigned char *bitflags, float *soln,
                              int xsize, int ysize, int *path_order) {
    int i, j, k, a, b, c, n=0, num_pieces=0;
    float value;
    int num_index=0, max_list_size=xsize*ysize;
    int *index_list;
    AllocateInt(&index_list, max_list_size + 1, "bookkeeping");

    for (j=0; j<ysize; j++) {
        for (i=0; i<xsize; i++) {
            k = j*xsize + i;
            if (!(bitflags[k] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
                bitflags[k] |= UNWRAPPED;
                if (bitflags[k] & POSTPONED) value = soln[k];
                else { ++num_pieces; value = soln[k] = phase[k]; }
                UpdateList(i, j, value, phase, soln, bitflags, xsize, ysize, index_list, &num_index);
                while (num_index > 0) {
                    ++n;
                    if (!GetNextOneToUnwrap(&a, &b, index_list, &num_index, xsize, ysize)) break;
                    c = b*xsize + a;
                    path_order[c] = n;
                    bitflags[c] |= UNWRAPPED;
                    value = soln[c];
                    UpdateList(a, b, value, phase, soln, bitflags, xsize, ysize, index_list, &num_index);
                }
            }
        }
    }
    free(index_list);

    for (j=1; j<ysize; j++) {
        for (i=1; i<xsize; i++) {
            k = j*xsize + i;
            if (bitflags[k] & AVOID) {
                if (!(bitflags[k-1] & AVOID)) {
                    soln[k] = soln[k-1] + Gradient(phase[k], phase[k-1]);
                    path_order[k] = ++n;
                }
                else if (!(bitflags[k-xsize] & AVOID)) {
                    soln[k] = soln[k-xsize] + Gradient(phase[k], phase[k-xsize]);
                    path_order[k] = ++n;
                }
            }
        }
    }
    return num_pieces;
}

int UnwrapAroundCutsFrontier(float *phase, unsigned char *bitflags, float *soln,
                             int xsize, int ysize, int *path_order,
                             float *grady, float *gradx, int *list, int length) {
    int k, kk, x, y, l, index, n=0, num_pieces=0;
    int flag, base_in, base_out, top_in, top_out;
    float value;

    for (k=0; k<length; k++) {
        if (!(bitflags[k] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
            ++num_pieces;
            soln[k] = phase[k];
            flag = 1;
            base_in = 0; base_out = xsize + ysize;
            top_in = base_in; top_out = base_out;
            list[top_in++] = k;

            while (flag) {
                for (l=base_in; l<top_in; l++) {
                    kk = list[l];
                    x = kk % xsize;
                    y = kk / xsize;
                    bitflags[kk] |= UNWRAPPED;
                    value = soln[kk];

                    index = kk - 1;
                    if (x-1 >= 0 && !(bitflags[index] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
                        bitflags[index] |= UNWRAPPED;
                        soln[index] = value + gradx[index];
                        list[top_out++] = index;
                    }
                    index = kk + 1;
                    if (x+1 < xsize && !(bitflags[index] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
                        bitflags[index] |= UNWRAPPED;
                        soln[index] = value - gradx[kk];
                        list[top_out++] = index;
                    }
                    index = kk - xsize;
                    if (y-1 >= 0 && !(bitflags[index] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
                        bitflags[index] |= UNWRAPPED;
                        soln[index] = value + grady[index];
                        list[top_out++] = index;
                    }
                    index = kk + xsize;
                    if (y+1 < ysize && !(bitflags[index] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
                        bitflags[index] |= UNWRAPPED;
                        soln[index] = value - grady[kk];
                        list[top_out++] = index;
                    }
                }
                if (base_out == top_out) flag = 0;
                else {
                    swap(&base_in, &base_out);
                    swap(&top_in, &top_out);
                    top_out = base_out;
                }
            }
        }
    }

    #pragma omp parallel for
    for (k=0; k<ysize*xsize; k++) {
        if (bitflags[k] & AVOID) {
            if (!(bitflags[k-1] & AVOID)) {
                soln[k] = soln[k-1] + Gradient(phase[k], phase[k-1]);
            }
            else if (!(bitflags[k-xsize] & AVOID)) {
                soln[k] = soln[k-xsize] + Gradient(phase[k], phase[k-xsize]);
            }
        }
    }
    return num_pieces;
}

/* ----------------- Branch Cuts ----------------- */

void PlaceCut(unsigned char *array, int a, int b, int c, int d, int xsize, int ysize, int code) {
    int i, j, istep, jstep;
    double r;
    if (c > a && a > 0) a++;
    else if (c < a && c > 0) c++;
    if (d > b && b > 0) b++;
    else if (d < b && d > 0) d++;
    if (a==c && b==d) { array[b*xsize + a] |= code; return; }
    if (abs(c-a) > abs(d-b)) {
        istep = (a < c) ? +1 : -1;
        r = ((double)(d - b))/((double)(c - a));
        for (i=a; i!=c+istep; i+=istep) {
            j = b + (i - a)*r + 0.5;
            array[j*xsize + i] |= code;
        }
    } else {
        jstep = (b < d) ? +1 : -1;
        r = ((double)(c - a))/((double)(d - b));
        for (j=b; j!=d+jstep; j+=jstep) {
            i = a + (j - b)*r + 0.5;
            array[j*xsize + i] |= code;
        }
    }
}

int DistToBorder(unsigned char *bitflags, int border_code, int a, int b,
                 int *ra, int *rb, int xsize, int ysize) {
    int besta=0, bestb=0, best_dist2=1000000;
    int i, j, k, dist2, found;
    for (int bs=0; bs<xsize+ysize; bs++) {
        found = 0;
        for (j=b-bs; j<=b+bs; j++) {
            for (i=a-bs; i<=a+bs; i++) {
                k = j*xsize + i;
                if (i<=0 || i>=xsize-1 || j<=0 || j>=ysize-1 || (bitflags[k] & border_code)) {
                    found = 1;
                    dist2 = (j-b)*(j-b) + (i-a)*(i-a);
                    if (dist2 < best_dist2) {
                        best_dist2 = dist2; besta=i; bestb=j;
                    }
                }
            }
        }
        if (found) { *ra=besta; *rb=bestb; break; }
    }
    return best_dist2;
}

void BranchCuts_parallel(unsigned char *bitflags, int MaxCutLen, int NumRes,
                         int xsize, int ysize, int iniy, int endy) {
    int i, j, k, ii, jj, kk, charge, boxctr_i, boxctr_j, boxsize, bs2;
    int dist, min_dist, rim_i, rim_j, near_i, near_j;
    int ka, num_active, max_active, *active_list;
    if (MaxCutLen < 2) MaxCutLen = 2;
    max_active = NumRes + 10;
    AllocateInt(&active_list, max_active + 1, "book keeping data");

    for (j=iniy; j<endy; j++) {
        for (i=0; i<xsize; i++) {
            k = j*xsize + i;
            if ((bitflags[k] & (POS_RES | NEG_RES)) && !(bitflags[k] & VISITED)) {
                bitflags[k] |= VISITED; bitflags[k] |= ACTIVE;
                charge = (bitflags[k] & POS_RES) ? 1 : -1;
                num_active = 0;
                active_list[num_active++] = k;
                if (num_active > max_active) num_active = max_active;

                for (boxsize = 3; boxsize<=2*MaxCutLen; boxsize += 2) {
                    bs2 = boxsize/2;
                    for (ka=0; ka<num_active; ka++) {
                        boxctr_i = active_list[ka]%xsize;
                        boxctr_j = active_list[ka]/xsize;
                        for (jj=boxctr_j - bs2; jj<=boxctr_j + bs2; jj++) {
                            for (ii=boxctr_i - bs2; ii<=boxctr_i + bs2; ii++) {
                                kk = jj*xsize + ii;
                                if (ii<0 || ii>=xsize || jj<0 || jj>=ysize) continue;
                                if (ii==0 || ii==xsize-1 || jj==0 || jj==ysize-1
                                    || (bitflags[kk] & BORDER)) {
                                    charge = 0;
                                    DistToBorder(bitflags, BORDER, boxctr_i, boxctr_j, &rim_i, &rim_j, xsize, ysize);
                                    PlaceCut(bitflags, rim_i, rim_j, boxctr_i, boxctr_j, xsize, ysize, BRANCH_CUT);
                                }
                                else if ((bitflags[kk] & (POS_RES | NEG_RES)) && !(bitflags[kk] & ACTIVE)) {
                                    if (!(bitflags[kk] & VISITED)) {
                                        charge += (bitflags[kk] & POS_RES) ? 1 : -1;
                                        bitflags[kk] |= VISITED;
                                    }
                                    active_list[num_active++] = kk;
                                    if (num_active > max_active) num_active = max_active;
                                    bitflags[kk] |= ACTIVE;
                                    PlaceCut(bitflags, ii, jj, boxctr_i, boxctr_j, xsize, ysize, BRANCH_CUT);
                                }
                                if (charge==0) goto continue_scan;
                            }
                        }
                    }
                }
                if (charge != 0) {
                    min_dist = xsize + ysize;
                    for (ka=0; ka<num_active; ka++) {
                        ii = active_list[ka]%xsize;
                        jj = active_list[ka]/xsize;
                        if ((dist = DistToBorder(bitflags, BORDER, ii, jj, &rim_i, &rim_j, xsize, ysize))<min_dist) {
                            min_dist = dist;
                            near_i = ii; near_j = jj;
                        }
                    }
                    PlaceCut(bitflags, near_i, near_j, rim_i, rim_j, xsize, ysize, BRANCH_CUT);
                }
                continue_scan:
                for (ka=0; ka<num_active; ka++)
                    bitflags[active_list[ka]] &= ~ACTIVE;
            }
        }
    }
    free(active_list);
}

void GoldsteinBranchCuts_parallel(unsigned char *bitflags, int MaxCutLen, int NumRes,
                                  int xsize, int ysize) {
    int band = ceil((double)ysize/(double)NUM_CORES);
    int b, iniy, endy;
    int MaxCutLen2 = (xsize + band)/2;
    #pragma omp parallel for private(b, iniy, endy)
    for (b=0; b<NUM_CORES; b++) {
        iniy = b*band;
        if (b<NUM_CORES-1) endy = iniy + band;
        else endy = ysize;
        BranchCuts_parallel(bitflags, MaxCutLen2, NumRes, xsize, ysize, iniy, endy);
    }
}

void GoldsteinBranchCuts_serial(unsigned char *bitflags, int MaxCutLen, int NumRes,
                                int xsize, int ysize) {
    BranchCuts_parallel(bitflags, MaxCutLen, NumRes, xsize, ysize, 0, ysize);
}

/* ----------------- Residues ----------------- */

int Residues_parallel(float *phase, unsigned char *bitflags, int xsize, int ysize) {
    int NumRes = 0;
    #pragma omp parallel for reduction(+ : NumRes)
    for (int j=0; j<ysize-1; j++) {
        for (int i=0; i<xsize-1; i++) {
            int k = j*xsize + i;
            if (bitflags && ((bitflags[k] & AVOID)
                          || (bitflags[k+1] & AVOID)
                          || (bitflags[k+1+xsize] & AVOID)
                          || (bitflags[k+xsize] & AVOID))) {
                continue;
            }
            double r = Gradient(phase[k+1], phase[k])
                     + Gradient(phase[k+1+xsize], phase[k+1])
                     + Gradient(phase[k+xsize], phase[k+1+xsize])
                     + Gradient(phase[k], phase[k+xsize]);
            if (bitflags) {
                if (r > 0.01) bitflags[k] |= POS_RES;
                else if (r < -0.01) bitflags[k] |= NEG_RES;
            }
            if (r*r > 0.01) ++NumRes;
        }
    }
    return NumRes;
}

int Residues_serial(float *phase, unsigned char *bitflags, int xsize, int ysize) {
    int NumRes = 0;
    for (int j=0; j<ysize-1; j++) {
        for (int i=0; i<xsize-1; i++) {
            int k = j*xsize + i;
            if (bitflags && ((bitflags[k] & AVOID)
                          || (bitflags[k+1] & AVOID)
                          || (bitflags[k+1+xsize] & AVOID)
                          || (bitflags[k+xsize] & AVOID))) {
                continue;
            }
            double r = Gradient(phase[k+1], phase[k])
                     + Gradient(phase[k+1+xsize], phase[k+1])
                     + Gradient(phase[k+xsize], phase[k+1+xsize])
                     + Gradient(phase[k], phase[k+xsize]);
            if (bitflags) {
                if (r > 0.01) bitflags[k] |= POS_RES;
                else if (r < -0.01) bitflags[k] |= NEG_RES;
            }
            if (r*r > 0.01) ++NumRes;
        }
    }
    return NumRes;
}


__declspec(dllexport) int goldstein_unwrap_array(float *phase, float *soln, int xsize, int ysize) {
    int length = xsize * ysize;

    // Allocate helper arrays
    float *grady, *gradx;
    unsigned char *bitflags;
    int *path_order, *list;

    AllocateFloat(&grady, length, "vertical gradient");
    AllocateFloat(&gradx, length, "horizontal gradient");
    AllocateByte(&bitflags, length, "flag array");
    AllocateInt(&path_order, length, "integration path");
    AllocateInt(&list, 2*(xsize+ysize), "frontier list");

    // Init flags
    for (int k=0; k<length; k++) {
        bitflags[k] = 0;   // nothing masked by default
    }

    // Pre-compute gradients
    Gradxy(phase, gradx, grady, xsize, ysize);

    // Find residues
    int NumRes = Residues_parallel(phase, bitflags, xsize, ysize);

    // Generate branch cuts
    int MaxCutLen = (xsize + ysize) / 2;
    NUM_CORES = omp_get_num_procs() / 2;
    omp_set_num_threads(NUM_CORES);
    GoldsteinBranchCuts_parallel(bitflags, MaxCutLen, NumRes, xsize, ysize);

    // Unwrap
    int num_pieces = UnwrapAroundCutsFrontier(
        phase, bitflags, soln, xsize, ysize, path_order,
        grady, gradx, list, length
    );

    // Cleanup
    free(grady);
    free(gradx);
    free(bitflags);
    free(path_order);
    free(list);

    return num_pieces;
}
