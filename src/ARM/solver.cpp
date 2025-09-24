#include "solver.h"
#include <cmath>
#include <cstdlib>
#include <cstring>

// =================== Core Solver ===================
int SolverGS_HQresidual(double *U_double, double *iRho_double, double *jRho_double,
                        double *iOmega_double, double *jOmega_double,
                        double *Msk_double, double *iW_double, double *jW_double,
                        double mu, double lambda, double tol,
                        int numIter, int ban_OmegaInit, int imageW, int imageH) {

    int i, j, t;
    double num, den;
    double val_Omega;
    double val_Omega_c;
    int idx, idx_11, idx_01, idx_21, idx_10, idx_12;

    // Initial condition for Omega
    if (ban_OmegaInit == 1) {
        for (j = 0; j < imageH; j++) {
            for (i = 0; i < imageW; i++) {
                idx = j * imageW + i;
                iOmega_double[idx] = iW_double[idx];
                jOmega_double[idx] = jW_double[idx];
            }
        }
    }

    for (t = 0; t < numIter; t++) {
        for (j = 0; j < imageH; j++) {
            for (i = 0; i < imageW; i++) {
                idx_11 = j * imageW + i;
                if (Msk_double[idx_11] > 0.0) {
                    num = 0.0;
                    den = 0.0;

                    // Left neighbor
                    if (i > 0) {
                        idx_01 = j * imageW + (i - 1);
                        if (Msk_double[idx_01] > 0.0) {
                            val_Omega = iOmega_double[idx_11];
                            val_Omega_c = val_Omega * val_Omega;
                            num += val_Omega_c * (U_double[idx_01] + iRho_double[idx_11] + lambda * U_double[idx_01]);
                            den += val_Omega_c * (1.0 + lambda);
                        }
                    }
                    // Right neighbor
                    if (i < imageW - 1) {
                        idx_21 = j * imageW + (i + 1);
                        if (Msk_double[idx_21] > 0.0) {
                            val_Omega = iOmega_double[idx_21];
                            val_Omega_c = val_Omega * val_Omega;
                            num += val_Omega_c * (U_double[idx_21] - iRho_double[idx_21] + lambda * U_double[idx_21]);
                            den += val_Omega_c * (1.0 + lambda);
                        }
                    }
                    // Top neighbor
                    if (j > 0) {
                        idx_10 = (j - 1) * imageW + i;
                        if (Msk_double[idx_10] > 0.0) {
                            val_Omega = jOmega_double[idx_11];
                            val_Omega_c = val_Omega * val_Omega;
                            num += val_Omega_c * (U_double[idx_10] + jRho_double[idx_11] + lambda * U_double[idx_10]);
                            den += val_Omega_c * (1.0 + lambda);
                        }
                    }
                    // Bottom neighbor
                    if (j < imageH - 1) {
                        idx_12 = (j + 1) * imageW + i;
                        if (Msk_double[idx_12] > 0.0) {
                            val_Omega = jOmega_double[idx_12];
                            val_Omega_c = val_Omega * val_Omega;
                            num += val_Omega_c * (U_double[idx_12] - jRho_double[idx_12] + lambda * U_double[idx_12]);
                            den += val_Omega_c * (1.0 + lambda);
                        }
                    }
                    if (den) U_double[idx_11] = num / den;
                }
            }
        }

        if (((t + 1) % 100) == 0) {
            computeOmega_HQ(iOmega_double, jOmega_double, U_double, iRho_double, jRho_double,
                            Msk_double, iW_double, jW_double, mu, lambda, imageW, imageH);
        }
    }
    return t;
}

// =================== Omega update ===================
void computeOmega_HQ(double *iOmega, double *jOmega, double *U,
                     double *iRho, double *jRho, double *Msk,
                     double *iW, double *jW,
                     double mu, double lambda, int imageW, int imageH) {
    int i, j, idx_11, idx_01, idx_10;
    double Uij, diff, diff_c, val;
    double epsilon = 1e-8;

    for (j = 0; j < imageH; j++) {
        for (i = 0; i < imageW; i++) {
            idx_11 = j * imageW + i;
            if (Msk[idx_11] > 0.0) {
                Uij = U[idx_11];
                if (i > 0) {
                    idx_01 = j * imageW + (i - 1);
                    if (Msk[idx_01] > 0.0) {
                        diff = Uij - U[idx_01];
                        diff_c = diff * diff;
                        val = (diff - iRho[idx_11]);
                        iOmega[idx_11] = mu / ((val * val + lambda * diff_c) + mu);
                    }
                }
                if (j > 0) {
                    idx_10 = (j - 1) * imageW + i;
                    if (Msk[idx_10] > 0.0) {
                        diff = Uij - U[idx_10];
                        diff_c = diff * diff;
                        val = (diff - jRho[idx_11]);
                        jOmega[idx_11] = mu / ((val * val + lambda * diff_c) + mu);
                    }
                }
            }
        }
    }
}

// =================== Helpers ===================
void ColumnRowBackwardDifferences(double *D_column, double *D_row, double *Data, int imageW, int imageH) {
    int i, j;
    long int idx1, idx2, idx3;

    // Top-left corner
    j = 0; i = imageW - 1;
    idx2 = imageW*j + i;
    D_column[0] = Data[0] - Data[idx2];
    j = imageH - 1; i = 0;
    idx2 = imageW*j + i;
    D_row[0] = Data[0] - Data[idx2];

    for (j = 1; j < imageH; j++) {
        idx1 = imageW*j;
        idx2 = imageW*j + (imageW - 1);
        idx3 = imageW*(j - 1);
        D_column[idx1] = Data[idx1] - Data[idx2];
        D_row[idx1] = Data[idx1] - Data[idx3];
    }
    for (i = 1; i < imageW; i++) {
        idx1 = i;
        idx2 = imageW*(imageH - 1) + i;
        idx3 = i - 1;
        D_row[idx1] = Data[idx1] - Data[idx2];
        D_column[idx1] = Data[idx1] - Data[idx3];
    }
    for (j = 1; j < imageH; j++) {
        for (i = 1; i < imageW; i++) {
            idx1 = imageW*j + i;
            idx2 = imageW*j + (i - 1);
            idx3 = imageW*(j - 1) + i;
            D_column[idx1] = Data[idx1] - Data[idx2];
            D_row[idx1] = Data[idx1] - Data[idx3];
        }
    }
}

void MatrixAddition(double *Dst_C, double *Src_A, double *Src_B, int imageW, int imageH) {
    int i, j;
    long int idx1;
    for (j = 0; j < imageH; j++) {
        for (i = 0; i < imageW; i++) {
            idx1 = imageW*j + i;
            Dst_C[idx1] = Src_A[idx1] + Src_B[idx1];
        }
    }
}

// =================== Top-level API ===================
double *UnwrapARM(double *G, double *Msk, double *iW, double *jW, double *F,
                  double mu, double lambda, int numIter, int ban_OmegaInit, int imageW, int imageH) {

    double tol = 1e-3;
    int i, j;
    long int idx1;
    double dx_G, dy_G, wrap_dx_G, wrap_dy_G;
    int iterations;

    size_t size_image = imageW * imageH;
    double *Dx_G = (double *)malloc(size_image * sizeof(double));
    double *Dy_G = (double *)malloc(size_image * sizeof(double));
    double *Dx_F = (double *)malloc(size_image * sizeof(double));
    double *Dy_F = (double *)malloc(size_image * sizeof(double));
    double *iRho = (double *)malloc(size_image * sizeof(double));
    double *jRho = (double *)malloc(size_image * sizeof(double));
    double *iOmega = (double *)malloc(size_image * sizeof(double));
    double *jOmega = (double *)malloc(size_image * sizeof(double));
    double *U = (double *)malloc(size_image * sizeof(double));

    for (size_t k = 0; k < size_image; ++k) {
    iOmega[k] = 1.0;
    jOmega[k] = 1.0;
}

    memset(U, 0, size_image * sizeof(double));

    ColumnRowBackwardDifferences(Dx_G, Dy_G, G, imageW, imageH);
    ColumnRowBackwardDifferences(Dx_F, Dy_F, F, imageW, imageH);

    for (j = 0; j < imageH; j++) {
        for (i = 0; i < imageW; i++) {
            idx1 = imageW*j + i;
            dx_G = Dx_G[idx1];
            dy_G = Dy_G[idx1];
            wrap_dx_G = atan2(sin(dx_G), cos(dx_G));
            wrap_dy_G = atan2(sin(dy_G), cos(dy_G));
            iRho[idx1] = wrap_dx_G - Dx_F[idx1];
            jRho[idx1] = wrap_dy_G - Dy_F[idx1];
        }
    }

    iterations = SolverGS_HQresidual(U, iRho, jRho, iOmega, jOmega,
                                     Msk, iW, jW, mu, lambda, tol,
                                     numIter, ban_OmegaInit, imageW, imageH);

    MatrixAddition(U, U, F, imageW, imageH);

    free(Dx_G); free(Dy_G); free(Dx_F); free(Dy_F);
    free(iRho); free(jRho); free(iOmega); free(jOmega);

    return U;
}
