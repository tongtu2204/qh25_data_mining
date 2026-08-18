#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

using Matrix = std::vector<double>;


struct JacobiResult {
    std::vector<double> eigenvalues;
    Matrix eigenvectors;
    int sweeps;
    double max_offdiag;
};


inline double &at(
    Matrix &A,
    int n,
    int i,
    int j
) {
    return A[
        static_cast<size_t>(i) * n + j
    ];
}


inline double at(
    const Matrix &A,
    int n,
    int i,
    int j
) {
    return A[
        static_cast<size_t>(i) * n + j
    ];
}


Matrix identity_matrix(
    int n
) {
    Matrix I(
        static_cast<size_t>(n) * n,
        0.0
    );

    for (int i = 0; i < n; ++i) {
        at(I, n, i, i) = 1.0;
    }

    return I;
}


Matrix generate_symmetric_matrix(
    int n
) {
    Matrix A(
        static_cast<size_t>(n) * n,
        0.0
    );

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {

            double value =
                std::sin(
                    0.017
                    * (i + 1)
                    * (j + 1)
                )
                +
                std::cos(
                    0.013
                    * (i + j + 2)
                );

            if (i == j) {
                value += static_cast<double>(n);
            }

            at(A, n, i, j) = value;
            at(A, n, j, i) = value;
        }
    }

    return A;
}


std::vector<
    std::vector<
        std::pair<int, int>
    >
>
build_round_robin_pairs(
    int n
) {
    int m =
        (n % 2 == 0)
        ? n
        : n + 1;

    std::vector<int> players(m);

    std::iota(
        players.begin(),
        players.end(),
        0
    );

    std::vector<
        std::vector<
            std::pair<int, int>
        >
    > rounds;

    rounds.reserve(
        m - 1
    );

    for (
        int round = 0;
        round < m - 1;
        ++round
    ) {
        std::vector<
            std::pair<int, int>
        > pairs;

        pairs.reserve(
            m / 2
        );

        for (
            int i = 0;
            i < m / 2;
            ++i
        ) {
            int p =
                players[i];

            int q =
                players[
                    m - 1 - i
                ];

            if (
                p < n
                &&
                q < n
            ) {
                pairs.emplace_back(
                    p,
                    q
                );
            }
        }

        rounds.push_back(
            std::move(pairs)
        );

        int last =
            players.back();

        for (
            int i = m - 1;
            i > 1;
            --i
        ) {
            players[i] =
                players[i - 1];
        }

        players[1] =
            last;
    }

    return rounds;
}


double max_offdiag(
    const Matrix &A,
    int n
) {
    double result = 0.0;

    #pragma omp parallel for reduction(max:result) schedule(static)
    for (int i = 0; i < n; ++i) {

        double local_max =
            0.0;

        for (
            int j = i + 1;
            j < n;
            ++j
        ) {
            local_max =
                std::max(
                    local_max,
                    std::abs(
                        at(
                            A,
                            n,
                            i,
                            j
                        )
                    )
                );
        }

        result =
            std::max(
                result,
                local_max
            );
    }

    return result;
}


double orthogonality_error(
    const Matrix &V,
    int n
) {
    double max_error =
        0.0;

    #pragma omp parallel for reduction(max:max_error) schedule(static)
    for (int i = 0; i < n; ++i) {

        double local_max =
            0.0;

        for (
            int j = 0;
            j < n;
            ++j
        ) {
            double dot =
                0.0;

            for (
                int k = 0;
                k < n;
                ++k
            ) {
                dot +=
                    at(
                        V,
                        n,
                        k,
                        i
                    )
                    *
                    at(
                        V,
                        n,
                        k,
                        j
                    );
            }

            double target =
                (i == j)
                ? 1.0
                : 0.0;

            local_max =
                std::max(
                    local_max,
                    std::abs(
                        dot
                        - target
                    )
                );
        }

        max_error =
            std::max(
                max_error,
                local_max
            );
    }

    return max_error;
}


JacobiResult jacobi_openmp(
    Matrix A,
    int n,
    int max_sweeps,
    double tolerance
) {
    Matrix V =
        identity_matrix(n);

    const auto rounds =
        build_round_robin_pairs(
            n
        );

    const int max_pairs =
        (n + 1) / 2;

    /*
     * Reuse these buffers.
     *
     * They are shared between OpenMP threads,
     * but each pair_id writes to a different
     * element.
     */
    std::vector<double> c(
        max_pairs,
        1.0
    );

    std::vector<double> s(
        max_pairs,
        0.0
    );

    int completed_sweeps =
        0;

    for (
        int sweep = 0;
        sweep < max_sweeps;
        ++sweep
    ) {

        /*
         * Important optimization:
         *
         * Only ONE parallel region is created
         * for the whole Jacobi sweep.
         *
         * All round operations reuse the same
         * OpenMP worker threads.
         */
        #pragma omp parallel shared(A, V, c, s, rounds)
        {
            for (
                size_t round_id = 0;
                round_id < rounds.size();
                ++round_id
            ) {
                const auto &pairs =
                    rounds[round_id];

                const int pair_count =
                    static_cast<int>(
                        pairs.size()
                    );

                /*
                 * --------------------------------
                 * STEP 1
                 * Compute Jacobi rotation angles
                 * --------------------------------
                 */

                #pragma omp for schedule(static)
                for (
                    int pair_id = 0;
                    pair_id < pair_count;
                    ++pair_id
                ) {
                    const int p =
                        pairs[
                            pair_id
                        ].first;

                    const int q =
                        pairs[
                            pair_id
                        ].second;

                    const double app =
                        at(
                            A,
                            n,
                            p,
                            p
                        );

                    const double aqq =
                        at(
                            A,
                            n,
                            q,
                            q
                        );

                    const double apq =
                        at(
                            A,
                            n,
                            p,
                            q
                        );

                    if (
                        std::abs(apq)
                        <= tolerance
                    ) {
                        c[pair_id] =
                            1.0;

                        s[pair_id] =
                            0.0;

                        continue;
                    }

                    const double theta =
                        0.5
                        *
                        std::atan2(
                            2.0 * apq,
                            aqq - app
                        );

                    c[pair_id] =
                        std::cos(
                            theta
                        );

                    s[pair_id] =
                        std::sin(
                            theta
                        );
                }

                /*
                 * Implicit barrier at the end
                 * of omp for guarantees that
                 * all c/s values are ready.
                 */


                /*
                 * --------------------------------
                 * STEP 2
                 * Right multiplication:
                 *
                 * A <- A J
                 *
                 * Parallel over rows.
                 * --------------------------------
                 */

                #pragma omp for schedule(static)
                for (
                    int row = 0;
                    row < n;
                    ++row
                ) {
                    for (
                        int pair_id = 0;
                        pair_id < pair_count;
                        ++pair_id
                    ) {
                        const int p =
                            pairs[
                                pair_id
                            ].first;

                        const int q =
                            pairs[
                                pair_id
                            ].second;

                        const double cv =
                            c[
                                pair_id
                            ];

                        const double sv =
                            s[
                                pair_id
                            ];

                        const double aip =
                            at(
                                A,
                                n,
                                row,
                                p
                            );

                        const double aiq =
                            at(
                                A,
                                n,
                                row,
                                q
                            );

                        at(
                            A,
                            n,
                            row,
                            p
                        ) =
                            cv * aip
                            -
                            sv * aiq;

                        at(
                            A,
                            n,
                            row,
                            q
                        ) =
                            sv * aip
                            +
                            cv * aiq;
                    }
                }


                /*
                 * --------------------------------
                 * STEP 3
                 * Left multiplication:
                 *
                 * A <- J^T A
                 *
                 * Each pair owns different rows,
                 * therefore pairs can safely run
                 * in parallel.
                 * --------------------------------
                 */

                #pragma omp for schedule(static)
                for (
                    int pair_id = 0;
                    pair_id < pair_count;
                    ++pair_id
                ) {
                    const int p =
                        pairs[
                            pair_id
                        ].first;

                    const int q =
                        pairs[
                            pair_id
                        ].second;

                    const double cv =
                        c[
                            pair_id
                        ];

                    const double sv =
                        s[
                            pair_id
                        ];

                    for (
                        int col = 0;
                        col < n;
                        ++col
                    ) {
                        const double apj =
                            at(
                                A,
                                n,
                                p,
                                col
                            );

                        const double aqj =
                            at(
                                A,
                                n,
                                q,
                                col
                            );

                        at(
                            A,
                            n,
                            p,
                            col
                        ) =
                            cv * apj
                            -
                            sv * aqj;

                        at(
                            A,
                            n,
                            q,
                            col
                        ) =
                            sv * apj
                            +
                            cv * aqj;
                    }
                }


                /*
                 * --------------------------------
                 * STEP 4
                 * Accumulate eigenvectors:
                 *
                 * V <- V J
                 *
                 * Parallel over rows.
                 * --------------------------------
                 */

                #pragma omp for schedule(static)
                for (
                    int row = 0;
                    row < n;
                    ++row
                ) {
                    for (
                        int pair_id = 0;
                        pair_id < pair_count;
                        ++pair_id
                    ) {
                        const int p =
                            pairs[
                                pair_id
                            ].first;

                        const int q =
                            pairs[
                                pair_id
                            ].second;

                        const double cv =
                            c[
                                pair_id
                            ];

                        const double sv =
                            s[
                                pair_id
                            ];

                        const double vip =
                            at(
                                V,
                                n,
                                row,
                                p
                            );

                        const double viq =
                            at(
                                V,
                                n,
                                row,
                                q
                            );

                        at(
                            V,
                            n,
                            row,
                            p
                        ) =
                            cv * vip
                            -
                            sv * viq;

                        at(
                            V,
                            n,
                            row,
                            q
                        ) =
                            sv * vip
                            +
                            cv * viq;
                    }
                }

                /*
                 * Implicit barrier here ensures
                 * the current round is completely
                 * finished before the next round
                 * begins.
                 */
            }
        }

        completed_sweeps =
            sweep + 1;

        const double offdiag =
            max_offdiag(
                A,
                n
            );

        if (
            offdiag
            <= tolerance
        ) {
            break;
        }
    }

    std::vector<double> eigenvalues(
        n
    );

    for (
        int i = 0;
        i < n;
        ++i
    ) {
        eigenvalues[i] =
            at(
                A,
                n,
                i,
                i
            );
    }

    return {
        eigenvalues,
        V,
        completed_sweeps,
        max_offdiag(
            A,
            n
        )
    };
}


void save_eigenvalues(
    const std::vector<double> &values,
    const std::string &path
) {
    std::vector<double> sorted =
        values;

    std::sort(
        sorted.begin(),
        sorted.end()
    );

    std::ofstream file(
        path
    );

    if (
        !file.is_open()
    ) {
        throw std::runtime_error(
            "Cannot open output file: "
            + path
        );
    }

    file <<
        std::setprecision(17);

    for (
        double value
        : sorted
    ) {
        file
            << value
            << "\n";
    }

    file.close();

    if (!file) {
        throw std::runtime_error(
            "Failed while writing output file: "
            + path
        );
    }
}


int main(
    int argc,
    char **argv
) {
    if (
        argc < 5
    ) {
        std::cerr
            << "Usage: jacobi_openmp "
            << "<n> "
            << "<max_sweeps> "
            << "<tolerance> "
            << "<eigenvalue_file>\n";

        return 1;
    }

    const int n =
        std::stoi(
            argv[1]
        );

    const int max_sweeps =
        std::stoi(
            argv[2]
        );

    const double tolerance =
        std::stod(
            argv[3]
        );

    const std::string output_file =
        argv[4];

    /*
     * Disable OpenMP dynamic adjustment.
     *
     * We want OMP_NUM_THREADS from the
     * Python benchmark to control exactly
     * how many threads are used.
     */
    omp_set_dynamic(0);

    Matrix A =
        generate_symmetric_matrix(
            n
        );

    const auto start =
        std::chrono
        ::high_resolution_clock
        ::now();

    JacobiResult result =
        jacobi_openmp(
            A,
            n,
            max_sweeps,
            tolerance
        );

    const auto end =
        std::chrono
        ::high_resolution_clock
        ::now();

    const double seconds =
        std::chrono
        ::duration<double>(
            end - start
        )
        .count();

    const double ortho_error =
        orthogonality_error(
            result.eigenvectors,
            n
        );

    save_eigenvalues(
        result.eigenvalues,
        output_file
    );

    std::cout
        << std::setprecision(12)

        << "n="
        << n

        << " seconds="
        << seconds

        << " sweeps="
        << result.sweeps

        << " max_offdiag="
        << result.max_offdiag

        << " orthogonality_error="
        << ortho_error

        << " threads="
        << omp_get_max_threads()

        << "\n";

    return 0;
}