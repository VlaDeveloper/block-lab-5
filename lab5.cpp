#include <iostream>
#include <time.h>
#include <vector>
#include <mpi.h>

using namespace std;

// размер матрицы
const int N = 200;

// часть строки матрицы А
const int num_i = 50;

// объявляем матрицы размером N
double matr[N][N];
double B[N];
double E[N][N];

double segmM[num_i][N];
double segmE[num_i][N];

double segmMK[N];
double segmEK[N];

double mini_segmM[num_i];
double mini_segmE[num_i];

// получаем рандомное число
double GetRandom(const int min, const int max)
{
    return rand() % (max - min + 1) + min;
}

int main(int argc, char* argv[])
{
    setlocale(LC_ALL, "Russian");

    int size, rank;
    MPI_Status status;
    MPI_Request request;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Заполнение матрицы A, B и E
    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                matr[i][j] = GetRandom(0, 100);
                if (i == j) E[i][j] = 1.0;
                else E[i][j] = 0.0;
            }
            B[i] = GetRandom(0, 100);
        }
    }

    // Прямой ход
    double t = clock();

    double div, multi;
    for (int k = 0; k < N; k++)
    {
        if (rank == 0)
        {
            if (matr[k][k] == 0.0)
            {
                bool changed = false;
                for (int i = k + 1; i < N; i++)
                {
                    if (matr[i][k] != 0)
                    {
                        swap(matr[k], matr[i]);
                        swap(E[k], E[i]);
                        changed = true;
                        break;
                    }
                }
                if (!changed)
                {
                    cout << endl << "Error: матрица не может быть найдена" << endl;
                    return -1;
                }
            }
            div = matr[k][k];
        }

        MPI_Scatter(matr[k], num_i, MPI_DOUBLE, mini_segmM, num_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(E[k], num_i, MPI_DOUBLE, mini_segmE, num_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&div, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int j = 0; j < num_i; j++)
        {
            mini_segmM[j] /= div;
            mini_segmE[j] /= div;
        }

        MPI_Gather(mini_segmM, num_i, MPI_DOUBLE, matr[k], num_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(mini_segmE, num_i, MPI_DOUBLE, E[k], num_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            for (int i = 0; i < N; i++)
            {
                segmMK[i] = matr[k][i];
                segmEK[i] = E[k][i];
            }
        }

        MPI_Bcast(segmMK, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(segmEK, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Scatter(matr, num_i * N, MPI_DOUBLE, segmM, num_i * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(E, num_i * N, MPI_DOUBLE, segmE, num_i * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (int i = 0; i < num_i; i++)
        {
            if ((rank * num_i) + i <= k)
                continue;

            multi = segmM[i][k];
            for (int j = 0; j < N; j++)
            {
                segmM[i][j] -= multi * segmMK[j];
                segmE[i][j] -= multi * segmEK[j];
            }
        }

        MPI_Gather(segmM, N * num_i, MPI_DOUBLE, matr, N * num_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(segmE, N * num_i, MPI_DOUBLE, E, N * num_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    //Обратный ход
    for (int k = N - 1; k > 0; k--)
    {
        if (rank == 0)
        {
            for (int i = 0; i < N; i++)
            {
                segmMK[i] = matr[k][i];
                segmEK[i] = E[k][i];
            }
        }

        MPI_Bcast(segmMK, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(segmEK, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(matr, num_i * N, MPI_DOUBLE, segmM, num_i * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(E, num_i * N, MPI_DOUBLE, segmE, num_i * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (int i = num_i - 1; i > -1; i--)
        {
            if ((rank * num_i) + i >= k)
                continue;

            multi = segmM[i][k];
            for (int j = 0; j < N; j++)
            {
                segmM[i][j] -= multi * segmMK[j];
                segmE[i][j] -= multi * segmEK[j];
            }
        }

        MPI_Gather(segmM, N * num_i, MPI_DOUBLE, matr, N * num_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(segmE, N * num_i, MPI_DOUBLE, E, N * num_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double X[N];
    double segmentX[num_i];

    MPI_Bcast(B, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(E, num_i * N, MPI_DOUBLE, segmE, num_i * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вычисление X
    for (int i = 0; i < num_i; i++)
    {
        segmentX[i] = 0;
        for (int j = 0; j < N; j++)
            segmentX[i] += segmE[i][j] * B[j];
    }

    MPI_Gather(segmentX, num_i, MPI_DOUBLE, X, num_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "\nСистема уравнений:";
        for (int i = 0; i < N; i++)
            cout << "\nx" << i + 1 << " = " << X[i];

        t = (clock() - t) / 1000;
        cout << "\n\nВремя, потраченное на вычисления: " << t << "с.\n";
    }
    MPI_Finalize();
    return 0;
}