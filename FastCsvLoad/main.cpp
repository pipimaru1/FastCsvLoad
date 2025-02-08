#define NOMINMAX // ���̒�`��Windows.h���C���N���[�h����O�ɒǉ����Ȃ��ƃG���[�ɂȂ�
#include <windows.h>
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>
//#include <emmintrin.h> // SSE2 (SIMD)
#include <immintrin.h> // AVX2 �w�b�_
#include <chrono> // �������Ԍv���p ���Ԍv�����Ȃ��ꍇ�͕s�v

#include "FastCsvLoad.h"


//"../../hoge/hoge.csv"
//////////////////////////////////////////////////////////////////////////////////////////////
//�g�p��
int main(int argc, char* argv[]) {
    // �R�}���h���C�������`�F�b�N
    if (argc != 2) {
        std::cerr << "Usage: FastCsvLoad.exe <input_file_path>" << std::endl;
        return 1;
    }

    // ���̓t�@�C�������擾
    std::string inputFilePath = argv[1];
    std::wstring wideFilePath(inputFilePath.begin(), inputFilePath.end()); // �}���`�o�C�g���烏�C�h������ɕϊ�

    std::wcout << L"FastCsvLoad.exe " << std::endl;
    std::wcout << L"Read File: " << wideFilePath << std::endl;

    int maxThreads = omp_get_max_threads();
    std::cout << "Max threads available: " << maxThreads << std::endl;

    std::vector<PointCloud> pointClouds;

    // �������Ԍv���̊J�n
    auto start = std::chrono::high_resolution_clock::now();
    if (FastCsvLoad(wideFilePath, pointClouds, COLUMN_SIZE) != 0) {
        std::cerr << "Failed to load the file: " << inputFilePath << std::endl;
        return 1;
    }

    // �������Ԍv���̏I��
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // �f�[�^�T�C�Y�̌v�Z
    size_t dataSizeBytes = pointClouds.size() * sizeof(PointCloud);

    // ���ʂ̏o��
    std::cout.imbue(std::locale("")); // �J���}��؂�̐��l�t�H�[�}�b�g��K�p
    std::cout << "Read Lines: " << pointClouds.size() << std::endl;
    std::cout << "Data Size (bytes): " << dataSizeBytes << " bytes" << std::endl;
    std::cout << "Processing Time: " << duration << " msec" << std::endl;

    return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////////
//�ȉ��̓e�X�g�pCSV�쐻�v���O���� 
// ���L�R�[�h��؂���Amain_make_csv_for_test()��main()�ɕύX���ăr���h
// 1���s �f�B�X�N��T�C�Y10GB�̃t�@�C���𐶐�
// �����ɂ�10~20��������
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>

int main_make_csv_for_test() {
    // �t�@�C�������w��
    const std::string output_file = "large_data.csv";

    // �s���Ɨ񐔂��w��
    const long long num_rows = 100000000; // 1���s
    const int num_cols = COLUMN_SIZE;          // COLUMN=10��

    // �o�̓t�@�C�����J��
    std::ofstream csvfile(output_file);
    if (!csvfile.is_open()) {
        std::cerr << "�t�@�C�����J���܂���ł���: " << output_file << std::endl;
        return 1;
    }

    // �����_�����������������
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1000.0);

    // �f�[�^�𐶐����ď�������
    for (long long i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            csvfile << std::fixed << std::setprecision(6) << dis(gen);
            if (j < num_cols - 1) {
                csvfile << ","; // �J���}��؂�
            }
        }
        csvfile << "\n"; // �s�̏I���

        // 100���s���Ƃ�#��\��
        if ((i + 1) % 1000000 == 0) {
            std::cout << "#";
            std::cout.flush();
        }
    }

    csvfile.close();
    std::cout << "\n" << output_file << " ��1���s�̃f�[�^�𐶐����܂����I" << std::endl;

    return 0;
}
