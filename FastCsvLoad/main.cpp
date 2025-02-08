#define NOMINMAX // この定義をWindows.hをインクルードする前に追加しないとエラーになる
#include <windows.h>
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>
//#include <emmintrin.h> // SSE2 (SIMD)
#include <immintrin.h> // AVX2 ヘッダ
#include <chrono> // 処理時間計測用 時間計測しない場合は不要

#include "FastCsvLoad.h"


//"../../hoge/hoge.csv"
//////////////////////////////////////////////////////////////////////////////////////////////
//使用例
int main(int argc, char* argv[]) {
    // コマンドライン引数チェック
    if (argc != 2) {
        std::cerr << "Usage: FastCsvLoad.exe <input_file_path>" << std::endl;
        return 1;
    }

    // 入力ファイル名を取得
    std::string inputFilePath = argv[1];
    std::wstring wideFilePath(inputFilePath.begin(), inputFilePath.end()); // マルチバイトからワイド文字列に変換

    std::wcout << L"FastCsvLoad.exe " << std::endl;
    std::wcout << L"Read File: " << wideFilePath << std::endl;

    int maxThreads = omp_get_max_threads();
    std::cout << "Max threads available: " << maxThreads << std::endl;

    std::vector<PointCloud> pointClouds;

    // 処理時間計測の開始
    auto start = std::chrono::high_resolution_clock::now();
    if (FastCsvLoad(wideFilePath, pointClouds, COLUMN_SIZE) != 0) {
        std::cerr << "Failed to load the file: " << inputFilePath << std::endl;
        return 1;
    }

    // 処理時間計測の終了
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // データサイズの計算
    size_t dataSizeBytes = pointClouds.size() * sizeof(PointCloud);

    // 結果の出力
    std::cout.imbue(std::locale("")); // カンマ区切りの数値フォーマットを適用
    std::cout << "Read Lines: " << pointClouds.size() << std::endl;
    std::cout << "Data Size (bytes): " << dataSizeBytes << " bytes" << std::endl;
    std::cout << "Processing Time: " << duration << " msec" << std::endl;

    return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////////
//以下はテスト用CSV作製プログラム 
// 下記コードを切り取り、main_make_csv_for_test()をmain()に変更してビルド
// 1億行 ディスク上サイズ10GBのファイルを生成
// 生成には10~20分かかる
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>

int main_make_csv_for_test() {
    // ファイル名を指定
    const std::string output_file = "large_data.csv";

    // 行数と列数を指定
    const long long num_rows = 100000000; // 1億行
    const int num_cols = COLUMN_SIZE;          // COLUMN=10列

    // 出力ファイルを開く
    std::ofstream csvfile(output_file);
    if (!csvfile.is_open()) {
        std::cerr << "ファイルを開けませんでした: " << output_file << std::endl;
        return 1;
    }

    // ランダム数生成器を初期化
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1000.0);

    // データを生成して書き込む
    for (long long i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            csvfile << std::fixed << std::setprecision(6) << dis(gen);
            if (j < num_cols - 1) {
                csvfile << ","; // カンマ区切り
            }
        }
        csvfile << "\n"; // 行の終わり

        // 100万行ごとに#を表示
        if ((i + 1) % 1000000 == 0) {
            std::cout << "#";
            std::cout.flush();
        }
    }

    csvfile.close();
    std::cout << "\n" << output_file << " に1億行のデータを生成しました！" << std::endl;

    return 0;
}
