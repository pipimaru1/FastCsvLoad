
#define NOMINMAX // この定義をWindows.hをインクルードする前に追加しないとエラーになる
#include <windows.h>
#include <vector>
#include <string>
#include <iostream>
// 必要であれば OpenMP を使用（並列化を行いたくない場合は削除）
// ※ Visual Studio などでコンパイル時に /openmp (あるいは /openmp:experimental) を指定
#include <omp.h>

#include <chrono> // 処理時間計測用 時間計測しない場合は不要

// fast_float ライブラリを使用
// 下記から入手
// https://github.com/fastfloat/fast_float
#include "../fast_float/include/fast_float/fast_float.h"  // fast_floatヘッダファイルのインクルード

//読み込みデータの構造体定義 点群データを想定しているが、ただのfloat[10]と考えてOK
struct PointCloud {
    union {
        struct {
            float x, y, z;    // 座標
            float acc;
            float r, g, b;    // 色
            float nx, ny, nz; // 法線ベクトル
        };
        float fields[10];
    };
};

/**
 * @brief XYZRGB 形式の CSV ファイルを読み込み、pointClouds に格納する
 * @param[in]  filename     入力ファイルパス（ワイド文字列）
 * @param[out] pointClouds  読み込んだ点群データを格納するベクター
 * @return                  成功時は 0、失敗時は非 0
 */
int FastCsvLoad(const std::wstring& filename,std::vector<PointCloud>& pointClouds)
{
    // ファイルを開く (Windows API)
    HANDLE hFile = CreateFileW(
        filename.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    if (hFile == INVALID_HANDLE_VALUE) {
        std::wcerr << L"ファイルを開けません: " << filename << std::endl;
        return 1;
    }

    // ファイルサイズを取得
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        std::wcerr << L"ファイルサイズの取得に失敗しました。" << std::endl;
        CloseHandle(hFile);
        return 1;
    }

    // ファイルをメモリにマップ
    HANDLE hMap = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMap == NULL) {
        std::wcerr << L"ファイルマッピングの作成に失敗しました。" << std::endl;
        CloseHandle(hFile);
        return 1;
    }

    // ファイル全体をマッピング
    LPCVOID pData = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (pData == NULL) {
        std::wcerr << L"ファイルのマッピングに失敗しました。" << std::endl;
        CloseHandle(hMap);
        CloseHandle(hFile);
        return 1;
    }

    // ファイル内容を文字列として扱う
    const char* fileContent = static_cast<const char*>(pData);
    size_t contentSize = static_cast<size_t>(fileSize.QuadPart);

    //--------------------------------------------------------------------------
    // 1) 行頭オフセットの取得
    //--------------------------------------------------------------------------
    std::vector<size_t> lineOffsets;
    lineOffsets.reserve(1024); // 大きめに予約（必要に応じて調整）
    {
        size_t pos = 0;
        while (pos < contentSize) {
            // 現在の pos を「この行の先頭」として記録
            lineOffsets.push_back(pos);

            // 行終端(改行文字 \n, \r) まで進める
            while (pos < contentSize &&
                fileContent[pos] != '\n' &&
                fileContent[pos] != '\r')
            {
                ++pos;
            }
            // 改行文字をスキップ (\r\n などまとめて飛ばす)
            while (pos < contentSize &&
                (fileContent[pos] == '\n' || fileContent[pos] == '\r'))
            {
                ++pos;
            }
        }
    }

    //--------------------------------------------------------------------------
    // 2) 結果を格納するベクターを行数分確保
    //--------------------------------------------------------------------------
    pointClouds.resize(lineOffsets.size());

    //--------------------------------------------------------------------------
    // 3) 各行を並列でパース（OpenMP 使用）
    //--------------------------------------------------------------------------
#pragma omp parallel for
    for (int lineIndex = 0; lineIndex < static_cast<int>(lineOffsets.size()); ++lineIndex)
    {
        // この行の開始位置と終了位置
        size_t startPos = lineOffsets[lineIndex];
        size_t endPos = (lineIndex + 1 < static_cast<int>(lineOffsets.size()))
            ? lineOffsets[lineIndex + 1]
            : contentSize;
        // ポインタをセット
        const char* ptr = &fileContent[startPos];
        const char* end = &fileContent[endPos];

        PointCloud p; // 一行分を格納する構造体

        // 単純に 10 個の float を CSV から読み込む
        for (int i = 0; i < 10; ++i) {
            // fast_float でパース
            auto result = fast_float::from_chars(ptr, end, p.fields[i]);
            ptr = result.ptr;
            // カンマがあればスキップ（行末近くで区切りがない場合もあるのでチェック）
            if (ptr < end && *ptr == ',') {
                ++ptr;
            }
        }

        // 出来上がった PointCloud をベクターに格納
        pointClouds[lineIndex] = p;
    }

    // メモリマップの後始末
    UnmapViewOfFile(pData);
    CloseHandle(hMap);
    CloseHandle(hFile);

    return 0; // 正常終了
}

//使用例

int main()
{
    std::vector<PointCloud> pointClouds;

    // 処理時間計測の開始
    auto start = std::chrono::high_resolution_clock::now();
    FastCsvLoad(L"../hoge/hoge.csv", pointClouds);

    // 処理時間計測の終了
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // データサイズの計算
    size_t dataSizeBytes = pointClouds.size() * sizeof(PointCloud);

    // 結果の出力
    std::cout.imbue(std::locale(""));
    std::cout << "Read Lines: " << pointClouds.size() << std::endl;
    std::cout << "Data Size (bytes): " << dataSizeBytes << " bytes" << std::endl;
    std::cout << "Processing Time: " << duration << " msec" << std::endl;

    return 0;
}
