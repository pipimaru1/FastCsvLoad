
#define NOMINMAX // この定義をWindows.hをインクルードする前に追加しないとエラーになる
#include <windows.h>
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>
#include <emmintrin.h> // SSE2 (SIMD)
#include <immintrin.h> // AVX2 ヘッダ
#include <chrono> // 処理時間計測用 時間計測しない場合は不要

// fast_float ライブラリを使用
// 下記から入手
// https://github.com/fastfloat/fast_float
#include "../fast_float/fast_float.h"  // fast_floatヘッダファイルのインクルード

#include "FastCsvLoad.h"

//////////////////////////////////////////////////////////////////////////////////////////////
//CSVファイル全体の「行の先頭位置（オフセット）」を取得
void GetLineOffsets(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets) 
{
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
}

void GetLineOffsets_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets) {
    const int numThreads = omp_get_max_threads(); // 使用可能な最大スレッド数
    std::vector<std::vector<size_t>> localOffsets(numThreads); // スレッドごとの結果を格納

#pragma omp parallel
    {
        int threadId = omp_get_thread_num(); // スレッドID
        size_t chunkSize = contentSize / numThreads; // 各スレッドが処理するデータ範囲
        size_t start = threadId * chunkSize;
        size_t end = (threadId == numThreads - 1) ? contentSize : start + chunkSize;

        // 先頭位置を調整（改行文字の途中から始まらないようにする）
        if (threadId != 0) {
            while (start < contentSize && (fileContent[start] == '\n' || fileContent[start] == '\r')) {
                ++start;
            }
        }

        for (size_t pos = start; pos < end;) {
            // 現在の pos を行の先頭として記録
            localOffsets[threadId].push_back(pos);

            // 改行文字 (\n, \r) 以外の文字まで進む
            while (pos < end && fileContent[pos] != '\n' && fileContent[pos] != '\r') {
                ++pos;
            }

            // 改行文字 (\n, \r) をまとめてスキップ
            while (pos < end && (fileContent[pos] == '\n' || fileContent[pos] == '\r')) {
                ++pos;
            }
        }
    }

    // 結果を統合（`localOffsets` を `lineOffsets` にマージ）
    for (const auto& offsets : localOffsets) {
        lineOffsets.insert(lineOffsets.end(), offsets.begin(), offsets.end());
    }
}

#include <immintrin.h>  // AVX2 用
#include <vector>
#include <omp.h>

// AVX2 + OpenMP による高速行オフセット取得
void GetLineOffsets_AVX2_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets) {
    const int numThreads = omp_get_max_threads();
    std::vector<std::vector<size_t>> localOffsets(numThreads);  // スレッドごとのオフセット格納

#pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        size_t chunkSize = contentSize / numThreads;
        size_t start = threadId * chunkSize;
        size_t end = (threadId == numThreads - 1) ? contentSize : start + chunkSize;

        // 改行途中でスレッドを開始しないように調整
        if (threadId != 0) {
            while (start < contentSize && (fileContent[start] == '\n' || fileContent[start] == '\r')) {
                ++start;
            }
        }

        // `\n` と `\r` を検索するための AVX2 ベクトル
        const __m256i newline_lf = _mm256_set1_epi8('\n');  // LF (Line Feed)
        const __m256i newline_cr = _mm256_set1_epi8('\r');  // CR (Carriage Return)

        size_t pos = start;
        while (pos + 32 <= end) {
            // 32バイトを読み込む
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fileContent + pos));

            // 改行文字 (`\n` or `\r`) の位置を探す
            __m256i cmp_lf = _mm256_cmpeq_epi8(chunk, newline_lf);
            __m256i cmp_cr = _mm256_cmpeq_epi8(chunk, newline_cr);
            __m256i cmp = _mm256_or_si256(cmp_lf, cmp_cr);  // 両者のビットマスクを統合

            // 改行位置のビットマスクを取得
            int mask = _mm256_movemask_epi8(cmp);

            // ビットマスクを解析してオフセットを記録（次の行の先頭位置を記録する）
            while (mask) {
                int offset = _tzcnt_u32(mask);  // 最初に見つかったビット位置
                size_t newOffset = pos + offset + 1; // **次の行の先頭位置** を記録

                // **Windows の `\r\n` を考慮**（`newOffset` が `\r\n` の `\r` に当たっている場合は `\n` もスキップ）
                if (newOffset < contentSize - 1 && fileContent[newOffset - 1] == '\r' && fileContent[newOffset] == '\n') {
                    ++newOffset; // `\r\n` の `\n` へ進める
                }

                localOffsets[threadId].push_back(newOffset);
                mask &= (mask - 1);  // すでに見つかったビットをクリア
            }

            pos += 32;  // 次の 32 バイトへ進む
        }

        // 残りの部分を処理（AVX2 で処理できなかった範囲）
        for (; pos < end; ++pos) {
            if (fileContent[pos] == '\n' || fileContent[pos] == '\r') {
                size_t newOffset = pos + 1;

                // **Windows の `\r\n` を考慮**
                if (newOffset < contentSize - 1 && fileContent[newOffset - 1] == '\r' && fileContent[newOffset] == '\n') {
                    ++newOffset;
                }

                localOffsets[threadId].push_back(newOffset);
            }
        }
    }

    // 結果を統合
    for (const auto& offsets : localOffsets) {
        lineOffsets.insert(lineOffsets.end(), offsets.begin(), offsets.end());
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////
// @brief 1行10要素のCSV ファイルを読み込み、pointClouds に格納する
// @param[in]  filename     入力ファイルパス（ワイド文字列）
// @param[out] pointClouds  読み込んだ点群データを格納するベクター
// @return                  成功時は 0、失敗時は非 0
int FastCsvLoad(const std::wstring& filename, std::vector<PointCloud>& pointClouds, int num_cols)
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

    std::cout.imbue(std::locale("")); // カンマ区切りの数値フォーマットを適用
    std::cout << "FileSize: " << fileSize.QuadPart << " byte" << std::endl;

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

    // 最初の行サイズを測定
    size_t firstLineSize = 0;
    while (firstLineSize < contentSize && fileContent[firstLineSize] != '\n') {
        ++firstLineSize;
    }
    std::cout << "firstLineSize: " << firstLineSize << " byte" << std::endl;

    // 推定行数を計算
    size_t estimatedLines = 0;
    if (firstLineSize > 0) {
        estimatedLines = (contentSize / firstLineSize) * MARGIN_RATIO;//5%多めに設定
    }
    std::cout << "estimatedLines: " << estimatedLines << " line" << std::endl;

    //--------------------------------------------------------------------------
    // 1) 行頭オフセットの取得
    // 固定長の場合、この処理は省ける
    //--------------------------------------------------------------------------
    std::vector<size_t> lineOffsets;

    // 推定行数で lineOffsets を事前予約
    lineOffsets.reserve(estimatedLines);

    //通常の方法 OK
    //GetLineOffsets(fileContent, contentSize, lineOffsets);

    //OpenMP使用
    GetLineOffsets_OpenMP(fileContent, contentSize, lineOffsets);

    //SSE使用
    //GetLineOffsets_SSE_OpenMP(fileContent, contentSize, lineOffsets);

    //AVX2使用
    //GetLineOffsets_AVX2_OpenMP(fileContent, contentSize, lineOffsets);

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

            // 単純に num_cols個の float を CSV から読み込む
            for (int i = 0; i < num_cols; ++i) {
                // fast_float でパース
                auto result = fast_float::from_chars(ptr, end, p.fields[i]);
                ptr = result.ptr;
                // カンマがあればスキップ（行末近くで区切りがない場合もあるのでチェック）
                if (ptr < end && *ptr == ',') {
                    ++ptr;
                }
            }

            std::cout << *ptr<<" "<< p.fields[0] << std::endl;

            // 出来上がった PointCloud をベクターに格納
            pointClouds[lineIndex] = p;

            std::cout << p.x << std::endl;
        }
    // メモリマップの後始末
    UnmapViewOfFile(pData);
    CloseHandle(hMap);
    CloseHandle(hFile);

    return 0; // 正常終了
}

