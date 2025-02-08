
#define NOMINMAX // この定義をWindows.hをインクルードする前に追加しないとエラーになる
#include <windows.h>
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>
#include <intrin.h>    // MSVCのビルトイン関数
#include <immintrin.h> // AVX2 ヘッダ
#include <chrono> // 処理時間計測用 時間計測しない場合は不要

// fast_float ライブラリを使用
// 下記から入手
// https://github.com/fastfloat/fast_float
#include "../fast_float/fast_float.h"  // fast_floatヘッダファイルのインクルード

#include "FastCsvLoad.h"

//////////////////////////////////////////////////////////////////////////////////////////////
//CSVファイル全体の「行の先頭位置（オフセット）」を取得
size_t GetLineOffsets(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets)
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
    return lineOffsets.size();
}
#define NEWLINETYPE_CRLF    2
#define NEWLINETYPE_LF      1
#define NEWLINETYPE_UNKNOWN 0

/////////////////////////////////////////////////////////////////////////
//改行コードの種類を最初の1行で判別
int DetectNewlineType(const char* fileContent, size_t contentSize) {
    for (size_t i = 0; i < contentSize; ++i) {
        if (fileContent[i] == '\n') {
            // 直前の文字が '\r' であれば CRLF と判断
            if (i > 0 && fileContent[i - 1] == '\r') {
                return NEWLINETYPE_CRLF;   //
            }
            else {
                return NEWLINETYPE_LF;   //LF
            }
        }
    }
    return NEWLINETYPE_UNKNOWN;
}

/////////////////////////////////////////////////////////////////////////
//メモリマップのCSVデータを行ごとに分解
size_t GetLineOffsets_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets)
{
    //改行コードの種類を最初の1行で判別
    int _nltype = DetectNewlineType(fileContent, contentSize);

    //改行コードの種類で処理を分ける 改行コードが混在しないと仮定
    if (_nltype == NEWLINETYPE_CRLF)
    {
        GetLineOffsets_CRLF_OpenMP(fileContent, contentSize, lineOffsets);
    }
    else if (_nltype == NEWLINETYPE_LF)
    {
        GetLineOffsets_LF_OpenMP(fileContent, contentSize, lineOffsets);
    }
    else
    {
        GetLineOffsets_LFCRLF_OpenMP(fileContent, contentSize, lineOffsets);
    }
    return lineOffsets.size();
}

/////////////////////////////////////////////////////////////////////////
//メモリマップのCSVデータを行ごとに分解　AVX2を使用 ほーんの少し速くなる
size_t GetLineOffsets_AVX2_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets)
{
    //改行コードの種類を最初の1行で判別
    int _nltype = DetectNewlineType(fileContent, contentSize);

    //改行コードの種類で処理を分ける 改行コードが混在しないと仮定
    if (_nltype == NEWLINETYPE_CRLF)
    {
        GetLineOffsets_CRLF_AVX2_OpenMP(fileContent, contentSize, lineOffsets);
    }
    else if (_nltype == NEWLINETYPE_LF)
    {
        GetLineOffsets_LF_AVX2_OpenMP(fileContent, contentSize, lineOffsets);
    }
    else
    {
        GetLineOffsets_LFCRLF_OpenMP(fileContent, contentSize, lineOffsets);
    }
    return lineOffsets.size();
}


//////////////////////////////////////////////////////////////////////////////////
//メモリマップのCSVデータを行ごとに分解　改行コードが混在している場合 汎用だが遅い
size_t GetLineOffsets_LFCRLF_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets) {
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
            while (start < contentSize && (fileContent[start] != '\r' || fileContent[start] != '\n')) {
                ++start;
            }
            while (start < contentSize && (fileContent[start] == '\r' || fileContent[start] == '\n')) {
                ++start;
            }
        }

        for (size_t pos = start; pos < end;) {
            // 現在の pos を行の先頭として記録
            localOffsets[threadId].push_back(pos);

            // 改行文字 (\n, \r) 以外の文字まで進む
            while (pos < end && fileContent[pos] != '\r' && fileContent[pos] != '\n') {
                ++pos;
            }

            // 改行文字 (\n, \r) をまとめてスキップ
            while (pos < end && (fileContent[pos] == '\r' || fileContent[pos] == '\n')) {
                ++pos;
            }
        }
    }

    // 結果を統合（`localOffsets` を `lineOffsets` にマージ）
    for (const auto& offsets : localOffsets) {
        lineOffsets.insert(lineOffsets.end(), offsets.begin(), offsets.end());
    }
    return lineOffsets.size();
}

//////////////////////////////////////////////////////////////////////////////////
//メモリマップのCSVデータを行ごとに分解　改行コードCRLF用
size_t GetLineOffsets_CRLF_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets) {
    const int numThreads = omp_get_max_threads();
    std::vector<std::vector<size_t>> localOffsets(numThreads);

#pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        size_t chunkSize = contentSize / numThreads;
        size_t start = threadId * chunkSize;
        size_t end = (threadId == numThreads - 1) ? contentSize : start + chunkSize;

        // 先頭が CRLF の途中にならないように調整
        if (threadId != 0) {
            while (start < contentSize && (fileContent[start] != '\r' || fileContent[start] != '\n')) {
                ++start;
            }
            while (start < contentSize && (fileContent[start] == '\r' || fileContent[start] == '\n')) {
                ++start;
            }
        }

        for (size_t pos = start; pos < end;) {
            // 現在の pos を行の先頭として記録
            localOffsets[threadId].push_back(pos);

            // CR (\r) 以外の文字まで進む
            while (pos < end && fileContent[pos] != '\r') {
                ++pos;
            }

            // CR が見つかった場合、次が LF (\n) であればまとめてスキップ
            if (pos < end && fileContent[pos] == '\r') {
                if ((pos + 1) < contentSize && fileContent[pos + 1] == '\n') {
                    pos += 2;
                }
                else {
                    pos++;
                }
            }
        }
    }

    // 各スレッドの結果を統合
    for (const auto& offsets : localOffsets) {
        lineOffsets.insert(lineOffsets.end(), offsets.begin(), offsets.end());
    }
    return lineOffsets.size();
}

//////////////////////////////////////////////////////////////////////////////////
//メモリマップのCSVデータを行ごとに分解　改行コードLF用
size_t GetLineOffsets_LF_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets) {
    const int numThreads = omp_get_max_threads();
    std::vector<std::vector<size_t>> localOffsets(numThreads);

#pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        size_t chunkSize = contentSize / numThreads;
        size_t start = threadId * chunkSize;
        size_t end = (threadId == numThreads - 1) ? contentSize : start + chunkSize;

        // 先頭が改行文字（\n）の途中にならないように調整
        if (threadId != 0) {
            while (start < contentSize && fileContent[start] != '\n') {
                ++start;
            }
            // 改行文字が続く場合はそれもスキップ
            while (start < contentSize && fileContent[start] == '\n') {
                ++start;
            }
        }

        for (size_t pos = start; pos < end;) {
            // 現在の pos を行の先頭として記録
            localOffsets[threadId].push_back(pos);

            // 改行文字以外の文字まで進む
            while (pos < end && fileContent[pos] != '\n') {
                ++pos;
            }

            // 改行文字（\n）をスキップ
            while (pos < end && fileContent[pos] == '\n') {
                ++pos;
            }
        }
    }

    // 各スレッドの結果を統合
    for (const auto& offsets : localOffsets) {
        lineOffsets.insert(lineOffsets.end(), offsets.begin(), offsets.end());
    }
    return lineOffsets.size();
}
//////////////////////////////////////////////////////////////////////////////////
// AVX2 + OpenMP による高速行オフセット取得
//メモリマップのCSVデータを行ごとに分解　改行コードLF用
size_t GetLineOffsets_LF_AVX2_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets) {
    const int numThreads = omp_get_max_threads();
    std::vector<std::vector<size_t>> localOffsets(numThreads);

#pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        size_t chunkSize = contentSize / numThreads;
        size_t start = threadId * chunkSize;
        size_t end = (threadId == numThreads - 1) ? contentSize : start + chunkSize;

        // 誤って改行文字の途中から処理しないように調整
        if (threadId != 0) {
            while (start < contentSize && fileContent[start] != '\n') {
                ++start;
            }
            while (start < contentSize && fileContent[start] == '\n') {
                ++start;
            }
        }

        size_t pos = start;
        const __m256i LF = _mm256_set1_epi8('\n');  // 改行文字'\n'を32バイトごとに比較するためのレジスタ

        while (pos < end) {
            // 現在のposを「行の先頭」として記録
            localOffsets[threadId].push_back(pos);

            // 次の改行文字を検索する
            size_t searchPos = pos;
            bool newlineFound = false;
            while (searchPos < end) {
                if (searchPos + 32 <= end) {
                    // 32バイト分をロードして一括比較
                    __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fileContent + searchPos));
                    __m256i cmp = _mm256_cmpeq_epi8(data, LF);
                    int mask = _mm256_movemask_epi8(cmp);
                    if (mask != 0) {
                        // マスクに1ビットが立っている＝改行文字が見つかった
                        unsigned long offset;
                        _BitScanForward(&offset, mask);  // 最下位の1ビット（＝最初の改行文字）の位置を取得
                        searchPos += offset;
                        newlineFound = true;
                        break;
                    }
                    else {
                        searchPos += 32;
                    }
                }
                else {
                    // 残りのバイト数が32未満の場合は逐次走査
                    if (fileContent[searchPos] == '\n') {
                        newlineFound = true;
                        break;
                    }
                    ++searchPos;
                }
            }

            if (!newlineFound) {
                // 改行が見つからなければこのブロック内は終了
                break;
            }

            // searchPosは改行文字が見つかった位置になっているので、
            // 連続する改行文字をスキップして次の行の開始位置を決定
            while (searchPos < end && fileContent[searchPos] == '\n') {
                ++searchPos;
            }
            pos = searchPos;
        }
    }

    // 各スレッドの結果を統合
    for (const auto& offsets : localOffsets) {
        lineOffsets.insert(lineOffsets.end(), offsets.begin(), offsets.end());
    }
    return lineOffsets.size();
}

//////////////////////////////////////////////////////////////////////////////////
//メモリマップのCSVデータを行ごとに分解　改行コードCRLF用
size_t GetLineOffsets_CRLF_AVX2_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets) {
    const int numThreads = omp_get_max_threads();
    std::vector<std::vector<size_t>> localOffsets(numThreads);

#pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        size_t chunkSize = contentSize / numThreads;
        size_t start = threadId * chunkSize;
        size_t end = (threadId == numThreads - 1) ? contentSize : start + chunkSize;

        // 先頭が CRLF の途中にならないように調整（前のスレッドで終わった改行をスキップ）
        if (threadId != 0) {
            while (start < contentSize && (fileContent[start] != '\r' || fileContent[start] != '\n')) {
                ++start;
            }
            while (start < contentSize && (fileContent[start] == '\r' || fileContent[start] == '\n')) {
                ++start;
            }
        }

        const __m256i vCR = _mm256_set1_epi8('\r');  // CR文字を全バイトにセット
        size_t pos = start;
        while (pos < end) {
            // 現在の pos を行の先頭として記録
            localOffsets[threadId].push_back(pos);

            // pos以降から CR を探す（AVX2による高速検索）
            size_t scanPos = pos;
            bool foundCR = false;
            while (scanPos < end) {
                if (scanPos + 32 <= end) {
                    // 32バイト分を一括読み込み
                    __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fileContent + scanPos));
                    // CRとの比較
                    __m256i cmp = _mm256_cmpeq_epi8(data, vCR);
                    // 各バイトの最上位ビットをまとめた32ビットのマスクを取得
                    int mask = _mm256_movemask_epi8(cmp);
                    if (mask != 0) {
                        // マスク内で最も下位の1ビットの位置を取得（最初に現れる CR）
                        unsigned long offset;
                        _BitScanForward(&offset, mask);
                        scanPos += offset;
                        foundCR = true;
                        break;
                    }
                    else {
                        scanPos += 32;
                    }
                }
                else {
                    // 32バイト未満の残り領域は逐次走査
                    if (fileContent[scanPos] == '\r') {
                        foundCR = true;
                        break;
                    }
                    ++scanPos;
                }
            }
            if (!foundCR) {
                // CRが見つからなかったので、このスレッドの処理は終了
                pos = end;
                break;
            }

            // scanPosはCRの位置を指している
            // CR が見つかったら、次が LF ('\n') であればまとめてスキップ
            if (scanPos + 1 < contentSize && fileContent[scanPos + 1] == '\n') {
                pos = scanPos + 2;
            }
            else {
                pos = scanPos + 1;
            }
        }
    }

    // 各スレッドの結果を統合
    for (const auto& offsets : localOffsets) {
        lineOffsets.insert(lineOffsets.end(), offsets.begin(), offsets.end());
    }

    return lineOffsets.size();
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
    //GetLineOffsets_LF_OpenMP(fileContent, contentSize, lineOffsets);
    //GetLineOffsets_CRLF_OpenMP(fileContent, contentSize, lineOffsets);

    //AVX2使用
    GetLineOffsets_AVX2_OpenMP(fileContent, contentSize, lineOffsets);

    //チェック
    //std::cout << "Read Lines by GetLine: " << lineOffsets.size() << std::endl;

    //--------------------------------------------------------------------------
    // 2) 結果を格納するベクターを行数分確保
    //--------------------------------------------------------------------------
    pointClouds.resize(lineOffsets.size());

    //--------------------------------------------------------------------------
    // 3) 各行を並列でパース（OpenMP 使用）
    //--------------------------------------------------------------------------
    //PointCloud p; // 一行分を格納する構造体
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

#ifdef _DEBUG
            std::cout << p.fields[0] << std::endl;
#endif
            // 出来上がった PointCloud をベクターに格納
            pointClouds[lineIndex] = p;

#ifdef _DEBUG
            std::cout << p.x << std::endl;
#endif
        }
    // メモリマップの後始末
    UnmapViewOfFile(pData);
    CloseHandle(hMap);
    CloseHandle(hFile);

    return 0; // 正常終了
}


