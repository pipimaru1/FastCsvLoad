#pragma once

#define COLUMN_SIZE 10 //CSVの列数が違う場合はここを変更
#define MARGIN_RATIO 1.01 //メモリ確保の時の余裕率

//////////////////////////////////////////////////////////////////////////////////////////////
// 読み込みデータの構造体定義
// 点群データとして扱いやすいようunionを定義
// 汎用的に使いたい場合は、unionブロックを消して、foatの配列だけの構造体にすればOK
#ifndef _POINTCLOUD
struct PointCloud {
    union {
        struct {
            float x, y, z;    // 座標
            float acc;
            float r, g, b;    // 色
            float nx, ny, nz; // 法線ベクトル
        };
        float fields[COLUMN_SIZE];
    };
};
#endif




//////////////////////////////////////////////////////////////////////////////////////////////
//xyz型の点群データを上記の構造体の配列に格納する関数
int FastCsvLoad(const std::wstring& filename, std::vector<PointCloud>& pointClouds, int num_cols);

//////////////////////////////////////////////////////////////////////////////////////////////
//CSVファイル全体の「行の先頭位置（オフセット）」を取得 特に高速化しない
size_t GetLineOffsets(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);

//////////////////////////////////////////////////////////////////////////////////////////////
// OpenMP による高速行オフセット取得
size_t GetLineOffsets_CRLF_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);
size_t GetLineOffsets_LF_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);
size_t GetLineOffsets_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);

// AVX2 + OpenMP による高速行オフセット取得
size_t GetLineOffsets_AVX2_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);
size_t GetLineOffsets_CRLF_AVX2_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);
size_t GetLineOffsets_LF_AVX2_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);

//両方混在の場合
size_t GetLineOffsets_LFCRLF_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);

int SlowCsvLoad(const std::wstring& filePath, std::vector<PointCloud>& pointClouds);

