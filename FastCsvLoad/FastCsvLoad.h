#pragma once

#define COLUMN_SIZE 10
#define MARGIN_RATIO 1.01

//////////////////////////////////////////////////////////////////////////////////////////////
//�ǂݍ��݃f�[�^�̍\���̒�` �_�Q�f�[�^��z�肵�Ă��邪�A������float[10]�ƍl����OK
#ifndef _POINTCLOUD
struct PointCloud {
    union {
        struct {
            float x, y, z;    // ���W
            float acc;
            float r, g, b;    // �F
            float nx, ny, nz; // �@���x�N�g��
        };
        float fields[COLUMN_SIZE];
    };
};
#endif

//////////////////////////////////////////////////////////////////////////////////////////////
//xyz�^�̓_�Q�f�[�^����L�̍\���̂̔z��Ɋi�[����֐�
int FastCsvLoad(const std::wstring& filename, std::vector<PointCloud>& pointClouds, int num_cols);

//////////////////////////////////////////////////////////////////////////////////////////////
//CSV�t�@�C���S�̂́u�s�̐擪�ʒu�i�I�t�Z�b�g�j�v���擾 ���ɍ��������Ȃ�
void GetLineOffsets(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);

//////////////////////////////////////////////////////////////////////////////////////////////
// OpenMP �ɂ�鍂���s�I�t�Z�b�g�擾
void GetLineOffsets_CRLF_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);
void GetLineOffsets_LF_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);
void GetLineOffsets_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);


// AVX2 + OpenMP �ɂ�鍂���s�I�t�Z�b�g�擾
void GetLineOffsets_AVX2_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);
void GetLineOffsets_CRLF_AVX2_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);
void GetLineOffsets_LF_AVX2_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);

//�������݂̏ꍇ
void GetLineOffsets_LFCRLF_OpenMP(const char* fileContent, size_t contentSize, std::vector<size_t>& lineOffsets);


