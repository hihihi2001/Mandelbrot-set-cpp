#pragma once
#include <iostream>
#include <stdio.h>
#include <chrono>

#define BYTE unsigned char
#define S 1
#define MS 2
#define US 3
#define NS 4
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

using namespace std;

class Measure
{
    chrono::steady_clock::time_point begin;
public:
    Measure()
    {
        begin = chrono::steady_clock::now();
    }
    void start()
    {
        begin = chrono::steady_clock::now();
    }
    auto stop(int mode = MS)
    {
        chrono::steady_clock::time_point end;
        end = chrono::steady_clock::now();
        switch (mode)
        {
        case S:
            return chrono::duration_cast<chrono::seconds>(end - begin).count();
            break;
        case MS:
            return chrono::duration_cast<chrono::milliseconds>(end - begin).count();
            break;
        case US:
            return chrono::duration_cast<chrono::microseconds>(end - begin).count();
            break;
        case NS:
            return chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
            break;
        default:
            return (long long int) - 1;
        }
    }
};

BYTE* createBitmapFileHeader(int height, int stride)
{
    int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    static BYTE fileHeader[] = {
        0,0,     /// signature
        0,0,0,0, /// image file size in bytes
        0,0,0,0, /// reserved
        0,0,0,0, /// start of pixel array
    };

    fileHeader[0] = (BYTE)('B');
    fileHeader[1] = (BYTE)('M');
    fileHeader[2] = (BYTE)(fileSize);
    fileHeader[3] = (BYTE)(fileSize >> 8);
    fileHeader[4] = (BYTE)(fileSize >> 16);
    fileHeader[5] = (BYTE)(fileSize >> 24);
    fileHeader[10] = (BYTE)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return fileHeader;
}

BYTE* createBitmapInfoHeader(int height, int width)
{
    static BYTE infoHeader[] = {
        0,0,0,0, /// header size
        0,0,0,0, /// image width
        0,0,0,0, /// image height
        0,0,     /// number of color planes
        0,0,     /// bits per pixel
        0,0,0,0, /// compression
        0,0,0,0, /// image size
        0,0,0,0, /// horizontal resolution
        0,0,0,0, /// vertical resolution
        0,0,0,0, /// colors in color table
        0,0,0,0, /// important color count
    };

    infoHeader[0] = (BYTE)(INFO_HEADER_SIZE);
    infoHeader[4] = (BYTE)(width);
    infoHeader[5] = (BYTE)(width >> 8);
    infoHeader[6] = (BYTE)(width >> 16);
    infoHeader[7] = (BYTE)(width >> 24);
    infoHeader[8] = (BYTE)(height);
    infoHeader[9] = (BYTE)(height >> 8);
    infoHeader[10] = (BYTE)(height >> 16);
    infoHeader[11] = (BYTE)(height >> 24);
    infoHeader[12] = (BYTE)(1);
    infoHeader[14] = (BYTE)(3 * 8);

    return infoHeader;
}

void generateBitmapImage(BYTE* image, int height, int width, char* imageFileName)
{
    int widthInBytes = width * 3;

    BYTE padding[3] = { 0, 0, 0 };
    int paddingSize = (4 - (widthInBytes) % 4) % 4;

    int stride = (widthInBytes)+paddingSize;

    FILE* imageFile;
    fopen_s(&imageFile, imageFileName, "wb");
    if (imageFile == NULL)
        printf("failed to save image\n");
    else
    {
        BYTE* fileHeader = createBitmapFileHeader(height, stride);
        fwrite(fileHeader, 1, FILE_HEADER_SIZE, imageFile);

        BYTE* infoHeader = createBitmapInfoHeader(height, width);
        fwrite(infoHeader, 1, INFO_HEADER_SIZE, imageFile);

        for (int i = 0; i < height; i++)
        {
            fwrite(image + (3 * i * width), 3, width, imageFile);
            fwrite(padding, 1, paddingSize, imageFile);
        }

        fclose(imageFile);
    }
}

__device__ BYTE colormap[256][3] = 
{
    {0, 0, 36},
    {0, 0, 36},
    {0, 0, 36},
    {0, 0, 36},
    {0, 0, 36},
    {0, 0, 36},
    {0, 0, 39},
    {0, 0, 39},
    {0, 0, 39},
    {0, 0, 39},
    {0, 0, 39},
    {0, 0, 39},
    {0, 0, 39},
    {0, 0, 42},
    {0, 0, 42},
    {0, 0, 42},
    {0, 0, 42},
    {0, 0, 44},
    {0, 0, 44},
    {0, 0, 44},
    {0, 0, 47},
    {0, 0, 47},
    {0, 0, 47},
    {0, 0, 47},
    {0, 0, 49},
    {0, 0, 49},
    {0, 0, 52},
    {0, 0, 52},
    {0, 0, 52},
    {0, 0, 55},
    {0, 0, 55},
    {0, 0, 57},
    {0, 0, 57},
    {0, 0, 60},
    {0, 0, 60},
    {0, 0, 63},
    {0, 0, 63},
    {0, 0, 65},
    {0, 0, 68},
    {0, 0, 68},
    {0, 0, 70},
    {0, 0, 70},
    {0, 0, 73},
    {0, 0, 76},
    {0, 0, 78},
    {0, 0, 78},
    {0, 0, 81},
    {0, 0, 81},
    {0, 0, 84},
    {0, 0, 86},
    {0, 0, 89},
    {0, 0, 91},
    {0, 0, 94},
    {0, 0, 97},
    {0, 0, 97},
    {0, 0, 99},
    {0, 0, 102},
    {0, 0, 105},
    {0, 0, 107},
    {0, 0, 110},
    {0, 0, 112},
    {0, 0, 112},
    {0, 0, 115},
    {0, 0, 118},
    {0, 0, 123},
    {0, 0, 126},
    {0, 0, 128},
    {0, 0, 131},
    {0, 0, 131},
    {0, 0, 133},
    {0, 0, 136},
    {0, 0, 141},
    {0, 0, 144},
    {0, 0, 147},
    {0, 0, 147},
    {0, 0, 152},
    {0, 0, 154},
    {0, 0, 160},
    {0, 0, 162},
    {0, 0, 165},
    {0, 0, 170},
    {0, 0, 170},
    {0, 0, 173},
    {0, 0, 178},
    {0, 0, 183},
    {0, 0, 186},
    {0, 0, 189},
    {0, 0, 194},
    {0, 0, 194},
    {0, 0, 196},
    {0, 0, 202},
    {0, 0, 207},
    {0, 0, 210},
    {0, 0, 212},
    {0, 0, 212},
    {0, 0, 217},
    {0, 0, 223},
    {0, 0, 225},
    {0, 0, 231},
    {0, 0, 236},
    {0, 0, 238},
    {0, 0, 238},
    {0, 0, 244},
    {0, 0, 249},
    {0, 0, 254},
    {0, 2, 255},
    {0, 7, 255},
    {0, 12, 255},
    {0, 12, 255},
    {0, 15, 255},
    {0, 20, 255},
    {0, 26, 255},
    {0, 28, 255},
    {0, 33, 255},
    {0, 39, 255},
    {0, 39, 255},
    {0, 44, 255},
    {0, 47, 255},
    {0, 52, 255},
    {0, 57, 255},
    {0, 60, 255},
    {0, 60, 255},
    {0, 68, 255},
    {0, 70, 255},
    {0, 75, 255},
    {0, 81, 255},
    {0, 83, 255},
    {0, 89, 255},
    {0, 89, 255},
    {0, 94, 255},
    {0, 99, 255},
    {0, 104, 255},
    {0, 107, 255},
    {0, 112, 255},
    {0, 117, 255},
    {0, 117, 255},
    {0, 120, 255},
    {0, 125, 255},
    {0, 131, 255},
    {0, 136, 255},
    {0, 138, 255},
    {0, 138, 255},
    {0, 144, 255},
    {0, 149, 255},
    {0, 151, 255},
    {0, 159, 255},
    {0, 162, 255},
    {0, 165, 255},
    {0, 165, 255},
    {0, 172, 255},
    {0, 175, 255},
    {0, 178, 255},
    {0, 186, 255},
    {0, 188, 255},
    {0, 193, 255},
    {0, 193, 255},
    {0, 199, 255},
    {0, 201, 255},
    {0, 207, 255},
    {0, 209, 255},
    {0, 214, 255},
    {0, 220, 255},
    {0, 220, 255},
    {0, 222, 255},
    {0, 228, 255},
    {0, 233, 255},
    {0, 235, 255},
    {0, 238, 255},
    {0, 238, 255},
    {0, 243, 255},
    {0, 246, 255},
    {0, 251, 255},
    {2, 255, 255},
    {6, 255, 255},
    {10, 255, 255},
    {10, 255, 255},
    {18, 255, 255},
    {22, 255, 255},
    {26, 255, 255},
    {34, 255, 255},
    {38, 255, 255},
    {46, 255, 255},
    {46, 255, 255},
    {50, 255, 255},
    {54, 255, 255},
    {62, 255, 255},
    {65, 255, 255},
    {69, 255, 255},
    {69, 255, 255},
    {77, 255, 255},
    {81, 255, 255},
    {85, 255, 255},
    {89, 255, 255},
    {93, 255, 255},
    {97, 255, 255},
    {97, 255, 255},
    {105, 255, 255},
    {109, 255, 255},
    {113, 255, 255},
    {117, 255, 255},
    {121, 255, 255},
    {125, 255, 255},
    {125, 255, 255},
    {128, 255, 255},
    {132, 255, 255},
    {136, 255, 255},
    {140, 255, 255},
    {140, 255, 255},
    {148, 255, 255},
    {148, 255, 255},
    {148, 255, 255},
    {152, 255, 255},
    {156, 255, 255},
    {160, 255, 255},
    {160, 255, 255},
    {160, 255, 255},
    {164, 255, 255},
    {168, 255, 255},
    {172, 255, 255},
    {172, 255, 255},
    {176, 255, 255},
    {180, 255, 255},
    {180, 255, 255},
    {180, 255, 255},
    {184, 255, 255},
    {184, 255, 255},
    {188, 255, 255},
    {188, 255, 255},
    {191, 255, 255},
    {191, 255, 255},
    {195, 255, 255},
    {195, 255, 255},
    {199, 255, 255},
    {199, 255, 255},
    {199, 255, 255},
    {199, 255, 255},
    {203, 255, 255},
    {203, 255, 255},
    {203, 255, 255},
    {207, 255, 255},
    {207, 255, 255},
    {207, 255, 255},
    {207, 255, 255},
    {211, 255, 255},
    {211, 255, 255},
    {211, 255, 255},
    {211, 255, 255},
    {211, 255, 255},
    {211, 255, 255},
    {211, 255, 255},
    {215, 255, 255},
    {215, 255, 255},
    {215, 255, 255},
    {215, 255, 255},
    {215, 255, 255},
    {0, 0, 0}
};

__forceinline__ __device__ void color(BYTE* pixel, int iteration, const int log2limit)
{
    if (log2limit < 8) iteration = (iteration << (8 - log2limit))+1;
    else if (log2limit > 8) iteration = iteration >> (log2limit - 8);
    pixel[0] = colormap[iteration][0];
    pixel[1] = colormap[iteration][1];
    pixel[2] = colormap[iteration][2];
}