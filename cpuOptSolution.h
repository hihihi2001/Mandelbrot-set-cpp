#pragma once
#include "otherFunctions.h"

using namespace cv;

void cpuOptMandelbrot(cv::Mat& image, int log2limit, double bottomLeftX = -2, double bottomLeftY = -1.5, double sizeX = 3, double sizeY = 3)
{
    // variables:
    const VECI limit = (1 << log2limit);
    const VECD infRadius = 4.0;
    const double dx = sizeX / image.cols;
    const double dy = sizeY / image.rows;

    VECD a;
    VECD b;
    VECD temp;
    VECD a_c;
    VECD b_c;
    VECI iteration;
    VECDB keepLooping;
    INT* extIteration = new INT[SIZE];

    // loop throught pixels:
    for (int r = 0; r < image.rows; r++) {
        cv::Vec3b* ptr = image.ptr<cv::Vec3b>(r);
        for (int c = 0; c < image.cols; c += SIZE) {
            // starting values:
            a = 0;
            b = 0;
            iteration = 0;
#ifndef USE_FLOAT
            a_c = VECD(
                bottomLeftX + c * dx, bottomLeftX + (c + 1) * dx,
                bottomLeftX + (c + 2) * dx, bottomLeftX + (c + 3) * dx
            );
#else
            a_c = VECD(
                bottomLeftX + c * dx, bottomLeftX + (c + 1) * dx,
                bottomLeftX + (c + 2) * dx, bottomLeftX + (c + 3) * dx,
                bottomLeftX + (c + 4) * dx, bottomLeftX + (c + 5) * dx,
                bottomLeftX + (c + 6) * dx, bottomLeftX + (c + 7) * dx
            );
#endif
            b_c = bottomLeftY + r * dy;
            keepLooping = true;

            // iterations:
            while (horizontal_or(keepLooping))
            {
                iteration = if_add(keepLooping, iteration, 1);
                keepLooping = VECDB(iteration < limit) && (a * a + b * b) < infRadius;
                temp = select(keepLooping, a * a - b * b + a_c, temp);
                b = select(keepLooping, 2.0 * a * b + b_c, b);
                a = temp;
            }
            // coloring:
            if (log2limit < 8) iteration = iteration << (8 - log2limit);
            else if (log2limit > 8) iteration = iteration >> (log2limit - 8);
            iteration.store(extIteration);
            for (int j = 0; j < SIZE; j++)
            {
                ptr[c + j] = color(extIteration[j]);
            }
        }
    }

    delete[] extIteration;
}

// slower version, runned out of registers, or idk
/*void cpuOptMandelbrot(cv::Mat& image, double bottomLeftX = -2, double bottomLeftY = -1.5, double sizeX = 3, double sizeY = 3)
{
    // set variables:
    const VECI limit = 255;
    const VECD infRadius = 4.0;
    double dx = sizeX / image.cols;
    double dy = sizeY / image.rows;

    // loop:
    VECD a1, a2, a3, a4;
    VECD b1, b2, b3, b4;
    VECD temp1, temp2, temp3, temp4;
    VECD a_c1, a_c2, a_c3, a_c4;
    VECD b_c1, b_c2, b_c3, b_c4;
    VECI iteration1, iteration2, iteration3, iteration4;
    VECDB keepLooping1, keepLooping2, keepLooping3, keepLooping4;
    INT* extIteration1 = new INT[SIZE];
    INT* extIteration2 = new INT[SIZE];
    INT* extIteration3 = new INT[SIZE];
    INT* extIteration4 = new INT[SIZE];

    for (int r = 0; r < image.rows; r++) {
        cv::Vec3b* ptr = image.ptr<cv::Vec3b>(r);
        int currentRow = r * image.cols;
        for (int c = 0; c < image.cols; c += 4 * SIZE) {
            a1 = a2 = a3 = a4 = 0;
            b1 = b2 = b3 = b4 = 0;
            iteration1 = iteration2 = iteration3 = iteration4 = 0;
            a_c1 = VECD(bottomLeftX + c * dx, bottomLeftX + (c + 1) * dx,
                bottomLeftX + (c + 2) * dx, bottomLeftX + (c + 3) * dx);
            a_c2 = VECD(bottomLeftX + (c + 4) * dx, bottomLeftX + (c + 5) * dx,
                bottomLeftX + (c + 6) * dx, bottomLeftX + (c + 7) * dx);
            a_c3 = VECD(bottomLeftX + (c + 8) * dx, bottomLeftX + (c + 9) * dx,
                bottomLeftX + (c + 10) * dx, bottomLeftX + (c + 11) * dx);
            a_c4 = VECD(bottomLeftX + (c + 12) * dx, bottomLeftX + (c + 13) * dx,
                bottomLeftX + (c + 14) * dx, bottomLeftX + (c + 15) * dx);
            b_c1 = b_c2 = b_c3 = b_c4 = bottomLeftY + r * dy;
            keepLooping1 = keepLooping2 = keepLooping3 = keepLooping4 = true;
            while (horizontal_or(keepLooping1) || horizontal_or(keepLooping2)
                || horizontal_or(keepLooping3) || horizontal_or(keepLooping4))
            {
                iteration1 = if_add(keepLooping1, iteration1, 1);
                iteration2 = if_add(keepLooping2, iteration2, 1);
                iteration3 = if_add(keepLooping3, iteration3, 1);
                iteration4 = if_add(keepLooping4, iteration4, 1);
                keepLooping1 = select(keepLooping1,
                    VECDB(iteration1 < limit) && (a1 * a1 + b1 * b1) < infRadius, keepLooping1);
                keepLooping2 = select(keepLooping2,
                    VECDB(iteration2 < limit) && (a2 * a2 + b2 * b2) < infRadius, keepLooping2);
                keepLooping3 = select(keepLooping3,
                    VECDB(iteration3 < limit) && (a3 * a3 + b3 * b3) < infRadius, keepLooping3);
                keepLooping4 = select(keepLooping4,
                    VECDB(iteration4 < limit) && (a4 * a4 + b4 * b4) < infRadius, keepLooping4);
                temp1 = select(keepLooping1, a1 * a1 - b1 * b1 + a_c1, temp1);
                temp2 = select(keepLooping2, a2 * a2 - b2 * b2 + a_c2, temp2);
                temp3 = select(keepLooping3, a3 * a3 - b3 * b3 + a_c3, temp3);
                temp4 = select(keepLooping4, a4 * a4 - b4 * b4 + a_c4, temp4);
                b1 = select(keepLooping1, 2.0 * a1 * b1 + b_c1, b1);
                b2 = select(keepLooping2, 2.0 * a2 * b2 + b_c2, b2);
                b3 = select(keepLooping3, 2.0 * a3 * b3 + b_c3, b3);
                b4 = select(keepLooping4, 2.0 * a4 * b4 + b_c4, b4);
                a1 = temp1;
                a2 = temp2;
                a3 = temp3;
                a4 = temp4;
            }
            iteration1.store(extIteration1);
            iteration2.store(extIteration2);
            iteration3.store(extIteration3);
            iteration4.store(extIteration4);
            for (int j = 0; j < SIZE; j++)
            {
                ptr[c + j] = cv::Vec3b(BYTE(extIteration1[j]), BYTE(extIteration1[j]), BYTE(extIteration1[j]));
                ptr[c + j + 4] = cv::Vec3b(BYTE(extIteration2[j]), BYTE(extIteration2[j]), BYTE(extIteration2[j]));
                ptr[c + j + 8] = cv::Vec3b(BYTE(extIteration3[j]), BYTE(extIteration3[j]), BYTE(extIteration3[j]));
                ptr[c + j + 12] = cv::Vec3b(BYTE(extIteration4[j]), BYTE(extIteration4[j]), BYTE(extIteration4[j]));
            }
        }
    }

    delete[] extIteration1;
    delete[] extIteration2;
    delete[] extIteration3;
    delete[] extIteration4;
}/**/