#pragma once
#include "otherFunctions.h"

using namespace cv;

void naivMandelbrot(Mat &image, int log2limit, NUMBER bottomLeftX = -2, NUMBER bottomLeftY = -1.5, NUMBER sizeX = 3, NUMBER sizeY = 3)
{
    // variables:
    const unsigned int limit = (1 << log2limit);
    const NUMBER infRadius = 4;
    const double dx = sizeX / image.cols;
    const double dy = sizeY / image.rows;

    // loop trought pixels:
    for (int r = 0; r < image.rows; r++) {
        cv::Vec3b* ptr = image.ptr<cv::Vec3b>(r);
        for (int c = 0; c < image.cols; c++) {
            // starting values:
            NUMBER a(0.0), b(0.0), temp;
            const NUMBER a_c = bottomLeftX + c * dx;
            const NUMBER b_c = bottomLeftY + r * dy;
            unsigned int iteration = 0;

            // iterations:
            while ((++iteration < limit) && ((a * a + b * b) < infRadius))
            {
                temp = a * a - b * b + a_c;
                b = 2.0 * a * b + b_c;
                a = temp;
            }
            // coloring:
            if (log2limit < 8) iteration = iteration << (8 - log2limit);
            else if (log2limit > 8) iteration = iteration >> (log2limit - 8);
            ptr[c] = color(iteration);
        }
    }
}
