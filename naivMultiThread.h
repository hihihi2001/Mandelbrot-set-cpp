#pragma once
#include "otherFunctions.h"

using namespace cv;

void naivMultThreadMandelbort(Mat &image, int log2limit, NUMBER bottomLeftX = -2, NUMBER bottomLeftY = -1.5, NUMBER sizeX = 3, NUMBER sizeY = 3)
{
    // variables:
    const auto threadCount = std::thread::hardware_concurrency();
    const NUMBER infRadius = 4;
    const NUMBER dx = sizeX / image.cols;
    const NUMBER dy = sizeY / image.rows;
    int currentRow = image.rows-1;

    // lambda function to run on each thread:
    auto f = [&currentRow](int log2limit, NUMBER infRadius, Mat &image, NUMBER bottomLeftX, NUMBER bottomLeftY, NUMBER dx, NUMBER dy)
    {
        const unsigned int limit = (1 << log2limit);
        int r = currentRow--;
        // start rendering a new row, if there is any left:
        while (r > -1) 
        {
            // loop trought pixels:
            cv::Vec3b* ptr = image.ptr<cv::Vec3b>(r);
            for (int c = 0; c < image.cols; c++) 
            {
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
            // next row:
            r = currentRow--;
        }
    };

    // run on each thread:
    vector<thread> threads(threadCount);
    for (int t = 0; t < threadCount; t++)
        threads[t] = thread(f, log2limit, infRadius, std::ref(image), bottomLeftX, bottomLeftY, dx, dy);
    for (int t = 0; t < threadCount; t++)
        threads[t].join();
}