#pragma once
#include "otherFunctions.h"

using namespace cv;

void cpuOptMultThreadMandelbrot(cv::Mat& image, int log2limit, NUMBER bottomLeftX = -2, NUMBER bottomLeftY = -1.5, NUMBER sizeX = 3, NUMBER sizeY = 3)
{
    // variables:
    const auto threadCount = std::thread::hardware_concurrency();
    
    NUMBER dx = sizeX / image.cols;
    NUMBER dy = sizeY / image.rows;
    int currentRow = image.rows - 1;

    // lambda function to run on each thread:
    auto f = [&currentRow](Mat& image, int log2limit, NUMBER bottomLeftX, NUMBER bottomLeftY, NUMBER dx, NUMBER dy)
    {
        // variables:
        const VECI limit = (1 << log2limit);
        const VECD infRadius = 4.0;
        int r = currentRow--;

        VECD a;
        VECD b;
        VECD temp;
        VECD a_c;
        VECD b_c;
        VECD sqra, sqrb;
        VECI iteration;
        VECDB keepLooping;
        INT* extIteration = new INT[SIZE];

        // start rendering a new row, if there is any left:
        while (r > -1) 
        {
            // loop trought pixels:
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
                    sqra = a * a;
                    sqrb = b * b;
                    iteration = if_add(keepLooping, iteration, 1);
                    keepLooping = VECDB(iteration < limit) && (sqra + sqrb) < infRadius;
                    temp = select(keepLooping, sqra - sqrb + a_c, temp);
                    b = select(keepLooping, 2.0 * a * b + b_c, b);
                    a = temp;
                }
                // coloring:
                if (log2limit < 8) iteration = iteration << (8 - log2limit);
                else if (log2limit > 8) iteration = iteration >> (log2limit - 8);
                iteration.store(extIteration);
                ptr[c] = color(extIteration[0]);
                ptr[c + 1] = color(extIteration[1]);
                ptr[c + 2] = color(extIteration[2]);
                ptr[c + 3] = color(extIteration[3]);
#ifdef USE_FLOAT
                ptr[c + 4] = color(extIteration[4]);
                ptr[c + 5] = color(extIteration[5]);
                ptr[c + 6] = color(extIteration[6]);
                ptr[c + 7] = color(extIteration[7]);
#endif
            }
            r = currentRow--;
        }

        delete[] extIteration;
    };
    
    // run on each thread:
    vector<thread> threads(threadCount);
    for (int t = 0; t < threadCount; t++)
    {
        threads[t] = thread(f, std::ref(image), log2limit, bottomLeftX, bottomLeftY, dx, dy);
    }
    for (int t = 0; t < threadCount; t++)
    {
        threads[t].join();
    }
    
}