#include <iostream>
#include "otherFunctions.h"
#include "naivSolution.h"
#include "naivMultiThread.h"
#include "cpuOptSolution.h"
#include "cpuOptMultiThread.h"

using std::cout;
using namespace std;
using namespace cv;

int main()
{
	system("title Mandelbrot set by Kozak Aron");
	// variables:
	Measure Stopper = Measure();
	Mat image = Mat::zeros(1024, 2048, CV_8UC3);
	char key;
	NUMBER bottomLeftX = -3;
	NUMBER bottomLeftY = -1;
	NUMBER sizeX = 4;
	NUMBER sizeY = 2;
	int log2limit = 8;

	// start rendering frames:
	cout << "OpenCV version is " << CV_VERSION << endl;
	do
	{
		Stopper.start();
	// draw mandelbrot: (uncomment exactly one)

		//naivMandelbrot(image, log2limit, bottomLeftX, bottomLeftY, sizeX, sizeY);	// 430 ms
		//naivMultThreadMandelbort(image, log2limit, bottomLeftX, bottomLeftY, sizeX, sizeY);	// 200 ms
		//cpuOptMandelbrot(image, log2limit, bottomLeftX, bottomLeftY, sizeX, sizeY);	// 320 ms
		cpuOptMultThreadMandelbrot(image, log2limit, bottomLeftX, bottomLeftY, sizeX, sizeY);	// 160 ms

	// print time, show image:
		long long int time = Stopper.stop(MS);
		cout << "rendertime: " << time << " ms" << endl;
		imshow("Mandelbrot set", image);
	// key input:
		key = (char)waitKey(0);
		switch (key)
		{
		case 'w':
			bottomLeftY -= sizeY / 5;
			break;
		case 'a':
			bottomLeftX -= sizeX / 5;
			break;
		case 's':
			bottomLeftY += sizeY / 5;
			break;
		case 'd':
			bottomLeftX += sizeX / 5;
			break;
		case 'q':
			bottomLeftX -= sizeX / 2;
			bottomLeftY -= sizeY / 2;
			sizeX *= 2;
			sizeY *= 2;
			cout << "width: " << sizeX << endl;
			break;
		case 'e':
			sizeX /= 2;
			sizeY /= 2;
			bottomLeftX += sizeX / 2;
			bottomLeftY += sizeY / 2;
			cout << "width: " << sizeX << endl;
			break;
		case 'r':
			bottomLeftX = -3;
			bottomLeftY = -1;
			sizeX = 4;
			sizeY = 2;
			log2limit = 8;
			break;
		case '1':
			if (log2limit > 5)log2limit--;
			cout << "iteration limit: " << (1 << log2limit) << endl;
			break;
		case '2':
			if (log2limit < 12)log2limit++;
			cout << "iteration limit: " << (1 << log2limit) << endl;
			break;
		}
	} while (key != (char)27);	// ESC
	return 0;
}

/*
avarage run times for default settings [milliseconds]

					+-----------+-----------+
					|   float	|	double  |
--------------------+-----------+-----------+
naiv				|	562		|	432		|	??
--------------------+-----------+-----------+
naiv multithread	|	182		|	143		|	??
--------------------+-----------+-----------+
cpu optimalized		|	142		|	271		|
--------------------+-----------+-----------+
cpu multi thread	|	59		|	129		|
--------------------+-----------+-----------+
*/

// x_0 = a_0 + i*b_0 = 0 + i0
// x_n = a_n + i*b_n
// x_(n+1) = x_n ^ 2 + c
// c = a_c + i*b_c
// x_(n+1) = (a_n^2 - b_n^2 + a_c) + i*(2*a_n*b_n + b_c)
