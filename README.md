# Mandelbrot set C++
Ez a program kirajzolja a [Mandelbrot halmazt](https://en.wikipedia.org/wiki/Mandelbrot_set) különböző hardvarre optimalizált módszerekkel.
## Futtatás
### Fordítás
* Mellékeltem 3 exe filet, amik közvetlenül futtathatók. (cpu_slowest.exe, cpu_fastest.exe, cpu_fastest_float.exe)
* Le lehet tölteni a teljes VS projektet (mandelbrot VS projekt.zip), ami kicsomagolás után szerkeszthető Visual Studioban, a megfelekő könyvtárak telepítése után
  * Le kell tölteni Agner Fog [Vector Class Library](https://github.com/vectorclass/version2)-ját. ([VCL manual](https://www.agner.org/optimize/vcl_manual.pdf))
  * Telepíteni kell az [OpenCV](https://learnopencv.com/code-opencv-in-visual-studio/)-t
  * Állítsd a fordítót x64-re, C++17-re, és Realeasre (Debug módban a VCL használhatatlanul lassú)
* A kódok más IDE-vel is fordíthatóak (mandelbrot window.cpp, otherFunctions.h, naivSolution.h, naivMultiThread.h, cpuOptSolution.h, cpuOptMultiThread.h)
### Beállítások
A kódban lehet állítani:
* felbontás: (*mandelbrot window.cpp* 18. sor) Mat image = Mat::zeros(1024, 2048, CV_8UC3); // Legyen a szélesség (2048) a 16 többszöröse
* kirajzolás módja: (*mandelbrot window.cpp* 33-36. sor) Pontosan az egyik kirajzoló függvény ne legyen kikommentezve
* float használata double helyett: (*otherFunctions.h* 15. sor) definiáld a USE_FLOAT konstanst, ha floatot szeretnél. (Így a maximális nagyítás 10^-15 helyett 10^-5; a VCL-es kirajzolás sokkal gyorsabb; összehasonlítható majd a GPU-val)
* colormap: (*otherFunctions.h*, 15-20. sor) definiáld a VIRIDIS, GIST_RAINBOW, HOT, PLASMA konstansok valamelyikét, vagy egyiket se, hogy grayscale képet kapj. A colormapeket a Python Matplotlib könyvtárából szedtem ki a *colormap stealer.ipynb* -vel
### Irányítás
A kép változtatható a billentyűzettel, amennyiben az OpenCV ablaka az aktív
* Mozgás: asdw
* Nagyítás: qe
* Iterációs limit állítása: 12
* Alaphelyzetbe állás: r
* Kilépés: ESC
* Renderelés: bármely billenytyű, kivéve az ESC
A konzolban megjeleník a nagyítás, iterációs limit, és a renderidő
## Optimalizálás
A különböző futási idők:
(Alapértelmezett kép, legalább 10 képkockából átlagolva, ezredmásodpercben)<br/>
| [millisec]        | float  | double |
|-------------------|--------|--------|
| naiv              | 562    | 432    |
| naiv multithread  | 182    | 143    |
| CPU optimalized   | 142    | 271    |
| CPU multi thread  | 59     | 129    |
| GPU               | coming | soon   |
Specs:
 Intel Core i5-4310M @ 2.70GHz  
 2 cores; 4 threads; 2.70-3.40 GHz  
 cache 128 KB, 512 KB, 3 MB;  
 AVX2 --> (256 bit vector registers)  



![image](https://user-images.githubusercontent.com/42745647/165688726-ef57f7ff-da7a-4a69-aa62-bc4c334fe46c.png)
