# Mandelbrot set C++
Ez a program kirajzolja a [Mandelbrot halmazt](https://en.wikipedia.org/wiki/Mandelbrot_set) különböző hardvarre optimalizált módszerekkel.
![image](https://user-images.githubusercontent.com/42745647/165781906-95ee7503-dd38-44bb-96b2-d117802596b8.png)
## Futtatás
### Fordítás
* Mellékeltem 3 exe filet, amik közvetlenül futtathatók. (cpu_slowest.exe, cpu_fastest.exe, cpu_fastest_float.exe) **TODO!!!!!!**
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

Computer specs:  
  Intel Core i5-4310M @ 2.70GHz  
  2 cores; 4 threads; 2.70-3.40 GHz  
  cache 128 KB, 512 KB, 3 MB;  
  AVX2 --> (256 bit vector registers)  
  
Megoldások:
* **Naiv megoldás:** Egyszálon fut, beépített (int, double) típusokat használ. Ez a leglasabb.
* **Multithread naiv megoldás:** Kihasználja az összes szálat (thread library), de ugyanúgy az alap típusokat (int, double) használja. Minden szál aktív, amíg van olyan sor, amit nem kezdett el renderelni egy másik szál. Nagyjából 3-szor gyorsabb a naiv megoldásnál (2 mag, 4 szál) float és double esetén is.
* **CPU-ra optimalizált megoldás:**  Kihasználja a Vector Class Library 256 bites vektor regisztereit (4 double vagy 8 float), azaz Single Intstuction Multiple Data (SIMD) elven gyorsítja a futást. Floattal kb 4-szer gyorsabb, doublelel pedig 2-szer.
* **CPU-ra optimalizált multithread megoldás:** A fentiekhez hasonlóan vektor regisztert használ, és többszálon fut. Float esetén 9,5-ször, double esetén 3,3-szor gyorsabb.
* **GPU-ra írt megoldás:** hamarsan

## Galéria
![image](https://user-images.githubusercontent.com/42745647/165774675-5cb76e5a-a502-4567-a780-0441fbef135c.png)
Az iterációs limit állításának hatása a képre.
![image](https://user-images.githubusercontent.com/42745647/165777869-7a24e55f-52f0-4261-b16f-a383c4fd9c7a.png)
A különböző colormapek.
![image](https://user-images.githubusercontent.com/42745647/165778969-5ef6dc73-a68a-4e7a-965a-a19b808f37e0.png)
![image](https://user-images.githubusercontent.com/42745647/165779859-4414b0d0-a69e-4095-b49f-a273f1719ac0.png)
A szám felbontási limitje (floatnál ~10^-5; doublenál ~10^-15)
