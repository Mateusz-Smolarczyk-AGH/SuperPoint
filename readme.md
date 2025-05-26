# Realizacja odometrii wizyjnej z wykorzystaniem sieci SuperPoint oraz zasobów sprzętowych Kria KV260

Projekt realizuje zadanie **odometrii wizyjnej** przy użyciu wstępnie wytrenowanej sieci [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork), która umożliwia: detekcję punktów charakterystycznych oraz obliczanie ich deskryptorów. Następnie zostaje wykonane dopasowanie punktów między kolejnymi klatkami obrazu z kamery. Umożliwia to estymację **rotacji** i **translacji** kamery na podstawie homografii.

## Struktura repozytorium

Projekt podzielony jest na kilka modułów:

### 📁 `src/`
Zawiera plik `launch_new.py`, który uruchamia cały pipeline na komputerze PC i wyznacza trajektorię przebytą przez kamerę.

### 📁 `kria_evaluation/`
Zawiera skrypt odpowiedzialny za:

- **kwantyzację** sieci,
- **kompilację** modelu do uruchomienia na **DPU** platformy Kria KV260.

> Do uruchomienia tej części należy wykorzystać obraz Dockera z **Vitis AI**.

### 📁 `Kria_run/`
Zawiera skrypt do:

- uruchomienia skompilowanej sieci,
- ewaluacji modelu na sprzęcie docelowym (Kria KV260).

## Zbiory danych

Projekt był testowany na następujących bazach danych:

- [HPatches](https://github.com/hpatches/hpatches-dataset),
- [KITTI](http://www.cvlibs.net/datasets/kitti/),
- [TUM RGB-D](https://vision.in.tum.de/data/datasets/rgbd-dataset).

---

> 💡 Jeśli masz pytania lub sugestie dotyczące tego repozytorium – zapraszam do kontaktu lub zgłaszania issue.

<!--

### Requiments (RTX 4070)
* python 3.11
* pytorch 2.6 + cuda 12.4

## Aktualne czasy:
| Platforma | Pre processing (ms)  | Sieć (ms)  | Post processing (ms)  | Matching (ms) | All (ms)   | Matches |
| --------- | --------- | --------- | ---------- | ------------- | ---------- | ------------ |
| **CPU**   | 12.889746 | 61.231204 | 111.977515 | 5.636664      | 191.735129 | 166.690566   |
| **KV260** | 30.233111 | 20.903206 | 69.634430  | 46.054518     | 166.825265 | 151.094340   |
| **CPU (MW)**| 5.811986 | 20.831012 | 30.555329 | 3.770221 |60.968548 | 166.690566|
| **RTX 4070** | 5.515795 |1.593777 | 0.913936 |2.197850 | 10.221359 | 166.735849 |

-->

