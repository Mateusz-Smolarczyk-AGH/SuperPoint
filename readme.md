W celu dokonania pomiaru czasu uruchomić plik python_network/launch.py

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



