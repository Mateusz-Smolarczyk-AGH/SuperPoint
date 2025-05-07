import matplotlib.pyplot as plt
import csv
import os

# Wpisz tutaj nazwy plików CSV do porównania
csv_files = [
    "80x60.csv",
    "160x120.csv",
    "320x240.csv",
    "640x480.csv",
    "1280x960.csv"
]

def load_power_values(file_path):
    values = []
    count = 0
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            count+=1
            if row:
                try:
                    value = float(row[0])
                    # if value >= 5000:
                    values.append(value - 4240)
                except ValueError:
                    continue
            if count > 50:
                break
    return values

plt.figure(figsize=(10, 6))

for file in csv_files:
    if os.path.exists(file):
        power_values = load_power_values(file)
        if power_values:
            avg_power = sum(power_values) / len(power_values)
            print(f"{file} — średnia moc: {avg_power:.2f} mW ({len(power_values)})")
            plt.plot(power_values, label=f"{file} (avg={avg_power:.1f} mW)")
        else:
            print(f"{file} — brak próbek ≥ 5000 mW")
    else:
        print(f"Plik nie istnieje: {file}")

plt.title("Porównanie mocy (mW)")
plt.xlabel("Numer próbki")
plt.ylabel("Moc (mW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()