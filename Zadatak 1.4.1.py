def total_euro(radni_sati, placa):
    ukupno = radni_sati * placa
    return ukupno

radni_sati = int(input("Radni_sati: " ))
placa = int(input("Placa: "))

total = total_euro(radni_sati, placa)
print("Ukupno: ", total, "eura")