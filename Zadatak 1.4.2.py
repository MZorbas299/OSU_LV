try:
    x = float(input("Ocjena: "))

    if x < 0.0 or x > 1.0:
        print("Ocjena izvan intervala")
    elif x >= 0.9:
        print("A")
    elif x >=0.8:
        print("B")
    elif x >=0.7:
        print("C")
    elif x >=0.6:
        print("D")
    else:
        print("F")

except ValueError:
    print("Error")
