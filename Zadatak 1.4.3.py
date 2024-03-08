list = []

while True:
    broj = input("Broj: ")
    if broj == "Done": break
    list.append(float(broj))
print(len(list))
print(sum(list) / len(list))
print(min(list))
print(max(list))
print(sorted(list))