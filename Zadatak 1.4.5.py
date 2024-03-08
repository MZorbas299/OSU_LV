file = open("SMSSpamCollection.txt")

spamCounter = 0
spamCounterCount = 0
hamCounter = 0
hamCounterCount = 0

exclamationCounter = 0

for line in file:
    if line.startswith("spam"):
        spamCounter += 1
        spamCounterCount += len(line.split()[1::])
        if line.endswith("!"):
            exclamationCounter += 1
    if line.startswith("ham"):
        hamCounter += 1
        hamCounterCount += len(line.split()[1::])
    

print(float(spamCounterCount / spamCounter))
print(float(hamCounterCount / hamCounter))
print(exclamationCounter)
    