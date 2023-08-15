#Parse the output into CSV format

filename = "1010Human.txt"



f = open(filename, 'r')
o = open("parsedOutput.txt", 'w')

lines = [line.strip() for line in f.readlines()]

for i in range(0, len(lines), 6):
    payload = (lines[i+j] for j in range(6))
    for thing in payload:
        o.write(thing.split(":")[1])
        o.write(",")
    o.write("\n")

o.close()