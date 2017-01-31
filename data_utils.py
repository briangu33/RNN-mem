import numpy as np

def char_counter(n, m=9):
    # Generate lines of the form "cccc4" "cccccc6" etc.
    f = open("data/char_count_input" + str(m) + ".txt", 'w')
    for i in range(n):
        k = np.random.randint(1, m+1)
        for j in range(k):
            f.write("@")
        f.write(to_digit(k))
        f.write("\n")
    f.close()

def to_digit(m):
    if m < 10:
        return str(m)
    return str(unichr(m+55))

def char_printer(n, m=9):
    # Generate lines of the form "5ccccc" "3ccc" etc.
    f = open("data/char_printer_input" + str(m) + ".txt", 'w')
    for i in range(n):
        k = np.random.randint(1, m+1)
        f.write(to_digit(k))
        for j in range(k):
            f.write("c")
        f.write("\n")
    f.close()
char_printer(100000)

def correct_print_line(line):
    k = 0
    if ord(line[0]) < 60:
        k = int(line[0])
    else:
        k = ord(line[0])-55
    if len(line) == k+1:
        return True
    return False

def acc_print(sample):
    lines = sample.split("\n")
    lines = lines[2:len(lines)-1]
    correct = 0.
    total = len(lines)
    for line in lines:
        print line
        if correct_print_line(line):
            correct += 1.
    return correct / total

def correct_count_line(line):
    k = 0
    if ord(line[len(line) - 1]) < 60:
        k = int(line[len(line) - 1])
    else:
        k = ord(line[len(line) - 1])-55
    if len(line) == k+1:
        return True
    return False

def acc_count(sample):
    lines = sample.split("\n")
    lines = lines[2:len(lines)-1]
    correct = 0.
    total = len(lines)
    for line in lines:
        if correct_count_line(line):
            correct += 1.
    return correct / total