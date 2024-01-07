import os
def extract_lines(input, output, lines = 15000):
    with open(input, 'r') as source_file:
        lines = source_file.readlines()[:lines]
    with open(output, 'w') as target_file:
        target_file.writelines(lines)
        
extract_lines(r'food-101\meta\train_og.txt', r'food-101\meta\train.txt')