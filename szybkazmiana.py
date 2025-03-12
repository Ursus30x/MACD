def reverse_lines_except_first(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if len(lines) > 1:
        lines = [lines[0]] + lines[1:][::-1]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

# Przykładowe użycie:
reverse_lines_except_first('data/BTC-Daily.csv', 'output.csv')
