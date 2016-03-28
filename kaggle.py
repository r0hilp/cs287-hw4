spaces_idx = []
with open('test_spaces_count.txt', 'r') as f:
    for line in f:
        spaces_idx.append(int(line) - 1)

i = 0
total = 0
with open('data/test_chars.txt', 'r') as f:
    with open('test_count.txt', 'w') as f_out:
        f_out.write("ID,Count\n")
        for j,line in enumerate(f):
            if i < len(spaces_idx):
                idx = spaces_idx[i]
            chars = line.strip().split()

            line_out = ''
            count = 0
            for c in chars:
                if total == idx:
                    line_out = line_out + ' '
                    count += 1
                    i += 1
                    if i < len(spaces_idx):
                        idx = spaces_idx[i]
                line_out = line_out + c
                total += 1

            f_out.write(str(j+1) + ',' + str(count) + '\n')
            # raw_input(line_out)

