with open('./output/lexical/dev.txt', 'w', encoding='utf8') as out:
    for i in range(1, 21):
        for line in open(f'./output/lexical/lookup-cleanup-w3/dev-{i}'):
            fname = line.split()[0]
            score = 0
            for s in line.split()[1:]:
                score = score / 100 + int(s)
            out.write(f'{fname} {score}\n')