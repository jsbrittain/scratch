import tokenize


result = [[]]
chunk = 0
with open("Snakefile", "rb") as file:
    tokens = tokenize.tokenize(file.readline)
    for toknum, tokval, _, _, _ in tokens:
        if toknum == tokenize.NAME and tokval == "rule":
            chunk += 1
            result.append([])
        result[chunk].append((toknum, tokval))

for r in result:
    print('---')
    print(tokenize.untokenize(r))
