seen = set()
with open("../data/peter_pan_labelled_novel_dataset.csv", "r") as file:
    for line in file:
        line_lower = line.lower()
        line_lower = line.split("\",")[0]
        if line_lower in seen:
            print(line)
        else:
            seen.add(line_lower)

# with open("data/peter_pan_labelled_novel_main_dataset.csv", "w+") as file:
#     for data in seen:
#         file.write(data)
