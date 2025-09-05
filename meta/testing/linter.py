from sys import argv

def lint_file(filename):
    lines = []
    three_dashes_count = 0
    with open(filename) as file:
        lines = [line.rstrip().replace("'", "") for line in file]

        for l in lines:
            if l == "---":
                three_dashes_count += 1
            if three_dashes_count < 2:
                continue
            if "$$" in l and l.strip() != "$$":
                print(filename + ": Error: $$ used in file not on its own line on line " + str(lines.index(l)))
            if "```" in l and l != "```" and l != "```{r}":
                print(filename + ": Error: ``` used in file not on its own line on line " + str(lines.index(l)))
            if "---" in l and "----" not in l and l != "---":
                print(filename + ": Error: ``` used in file not on its own line on line " + str(lines.index(l))
)
            """
            if len(l) > 0 and len(lines[lines.index(l)-1]) > 0 and l[0] in ["*", "-"] and l != "---" and lines[lines.index(l)-1][0] != l[0] and lines[lines.index(l)-1] != '':
                print(filename + ": Error: no blank line before list starting on line " + str(lines.index(l)))
            """
    if three_dashes_count != 2:
        print(filename + ": Error: --- occurs too few or too many times. It must occur exactly twice at the beginning and end of the Quarto header.")

    joined_lines = "".join(lines)
    if joined_lines.count("<div>") > joined_lines.count("</div>"):
        print(filename + ": Error: more opening than closing diff tags.")
    if joined_lines.count("<div>") < joined_lines.count("</div>"):
        print(filename + ": Error: more closing than opening diff tags.")

for i in range(1, len(argv)):
    lint_file(argv[i])
