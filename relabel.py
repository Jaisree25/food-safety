import os

# Define class mappings
map_to_0 = {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 25}
map_to_1 = {2, 21, 22, 27, 28, 29, 30, 31, 32, 33, 34}

# Root directory
directory = "D:/food-safety/Dataset"

# Walk through directory and subdirectories
for root, _, files in os.walk(directory):
    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(root, filename)

            with open(file_path, "r") as f:
                lines = f.readlines()

            if not lines:
                os.remove(file_path)
                continue

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts or len(parts) != 5:
                    continue
                cls = int(parts[0])
                if cls in map_to_0:
                    parts[0] = "0"
                elif cls in map_to_1:
                    parts[0] = "1"
                else:
                    continue
                new_lines.append(" ".join(parts) + "\n")

            if new_lines:
                with open(file_path, "w") as f:
                    f.writelines(new_lines)
            else:
                os.remove(file_path)
