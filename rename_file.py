import os

path = f"{os.getcwd()}\\novel_text"

for file in os.listdir(path):
    new_file_name = file.lower() \
        .replace(" ", "_") .replace(".", "_") \
        .replace("-", "").replace(";", "").replace(",", "") \
        .replace("'", "").replace("__", "_").replace("_txt", ".txt")
    os.rename(f"{path}\\{file}", f"{path}\\{new_file_name}")
