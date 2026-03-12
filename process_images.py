import os

CONTER_FILE = "ref_counter.txt"
IMAGE_PATH = "data/test_sar/s1_7/"
PREFIX = "scene"

def read_counter():
    if not os.path.exists(CONTER_FILE):
        with open(CONTER_FILE, "w") as f:
            f.write("1")
        return 1
    
    with open(CONTER_FILE, "r"):
        return int(f.read().strip())
    
def write_counter(value):
    with open(CONTER_FILE, "w") as f:
        f.write(str(value))

def rename_images():
    counter = read_counter()

    for filename in sorted(os.listdir(IMAGE_PATH)):
        old_path = os.path.join(IMAGE_PATH, filename)

        if not os.path.isfile(old_path):
            continue

        ext = os.path.splitext(filename)[1]
        new_name = f"{PREFIX}{counter}{ext}"
        new_path = os.path.join(IMAGE_PATH, new_name)

        if os.path.exists(new_path):
            print("skipping(exists): {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

        counter += 1
    
    write_counter(counter)

if __name__ == "__main__":
    rename_images()