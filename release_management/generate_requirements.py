import importlib.metadata


def generate_requirements_txt(filename="requirements_master_full.txt"):
    with open(filename, "w") as f:
        for dist in importlib.metadata.distributions():
            f.write(f"{dist.metadata['Name']}=={dist.version}\n")


if __name__ == "__main__":
    generate_requirements_txt()
    print("Requirements saved to requirements_master_full.txt")
