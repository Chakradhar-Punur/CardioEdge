import sys
import subprocess

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|preprocessing")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "train":
        result = subprocess.run([sys.executable, "train.py"])
    elif mode == "preprocessing":
        result = subprocess.run([sys.executable, "preprocessing.py"])
    else:
        print("Invalid mode")
