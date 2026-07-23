import os
import sys

def get_dir_size(path):
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    except Exception:
        pass
    return total

def main():
    root = "C:\\Users\\hp"
    print(f"Scanning subdirectories under '{root}' (showing folders > 500MB)...")
    
    results = []
    try:
        for entry in os.scandir(root):
            if entry.is_dir(follow_symlinks=False):
                # Skip system junctions or appdata initially to make it faster, or scan everything
                size = get_dir_size(entry.path)
                size_gb = size / (1024**3)
                if size_gb > 0.5:
                    results.append((entry.path, size_gb))
                    print(f"Found: {entry.name} -> {size_gb:.2f} GB")
    except Exception as e:
        print(f"Error scanning root: {e}")
        
    print("\n--- SCAN SUMMARY (SORTED BY SIZE) ---")
    results.sort(key=lambda x: x[1], reverse=True)
    for path, size_gb in results:
        print(f"{path}: {size_gb:.2f} GB")

if __name__ == "__main__":
    main()
