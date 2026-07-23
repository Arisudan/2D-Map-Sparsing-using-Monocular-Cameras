import os
import sys

def main():
    print("=== MASt3R-SLAM Inspector ===")
    
    # Locate MASt3R-SLAM folder
    paths_to_check = [
        "./MASt3R-SLAM",
        "../MASt3R-SLAM",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../MASt3R-SLAM")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../MASt3R-SLAM")),
        "~/Desktop/SLAM/MASt3R-SLAM"
    ]
    
    mast3r_dir = None
    for p in paths_to_check:
        expanded = os.path.expanduser(p)
        if os.path.isdir(expanded):
            mast3r_dir = expanded
            break
            
    if not mast3r_dir:
        print("Error: Could not locate MASt3R-SLAM directory.")
        return
        
    print(f"Located MASt3R-SLAM directory at: {mast3r_dir}")
    
    # List directories
    print("\n=== Directory Structure ===")
    for root, dirs, files in os.walk(os.path.join(mast3r_dir, "mast3r_slam")):
        rel_path = os.path.relpath(root, mast3r_dir)
        print(f"[{rel_path}]")
        for f in files:
            if f.endswith('.py'):
                print(f"  - {f}")
                
    # Read main.py
    main_py_path = os.path.join(mast3r_dir, "main.py")
    if os.path.isfile(main_py_path):
        print("\n=== main.py Contents ===")
        with open(main_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
            
            # Also write it to a report file for easy review
            report_path = "mast3r_inspection_report.txt"
            with open(report_path, 'w', encoding='utf-8') as rf:
                rf.write(content)
            print(f"\nWritten main.py contents to {os.path.abspath(report_path)}")
    else:
        print(f"\nError: main.py not found at {main_py_path}")

if __name__ == "__main__":
    main()
