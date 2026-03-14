import os

def get_project_stats(root_dir, ignore_dirs):
    stats = {
        'total_files': 0,
        'total_folders': 0,
        'total_lines': 0,
        'extension_counts': {}
    }

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        stats['total_folders'] += len(dirs)
        stats['total_files'] += len(files)

        for file in files:
            ext = os.path.splitext(file)[1].lower() or 'no extension'
            stats['extension_counts'][ext] = stats['extension_counts'].get(ext, 0) + 1

            file_path = os.path.join(root, file)
            if ext in {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.css', '.js', '.html'}:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        stats['total_lines'] += sum(1 for _ in f)
                except Exception:
                    pass
    return stats

def generate_tree(root_dir, ignore_dirs, prefix=""):
    """
    Generates a tree representation of the directory structure.
    """
    tree_lines = []

    # Get items in current directory, filtered by ignore_dirs
    try:
        items = sorted([item for item in os.listdir(root_dir) if item not in ignore_dirs])
    except PermissionError:
        return []

    for i, item in enumerate(items):
        item_path = os.path.join(root_dir, item)
        is_last = (i == len(items) - 1)

        connector = "└── " if is_last else "├── "
        tree_lines.append(f"{prefix}{connector}{item}")

        if os.path.isdir(item_path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            tree_lines.extend(generate_tree(item_path, ignore_dirs, new_prefix))

    return tree_lines

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    ignore_dirs = {'.git', '__pycache__', '.pytest_cache', '.cursor', '.gemini', 'venv', 'env', '.vscode', '.idea'}

    print(f"Analyzing project at: {project_root}\n")

    # Generate and print tree
    print("Project Structure:")
    print(os.path.basename(project_root) + "/")
    tree = generate_tree(project_root, ignore_dirs)
    for line in tree:
        print(line)

    # Get and print stats
    stats = get_project_stats(project_root, ignore_dirs)

    print("\n" + "-" * 30)
    print(f"Total Folders: {stats['total_folders']}")
    print(f"Total Files:   {stats['total_files']}")
    print(f"Total Lines:   {stats['total_lines']} (approx. code/text)")
    print("-" * 30)

    print("\nFiles by extension:")
    for ext, count in sorted(stats['extension_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f" {ext:15}: {count}")

if __name__ == "__main__":
    main()
