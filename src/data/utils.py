import os

def select_subfolder(folder: str, header_str: str = "\nChoose a subfolder:", allow_create_new: bool = False) -> str | None:
    """
    Lists subfolders and lets user select one by number.
    Optionally allows creating a new one with 'N', or canceling with 'C'.

    Args:
        folder: Path to the parent folder.
        header_str: Header to display before the subfolder list.
        allow_create_new: If True, shows the 'N' option to create a new folder.

    Returns:
        Full path to the selected (or newly created) subfolder, or None if canceled.
    """
    while True:
        entries = os.listdir(folder)
        subfolders = sorted([f for f in entries if os.path.isdir(os.path.join(folder, f))])

        print(header_str)
        for i, sub in enumerate(subfolders, start=1):
            print(f"{i}: {sub}")
        if allow_create_new:
            print("N: Create a new folder")
        print("C: Cancel")

        selection = input("Enter selection: ").strip()

        if selection.lower() == 'c':
            print("Selection canceled.")
            return None

        elif allow_create_new and selection.lower() == 'n':
            new_name = input("Enter name for new folder: ").strip()
            new_path = os.path.join(folder, new_name)
            if os.path.exists(new_path):
                print("A file or folder with that name already exists. Try again.")
                continue
            os.makedirs(new_path)
            return new_path

        try:
            idx = int(selection)
            if 1 <= idx <= len(subfolders):
                return os.path.join(folder, subfolders[idx - 1])
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Invalid input.")
