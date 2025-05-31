import os

def choose_or_create_subfolder(folder: str) -> str | None:
    """
    Lists subfolders and lets user select one by number,
    create a new one with 'N', or cancel with 'C'.
    """
    while True:
        entries = os.listdir(folder)
        subfolders = sorted([f for f in entries if os.path.isdir(os.path.join(folder, f))])

        print("\nChoose a subfolder:")
        for i, sub in enumerate(subfolders, start=1):
            print(f"{i}: {sub}")
        print("N: Create a new folder")
        print("C: Cancel")

        selection = input("Enter a number, 'N' for new folder, or 'C' to cancel: ").strip()

        if selection.lower() == 'c':
            print("Selection canceled.")
            return None

        elif selection.lower() == 'n':
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
            print("Invalid input. Please enter a number, 'N', or 'C'.")
