import os

def get_valid_labels(directory):
    """
    A generator that yields valid directory names, 
    skipping hidden files (.DS_Store) and non-directories.
    """
    # We sort to ensure deterministic mapping (ID 0 is always the same person)
    for entry in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, entry)
        if not entry.startswith('.') and os.path.isdir(full_path):
            yield entry


def get_valid_files(directory):
    """
    A generator that yields valid files names, 
    skipping hidden files (.DS_Store)
    """
    # We sort to ensure deterministic mapping (ID 0 is always the same person)
    for entry in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, entry)
        if not entry.startswith('.'):
            yield entry