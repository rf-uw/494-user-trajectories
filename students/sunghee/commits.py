import subprocess

def get_commit(repo_path, date_str):
    """
    Returns the latest commit BEFORE or ON the given date.
    date_str format: 'YYYY-MM-DD'
    repo_path: path to the git repo
    """
    cmd = [
        "git", "-C", repo_path,
        "rev-list", "-1",
        f'--before="{date_str}"',
        "HEAD"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    commit = result.stdout.strip()

    if not commit:
        raise ValueError(f"No commit found before {date_str}")

    return commit
