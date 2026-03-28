import argparse
import getpass

import bcrypt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a bcrypt hash for an admin password."
    )
    parser.add_argument(
        "-p",
        "--password",
        help="Password to hash. If omitted, a secure prompt is shown.",
    )
    args = parser.parse_args()

    password = args.password
    if password is None:
        password = getpass.getpass("Enter password: ")

    if password is None or password == "":
        raise SystemExit("Error: password cannot be empty.")

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(12)).decode("utf-8")
    print(hashed)


if __name__ == "__main__":
    main()
