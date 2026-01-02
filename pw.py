import bcrypt

print(bcrypt.hashpw(b'', bcrypt.gensalt(12)).decode())