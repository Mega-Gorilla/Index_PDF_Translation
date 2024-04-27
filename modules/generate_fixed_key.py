import secrets
import base64

# 32バイトの乱数を生成
random_bytes = secrets.token_bytes(32)

# 乱数をBase64エンコードして固定キーを生成
fixed_key = base64.urlsafe_b64encode(random_bytes).decode()

print("Generated Fixed Key:", fixed_key)