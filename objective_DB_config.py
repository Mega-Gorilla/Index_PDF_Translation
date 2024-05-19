import os

# Black BlazeオブジェクトDB設定
BLACK_BLAZE_CONFIG = {
    'public_key_id': os.environ["blackblaze_public_id"],
    'public_key' : os.environ["blackblaze_public_key"],
    'private_key_id': os.environ["blackblaze_private_id"],
    'private_key' : os.environ["blackblaze_private_key"],
    'public_bucket' : 'pdf-public',
    'private_bucket' : 'pdf-private'
}