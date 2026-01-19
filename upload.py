from huggingface_hub import HfApi
import getpass

token = getpass.getpass('Enter your HF token: ')

api = HfApi()
print('Uploading attention_patterns.pkl...')

api.upload_file(
    path_or_fileobj='outputs/attention_patterns.pkl',
    path_in_repo='outputs/attention_patterns.pkl',
    repo_id='Faruna01/igala-mbert-interpretability',
    repo_type='space',
    token=token
)

print('Upload complete!')
