"""Diagnostic: checks which HF account a token belongs to and whether it can
access the two gated pyannote models the app needs. Run it yourself — it
only prints info about your own account, the token itself is never sent
anywhere except to huggingface.co (the same place the app sends it).

Usage: .venv\\Scripts\\python.exe check_hf_access.py hf_xxxxxxxx
"""
import sys

from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

MODELS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/speaker-diarization-community-1",
]


def main():
    if len(sys.argv) != 2:
        print("Usage: check_hf_access.py <hf_token>")
        sys.exit(1)
    token = sys.argv[1]
    api = HfApi(token=token)

    try:
        who = api.whoami()
        print(f"Токен принадлежит аккаунту: {who['name']} ({who.get('email', '?')})")
        print(f"Тип токена: {who.get('auth', {}).get('accessToken', {}).get('role', '?')}")
    except Exception as exc:
        print(f"НЕ УДАЛОСЬ определить владельца токена — токен невалиден: {exc}")
        sys.exit(1)

    for model in MODELS:
        try:
            info = api.model_info(model)
            print(f"[OK]  {model} — доступ есть (gated={info.gated})")
        except GatedRepoError:
            print(f"[БЛОК] {model} — 403 Gated: этому аккаунту доступ НЕ выдан")
        except HfHubHTTPError as exc:
            print(f"[ОШИБКА] {model} — {exc}")


if __name__ == "__main__":
    main()
