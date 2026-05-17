import os
import zipfile
import shutil
import gdown

BASE_DIR = "/app"

DATABASE_DIR = os.path.join(BASE_DIR, "Database")
DOCUMENTS_DIR = os.path.join(BASE_DIR, "Documents")

TEMP_DIR = os.path.join(BASE_DIR, "temp_downloads")

os.makedirs(TEMP_DIR, exist_ok=True)

# =========================================================
# GOOGLE DRIVE FILE IDS
# =========================================================

VECTOR_DB_FILE_ID = "1ky5upCEaUSZgAQTGsMS365KmLELmoxLW"
DOCUMENTS_FILE_ID = "1hmzNBNIaNGlRXkJvwySyBGa4dloE18fs"
VECTOR_DB_ZIP = os.path.join(TEMP_DIR, "Vector_database.zip")
DOCUMENTS_ZIP = os.path.join(TEMP_DIR, "Documents.zip")
def download_file(file_id: str, output_path: str):

    print(f"Downloading: {output_path}")

    gdown.download(id=file_id, output_path=output_path, quiet=False)


def extract_zip(zip_path: str, target_dir: str):
    print(f"Extracting {zip_path} -> {target_dir}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)


# =========================================================
# DOWNLOAD VECTOR DB
# =========================================================

db_exists = (
    os.path.exists(DATABASE_DIR)
    and len(os.listdir(DATABASE_DIR)) > 0
)

if not db_exists:
    print("Vector DB missing. Downloading...")

    os.makedirs(DATABASE_DIR, exist_ok=True)

    download_file(VECTOR_DB_FILE_ID, VECTOR_DB_ZIP)

    extract_zip(VECTOR_DB_ZIP, DATABASE_DIR)

else:
    print("Vector DB already exists. Skipping download.")


# =========================================================
# DOWNLOAD DOCUMENTS
# =========================================================

docs_exist = (
    os.path.exists(DOCUMENTS_DIR)
    and len(os.listdir(DOCUMENTS_DIR)) > 0
)

if not docs_exist:
    print("Documents missing. Downloading...")

    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    download_file(DOCUMENTS_FILE_ID, DOCUMENTS_ZIP)

    extract_zip(DOCUMENTS_ZIP, DOCUMENTS_DIR)

else:
    print("Documents already exist. Skipping download.")


# =========================================================
# CLEANUP
# =========================================================

if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

print("Asset setup complete.")

