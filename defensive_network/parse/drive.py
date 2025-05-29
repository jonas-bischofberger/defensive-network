import os
import io
import time

import pandas as pd
import streamlit
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import streamlit as st
from googleapiclient.http import MediaIoBaseDownload


# If modifying scopes, delete token.json to re-authenticate
SCOPES = ['https://www.googleapis.com/auth/drive.file']
ROOT_FOLDER_ID = "1RUB3vrYA3GwQSCgKao9IAZHxr4TTKeq3"  # folder "data" in jb univie drive
CLIENT_ID = "228954729014-58299oavv1ekhiiejqbu0u0q03hbabh3.apps.googleusercontent.com"
CLIENT_SECRET = "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"

TOKEN_FPATH = os.path.join(os.path.dirname(__file__), "../../secrets/token.json")
CLIENT_SECRETS_FPATH = os.path.join(os.path.dirname(__file__), "../../secrets/client_secret_228954729014-58299oavv1ekhiiejqbu0u0q03hbabh3.apps.googleusercontent.com.json")
# CLIENT_SECRETS_FPATH = "secrets/client_secret_228954729014-58299oavv1ekhiiejqbu0u0q03hbabh3.apps.googleusercontent.com.json"


def _authenticate():
    creds = None
    # token.json stores user access/refresh tokens.
    if os.path.exists(TOKEN_FPATH):
        creds = Credentials.from_authorized_user_file(TOKEN_FPATH, SCOPES)
    if not creds or not creds.valid:
        # if creds and creds.expired and creds.refresh_token:
        #     x = Request()
        #     creds.refresh(x)
        # else:
            # flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FPATH, SCOPES)
        creds = flow.run_local_server(port=0)
        # Save credentials for future use
        with open(TOKEN_FPATH, 'w') as token:
            token.write(creds.to_json())
    return creds


# def get_or_create_folder(service, folder_name, parent_id=ROOT_FOLDER_ID):
#     """
#     Get the folder ID with name `folder_name` under `parent_id`.
#     If it doesn't exist, create it.
#
#     Returns:
#         folder_id (str)
#     """
#     query = (
#         f"mimeType='application/vnd.google-apps.folder' and "
#         f"name='{folder_name}' and '{parent_id}' in parents and trashed = false"
#     )
#     response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
#     files = response.get('files', [])
#
#     if files:
#         return files[0]['id']
#
#     # Folder not found, create it
#     file_metadata = {
#         'name': folder_name,
#         'mimeType': 'application/vnd.google-apps.folder',
#         'parents': [parent_id]
#     }
#     folder = service.files().create(body=file_metadata, fields='id').execute()
#     return folder.get('id')


# def get_folder_id_from_path_create(service, folder_path, root_folder_id=ROOT_FOLDER_ID):
#     """
#     Resolve and create folders along the path if they don't exist.
#
#     Args:
#         folder_path (str): e.g. 'raw/2025/April'
#         root_folder_id (str): Drive root by default
#
#     Returns:
#         folder_id (str): ID of the last folder in the path
#     """
#     folder_names = folder_path.strip('/').split('/')
#     parent_id = root_folder_id
#
#     for folder_name in folder_names:
#         parent_id = get_or_create_folder(service, folder_name, parent_id)
#
#     return parent_id


# def upload_df_to_drive_folder_path(df, filename, folder_path, service=None):
#     """
#     Upload DataFrame as parquet to Google Drive folder path, creating folders as needed.
#
#     Args:
#         df (pd.DataFrame)
#         filename (str): e.g. 'data.parquet'
#         folder_path (str): e.g. 'raw/2025/April'
#         service: Google Drive API service object. If None, it authenticates.
#     """
#     import io
#     import os
#
#     if service is None:
#         service = build('drive', 'v3', credentials=authenticate())
#
#     folder_id = get_folder_id_from_path_create(service, folder_path)
#
#     # Save DataFrame locally to a buffer
#     buffer = io.BytesIO()
#     df.to_parquet(buffer, engine='pyarrow', index=False)
#     buffer.seek(0)
#
#     file_metadata = {'name': filename, 'parents': [folder_id]}
#     media = MediaIoBaseUpload(buffer, mimetype='application/octet-stream')
#
#     file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
#     print(f"Uploaded file '{filename}' to folder '{folder_path}' with ID: {file.get('id')}")


def _get_or_create_drive_folder(path: str, root_folder_id: str, service) -> str:
    """
    Recursively finds or creates nested subfolders in Google Drive.

    Args:
        path (str): Relative path like 'level1/level2'.
        root_folder_id (str): The ID of the root folder to start from.
        service: Authenticated Google Drive service instance.

    Returns:
        str: ID of the final subfolder.
    """
    st.write("get_or_create_drive_folder")
    parent_id = root_folder_id
    for folder_name in path.strip("/").split("/"):
        query = (
            f"'{parent_id}' in parents and "
            f"name = '{folder_name}' and "
            f"mimeType = 'application/vnd.google-apps.folder' and "
            f"trashed = false"
        )
        results = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        files = results.get('files', [])

        if files:
            parent_id = files[0]['id']
        else:
            metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id]
            }
            new_folder = service.files().create(body=metadata, fields='id').execute()
            parent_id = new_folder['id']

    return parent_id


def upload_file_to_drive(local_path: str, relative_drive_path: str, root_folder_id: str = ROOT_FOLDER_ID, creds=None) -> str:
    """
    Uploads a file to Google Drive, creating folders as needed, and overwriting if it already exists.

    Args:
        local_path (str): Path to the local file to upload.
        relative_drive_path (str): Path including folders and filename, e.g., 'data/2025/05/21/file.parquet'.
        root_folder_id (str): The ID of the root folder in Drive to upload into.
        creds: (optional) Authenticated credentials.

    Returns:
        str: ID of the uploaded or updated file.
    """
    if creds is None:
        creds = _authenticate()

    service = build('drive', 'v3', credentials=creds)

    # Split relative path into folder path and filename
    path_parts = relative_drive_path.strip("/").split("/")
    filename = path_parts[-1]
    folder_path = "/".join(path_parts[:-1])

    # Create/get target folder
    target_folder_id = _get_or_create_drive_folder(folder_path, root_folder_id, service) if folder_path else root_folder_id

    # Check for existing file
    query = (
        f"name = '{filename}' and "
        f"'{target_folder_id}' in parents and "
        f"mimeType != 'application/vnd.google-apps.folder' and "
        f"trashed = false"
    )
    results = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
    files = results.get('files', [])

    media = MediaIoBaseUpload(io.FileIO(local_path, 'rb'), mimetype='application/octet-stream')


    if files:
        # Overwrite
        file_id = files[0]['id']
        updated_file = service.files().update(fileId=file_id, media_body=media).execute()
        return updated_file['id']
    else:
        # Upload new
        metadata = {'name': filename, 'parents': [target_folder_id]}
        uploaded_file = service.files().create(body=metadata, media_body=media, fields='id').execute()
        return uploaded_file['id']



def upload_parquet_to_drive(df: pd.DataFrame, fpath: str, folder_id: str = ROOT_FOLDER_ID):
    """
    Save DataFrame to a parquet file and upload it to Google Drive.

    Args:
        df (pd.DataFrame): DataFrame to save and upload.
        fpath (str): Name of the file on Drive.
        folder_id (str): Folder ID to upload to.
    """
    local_path = os.path.basename(fpath)
    df.to_parquet(local_path, engine='pyarrow', index=False)

    try:
        st.write(local_path)
        file_id = upload_file_to_drive(local_path, fpath, folder_id)
        streamlit.write(f"Uploaded file with ID: {file_id}")
    finally:
        for i in range(10):
            try:
                os.remove(local_path)
                return
            except PermissionError:
                streamlit.write(f"PermissionError: File {local_path} is still in use. Retrying...")
                time.sleep(1)

        raise RuntimeError(f"Failed to delete file {local_path} after multiple attempts.")


def upload_csv_to_drive(df: pd.DataFrame, fpath: str, folder_id: str = ROOT_FOLDER_ID):
    """
    Save DataFrame to a CSV file and upload it to Google Drive.

    Args:
        df (pd.DataFrame): DataFrame to save and upload.
        fpath (str): Path including subfolders and filename on Drive.
        folder_id (str): Folder ID to upload to.
    """
    local_path = os.path.basename(fpath)
    df.to_csv(local_path, index=False)

    try:
        st.write(local_path)
        file_id = upload_file_to_drive(local_path, fpath, folder_id)
        streamlit.write(f"Uploaded file with ID: {file_id}")
    finally:
        for i in range(10):
            try:
                os.remove(local_path)
                return
            except PermissionError:
                streamlit.write(f"PermissionError: File {local_path} is still in use. Retrying...")
                time.sleep(1)

        raise RuntimeError(f"Failed to delete file {local_path} after multiple attempts.")


def _get_folder_id_by_path(path: str, root_folder_id: str, service) -> str:
    """
    Recursively find (or return None) the folder ID in Google Drive for a given path starting at root_folder_id.
    """
    if not path:
        return root_folder_id

    folder_names = path.strip("/").split("/")
    current_folder_id = root_folder_id

    for folder_name in folder_names:
        query = (
            f"mimeType='application/vnd.google-apps.folder' and "
            f"'{current_folder_id}' in parents and "
            f"name='{folder_name}' and trashed = false"
        )
        st.write(f"Query: {query}")
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = response.get('files', [])
        if not files:
            return None  # folder does not exist
        current_folder_id = files[0]['id']
    return current_folder_id


def download_csv_from_drive(relative_path: str, root_folder_id: str = ROOT_FOLDER_ID):
    """
    Download CSV file from Google Drive by path and load into DataFrame.

    Args:
        relative_path (str): path like 'folder/subfolder/file.csv'
        root_folder_id (str): Drive root folder ID

    Returns:
        pd.DataFrame
    """
    creds = _authenticate()
    service = build('drive', 'v3', credentials=creds)

    path_parts = relative_path.strip("/").split("/")
    filename = path_parts[-1]
    folder_path = "/".join(path_parts[:-1])

    folder_id = _get_folder_id_by_path(folder_path, root_folder_id, service)
    if folder_id is None:
        raise FileNotFoundError(f"Folder path '{folder_path}' not found on Drive.")

    # Find file in folder
    query = (
        f"name = '{filename}' and "
        f"'{folder_id}' in parents and "
        f"mimeType != 'application/vnd.google-apps.folder' and "
        f"trashed = false"
    )
    response = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
    files = response.get('files', [])
    if not files:
        raise FileNotFoundError(f"File '{filename}' not found in folder '{folder_path}'.")

    file_id = files[0]['id']

    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()

    fh.seek(0)
    return pd.read_csv(fh)



def download_parquet_from_drive(relative_path: str, root_folder_id: str = ROOT_FOLDER_ID):
    """
    Download Parquet file from Google Drive by path and load into DataFrame.

    Args:
        relative_path (str): path like 'folder/subfolder/file.parquet'
        root_folder_id (str): Drive root folder ID

    Returns:
        pd.DataFrame
    """
    creds = _authenticate()
    service = build('drive', 'v3', credentials=creds)

    path_parts = relative_path.strip("/").split("/")
    filename = path_parts[-1]
    folder_path = "/".join(path_parts[:-1])

    folder_id = _get_folder_id_by_path(folder_path, root_folder_id, service)
    if folder_id is None:
        raise FileNotFoundError(f"Folder path '{folder_path}' not found on Drive.")

    # Find file in folder
    query = (
        f"name = '{filename}' and "
        f"'{folder_id}' in parents and "
        f"mimeType != 'application/vnd.google-apps.folder' and "
        f"trashed = false"
    )
    response = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
    files = response.get('files', [])
    if not files:
        raise FileNotFoundError(f"File '{filename}' not found in folder '{folder_path}'.")

    file_id = files[0]['id']

    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()

    fh.seek(0)
    import pyarrow.parquet as pq
    table = pq.read_table(fh)
    return table.to_pandas()


def delete_folder_by_path(folder_path: str, root_folder_id: str=ROOT_FOLDER_ID):
    """
    Authenticate, find folder by path relative to root_folder_id, and recursively delete it and contents.
    """
    def get_folder_id_by_path(path: str, root_id: str, service) -> str:
        if not path:
            return root_id
        folder_names = path.strip("/").split("/")
        current_id = root_id
        for name in folder_names:
            query = (
                f"mimeType='application/vnd.google-apps.folder' and "
                f"'{current_id}' in parents and "
                f"name='{name}' and trashed = false"
            )
            response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
            files = response.get('files', [])
            if not files:
                return None
            current_id = files[0]['id']
        return current_id

    def recursive_delete(service, folder_id: str):
        query = f"'{folder_id}' in parents and trashed = false"
        response = service.files().list(q=query, fields='files(id, mimeType)').execute()
        items = response.get('files', [])
        for item in items:
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                recursive_delete(service, item['id'])
            else:
                service.files().delete(fileId=item['id']).execute()
        service.files().delete(fileId=folder_id).execute()

    creds = _authenticate()
    service = build('drive', 'v3', credentials=creds)

    folder_id = get_folder_id_by_path(folder_path, root_folder_id, service)
    if folder_id is None:
        return
    recursive_delete(service, folder_id)


def append_to_parquet_on_drive(df_to_append, fpath: str, key_cols, overwrite_key_cols=True, root_folder_id=ROOT_FOLDER_ID):
    """
    Append to a Parquet file stored on Google Drive. Deduplicates based on key_cols.

    Args:
        df_to_append (pd.DataFrame): DataFrame to append.
        fpath (str): Path on Google Drive (relative to root folder), e.g. 'subdir/data.parquet'.
        key_cols (list): List of columns that define unique keys.
        overwrite_key_cols (bool): If True, overwrite matching keys. If False, keep first.
        root_folder_id (str): Google Drive folder ID where the path starts.
    """
    from googleapiclient.discovery import build
    import pandas as pd
    import os

    # --- Auth & Setup
    creds = _authenticate()
    service = build('drive', 'v3', credentials=creds)
    parent_path, fname = os.path.split(fpath)

    # --- Resolve folder ID
    st.write(parent_path, root_folder_id, service)
    folder_id = _get_folder_id_by_path(parent_path, root_folder_id, service)
    assert folder_id is not None, f"Folder path '{parent_path}' not found on Drive."

    # --- Download existing file
    try:
        df_existing = download_parquet_from_drive(fpath, root_folder_id)
    except FileNotFoundError:
        df_existing = pd.DataFrame(columns=df_to_append.columns)

    # --- Deduplication assertions
    def assert_no_duplicate_keys(df, keys):
        if df.duplicated(keys).any():
            raise ValueError("Duplicate keys found")

    def assert_no_duplicate_columns(df):
        if df.columns.duplicated().any():
            raise ValueError("Duplicate columns found")

    assert_no_duplicate_keys(df_to_append, key_cols)
    assert_no_duplicate_columns(df_to_append)
    assert_no_duplicate_keys(df_existing, key_cols)
    assert_no_duplicate_columns(df_existing)

    # --- Append and deduplicate
    df_combined = pd.concat([df_existing, df_to_append], axis=0)
    df_combined = df_combined[~df_combined.duplicated(key_cols, keep="last" if overwrite_key_cols else "first")]

    assert_no_duplicate_keys(df_combined, key_cols)
    assert_no_duplicate_columns(df_combined)

    # --- Save locally
    local_temp_path = "_temp_append.parquet"
    df_combined.to_parquet(local_temp_path, index=False)

    # --- Upload back to Drive
    file_id = upload_file_to_drive(local_temp_path, fpath, root_folder_id)

    os.remove(local_temp_path)

    return file_id




if __name__ == '__main__':
    import pandas as pd
    delete_folder_by_path("test/")

    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"]
    })
    st.write("Source data")
    st.write(df)

    upload_parquet_to_drive(df, "test/output.parquet")
    del df

    df = download_parquet_from_drive("test/output.parquet")
    st.write("Parquet target")
    st.write(df)

    df = pd.DataFrame({
        "a": [1],
        "b": ["xxxxxxxxxxxxxxxxxxxx"]
    })
    append_to_parquet_on_drive(df, "test/output.parquet", ["a"], overwrite_key_cols=True)
    del df

    df = download_parquet_from_drive("test/output.parquet")
    st.write("Parquet target after append")
    st.write(df)

    delete_folder_by_path("test/")
    delete_folder_by_path("test/")
