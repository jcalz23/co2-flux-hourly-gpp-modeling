
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials

import os.path
import json

class GoogleDriveClient:
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    def __init__(self, cred_folder):
        self.creds = None
        self.client = None
        if os.path.exists(cred_folder+'\\token.json'):
            creds = Credentials.from_authorized_user_file(cred_folder+'\\token.json', self.SCOPES)
       
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    cred_folder+'\\credentials.json', self.SCOPES)
                creds = flow.run_local_server(port=8001)
            # Save the credentials for the next run
            with open(cred_folder+'\\token.json', 'w') as token:
                token.write(creds.to_json())
        
        # initializer Google API service client
        self.client = build('drive', 'v3', credentials=creds)
    
    def getFolder(self, folder_name, verbose=True):
        queryFolder = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}'"
        results = self.client.files().list(q=queryFolder, pageSize=10, fields="nextPageToken, files(id, name)").execute()
        folders = results.get('files', [])
        if not folders:
            print('No folder found.')
        else:
            folder = folders[0]
            if verbose:
                print(f'Retrieved `{folder["name"]}` folder metadata.')
            return folder
    
    def getFilesInFolder(self, folder, verbose=True):
        queryPageSize = 512
        queryFields = "nextPageToken, files(id, name, size)"
        folder_id = folder["id"]
        query = f"'{folder_id}' in parents and trashed = false"
        results = self.client.files().list(q=query, pageSize=queryPageSize, fields=queryFields).execute()
        files = results.get('files', [])
        
        if not files:
            print('No files found.')
        else:
            fileCount = 0
            if verbose:
                for file in files:
                    print(f'{fileCount+1:3}. {file["name"]} ({file["size"]:>10} bytes)')
                
            return files
    
    def downloadFile(self, file, local_filename, verbose=True):
        request = self.client.files().get_media(fileId=file["id"])
        file_handle = request.execute()
        with open(local_filename, "wb") as f:
            f.write(file_handle)
            if verbose:
                print(f'Downloaded "{file["name"]}" to "{local_filename}".')