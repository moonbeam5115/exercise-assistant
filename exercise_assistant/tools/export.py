from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def connect_to_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
    drive = GoogleDrive(gauth)
    fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file in fileList:
      print('Title: %s, ID: %s' % (file['title'], file['id']))
      # Get the folder ID that you want
      if(file['title'] == "To Share"):
          fileID = file['id']

#connect_to_drive()


print("success!")

