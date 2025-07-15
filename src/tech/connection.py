import httplib2
import apiclient.discovery
from oauth2client.service_account import ServiceAccountCredentials
from typing import List


def extract_data(CRED_FILE_PATH: str, SERVICES: List[str]) -> apiclient.discovery.Resource:
    """Gets data from google sheets

    Args:
        CRED_FILE_PATH (str): path to json with google sheets creds which were recieved while creating api key.
        SERVICES (list[str]): list of urls to google services.

    Returns:
        apiclient.discovery.Resource: special format that could be returned with api, such decision was made with
            necessity of taking different sheets.
    """
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        CRED_FILE_PATH,
        SERVICES
    )
    http_auth = credentials.authorize(httplib2.Http())
    service = apiclient.discovery.build('sheets', 'v4', http=http_auth)

    return service.spreadsheets().values()
