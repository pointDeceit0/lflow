"""
"""
import sys
import configparser
import time
import visualization.plots as pl
import tech.data as dt
import tech.connection as con
import tech.launch as launch
import tech.sending as snd


kwargs = {
    "habbits_radar": {'exclude': ('creatine', 'was_in_university')},
    "habbits_linear": {'grannulation': 'week',
                       'exclude': ('creatine', 'was_in_university'),
                       'number_of_weeks': 10},
    "nutrition_heatmap": None,
}

unions = {
    'frequencies': ['habbits_radar', 'habbits_linear']
}


def main(CRED_FILE_PATH, SERVICES, SPREADSHEET_ID):
    print('-------------------------------------')
    start1 = time.time()
    resource = con.extract_data(CRED_FILE_PATH, SERVICES)
    t1 = time.time() - start1
    print(f'---> Data was extracted for {round(t1, 2)} s.')
    start2 = time.time()
    df, metadata, trains = dt.transform_enrichment_data(resource, SPREADSHEET_ID)
    t2 = time.time() - start2
    print(f'---> Data was transformed for {round(t2, 2)} s.')
    start3 = time.time()
    snd.send_messages(
        bot_token,
        chat_id,
        launch.launch_plot_functions(functions, df, metadata, trains, **kwargs),
        functions
    )
    t3 = time.time() - start3
    print(f'---> Report was sent for {round(t3, 2)} s.\n')
    print(f'Total execution time is {round(t1 + t2 + t3, 2)} s.')
    print('-------------------------------------')


if __name__ == '__main__':
    bot_token = sys.argv[1]
    chat_id = sys.argv[2]

    config = configparser.ConfigParser()
    config.read("/abc/py_conf/global_config.cfg")
    SPREADSHEET_ID = str(config["GOOGLE_SPREADSHEETS"]["lflow"])
    CRED_FILE_PATH = r'C:\abc\creds_googledrive_lflow.json'

    SERVICES = ['https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive']

    # check what functions are exist and add them
    # functions is something like [[func, func, ...], func, ...]
    functions = []
    for f in sys.argv[3:]:
        if (x := unions.get(f, False)):
            functions.append([getattr(pl, subf) for subf in x])
        elif f in dir(pl):
            functions.append(getattr(pl, f))
        else:
            print(f"Warning! Function \"{f}\" doesn't exist!")
    print()

    if functions:
        main(CRED_FILE_PATH, SERVICES, SPREADSHEET_ID)
    else:
        print('No valid functions recieved.')
