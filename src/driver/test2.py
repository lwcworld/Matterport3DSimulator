import os
import errno

def create_dir(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

dir_log = 'datalog_roomnav/test3'

create_dir(dir_log)


print(os.getcwd())