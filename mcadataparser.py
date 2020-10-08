"""
Data parser for Amptek PX5 MCA output files.

So far the module consist of a single class, parsing is performed by calling a proper
method on an instance and providing either a file or a directory path as their parameter.

To extract data, the target directory/filepath must contain the extension ".mca".
A precaution has been taken to avoid duplicite data by ignoring the "v.mca" substring
"""

import os.path
from os import walk


class McaDataParser:

    def __init__(self):
        self.data = []

    def parse_from_mca_file(self, filepath):
        initial_string = "<<DATA>>"
        terminal_string = "<<END>>"
        parse_flag = False
        parsed_data = []

        if os.path.isfile(filepath) and ".mca" in filepath and "v.mca" not in filepath:
            with open(filepath) as mca_file:
                for line in mca_file:
                    if initial_string in line:
                        parse_flag = True
                        continue
                    elif terminal_string in line:
                        break
                    if parse_flag:
                        parsed_data.append(int(line))
        else:
            print("Provided filepath is invalid")
        if parsed_data:
            self.data.append(parsed_data)

    def get_data_from_directory(self, directory_path):
        files = []
        for root, directories, filenames in walk(directory_path):
            for name in filenames:
                if ".mca" in name and "v.mca" not in name:
                    files.append(directory_path + '\\' + name)
        for file in files:
            self.parse_from_mca_file(file)
