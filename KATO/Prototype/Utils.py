import pathlib

class Consts:
    data_folder = "KATO\\Prototype\\Data\\"
    
Consts.data_folder = pathlib.Path(__file__).parent.parent / "Data"

class Utils:
    def make_transcription_file_name(index, model):
        return str(index) + "-" + str(model) + ".csv"