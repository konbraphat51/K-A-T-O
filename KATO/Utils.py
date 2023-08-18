import pathlib

class Consts:
    data_folder = pathlib.Path(__file__).parent / "data"
    
class Utils:
    def make_transcription_file_name(index, model):
        return str(index) + "-" + str(model) + ".csv"